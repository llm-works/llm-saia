# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Tests for parse retry via IterationGuard."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import pytest

from llm_saia import (
    IterationContext,
    IterationGuard,
    StructuredOutputError,
    TruncatedResponseError,
)
from llm_saia.core.backend import ChatResponse
from llm_saia.core.config import Config
from llm_saia.core.conversation import Message, Role, ToolCall
from llm_saia.core.logger import NullLogger
from llm_saia.core.types import ToolDef
from llm_saia.guards import schema_retry
from llm_saia.verbs import Extract
from tests.unit.conftest import MockBackend

pytestmark = pytest.mark.unit


class SequencedMockBackend(MockBackend):
    """Mock backend that returns a sequence of responses and records each call's messages."""

    def __init__(self) -> None:
        super().__init__()
        self._response_sequence: list[ChatResponse] = []
        self._sequence_index = 0
        self.calls: list[list[Message]] = []

    @property
    def call_count(self) -> int:
        """Number of chat calls made to this backend."""
        return self._sequence_index

    def queue_response(
        self,
        content: str,
        finish_reason: str | None = None,
        output_tokens: int = 0,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        """Queue a response."""
        self._response_sequence.append(
            ChatResponse(
                content=content,
                tool_calls=tool_calls or [],
                finish_reason=finish_reason,
                output_tokens=output_tokens,
            )
        )

    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        context: dict[str, Any] | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> ChatResponse:
        """Return next queued response."""
        if self._sequence_index >= len(self._response_sequence):
            raise RuntimeError("No more responses queued")
        self.calls.append(list(messages))
        response = self._response_sequence[self._sequence_index]
        self._sequence_index += 1
        return response


@dataclass
class Person:
    """Simple test schema."""

    name: str
    age: int


@dataclass
class Member:
    """Nested item for field-path tests."""

    name: str
    role: Literal["lead", "dev"]


@dataclass
class Team:
    """Schema with a list of nested dataclasses."""

    members: list[Member]


def _make_config(backend: MockBackend, json_parser: Callable[[str], Any] | None = None) -> Config:
    """Create a Config with the given backend."""
    return Config(
        lg=NullLogger(),
        backend=backend,
        tools=[],
        executor=None,
        json_parser=json_parser,
    )


async def _first_feedback(
    content: str,
    schema: type = Person,
    finish_reason: str | None = None,
    output_tokens: int = 0,
    json_parser: Callable[[str], Any] | None = None,
) -> str:
    """Return the retry feedback sent after ``content`` fails to parse."""
    backend = SequencedMockBackend()
    backend.queue_response(content, finish_reason=finish_reason, output_tokens=output_tokens)
    backend.queue_response('{"name": "Alice", "age": 30, "members": []}')
    extract = Extract(_make_config(backend, json_parser)).with_guard(schema_retry(max_retries=1))
    await extract("Extract it", schema)
    return backend.calls[1][-1].content


class TestIterationContextForParse:
    """Test IterationContext in parse retry context."""

    def test_iteration_context_properties(self) -> None:
        """IterationContext provides expected properties."""
        error = StructuredOutputError(
            "Invalid JSON",
            raw_content='{"name": "test"',
            schema_name="Person",
            parse_error="Unterminated string",
        )
        response = ChatResponse(content='{"name": "test"', tool_calls=[])
        ctx = IterationContext(
            response=response,
            iteration=1,
            max_iterations=3,
            parse_error=error,
        )
        assert ctx.iteration == 1
        assert ctx.max_iterations == 3
        assert ctx.remaining == 2
        assert ctx.parse_error is error
        assert ctx.parse_error.parse_error == "Unterminated string"

    def test_iteration_context_remaining_unlimited(self) -> None:
        """IterationContext.remaining returns UNLIMITED when max_iterations=0."""
        from llm_saia.core.guard import UNLIMITED

        response = ChatResponse(content="test", tool_calls=[])
        ctx = IterationContext(
            response=response,
            iteration=5,
            max_iterations=0,  # unlimited
            parse_error=None,
        )
        assert ctx.remaining == UNLIMITED


class TestSchemaRetryGuard:
    """Test the built-in schema_retry guard."""

    def test_schema_retry_defaults(self) -> None:
        """schema_retry has expected defaults."""
        guard = schema_retry()
        assert guard.parse_max_retries == 2
        assert guard.name == "schema_retry"

    def test_schema_retry_custom_retries(self) -> None:
        """schema_retry accepts custom max_retries."""
        guard = schema_retry(max_retries=5)
        assert guard.parse_max_retries == 5

    def test_iteration_guard_rejects_negative_parse_max_retries(self) -> None:
        """IterationGuard raises ValueError for negative parse_max_retries."""
        with pytest.raises(ValueError, match="parse_max_retries must be >= 0"):
            IterationGuard(validator=lambda ctx: None, parse_max_retries=-1)

    def test_schema_retry_returns_none_without_parse_error(self) -> None:
        """schema_retry does nothing in tool loop context (no parse_error)."""
        guard = schema_retry()
        response = ChatResponse(content="some text", tool_calls=[])
        ctx = IterationContext(
            response=response,
            iteration=0,
            max_iterations=10,
            parse_error=None,  # Not a parse error
        )
        # Should return None (no feedback) when not in parse context
        assert guard.validator(ctx) is None

    def test_schema_retry_returns_feedback_on_parse_error(self) -> None:
        """schema_retry returns feedback when parse_error is set."""
        guard = schema_retry()
        error = StructuredOutputError(
            "Invalid JSON", raw_content="not json", parse_error="Syntax error"
        )
        response = ChatResponse(content="not json", tool_calls=[])
        ctx = IterationContext(
            response=response,
            iteration=0,
            max_iterations=3,
            parse_error=error,
        )
        feedback = guard.validator(ctx)
        assert feedback is not None
        assert "could not be parsed" in feedback
        assert "Syntax error" in feedback

    def test_schema_retry_escalates_instruction(self) -> None:
        """schema_retry says earlier attempts failed from the second retry on."""
        guard = schema_retry()
        error = StructuredOutputError(
            "Invalid JSON", raw_content="not json", parse_error="Invalid JSON"
        )
        response = ChatResponse(content="not json", tool_calls=[])

        ctx1 = IterationContext(response=response, iteration=0, max_iterations=3, parse_error=error)
        instr1 = guard.validator(ctx1)
        assert instr1 is not None
        assert "Earlier attempts" not in instr1

        ctx2 = IterationContext(response=response, iteration=1, max_iterations=3, parse_error=error)
        instr2 = guard.validator(ctx2)
        assert instr2 is not None
        assert "Earlier attempts failed as well (2 so far)" in instr2

    def test_schema_retry_without_escalation(self) -> None:
        """escalate=False gives the same feedback on every attempt."""
        guard = schema_retry(escalate=False)
        error = StructuredOutputError("Invalid JSON", parse_error="Invalid JSON")
        response = ChatResponse(content="not json", tool_calls=[])
        ctx = IterationContext(response=response, iteration=1, max_iterations=3, parse_error=error)
        feedback = guard.validator(ctx)
        assert feedback is not None
        assert "Earlier attempts" not in feedback

    def test_schema_retry_returns_feedback_on_any_attempt(self) -> None:
        """schema_retry returns feedback on any attempt (caller handles bounds)."""
        guard = schema_retry()
        error = StructuredOutputError("Invalid JSON", raw_content="not json")
        response = ChatResponse(content="not json", tool_calls=[])
        # Even on last attempt, guard returns feedback - caller decides whether to use it
        ctx = IterationContext(
            response=response,
            iteration=2,  # Last attempt (max_iterations - 1)
            max_iterations=3,
            parse_error=error,
        )
        # Guard always provides feedback; caller handles iteration bounds
        feedback = guard.validator(ctx)
        assert feedback is not None
        assert "Earlier attempts" in feedback  # Escalated message on attempt > 0


class TestParseGuardIntegration:
    """Integration tests for parse guards with verbs."""

    @pytest.mark.asyncio
    async def test_parse_retry_on_invalid_json(self) -> None:
        """schema_retry retries on invalid JSON and succeeds."""
        backend = SequencedMockBackend()
        # First response: invalid JSON
        backend.queue_response('{"name": "Alice"')  # Missing closing brace
        # Second response: valid JSON
        backend.queue_response('{"name": "Alice", "age": 30}')

        config = _make_config(backend)
        extract = Extract(config).with_guard(schema_retry(max_retries=2))

        result = await extract("Extract person", Person)
        assert result.value.name == "Alice"
        assert result.value.age == 30
        assert backend.call_count == 2

    @pytest.mark.asyncio
    async def test_parse_retry_on_schema_mismatch(self) -> None:
        """schema_retry retries when JSON is valid but doesn't match schema."""
        backend = SequencedMockBackend()
        # First response: valid JSON but missing required field (no 'age')
        backend.queue_response('{"name": "Alice"}')
        # Second response: valid JSON with all required fields
        backend.queue_response('{"name": "Alice", "age": 30}')

        config = _make_config(backend)
        extract = Extract(config).with_guard(schema_retry(max_retries=2))

        result = await extract("Extract person", Person)
        assert result.value.name == "Alice"
        assert result.value.age == 30
        assert backend.call_count == 2

    @pytest.mark.asyncio
    async def test_parse_retry_exhausted(self) -> None:
        """schema_retry raises after exhausting retries."""
        backend = SequencedMockBackend()
        # All responses invalid
        backend.queue_response("not json at all")
        backend.queue_response("still not json")
        backend.queue_response("nope")

        config = _make_config(backend)
        extract = Extract(config).with_guard(schema_retry(max_retries=2))

        with pytest.raises(StructuredOutputError):
            await extract("Extract person", Person)
        # 1 initial + 2 retries = 3 attempts
        assert backend.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_without_guard(self) -> None:
        """Without parse guard, invalid JSON fails immediately."""
        backend = SequencedMockBackend()
        backend.queue_response('{"name": "Alice"')  # Invalid JSON

        config = _make_config(backend)
        extract = Extract(config)  # No guard

        with pytest.raises(StructuredOutputError):
            await extract("Extract person", Person)
        assert backend.call_count == 1  # No retry

    @pytest.mark.asyncio
    async def test_multiple_parse_guards_stack_retries(self) -> None:
        """Multiple parse guards stack their parse_max_retries."""
        backend = SequencedMockBackend()
        # Queue 4 invalid responses, then valid
        for _ in range(4):
            backend.queue_response("invalid")
        backend.queue_response('{"name": "Bob", "age": 25}')

        config = _make_config(backend)
        # Two guards with 2 retries each = 4 total retries = 5 attempts max
        extract = Extract(config).with_guards(
            schema_retry(max_retries=2),
            schema_retry(max_retries=2),
        )

        result = await extract("Extract person", Person)
        assert result.value.name == "Bob"
        assert backend.call_count == 5


class TestCustomIterationGuardForParse:
    """Test custom iteration guard implementations for parse retry."""

    @pytest.mark.asyncio
    async def test_custom_guard_can_reject_retry(self) -> None:
        """Custom guard can decline to retry based on error content."""

        def only_retry_truncated(ctx: IterationContext) -> str | None:
            # Only retry if in parse context AND error looks like truncation
            if ctx.parse_error is None:
                return None  # Not in parse context
            parse_err = getattr(ctx.parse_error, "parse_error", None) or ""
            if "Unterminated" in parse_err:
                return "Your response was truncated. Please complete it."
            return None  # Don't retry other JSON errors

        guard = IterationGuard(
            validator=only_retry_truncated,
            name="truncation_retry",
            parse_max_retries=2,
        )

        backend = SequencedMockBackend()
        # First response: not truncated, just invalid JSON (different error type)
        backend.queue_response("not valid json at all")  # Syntax error, not truncation

        config = _make_config(backend)
        extract = Extract(config).with_guard(guard)

        # Should fail because guard only retries truncation errors
        with pytest.raises(StructuredOutputError):
            await extract("Extract person", Person)
        assert backend.call_count == 1  # No retry attempted


_RUNAWAY = '{"name": "' + "la " * 400
_LONG_ANSWER = '{"name": "Alice", "bio": "' + " ".join(f"word{i}" for i in range(400))


class TestRetryInput:
    """A retry starts from the original prompt; failed answers are never resent."""

    @pytest.mark.asyncio
    async def test_cut_off_retry_sends_prompt_and_feedback_only(self) -> None:
        backend = SequencedMockBackend()
        backend.queue_response(_RUNAWAY, finish_reason="length", output_tokens=1200)
        backend.queue_response('{"name": "Alice", "age": 30}')
        extract = Extract(_make_config(backend)).with_guard(schema_retry(max_retries=1))

        await extract("Extract person", Person)

        first, retry = backend.calls
        assert retry[:-1] == first
        assert retry[-1].role == Role.USER
        assert all(m.role != Role.ASSISTANT for m in retry)

    @pytest.mark.asyncio
    async def test_malformed_answer_is_not_resent(self) -> None:
        backend = SequencedMockBackend()
        backend.queue_response('{"name": "Alice" "age": 30}')
        backend.queue_response('{"name": "Alice", "age": 30}')
        extract = Extract(_make_config(backend)).with_guard(schema_retry(max_retries=1))

        await extract("Extract person", Person)

        first, retry = backend.calls
        assert retry[:-1] == first
        assert all('"Alice" "age"' not in m.content for m in retry[:-1])

    @pytest.mark.asyncio
    async def test_each_retry_replaces_the_previous_feedback(self) -> None:
        backend = SequencedMockBackend()
        backend.queue_response(_RUNAWAY, finish_reason="length", output_tokens=1200)
        backend.queue_response('{"name": "Alice" "age": 30}')
        backend.queue_response('{"name": "Alice", "age": 30}')
        extract = Extract(_make_config(backend)).with_guard(schema_retry(max_retries=2))

        await extract("Extract person", Person)

        first, retry1, retry2 = backend.calls
        assert len(retry1) == len(retry2) == len(first) + 1
        assert retry2[:-1] == first
        assert retry2[-1].content != retry1[-1].content

    @pytest.mark.asyncio
    async def test_tool_exchange_after_feedback_is_kept(self) -> None:
        """A tool call made during a retry survives the next retry; failed answers do not."""
        calls: list[str] = []

        async def executor(name: str, args: dict[str, Any]) -> Any:
            calls.append(name)
            return {"found": True}

        backend = SequencedMockBackend()
        backend.queue_response(_RUNAWAY, finish_reason="length", output_tokens=1200)
        lookup = ToolCall(id="t1", name="lookup", arguments={})
        backend.queue_response("", tool_calls=[lookup])
        backend.queue_response('{"name": "Alice" "age": 30}')
        backend.queue_response('{"name": "Alice", "age": 30}')
        config = Config(
            lg=NullLogger(),
            backend=backend,
            tools=[ToolDef(name="lookup", description="Look up.", parameters={"type": "object"})],
            executor=executor,
        )
        # Tool rounds and parse attempts share the loop's iteration budget.
        extract = Extract(config).with_max_iterations(4).with_guard(schema_retry(max_retries=2))

        await extract("Extract person", Person)

        _, _, after_tool, last = backend.calls
        assert last[: len(after_tool)] == after_tool
        assert len(last) == len(after_tool) + 1
        assert last[-1].role == Role.USER
        assert any(m.role == Role.TOOL for m in last)
        failed = (_RUNAWAY, '{"name": "Alice" "age": 30}')
        assert all(m.content not in failed for m in last)
        assert calls == ["lookup"]


class TestCutOffAnswers:
    """finish_reason marks an answer cut off at the token limit."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
    async def test_exhausted_retries_raise_truncated(self, finish_reason: str) -> None:
        backend = SequencedMockBackend()
        for _ in range(2):
            backend.queue_response(_RUNAWAY, finish_reason=finish_reason, output_tokens=1200)
        extract = Extract(_make_config(backend)).with_guard(schema_retry(max_retries=1))

        with pytest.raises(TruncatedResponseError):
            await extract("Extract person", Person)

    @pytest.mark.asyncio
    async def test_cut_off_without_retry_guard_raises_truncated(self) -> None:
        backend = SequencedMockBackend()
        backend.queue_response('{"name": "Alice", "age": 3', finish_reason="length")

        with pytest.raises(TruncatedResponseError):
            await Extract(_make_config(backend))("Extract person", Person)

    @pytest.mark.asyncio
    async def test_answer_at_limit_that_parses_is_accepted(self) -> None:
        backend = SequencedMockBackend()
        backend.queue_response('{"name": "Alice", "age": 30}', finish_reason="length")

        result = await Extract(_make_config(backend))("Extract person", Person)

        assert result.value.age == 30

    @pytest.mark.asyncio
    async def test_stopped_answer_is_not_called_truncated(self) -> None:
        """A model that stopped on its own mid-JSON gets a syntax error, not a truncation."""
        backend = SequencedMockBackend()
        backend.queue_response('{"name": "Alice", "age": 3', finish_reason="stop")

        with pytest.raises(StructuredOutputError) as exc:
            await Extract(_make_config(backend))("Extract person", Person)
        assert not isinstance(exc.value, TruncatedResponseError)

    @pytest.mark.asyncio
    async def test_unknown_finish_reason_falls_back_to_content_check(self) -> None:
        backend = SequencedMockBackend()
        backend.queue_response('{"name": "Alice", "age": 3')

        with pytest.raises(TruncatedResponseError):
            await Extract(_make_config(backend))("Extract person", Person)

    @pytest.mark.asyncio
    async def test_unknown_finish_reason_cut_inside_string(self) -> None:
        """An answer that ends inside a string counts as cut off."""
        backend = SequencedMockBackend()
        backend.queue_response('{"name": "Al')

        with pytest.raises(TruncatedResponseError):
            await Extract(_make_config(backend))("Extract person", Person)

    @pytest.mark.asyncio
    async def test_unknown_finish_reason_cut_after_comma(self) -> None:
        """An object cut off before its next key counts as cut off too."""
        backend = SequencedMockBackend()
        backend.queue_response('{"name": "Alice",')

        with pytest.raises(TruncatedResponseError):
            await Extract(_make_config(backend))("Extract person", Person)


class TestRetryFeedback:
    """Feedback names what went wrong in the failed answer."""

    @pytest.mark.asyncio
    async def test_cut_off_while_repeating(self) -> None:
        feedback = await _first_feedback(_RUNAWAY, finish_reason="length", output_tokens=1200)
        assert "1200 tokens" in feedback
        assert "repeating" in feedback
        assert "la " in feedback

    @pytest.mark.asyncio
    async def test_cut_off_long_answer(self) -> None:
        feedback = await _first_feedback(_LONG_ANSWER, finish_reason="length", output_tokens=1200)
        assert "1200 tokens" in feedback
        assert "Keep every item" in feedback
        assert "shorter string values" in feedback
        assert "fewer" not in feedback
        assert "repeating" not in feedback

    @pytest.mark.asyncio
    async def test_empty_answer(self) -> None:
        feedback = await _first_feedback("   ")
        assert "empty" in feedback

    @pytest.mark.asyncio
    async def test_code_fence(self) -> None:
        feedback = await _first_feedback('```json\n{"name": "Alice", "age": 30}\n```')
        assert "code block" in feedback

    @pytest.mark.asyncio
    async def test_text_before_json(self) -> None:
        feedback = await _first_feedback('Here is the person: {"name": "Alice", "age": 30}')
        assert "Here is the person" in feedback

    @pytest.mark.asyncio
    async def test_text_after_json(self) -> None:
        feedback = await _first_feedback('{"name": "Alice", "age": 30} Hope this helps')
        assert "after the JSON" in feedback
        assert "Hope this helps" in feedback

    @pytest.mark.asyncio
    async def test_syntax_error_shows_the_spot(self) -> None:
        feedback = await _first_feedback('{"name": "Alice" "age": 30}')
        lines = feedback.splitlines()
        excerpt = next(line for line in lines if '"Alice" "age"' in line)
        caret = lines[lines.index(excerpt) + 1]
        assert caret.strip() == "^"
        assert excerpt[caret.index("^")] == '"'
        assert "line 1, column 18" in feedback

    @pytest.mark.asyncio
    async def test_syntax_error_excerpt_is_bounded(self) -> None:
        padding = ", ".join(f'"k{i}": {i}' for i in range(2000))
        feedback = await _first_feedback('{"name": "Alice", ' + padding + ' "age" 30}')
        assert len(feedback) < 800

    @pytest.mark.asyncio
    async def test_wrong_top_level_type(self) -> None:
        feedback = await _first_feedback("[1, 2]")
        assert "Expected dict, got list" in feedback

    @pytest.mark.asyncio
    async def test_missing_field(self) -> None:
        feedback = await _first_feedback('{"name": "Alice"}')
        assert "missing required field 'age'" in feedback

    @pytest.mark.asyncio
    async def test_bad_value_names_the_field(self) -> None:
        content = '{"members": [{"name": "a", "role": "dev"}, {"name": "b", "role": "boss"}]}'
        feedback = await _first_feedback(content, schema=Team)
        assert "members[1].role" in feedback
        assert "'boss'" in feedback

    @pytest.mark.asyncio
    async def test_pydantic_error_names_the_field(self) -> None:
        pydantic = pytest.importorskip("pydantic")

        # create_model keeps real types; class bodies here would hold strings
        # (future annotations) that get_type_hints cannot resolve for local classes.
        p_member = pydantic.create_model(
            "PMember", name=(str, ...), role=(Literal["lead", "dev"], ...)
        )
        p_team = pydantic.create_model("PTeam", members=(list[p_member], ...))  # type: ignore[valid-type]
        content = '{"members": [{"name": "a", "role": "dev"}, {"name": "b", "role": "boss"}]}'
        feedback = await _first_feedback(content, schema=p_team)
        assert "members[1].role" in feedback
        assert "'boss'" in feedback

    @pytest.mark.asyncio
    async def test_custom_parser_error_is_passed_on(self) -> None:
        calls = 0

        def parser(text: str) -> Any:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("custom parser rejected it")
            return {"name": "Alice", "age": 30}

        feedback = await _first_feedback("whatever", json_parser=parser)
        assert "custom parser rejected it" in feedback

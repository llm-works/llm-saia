"""Tests for parse retry via IterationGuard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from llm_saia import IterationContext, IterationGuard, StructuredOutputError
from llm_saia.core.backend import AgentResponse
from llm_saia.core.config import Config
from llm_saia.core.conversation import Message
from llm_saia.core.logger import NullLogger
from llm_saia.core.types import ToolDef
from llm_saia.guards import schema_retry
from llm_saia.verbs import Extract
from tests.unit.conftest import MockBackend

pytestmark = pytest.mark.unit


class SequencedMockBackend(MockBackend):
    """Mock backend that can return a sequence of responses."""

    def __init__(self) -> None:
        super().__init__()
        self._response_sequence: list[str] = []
        self._sequence_index = 0

    @property
    def call_count(self) -> int:
        """Number of chat calls made to this backend."""
        return self._sequence_index

    def queue_response(self, content: str) -> None:
        """Queue a response."""
        self._response_sequence.append(content)

    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AgentResponse:
        """Return next queued response."""
        if self._sequence_index >= len(self._response_sequence):
            raise RuntimeError("No more responses queued")
        content = self._response_sequence[self._sequence_index]
        self._sequence_index += 1
        return AgentResponse(content=content, tool_calls=[])


@dataclass
class Person:
    """Simple test schema."""

    name: str
    age: int


def _make_config(backend: MockBackend) -> Config:
    """Create a Config with the given backend."""
    return Config(
        lg=NullLogger(),
        backend=backend,
        tools=[],
        executor=None,
    )


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
        response = AgentResponse(content='{"name": "test"', tool_calls=[])
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

        response = AgentResponse(content="test", tool_calls=[])
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
        response = AgentResponse(content="some text", tool_calls=[])
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
        response = AgentResponse(content="not json", tool_calls=[])
        ctx = IterationContext(
            response=response,
            iteration=0,
            max_iterations=3,
            parse_error=error,
        )
        feedback = guard.validator(ctx)
        assert feedback is not None
        assert "not valid JSON" in feedback
        assert "Syntax error" in feedback

    def test_schema_retry_escalates_instruction(self) -> None:
        """schema_retry escalates on repeated failures."""
        guard = schema_retry()
        error = StructuredOutputError(
            "Invalid JSON", raw_content="not json", parse_error="Invalid JSON"
        )
        response = AgentResponse(content="not json", tool_calls=[])

        # First attempt - polite
        ctx1 = IterationContext(response=response, iteration=0, max_iterations=3, parse_error=error)
        instr1 = guard.validator(ctx1)
        assert instr1 is not None
        assert "not valid JSON" in instr1
        assert "FAILED" not in instr1

        # Second attempt - escalated
        ctx2 = IterationContext(response=response, iteration=1, max_iterations=3, parse_error=error)
        instr2 = guard.validator(ctx2)
        assert instr2 is not None
        assert "FAILED" in instr2
        assert "2 TIMES" in instr2

    def test_schema_retry_returns_feedback_on_any_attempt(self) -> None:
        """schema_retry returns feedback on any attempt (caller handles bounds)."""
        guard = schema_retry()
        error = StructuredOutputError("Invalid JSON", raw_content="not json")
        response = AgentResponse(content="not json", tool_calls=[])
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
        assert "FAILED" in feedback  # Escalated message on attempt > 0


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

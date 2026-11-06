# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Structured output with parse retries, driven through the public API.

Every scenario runs a structured call against a :class:`StrictBackend` and
checks the same rules: no request breaks a provider rule, resends a failed
answer, drops a tool result or piles up retries; each retry's feedback names
what was wrong with the answer before it; each tool runs once per call; the
configured number of parse attempts is available whatever tool rounds precede
them; and a caller's conversation receives only the prompt and the final
answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pydantic
import pytest

from llm_saia import (
    SAIA,
    IterationContext,
    IterationGuard,
    OutputGuard,
    ParseFailureKind,
    PauseRequested,
    StructuredOutputError,
    TruncatedResponseError,
)
from llm_saia.core.conversation import ListConversation, Message, Role, ToolCall
from llm_saia.core.types import ToolDef
from llm_saia.guards import schema_retry

from .strict_backend import StrictBackend, Turn

pytestmark = pytest.mark.asyncio

TRUNCATED = ParseFailureKind.TRUNCATED


@dataclass
class Person:
    """Schema the scripted answers are parsed into."""

    name: str
    age: int


@dataclass
class Team:
    """Schema with a nested list, for field paths in schema errors."""

    members: list[Person]


GOOD = '{"name": "Alice", "age": 30}'
LOOKUP = ToolDef(name="lookup", description="Look something up.", parameters={"type": "object"})
PROMPT = "Extract the person."


@dataclass(frozen=True)
class Failure:
    """An answer that fails to parse, and what the retry feedback must say."""

    content: str
    kind: ParseFailureKind
    feedback: str
    finish_reason: str | None = None
    output_tokens: int = 0

    def turn(self, *also: str) -> Turn:
        """The scripted turn; the next request's feedback must carry ``also`` too."""
        return Turn(
            self.content,
            finish_reason=self.finish_reason,
            output_tokens=self.output_tokens,
            failed=True,
            feedback=(self.feedback, *also),
        )


FAILURES = {
    # Cut off, reported by the backend.
    "length_runaway": Failure(
        '{"name": "' + "la " * 400, TRUNCATED, "repeating the same text", "length", 1200
    ),
    "length_mid_value": Failure(
        '{"name": "Alice", "age": 3', TRUNCATED, "write shorter string values", "length", 900
    ),
    "max_tokens": Failure('{"name": "Alice", "age"', TRUNCATED, "(900 tokens)", "max_tokens", 900),
    "length_nothing_written": Failure("", TRUNCATED, "cut off at the output limit", "length", 4096),
    # Cut off, not reported: the content has to show it.
    "unreported_in_key": Failure('{"name"', TRUNCATED, "cut off"),
    "unreported_in_string": Failure('{"name": "Ali', TRUNCATED, "cut off"),
    "unreported_after_comma": Failure('{"name": "Alice", ', TRUNCATED, "cut off"),
    "unreported_before_value": Failure('{"name": "Alice", "age": ', TRUNCATED, "cut off"),
    "unreported_after_value": Failure('{"name": "Alice", "age": 30', TRUNCATED, "cut off"),
    # Complete JSON, then text until the limit: the JSON was not cut off.
    "length_after_json": Failure(
        GOOD + "\nTo explain:" + " more" * 300,
        ParseFailureKind.TEXT_AFTER,
        "text after the JSON",
        "length",
        1200,
    ),
    # Ended early but reported as a normal stop: malformed, not cut off.
    "stop_mid_json": Failure(
        '{"name": "Alice", ', ParseFailureKind.SYNTAX, "not valid JSON", "stop"
    ),
    "empty": Failure("", ParseFailureKind.EMPTY, "was empty", "stop"),
    "blank": Failure("  \n", ParseFailureKind.EMPTY, "was empty"),
    "code_fence": Failure(f"```json\n{GOOD}\n```", ParseFailureKind.CODE_FENCE, "code block"),
    "text_before": Failure(
        f"Here it is: {GOOD}", ParseFailureKind.TEXT_BEFORE, "started with text"
    ),
    "text_after": Failure(f"{GOOD}\nHope this helps.", ParseFailureKind.TEXT_AFTER, "text after"),
    "malformed": Failure('{"name": "Alice" "age": 30}', ParseFailureKind.SYNTAX, "not valid JSON"),
    "wrong_closer": Failure('{"name": "Alice", "age": 30]', ParseFailureKind.SYNTAX, "not valid"),
    "missing_field": Failure('{"name": "Alice"}', ParseFailureKind.SCHEMA, "'age'"),
    "top_level_array": Failure(f"[{GOOD}]", ParseFailureKind.SCHEMA, "does not match"),
}


class ToolLog:
    """Executor that records each tool call it runs."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[str] = []
        self._fail = fail

    async def __call__(self, name: str, args: dict[str, Any]) -> Any:
        self.calls.append(name)
        if self._fail:
            raise RuntimeError("lookup service unavailable")
        return {"found": True}


def _call(n: int) -> ToolCall:
    return ToolCall(id=f"t{n}", name="lookup", arguments={})


def _tool_turn(n: int, content: str = "") -> Turn:
    return Turn(content, tool_calls=[_call(n)])


def _saia(backend: StrictBackend, log: ToolLog, retries: int = 2) -> SAIA:
    """SAIA with tools configured and the default max_iterations."""
    return (
        SAIA.builder()
        .backend(backend)
        .tools([LOOKUP], log)
        .build()
        .with_guard(schema_retry(retries))
    )


def _assert_clean(backend: StrictBackend) -> None:
    assert backend.violations == []
    assert backend.remaining == 0


def _placed(placement: str, failure: Failure) -> list[Turn]:
    """Two failed answers placed among tool rounds, then a good answer."""
    first, second = failure.turn(), failure.turn("(2 so far)")
    return {
        "no_tools": [first, second, Turn(GOOD)],
        "tool_first": [_tool_turn(1), first, second, Turn(GOOD)],
        "tool_between": [first, _tool_turn(1), second, Turn(GOOD)],
        "tools_twice": [_tool_turn(1), first, _tool_turn(2), second, Turn(GOOD)],
    }[placement]


# -- every failure kind in every placement --


@pytest.mark.parametrize("kind", sorted(FAILURES))
@pytest.mark.parametrize("placement", ["no_tools", "tool_first", "tool_between", "tools_twice"])
async def test_retries_succeed(placement: str, kind: str) -> None:
    """Two failed answers, then a good one: parsed, and every request clean."""
    script = _placed(placement, FAILURES[kind])
    backend, log = StrictBackend(script), ToolLog()

    result = await _saia(backend, log).complete_structured(PROMPT, Person)

    assert result.value == Person("Alice", 30)
    _assert_clean(backend)
    assert len(log.calls) == sum(len(t.tool_calls) for t in script)


@pytest.mark.parametrize("kind", sorted(FAILURES))
@pytest.mark.parametrize("tool_rounds", [0, 1, 2])
async def test_every_parse_attempt_is_available(tool_rounds: int, kind: str) -> None:
    """schema_retry(2) gives three parse attempts however many tool rounds come first."""
    failure = FAILURES[kind]
    attempts = [failure.turn(), failure.turn("(2 so far)"), failure.turn()]
    script = [_tool_turn(n) for n in range(tool_rounds)] + attempts
    backend, log = StrictBackend(script), ToolLog()

    with pytest.raises(StructuredOutputError) as caught:
        await _saia(backend, log).complete_structured(PROMPT, Person)

    _assert_clean(backend)
    assert caught.value.failure is not None
    assert caught.value.failure.kind is failure.kind
    assert isinstance(caught.value, TruncatedResponseError) == (failure.kind is TRUNCATED)


# -- tool turns --


async def test_several_tool_calls_in_one_turn() -> None:
    both = Turn(tool_calls=[_call(1), _call(2)])
    script = [
        FAILURES["malformed"].turn(),
        both,
        FAILURES["unreported_in_string"].turn(),
        Turn(GOOD),
    ]
    backend, log = StrictBackend(script), ToolLog()

    result = await _saia(backend, log).complete_structured(PROMPT, Person)

    assert result.value == Person("Alice", 30)
    _assert_clean(backend)
    assert log.calls == ["lookup", "lookup"]


async def test_text_alongside_tool_calls() -> None:
    script = [
        _tool_turn(1, "Let me look that up."),
        FAILURES["code_fence"].turn(),
        _tool_turn(2, "One more check."),
        FAILURES["length_runaway"].turn(),
        Turn(GOOD),
    ]
    backend, log = StrictBackend(script), ToolLog()

    result = await _saia(backend, log).complete_structured(PROMPT, Person)

    assert result.value == Person("Alice", 30)
    _assert_clean(backend)


async def test_failing_tool() -> None:
    """A tool that raises still gets a result, and retries stay clean."""
    script = [_tool_turn(1), FAILURES["missing_field"].turn(), _tool_turn(2), Turn(GOOD)]
    backend, log = StrictBackend(script), ToolLog(fail=True)

    result = await _saia(backend, log).complete_structured(PROMPT, Person)

    assert result.value == Person("Alice", 30)
    _assert_clean(backend)
    tool_results = [m.content for m in backend.requests[-1] if m.role == Role.TOOL]
    assert tool_results == ["Error: lookup service unavailable"] * 2


# -- iteration guards --


async def test_blocking_guard_between_retries() -> None:
    """A blocking guard's exchange and parse retries keep every request clean."""

    def block_once(ctx: IterationContext) -> str | None:
        return "Do not say BLOCK." if ctx.response.content == "BLOCK" else None

    script = [
        FAILURES["malformed"].turn(),
        Turn("BLOCK", feedback=("Do not say BLOCK.",)),
        FAILURES["missing_field"].turn(),
        Turn(GOOD),
    ]
    backend, log = StrictBackend(script), ToolLog()
    saia = _saia(backend, log).with_guard(IterationGuard(validator=block_once, name="block"))

    result = await saia.complete_structured(PROMPT, Person)

    assert result.value == Person("Alice", 30)
    _assert_clean(backend)


async def test_blocking_guard_on_tool_calls() -> None:
    """Blocked tool calls are acknowledged, not run, and parse retries follow cleanly."""

    def no_tools(ctx: IterationContext) -> str | None:
        return "Answer without tools." if ctx.response.tool_calls else None

    script = [
        FAILURES["text_before"].turn(),
        Turn(tool_calls=[_call(1)], feedback=("Answer without tools.",)),
        FAILURES["unreported_after_comma"].turn("(2 so far)"),
        Turn(GOOD),
    ]
    backend, log = StrictBackend(script), ToolLog()
    saia = _saia(backend, log).with_guard(IterationGuard(validator=no_tools, name="no_tools"))

    result = await saia.complete_structured(PROMPT, Person)

    assert result.value == Person("Alice", 30)
    _assert_clean(backend)
    assert log.calls == []


async def test_advisory_guard_with_tools() -> None:
    """Advisory feedback after tool results and parse retries keep requests clean."""

    def explain(ctx: IterationContext) -> str | None:
        return "Say why you call a tool." if ctx.response.tool_calls else None

    script = [
        Turn(tool_calls=[_call(1)], feedback=("Say why you call a tool.",)),
        FAILURES["malformed"].turn(),
        Turn(tool_calls=[_call(2)], feedback=("Say why you call a tool.",)),
        FAILURES["length_runaway"].turn("(2 so far)"),
        Turn(GOOD),
    ]
    backend, log = StrictBackend(script), ToolLog()
    guard = IterationGuard(validator=explain, name="explain", blocking=False)
    saia = _saia(backend, log).with_guard(guard)

    result = await saia.complete_structured(PROMPT, Person)

    assert result.value == Person("Alice", 30)
    _assert_clean(backend)
    assert log.calls == ["lookup", "lookup"]


# -- budget --


async def test_unlimited_iterations_keep_the_parse_budget() -> None:
    """max_iterations=0 runs any number of tool rounds; parse attempts stay at three."""
    failure = FAILURES["malformed"]
    script = [_tool_turn(n) for n in range(6)]
    script += [failure.turn(), failure.turn(), failure.turn()]
    backend, log = StrictBackend(script), ToolLog()

    with pytest.raises(StructuredOutputError):
        await _saia(backend, log).with_max_iterations(0).complete_structured(PROMPT, Person)

    _assert_clean(backend)
    assert len(log.calls) == 6


async def test_stacked_schema_retry_guards_sum() -> None:
    """Two schema_retry(1) guards allow three attempts, with tools between."""
    failure = FAILURES["missing_field"]
    script = [failure.turn(), _tool_turn(1), failure.turn(), failure.turn()]
    backend, log = StrictBackend(script), ToolLog()
    saia = _saia(backend, log, retries=1).with_guard(schema_retry(1))

    with pytest.raises(StructuredOutputError):
        await saia.complete_structured(PROMPT, Person)

    _assert_clean(backend)


async def test_small_max_iterations_still_gets_its_retries() -> None:
    """max_iterations=1 with schema_retry(2): one answer plus two retries."""
    failure = FAILURES["code_fence"]
    script = [failure.turn(), failure.turn(), Turn(GOOD)]
    backend, log = StrictBackend(script), ToolLog()

    result = await _saia(backend, log).with_max_iterations(1).complete_structured(PROMPT, Person)

    assert result.value == Person("Alice", 30)
    _assert_clean(backend)


# -- schemas and parsers --


async def test_nested_field_path_in_feedback() -> None:
    bad = '{"members": [{"name": "A", "age": 1}, {"name": "B"}]}'
    good = '{"members": [{"name": "A", "age": 1}, {"name": "B", "age": 2}]}'
    script = [Turn(bad, failed=True, feedback=("members[1]", "'age'")), _tool_turn(1), Turn(good)]
    backend, log = StrictBackend(script), ToolLog()

    result = await _saia(backend, log).complete_structured("Extract the team.", Team)

    assert result.value == Team([Person("A", 1), Person("B", 2)])
    _assert_clean(backend)


class Adult(pydantic.BaseModel):
    """Pydantic schema with a validator."""

    name: str
    age: int = pydantic.Field(ge=18)


async def test_pydantic_validation_feedback() -> None:
    script = [
        Turn('{"name": "Bob", "age": 7}', failed=True, feedback=("age:", "(got 7)")),
        _tool_turn(1),
        Turn('{"name": "Bob", "age": 70}'),
    ]
    backend, log = StrictBackend(script), ToolLog()

    result = await _saia(backend, log).complete_structured("Extract the adult.", Adult)

    assert result.value == Adult(name="Bob", age=70)
    _assert_clean(backend)


async def test_custom_json_parser_error() -> None:
    def strict_parser(text: str) -> Any:
        if "'" in text:
            raise ValueError("single quotes are not allowed")
        return json.loads(text)

    script = [
        Turn("{'name': 'Alice', 'age': 30}", failed=True, feedback=("single quotes",)),
        Turn(GOOD),
    ]
    backend, log = StrictBackend(script), ToolLog()
    saia = _saia(backend, log).with_json_parser(strict_parser)

    result = await saia.complete_structured(PROMPT, Person)

    assert result.value == Person("Alice", 30)
    _assert_clean(backend)


# -- other entry points --


async def test_without_tools_configured() -> None:
    failure = FAILURES["unreported_in_string"]
    script = [failure.turn(), failure.turn("(2 so far)"), Turn(GOOD)]
    backend = StrictBackend(script)
    saia = SAIA.builder().backend(backend).build().with_guard(schema_retry(2))

    result = await saia.complete_structured(PROMPT, Person)

    assert result.value == Person("Alice", 30)
    _assert_clean(backend)


async def test_extract_verb() -> None:
    script = [_tool_turn(1), FAILURES["length_runaway"].turn(), _tool_turn(2), Turn(GOOD)]
    backend, log = StrictBackend(script), ToolLog()

    result = await _saia(backend, log).extract("Alice is 30.", Person)

    assert result.value == Person("Alice", 30)
    _assert_clean(backend)


async def test_output_guard_retry_with_tools() -> None:
    """An output guard's retry runs its own tool round and parses cleanly."""
    older = OutputGuard(
        validator=lambda p: "Age must be over 40." if p.age <= 40 else None,
        retry_instruction="Look up the age again.",
    )
    script = [
        FAILURES["malformed"].turn(),
        Turn(GOOD),
        _tool_turn(1),
        Turn('{"name": "A", "age": 50}'),
    ]
    backend, log = StrictBackend(script), ToolLog()

    result = await _saia(backend, log).with_guard(older).complete_structured(PROMPT, Person)

    assert result.value == Person("A", 50)
    _assert_clean(backend)


async def test_output_guard_retry_that_fails_to_parse() -> None:
    """The output guard's retry is a single parse attempt: a failure raises at once."""
    older = OutputGuard(
        validator=lambda p: "Age must be over 40." if p.age <= 40 else None,
        retry_instruction="Look up the age again.",
    )
    script = [Turn(GOOD), FAILURES["malformed"].turn()]
    backend, log = StrictBackend(script), ToolLog()

    with pytest.raises(StructuredOutputError):
        await _saia(backend, log).with_guard(older).complete_structured(PROMPT, Person)

    _assert_clean(backend)


# -- caller conversation, pause and resume --


async def test_caller_conversation_gets_prompt_and_answer_only() -> None:
    script = [
        FAILURES["length_runaway"].turn(),
        _tool_turn(1),
        FAILURES["malformed"].turn(),
        Turn(GOOD),
    ]
    backend, log = StrictBackend(script), ToolLog()
    conv = ListConversation()
    conv.append(Message(role=Role.USER, content="earlier question"))
    conv.append(Message(role=Role.ASSISTANT, content="earlier answer"))

    await _saia(backend, log).complete_structured(PROMPT, Person, conversation=conv)

    contents = [m.content for m in conv.as_messages()]
    assert contents == ["earlier question", "earlier answer", PROMPT, GOOD]
    _assert_clean(backend)


async def test_pause_and_resume_during_retries() -> None:
    """Pausing on a retry and resuming finishes cleanly, with each tool run once.

    The pause discards the answer it was handed, so no feedback follows it.
    """
    script = [
        FAILURES["length_runaway"].turn(),
        _tool_turn(1),
        Turn(FAILURES["missing_field"].content),
        Turn(GOOD),
    ]
    backend, log = StrictBackend(script), ToolLog()
    saia = _saia(backend, log)
    conv = ListConversation()

    async def pause_on_third(iteration: int, response: Any) -> None:
        if len(backend.requests) == 3:
            raise PauseRequested()

    with pytest.raises(PauseRequested):
        await saia.complete_structured(
            PROMPT, Person, conversation=conv, on_iteration=pause_on_third
        )
    result = await saia.complete_structured("", Person, conversation=conv, resume=True)

    assert result.value == Person("Alice", 30)
    _assert_clean(backend)
    assert log.calls == ["lookup"]


async def test_abort_during_retry_and_resume() -> None:
    """A call aborted right after a retry resumes without piling up retries."""
    script = [
        FAILURES["malformed"].turn(),
        Turn(abort=True),
        FAILURES["length_runaway"].turn(),
        Turn(GOOD),
    ]
    backend, log = StrictBackend(script), ToolLog()
    saia = _saia(backend, log)
    conv = ListConversation()

    with pytest.raises(PauseRequested):
        await saia.complete_structured(PROMPT, Person, conversation=conv)
    result = await saia.complete_structured("", Person, conversation=conv, resume=True)

    assert result.value == Person("Alice", 30)
    _assert_clean(backend)


async def test_pause_between_tool_calls_and_resume() -> None:
    """Pausing inside a tool batch during retries resumes cleanly."""
    script = [
        FAILURES["missing_field"].turn(),
        Turn(tool_calls=[_call(1), _call(2)]),
        Turn(GOOD),
    ]
    backend, log = StrictBackend(script), ToolLog()
    saia = _saia(backend, log)
    conv = ListConversation()

    async def pause_after_first_tool() -> bool:
        return len(log.calls) == 1

    with pytest.raises(PauseRequested):
        await saia.complete_structured(
            PROMPT, Person, conversation=conv, pause_check=pause_after_first_tool
        )
    result = await saia.complete_structured("", Person, conversation=conv, resume=True)

    assert result.value == Person("Alice", 30)
    _assert_clean(backend)
    assert log.calls == ["lookup"]


# -- trace --


async def test_trace_has_one_step_per_call() -> None:
    script = [
        _tool_turn(1),
        FAILURES["length_runaway"].turn(),
        _tool_turn(2),
        FAILURES["malformed"].turn(),
        Turn(GOOD),
    ]
    backend, log = StrictBackend(script), ToolLog()

    result = await _saia(backend, log).complete_structured(PROMPT, Person)

    steps = result.trace.steps
    assert [s.phase for s in steps] == [
        "attempt",
        "attempt",
        "attempt",
        "parse_retry",
        "parse_retry",
    ]
    assert [s.parsed for s in steps] == [True, False, True, False, True]
    assert [s.parse_error is not None for s in steps] == [False, True, False, True, False]
    _assert_clean(backend)

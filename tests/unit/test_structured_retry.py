# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Structured output with parse retries, driven through the public API.

Every scenario runs ``SAIA.complete_structured`` against a :class:`StrictBackend`
and checks the same rules: no request breaks a provider rule or resends a
failed answer, each tool runs once per call, the configured number of parse
attempts is available whatever tool rounds precede them, and a caller's
conversation receives only the prompt and the final answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from llm_saia import SAIA, IterationContext, IterationGuard, StructuredOutputError
from llm_saia.core.conversation import ListConversation, Message, Role, ToolCall
from llm_saia.core.errors import PauseRequested
from llm_saia.core.types import ToolDef
from llm_saia.guards import schema_retry

from .strict_backend import StrictBackend, Turn

pytestmark = pytest.mark.asyncio


@dataclass
class Person:
    """Schema the scripted answers are parsed into."""

    name: str
    age: int


GOOD = '{"name": "Alice", "age": 30}'
FAILURES = {
    "cut_off": Turn(
        '{"name": "' + "la " * 400, finish_reason="length", output_tokens=1200, failed=True
    ),
    "malformed": Turn('{"name": "Alice" "age": 30}', failed=True),
    "schema": Turn('{"name": "Alice"}', failed=True),
}
LOOKUP = ToolDef(name="lookup", description="Look something up.", parameters={"type": "object"})


class ToolLog:
    """Executor that records each tool call it runs."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, name: str, args: dict[str, Any]) -> Any:
        self.calls.append(name)
        return {"found": True}


def _tool_turn(n: int) -> Turn:
    return Turn(tool_calls=[ToolCall(id=f"t{n}", name="lookup", arguments={})])


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


def _failing(kind: str) -> Turn:
    turn = FAILURES[kind]
    return Turn(
        turn.content,
        finish_reason=turn.finish_reason,
        output_tokens=turn.output_tokens,
        failed=True,
    )


PLACEMENTS = {
    "no_tools": lambda f: [f(), f(), Turn(GOOD)],
    "tool_first": lambda f: [_tool_turn(1), f(), f(), Turn(GOOD)],
    "tool_between": lambda f: [f(), _tool_turn(1), f(), Turn(GOOD)],
    "tools_twice": lambda f: [_tool_turn(1), f(), _tool_turn(2), f(), Turn(GOOD)],
}


@pytest.mark.parametrize("kind", sorted(FAILURES))
@pytest.mark.parametrize("placement", sorted(PLACEMENTS))
async def test_retries_succeed(placement: str, kind: str) -> None:
    """Two failed answers, then a good one: parsed, and every request clean."""
    script = PLACEMENTS[placement](lambda: _failing(kind))
    backend, log = StrictBackend(script), ToolLog()

    result = await _saia(backend, log).complete_structured("Extract the person.", Person)

    assert result.value == Person("Alice", 30)
    _assert_clean(backend)
    assert len(log.calls) == sum(len(t.tool_calls) for t in script)


@pytest.mark.parametrize("kind", sorted(FAILURES))
@pytest.mark.parametrize("tool_rounds", [0, 1, 2])
async def test_every_parse_attempt_is_available(tool_rounds: int, kind: str) -> None:
    """schema_retry(2) gives three parse attempts however many tool rounds come first."""
    script = [_tool_turn(n) for n in range(tool_rounds)] + [_failing(kind) for _ in range(3)]
    backend, log = StrictBackend(script), ToolLog()

    with pytest.raises(StructuredOutputError):
        await _saia(backend, log).complete_structured("Extract the person.", Person)

    _assert_clean(backend)


async def test_blocking_guard_between_retries() -> None:
    """A blocking guard's exchange and parse retries keep every request clean."""

    def block_once(ctx: IterationContext) -> str | None:
        return "Do not say BLOCK." if ctx.response.content == "BLOCK" else None

    script = [_failing("malformed"), Turn("BLOCK"), _failing("schema"), Turn(GOOD)]
    backend, log = StrictBackend(script), ToolLog()
    saia = _saia(backend, log).with_guard(IterationGuard(validator=block_once, name="block"))

    result = await saia.complete_structured("Extract the person.", Person)

    assert result.value == Person("Alice", 30)
    _assert_clean(backend)


async def test_caller_conversation_gets_prompt_and_answer_only() -> None:
    script = [_failing("cut_off"), _tool_turn(1), _failing("malformed"), Turn(GOOD)]
    backend, log = StrictBackend(script), ToolLog()
    conv = ListConversation()
    conv.append(Message(role=Role.USER, content="earlier question"))
    conv.append(Message(role=Role.ASSISTANT, content="earlier answer"))

    await _saia(backend, log).complete_structured("Extract the person.", Person, conversation=conv)

    contents = [m.content for m in conv.as_messages()]
    assert contents == ["earlier question", "earlier answer", "Extract the person.", GOOD]
    _assert_clean(backend)


async def test_pause_and_resume_during_retries() -> None:
    """Pausing on a retry and resuming finishes cleanly, with each tool run once."""
    script = [_failing("cut_off"), _tool_turn(1), _failing("schema"), Turn(GOOD)]
    backend, log = StrictBackend(script), ToolLog()
    saia = _saia(backend, log)
    conv = ListConversation()

    async def pause_on_third(iteration: int, response: Any) -> None:
        if len(backend.requests) == 3:
            raise PauseRequested()

    with pytest.raises(PauseRequested):
        await saia.complete_structured(
            "Extract the person.", Person, conversation=conv, on_iteration=pause_on_third
        )
    result = await saia.complete_structured("", Person, conversation=conv, resume=True)

    assert result.value == Person("Alice", 30)
    _assert_clean(backend)
    assert log.calls == ["lookup"]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""A scripted backend that checks every request the way a strict provider would.

The backend answers from a fixed script and records each request it receives.
Every request is checked against rules real providers and chat templates
enforce, and against rules SAIA promises to keep:

- no two user messages in a row (strict chat templates reject them);
- every tool result answers a tool call of the assistant turn right before it,
  and every tool call of a turn is answered before the next turn;
- no answer the script marked as failed is ever sent back to the model;
- every request starts with a user message, and a tool result, once sent, is
  in every later request;
- an assistant message the script did not write (a stand-in for a dropped
  answer) is never followed by a user message and another such stand-in, so
  retries do not pile up;
- after a turn with ``feedback``, the next request ends with a user message
  containing each of its phrases.

Violations are collected, not raised, so a test can report all of them.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from llm_saia import Backend, ChatResponse, PauseRequested
from llm_saia.core.conversation import Message, Role, ToolCall
from llm_saia.core.types import ToolDef


@dataclass
class Turn:
    """One scripted model turn.

    Attributes:
        content: The answer text.
        tool_calls: Tool calls the model makes in this turn.
        finish_reason: Reported finish reason.
        output_tokens: Reported output token count.
        failed: The answer is expected to fail parsing and must never be sent
            back to the model.
        feedback: Phrases the user message ending the next request must contain.
        abort: Raise ``PauseRequested`` instead of answering, as a backend does
            when the call's abort signal is set.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    output_tokens: int = 0
    failed: bool = False
    feedback: tuple[str, ...] = ()
    abort: bool = False


class StrictBackend(Backend):
    """Answers from a script and checks every request it receives."""

    def __init__(self, script: list[Turn]) -> None:
        self._script = list(script)
        # Empty content would match every tool-call turn, so it is not tracked.
        self._failed = {t.content for t in script if t.failed and t.content}
        self._scripted = {t.content for t in script}
        self._expected: tuple[str, ...] = ()
        self._sent_results: set[str] = set()
        self.requests: list[list[Message]] = []
        self.violations: list[str] = []

    @property
    def remaining(self) -> int:
        """Script turns not yet answered."""
        return len(self._script)

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
        """Check the request, record it, and answer with the next script turn."""
        request = list(messages)
        self.requests.append(request)
        self._check(len(self.requests), request)
        if not self._script:
            raise RuntimeError(f"request {len(self.requests)}: script exhausted")
        turn = self._script.pop(0)
        self._expected = turn.feedback
        if turn.abort:
            raise PauseRequested()
        return ChatResponse(
            content=turn.content,
            tool_calls=list(turn.tool_calls),
            finish_reason=turn.finish_reason,
            output_tokens=turn.output_tokens,
        )

    def _check(self, number: int, messages: list[Message]) -> None:
        """Record every rule the request breaks."""
        for i, msg in enumerate(messages):
            if msg.role == Role.USER and i > 0 and messages[i - 1].role == Role.USER:
                self.violations.append(f"request {number}: user message follows user at {i}")
            if msg.role == Role.ASSISTANT and msg.content in self._failed:
                self.violations.append(f"request {number}: failed answer resent at {i}")
        self._check_tool_pairing(number, messages)
        self._check_history(number, messages)
        self._check_stand_ins(number, messages)
        self._check_feedback(number, messages)

    def _check_tool_pairing(self, number: int, messages: list[Message]) -> None:
        """Each tool result answers the open tool calls; none stay open."""
        open_ids: set[str] = set()
        for i, msg in enumerate(messages):
            if msg.role == Role.TOOL:
                if msg.tool_call_id not in open_ids:
                    self.violations.append(f"request {number}: orphan tool result at {i}")
                open_ids.discard(msg.tool_call_id or "")
                continue
            if open_ids:
                self.violations.append(f"request {number}: unanswered tool calls before {i}")
                open_ids = set()
            if msg.role == Role.ASSISTANT and msg.tool_calls:
                open_ids = {tc.id for tc in msg.tool_calls}
        if open_ids:
            self.violations.append(f"request {number}: unanswered tool calls at the end")

    def _check_history(self, number: int, messages: list[Message]) -> None:
        """The request starts with a user message; tool results sent once stay."""
        if messages and messages[0].role != Role.USER:
            self.violations.append(f"request {number}: does not start with a user message")
        results = {m.tool_call_id for m in messages if m.role == Role.TOOL and m.tool_call_id}
        for missing in sorted(self._sent_results - results):
            self.violations.append(f"request {number}: tool result {missing} dropped")
        self._sent_results |= results

    def _check_stand_ins(self, number: int, messages: list[Message]) -> None:
        """No stand-in, user message, stand-in run after the first request's messages."""
        start = len(self.requests[0])
        for i in range(start, len(messages) - 2):
            if (
                self._is_stand_in(messages[i])
                and messages[i + 1].role == Role.USER
                and self._is_stand_in(messages[i + 2])
            ):
                self.violations.append(f"request {number}: retries pile up at {i}")

    def _is_stand_in(self, msg: Message) -> bool:
        return (
            msg.role == Role.ASSISTANT and not msg.tool_calls and msg.content not in self._scripted
        )

    def _check_feedback(self, number: int, messages: list[Message]) -> None:
        """The request after a turn with feedback ends with a user message carrying it."""
        if not self._expected:
            return
        last = messages[-1] if messages else None
        if last is None or last.role != Role.USER:
            self.violations.append(f"request {number}: no feedback message at the end")
            return
        for phrase in self._expected:
            if phrase not in last.content:
                self.violations.append(f"request {number}: feedback lacks {phrase!r}")

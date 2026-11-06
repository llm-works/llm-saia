# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""A scripted backend that checks every request the way a strict provider would.

The backend answers from a fixed script and records each request it receives.
Every request is checked against rules real providers and chat templates
enforce, and against rules SAIA promises to keep:

- no two user messages in a row (strict chat templates reject them);
- every tool result answers a tool call of the assistant turn right before it,
  and every tool call of a turn is answered before the next turn;
- no answer the script marked as failed is ever sent back to the model.

Violations are collected, not raised, so a test can report all of them.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from llm_saia import Backend, ChatResponse
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
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    output_tokens: int = 0
    failed: bool = False


class StrictBackend(Backend):
    """Answers from a script and checks every request it receives."""

    def __init__(self, script: list[Turn]) -> None:
        self._script = list(script)
        # Empty content would match every tool-call turn, so it is not tracked.
        self._failed = {t.content for t in script if t.failed and t.content}
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

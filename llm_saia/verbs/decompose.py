# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""DECOMPOSE verb: Break down task into subtasks."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.types import VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.backend import ChatResponse
    from ..core.conversation import ConversationLike


@dataclass
class DecomposeResult:
    """Internal schema for decompose structured output."""

    subtasks: list[str]


class Decompose(Verb):
    """Break down task into subtasks."""

    async def __call__(
        self,
        task: str,
        *,
        conversation: ConversationLike | None = None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None = None,
        abort_signal: asyncio.Event | None = None,
        pause_check: Callable[[], Awaitable[bool]] | None = None,
        resume: bool = False,
    ) -> VerbResult[list[str]]:
        """Break down a task into a list of subtasks."""
        trace = self._init_verb_trace()
        try:
            prompt = f"Break down this task into subtasks:\n\n{task}"
            result = await self._complete_structured(
                prompt,
                DecomposeResult,
                conversation=conversation,
                _trace=trace,
                on_iteration=on_iteration,
                abort_signal=abort_signal,
                pause_check=pause_check,
                resume=resume,
            )
            return VerbResult(value=result.subtasks, trace=trace)
        finally:
            self._emit_verb_trace(trace)

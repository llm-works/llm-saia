# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""ASK verb: Query an artifact with a question."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from ..core.types import VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.backend import ChatResponse
    from ..core.conversation import ConversationLike


class Ask(Verb):
    """Query an artifact with a question."""

    async def __call__(
        self,
        artifact: Any,
        question: str,
        *,
        conversation: ConversationLike | None = None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None = None,
        abort_signal: asyncio.Event | None = None,
        pause_check: Callable[[], Awaitable[bool]] | None = None,
        resume: bool = False,
    ) -> VerbResult[str]:
        """Query an artifact with a question and return the answer."""
        trace = self._init_verb_trace()
        try:
            prompt = f"Given this artifact:\n{artifact}\n\nAnswer this question: {question}"
            value = await self._complete(
                prompt,
                conversation=conversation,
                _trace=trace,
                on_iteration=on_iteration,
                abort_signal=abort_signal,
                pause_check=pause_check,
                resume=resume,
            )
            return VerbResult(value=value, trace=trace)
        finally:
            self._emit_verb_trace(trace)

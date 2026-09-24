# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""INSTRUCT verb: Give a directive and get a response."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from ..core.types import VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.backend import ChatResponse
    from ..core.conversation import ConversationLike


class Instruct(Verb):
    """Give a directive and get a response."""

    async def __call__(
        self,
        directive: str,
        context: str | None = None,
        *,
        conversation: ConversationLike | None = None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None = None,
        abort_signal: asyncio.Event | None = None,
        pause_check: Callable[[], Awaitable[bool]] | None = None,
        resume: bool = False,
    ) -> VerbResult[str]:
        """Execute a directive and return the response."""
        trace = self._init_verb_trace()
        try:
            prompt = directive
            if context:
                prompt += f"\n\nContext: {context}"
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

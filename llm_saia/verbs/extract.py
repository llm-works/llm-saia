# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""EXTRACT verb: Extract structured data from content."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from ..core.types import VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.backend import ChatResponse
    from ..core.conversation import ConversationLike

T = TypeVar("T")


class Extract(Verb):
    """Extract structured data from unstructured content."""

    async def __call__(
        self,
        content: str,
        schema: type[T],
        instructions: str | None = None,
        *,
        conversation: ConversationLike | None = None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None = None,
        abort_signal: asyncio.Event | None = None,
        pause_check: Callable[[], Awaitable[bool]] | None = None,
        resume: bool = False,
    ) -> VerbResult[T]:
        """Extract structured data from content according to the schema."""
        trace = self._init_verb_trace()
        try:
            prompt = f"Extract the following information from this content:\n\n{content}"
            if instructions:
                prompt += f"\n\nExtraction guidance: {instructions}"
            value = await self._complete_structured(
                prompt,
                schema,
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

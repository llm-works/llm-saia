# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""CLASSIFY verb: Classify text into categories."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from ..core.types import ClassifyResult, VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.backend import ChatResponse
    from ..core.conversation import ConversationLike


class Classify(Verb):
    """Classify text into one of the given categories."""

    async def __call__(
        self,
        text: str,
        categories: list[str],
        criteria: str | None = None,
        *,
        conversation: ConversationLike | None = None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None = None,
        abort_signal: asyncio.Event | None = None,
        pause_check: Callable[[], Awaitable[bool]] | None = None,
        resume: bool = False,
    ) -> VerbResult[ClassifyResult]:
        """Classify text into one of the specified categories."""
        trace = self._init_verb_trace()
        try:
            cats = ", ".join(categories)
            prompt = f"Classify this text into one of: {cats}\n\nText: {text}"
            if criteria:
                prompt += f"\n\nCriteria: {criteria}"
            value = await self._complete_structured(
                prompt,
                ClassifyResult,
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

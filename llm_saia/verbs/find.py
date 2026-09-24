# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""FIND verb: Filter items matching criteria."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.types import FindResult, VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.backend import ChatResponse
    from ..core.conversation import ConversationLike

# Maximum items to process in a single call
MAX_ITEMS = 100


class Find(Verb):
    """Filter items to those matching criteria."""

    async def __call__(
        self,
        items: list[str],
        criteria: str,
        *,
        conversation: ConversationLike | None = None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None = None,
        abort_signal: asyncio.Event | None = None,
        pause_check: Callable[[], Awaitable[bool]] | None = None,
        resume: bool = False,
    ) -> VerbResult[FindResult]:
        """Find items matching criteria.

        Raises:
            ValueError: If items exceeds MAX_ITEMS.
        """
        trace = self._init_verb_trace()
        try:
            if not items:
                return VerbResult(
                    value=FindResult(indices=[], reason="No items provided"), trace=trace
                )
            if len(items) > MAX_ITEMS:
                raise ValueError(f"Too many items: {len(items)} exceeds max of {MAX_ITEMS}")
            result = await self._complete_structured(
                self._build_prompt(items, criteria),
                _FindResponse,
                conversation=conversation,
                _trace=trace,
                on_iteration=on_iteration,
                abort_signal=abort_signal,
                pause_check=pause_check,
                resume=resume,
            )
            indices = sorted({i - 1 for i in result.matching_numbers if 1 <= i <= len(items)})
            return VerbResult(value=FindResult(indices=indices, reason=result.reason), trace=trace)
        finally:
            self._emit_verb_trace(trace)

    @staticmethod
    def _build_prompt(items: list[str], criteria: str) -> str:
        items_list = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))
        return (
            f"Which of these items match the criteria?\n\n"
            f"Items:\n{items_list}\n\n"
            f"Criteria: {criteria}\n\n"
            f"Return:\n"
            f"- matching_numbers: 1-indexed numbers of ALL matching items "
            f"(empty list if none)\n"
            f"- reason: brief explanation of why those items match"
        )


@dataclass
class _FindResponse:
    """Internal response schema for structured output."""

    matching_numbers: list[int]  # 1-indexed numbers from the prompt
    reason: str

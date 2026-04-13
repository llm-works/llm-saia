"""FIND verb: Filter items matching criteria."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.types import FindResult, VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
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
    ) -> VerbResult[FindResult]:
        """Find items matching criteria.

        Args:
            items: List of items to filter (max 100).
            criteria: Criteria for matching (e.g., "relevant to AI research").

        Returns:
            VerbResult wrapping FindResult with 0-indexed indices and reason.

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

            items_list = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))
            prompt = (
                f"Which of these items match the criteria?\n\n"
                f"Items:\n{items_list}\n\n"
                f"Criteria: {criteria}\n\n"
                f"Return:\n"
                f"- matching_numbers: 1-indexed numbers of ALL matching items "
                f"(empty list if none)\n"
                f"- reason: brief explanation of why those items match"
            )
            result = await self._complete_structured(
                prompt, _FindResponse, conversation=conversation, _trace=trace
            )
            # Convert 1-indexed to 0-indexed, filter invalid indices, deduplicate
            indices = sorted({i - 1 for i in result.matching_numbers if 1 <= i <= len(items)})
            return VerbResult(value=FindResult(indices=indices, reason=result.reason), trace=trace)
        finally:
            self._emit_verb_trace(trace)


@dataclass
class _FindResponse:
    """Internal response schema for structured output."""

    matching_numbers: list[int]  # 1-indexed numbers from the prompt
    reason: str

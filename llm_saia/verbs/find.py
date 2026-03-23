"""FIND verb: Filter items matching criteria."""

from dataclasses import dataclass

from llm_saia.core.types import FindResult
from llm_saia.core.verb import Verb


class Find(Verb):
    """Filter items to those matching criteria."""

    async def __call__(
        self,
        items: list[str],
        criteria: str,
    ) -> FindResult:
        """Find items matching criteria.

        Args:
            items: List of items to filter.
            criteria: Criteria for matching (e.g., "relevant to AI research").

        Returns:
            FindResult with 0-indexed indices of matching items and reason.
        """
        if not items:
            return FindResult(indices=[], reason="No items provided")

        items_list = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))
        prompt = (
            f"Which of these items match the criteria?\n\n"
            f"Items:\n{items_list}\n\n"
            f"Criteria: {criteria}\n\n"
            f"Return the numbers (1-indexed) of ALL matching items. "
            f"If none match, return an empty list."
        )
        result = await self._complete_structured(prompt, _FindResponse)
        # Convert 1-indexed to 0-indexed, filter invalid indices
        indices = [i - 1 for i in result.matching_numbers if 1 <= i <= len(items)]
        return FindResult(indices=indices, reason=result.reason)


@dataclass
class _FindResponse:
    """Internal response schema for structured output."""

    matching_numbers: list[int]  # 1-indexed numbers from the prompt
    reason: str

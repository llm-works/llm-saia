"""CLASSIFY verb: Classify text into categories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.types import ClassifyResult
from ..core.verb import Verb

if TYPE_CHECKING:
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
    ) -> ClassifyResult:
        """Classify text into one of the specified categories."""
        cats = ", ".join(categories)
        prompt = f"Classify this text into one of: {cats}\n\nText: {text}"
        if criteria:
            prompt += f"\n\nCriteria: {criteria}"
        return await self._complete_structured(prompt, ClassifyResult, conversation=conversation)

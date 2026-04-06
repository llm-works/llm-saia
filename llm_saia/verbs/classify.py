"""CLASSIFY verb: Classify text into categories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.types import ClassifyResult, VerbResult
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
    ) -> VerbResult[ClassifyResult]:
        """Classify text into one of the specified categories."""
        trace = self._init_verb_trace()
        cats = ", ".join(categories)
        prompt = f"Classify this text into one of: {cats}\n\nText: {text}"
        if criteria:
            prompt += f"\n\nCriteria: {criteria}"
        value = await self._complete_structured(
            prompt, ClassifyResult, conversation=conversation, _trace=trace
        )
        self._emit_verb_trace(trace)
        return VerbResult(value=value, trace=trace)

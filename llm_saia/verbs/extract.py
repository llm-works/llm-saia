"""EXTRACT verb: Extract structured data from content."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from ..core.types import VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
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
    ) -> VerbResult[T]:
        """Extract structured data from content according to the schema."""
        trace = self._init_verb_trace()
        prompt = f"Extract the following information from this content:\n\n{content}"
        if instructions:
            prompt += f"\n\nExtraction guidance: {instructions}"
        value = await self._complete_structured(
            prompt, schema, conversation=conversation, _trace=trace
        )
        self._emit_verb_trace(trace)
        return VerbResult(value=value, trace=trace)

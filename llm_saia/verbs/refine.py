"""REFINE verb: Improve artifact based on feedback."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.types import VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.conversation import ConversationLike


class Refine(Verb):
    """Improve artifact based on feedback."""

    async def __call__(
        self, artifact: Any, feedback: str, *, conversation: ConversationLike | None = None
    ) -> VerbResult[str]:
        """Improve an artifact based on the provided feedback."""
        trace = self._init_verb_trace()
        prompt = (
            f"Improve this artifact based on the feedback.\n\n"
            f"Artifact: {artifact}\n\nFeedback: {feedback}"
        )
        value = await self._complete(prompt, conversation=conversation, _trace=trace)
        self._emit_verb_trace(trace)
        return VerbResult(value=value, trace=trace)

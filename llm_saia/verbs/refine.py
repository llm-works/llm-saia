"""REFINE verb: Improve artifact based on feedback."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.conversation import ConversationLike


class Refine(Verb):
    """Improve artifact based on feedback."""

    async def __call__(
        self, artifact: Any, feedback: str, *, conversation: ConversationLike | None = None
    ) -> str:
        """Improve an artifact based on the provided feedback."""
        prompt = (
            f"Improve this artifact based on the feedback.\n\n"
            f"Artifact: {artifact}\n\nFeedback: {feedback}"
        )
        return await self._complete(prompt, conversation=conversation)

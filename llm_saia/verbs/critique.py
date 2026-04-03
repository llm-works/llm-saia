"""CRITIQUE verb: Generate strongest counter-argument."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llm_saia.core.types import Critique
from llm_saia.core.verb import Verb

if TYPE_CHECKING:
    from llm_saia.core.conversation import ConversationLike


class Critique_(Verb):
    """Generate strongest counter-argument."""

    async def __call__(
        self, artifact: Any, *, conversation: ConversationLike | None = None
    ) -> Critique:
        """Generate the strongest counter-argument to the artifact."""
        prompt = f"Generate the strongest counter-argument to this:\n\n{artifact}"
        return await self._complete_structured(prompt, Critique, conversation=conversation)

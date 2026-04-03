"""INSTRUCT verb: Give a directive and get a response."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.conversation import ConversationLike


class Instruct(Verb):
    """Give a directive and get a response."""

    async def __call__(
        self,
        directive: str,
        context: str | None = None,
        *,
        conversation: ConversationLike | None = None,
    ) -> str:
        """Execute a directive and return the response."""
        prompt = directive
        if context:
            prompt += f"\n\nContext: {context}"
        return await self._complete(prompt, conversation=conversation)

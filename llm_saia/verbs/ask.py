"""ASK verb: Query an artifact with a question."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llm_saia.core.verb import Verb

if TYPE_CHECKING:
    from llm_saia.core.conversation import ConversationLike


class Ask(Verb):
    """Query an artifact with a question."""

    async def __call__(
        self, artifact: Any, question: str, *, conversation: ConversationLike | None = None
    ) -> str:
        """Query an artifact with a question and return the answer."""
        prompt = f"Given this artifact:\n{artifact}\n\nAnswer this question: {question}"
        return await self._complete(prompt, conversation=conversation)

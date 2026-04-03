"""CHOOSE verb: Force a choice between options."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.types import ChooseResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.conversation import ConversationLike


class Choose(Verb):
    """Force a choice between given options."""

    async def __call__(
        self,
        options: list[str],
        context: str | None = None,
        criteria: str | None = None,
        *,
        conversation: ConversationLike | None = None,
    ) -> ChooseResult:
        """Select one option from the given choices."""
        opts = "\n".join(f"- {o}" for o in options)
        prompt = f"Choose one of these options:\n{opts}"
        if context:
            prompt += f"\n\nContext: {context}"
        if criteria:
            prompt += f"\n\nCriteria: {criteria}"
        return await self._complete_structured(prompt, ChooseResult, conversation=conversation)

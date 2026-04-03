"""CONSTRAIN verb: Enforce rules and boundaries on text."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_saia.core.verb import Verb

if TYPE_CHECKING:
    from llm_saia.core.conversation import ConversationLike


class Constrain(Verb):
    """Enforce rules and boundaries on text."""

    async def __call__(
        self, text: str, rules: list[str], *, conversation: ConversationLike | None = None
    ) -> str:
        """Rewrite text to comply with the specified rules."""
        if not rules:
            return text
        rules_str = "\n".join(f"- {r}" for r in rules)
        prompt = f"Rewrite this text to comply with these rules:\n{rules_str}\n\nText:\n{text}"
        return await self._complete(prompt, conversation=conversation)

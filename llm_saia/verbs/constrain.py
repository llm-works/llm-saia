"""CONSTRAIN verb: Enforce rules and boundaries on text."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.types import VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.conversation import ConversationLike


class Constrain(Verb):
    """Enforce rules and boundaries on text."""

    async def __call__(
        self, text: str, rules: list[str], *, conversation: ConversationLike | None = None
    ) -> VerbResult[str]:
        """Rewrite text to comply with the specified rules."""
        trace = self._init_verb_trace()
        if not rules:
            self._emit_verb_trace(trace)
            return VerbResult(value=text, trace=trace)
        rules_str = "\n".join(f"- {r}" for r in rules)
        prompt = f"Rewrite this text to comply with these rules:\n{rules_str}\n\nText:\n{text}"
        value = await self._complete(prompt, conversation=conversation, _trace=trace)
        self._emit_verb_trace(trace)
        return VerbResult(value=value, trace=trace)

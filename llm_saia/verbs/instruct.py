"""INSTRUCT verb: Give a directive and get a response."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.types import VerbResult
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
    ) -> VerbResult[str]:
        """Execute a directive and return the response."""
        trace = self._init_verb_trace()
        try:
            prompt = directive
            if context:
                prompt += f"\n\nContext: {context}"
            value = await self._complete(prompt, conversation=conversation, _trace=trace)
            return VerbResult(value=value, trace=trace)
        finally:
            self._emit_verb_trace(trace)

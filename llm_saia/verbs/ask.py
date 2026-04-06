"""ASK verb: Query an artifact with a question."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.types import VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.conversation import ConversationLike


class Ask(Verb):
    """Query an artifact with a question."""

    async def __call__(
        self, artifact: Any, question: str, *, conversation: ConversationLike | None = None
    ) -> VerbResult[str]:
        """Query an artifact with a question and return the answer."""
        trace = self._init_verb_trace()
        prompt = f"Given this artifact:\n{artifact}\n\nAnswer this question: {question}"
        value = await self._complete(prompt, conversation=conversation, _trace=trace)
        self._emit_verb_trace(trace)
        return VerbResult(value=value, trace=trace)

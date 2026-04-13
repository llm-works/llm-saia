"""CRITIQUE verb: Generate strongest counter-argument."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.types import Critique, VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.conversation import ConversationLike


class Critique_(Verb):
    """Generate strongest counter-argument."""

    async def __call__(
        self, artifact: Any, *, conversation: ConversationLike | None = None
    ) -> VerbResult[Critique]:
        """Generate the strongest counter-argument to the artifact."""
        trace = self._init_verb_trace()
        try:
            prompt = f"Generate the strongest counter-argument to this:\n\n{artifact}"
            value = await self._complete_structured(
                prompt, Critique, conversation=conversation, _trace=trace
            )
            return VerbResult(value=value, trace=trace)
        finally:
            self._emit_verb_trace(trace)

"""VERIFY verb: Check if artifact satisfies predicate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.types import VerbResult, VerifyResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.conversation import ConversationLike


class Verify(Verb):
    """Check if artifact satisfies predicate."""

    async def __call__(
        self,
        artifact: Any,
        predicate: str = "factually accurate",
        *,
        conversation: ConversationLike | None = None,
    ) -> VerbResult[VerifyResult]:
        """Check whether an artifact satisfies a given predicate."""
        trace = self._init_verb_trace()
        try:
            prompt = (
                f"Verify that this artifact satisfies the predicate.\n\n"
                f"Artifact: {artifact}\n\nPredicate: {predicate}"
            )
            value = await self._complete_structured(
                prompt, VerifyResult, conversation=conversation, _trace=trace
            )
            return VerbResult(value=value, trace=trace)
        finally:
            self._emit_verb_trace(trace)

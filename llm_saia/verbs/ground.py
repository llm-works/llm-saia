"""GROUND verb: Anchor artifact against sources for evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.types import Evidence, VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.conversation import ConversationLike


class Ground(Verb):
    """Anchor artifact against sources for evidence."""

    async def __call__(
        self, artifact: Any, sources: list[Any], *, conversation: ConversationLike | None = None
    ) -> VerbResult[list[Evidence]]:
        """Find evidence in sources that supports or refutes the artifact."""
        trace = self._init_verb_trace()
        try:
            # Snapshot conversation before the loop so each source starts from
            # the same baseline (no cross-source context leakage).
            baseline = self._fork_conversation(conversation)
            results: list[Evidence] = []
            for source in sources:
                prompt = (
                    f"Find evidence in this source for the artifact.\n\n"
                    f"Artifact: {artifact}\n\nSource: {source}"
                )
                source_conv = self._fork_conversation(baseline)
                results.append(
                    await self._complete_structured(
                        prompt, Evidence, conversation=source_conv, _trace=trace
                    )
                )
                self._merge_conversation(conversation, source_conv)
            return VerbResult(value=results, trace=trace)
        finally:
            self._emit_verb_trace(trace)

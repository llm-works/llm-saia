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
            baseline_len = len(baseline.as_messages()) if baseline is not None else 0
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
                # Merge new messages from this source into the caller's conversation.
                # Use baseline_len (not target length) as offset since each source_conv
                # is forked from the same baseline snapshot.
                if conversation is not None and source_conv is not None:
                    for msg in source_conv.as_messages()[baseline_len:]:
                        conversation.append(msg)
            return VerbResult(value=results, trace=trace)
        finally:
            self._emit_verb_trace(trace)

"""GROUND verb: Anchor artifact against sources for evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llm_saia.core.types import Evidence
from llm_saia.core.verb import Verb

if TYPE_CHECKING:
    from llm_saia.core.conversation import ConversationLike


class Ground(Verb):
    """Anchor artifact against sources for evidence."""

    async def __call__(
        self, artifact: Any, sources: list[Any], *, conversation: ConversationLike | None = None
    ) -> list[Evidence]:
        """Find evidence in sources that supports or refutes the artifact."""
        results: list[Evidence] = []
        for source in sources:
            prompt = (
                f"Find evidence in this source for the artifact.\n\n"
                f"Artifact: {artifact}\n\nSource: {source}"
            )
            # Fork per source so each evaluation is independent
            source_conv = self._fork_conversation(conversation)
            results.append(
                await self._complete_structured(prompt, Evidence, conversation=source_conv)
            )
            self._merge_conversation(conversation, source_conv)
        return results

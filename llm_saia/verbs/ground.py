"""GROUND verb: Anchor artifact against sources for evidence."""

from typing import Any

from llm_saia.core.types import Evidence, LoopConfig
from llm_saia.verbs._base import _Verb


class Ground(_Verb):
    """Anchor artifact against sources for evidence."""

    async def __call__(
        self, artifact: Any, sources: list[Any], loop: LoopConfig | None = None
    ) -> list[Evidence]:
        results: list[Evidence] = []
        for source in sources:
            prompt = (
                f"Find evidence in this source for the artifact.\n\n"
                f"Artifact: {artifact}\n\nSource: {source}"
            )
            results.append(await self._complete_structured(prompt, Evidence, loop))
        return results

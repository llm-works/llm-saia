"""GROUND verb: Anchor artifact against sources for evidence."""

from typing import Any

from llm_saia.core.types import Evidence
from llm_saia.core.verb import Verb


class Ground(Verb):
    """Anchor artifact against sources for evidence."""

    async def __call__(self, artifact: Any, sources: list[Any]) -> list[Evidence]:
        results: list[Evidence] = []
        for source in sources:
            prompt = (
                f"Find evidence in this source for the artifact.\n\n"
                f"Artifact: {artifact}\n\nSource: {source}"
            )
            results.append(await self._complete_structured(prompt, Evidence))
        return results

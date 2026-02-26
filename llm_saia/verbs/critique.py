"""CRITIQUE verb: Generate strongest counter-argument."""

from typing import Any

from llm_saia.core.types import Critique
from llm_saia.core.verb import Verb


class Critique_(Verb):
    """Generate strongest counter-argument."""

    async def __call__(self, artifact: Any) -> Critique:
        """Generate the strongest counter-argument to the artifact."""
        prompt = f"Generate the strongest counter-argument to this:\n\n{artifact}"
        return await self._complete_structured(prompt, Critique)

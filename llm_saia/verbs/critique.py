"""CRITIQUE verb: Generate strongest counter-argument."""

from typing import Any

from llm_saia.core.types import Critique
from llm_saia.verbs._base import _Verb


class Critique_(_Verb):
    """Generate strongest counter-argument."""

    async def __call__(self, artifact: Any) -> Critique:
        prompt = f"Generate the strongest counter-argument to this:\n\n{artifact}"
        return await self._complete_structured(prompt, Critique)

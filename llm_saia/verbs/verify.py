"""VERIFY verb: Check if artifact satisfies predicate."""

from typing import Any

from llm_saia.core.types import VerifyResult
from llm_saia.core.verb import Verb


class Verify(Verb):
    """Check if artifact satisfies predicate."""

    async def __call__(self, artifact: Any, predicate: str = "factually accurate") -> VerifyResult:
        """Check whether an artifact satisfies a given predicate."""
        prompt = (
            f"Verify that this artifact satisfies the predicate.\n\n"
            f"Artifact: {artifact}\n\nPredicate: {predicate}"
        )
        return await self._complete_structured(prompt, VerifyResult)

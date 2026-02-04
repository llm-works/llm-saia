"""VERIFY verb: Check if artifact satisfies predicate."""

from typing import Any

from llm_saia.core.types import LoopConfig, VerifyResult
from llm_saia.verbs._base import _Verb


class Verify(_Verb):
    """Check if artifact satisfies predicate."""

    async def __call__(
        self, artifact: Any, predicate: str, loop: LoopConfig | None = None
    ) -> VerifyResult:
        prompt = (
            f"Verify that this artifact satisfies the predicate.\n\n"
            f"Artifact: {artifact}\n\nPredicate: {predicate}"
        )
        return await self._complete_structured(prompt, VerifyResult, loop)

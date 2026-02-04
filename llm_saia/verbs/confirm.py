"""CONFIRM verb: Get yes/no confirmation of a claim."""

from llm_saia.core.types import ConfirmResult
from llm_saia.verbs._base import _Verb


class Confirm(_Verb):
    """Ask for yes/no confirmation of a claim."""

    async def __call__(self, claim: str, context: str | None = None) -> ConfirmResult:
        prompt = f"Confirm whether this claim is true: {claim}"
        if context:
            prompt += f"\n\nContext: {context}"
        return await self._complete_structured(prompt, ConfirmResult)

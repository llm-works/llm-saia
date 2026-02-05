"""CONFIRM verb: Get yes/no confirmation of a claim."""

from llm_saia.core.types import ConfirmResult
from llm_saia.core.verb import Verb


class Confirm(Verb):
    """Ask for yes/no confirmation of a claim."""

    async def __call__(self, claim: str, context: str | None = None) -> ConfirmResult:
        """Confirm whether a claim is true, returning confirmed status and reason."""
        if self._lg:
            self._lg.trace(
                "checking confirmation...",
                extra={"claim": claim, "context": context},
            )

        prompt = f"Confirm whether this claim is true: {claim}"
        if context:
            prompt += f"\n\nContext: {context}"
        result = await self._complete_structured(prompt, ConfirmResult)

        if self._lg:
            self._lg.trace(
                "confirmation result",
                extra={"confirmed": result.confirmed, "reason": result.reason},
            )

        return result

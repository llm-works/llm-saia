"""CLASSIFY verb: Classify text into categories."""

from llm_saia.core.types import ClassifyResult
from llm_saia.core.verb import Verb


class Classify(Verb):
    """Classify text into one of the given categories."""

    async def __call__(
        self,
        text: str,
        categories: list[str],
        criteria: str | None = None,
    ) -> ClassifyResult:
        cats = ", ".join(categories)
        prompt = f"Classify this text into one of: {cats}\n\nText: {text}"
        if criteria:
            prompt += f"\n\nCriteria: {criteria}"
        return await self._complete_structured(prompt, ClassifyResult)

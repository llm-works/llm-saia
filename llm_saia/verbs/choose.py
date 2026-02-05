"""CHOOSE verb: Force a choice between options."""

from llm_saia.core.types import ChooseResult
from llm_saia.core.verb import Verb


class Choose(Verb):
    """Force a choice between given options."""

    async def __call__(
        self,
        options: list[str],
        context: str | None = None,
        criteria: str | None = None,
    ) -> ChooseResult:
        opts = "\n".join(f"- {o}" for o in options)
        prompt = f"Choose one of these options:\n{opts}"
        if context:
            prompt += f"\n\nContext: {context}"
        if criteria:
            prompt += f"\n\nCriteria: {criteria}"
        return await self._complete_structured(prompt, ChooseResult)

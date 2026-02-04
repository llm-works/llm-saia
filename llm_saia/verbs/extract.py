"""EXTRACT verb: Extract structured data from content."""

from typing import TypeVar

from llm_saia.verbs._base import _Verb

T = TypeVar("T")


class Extract(_Verb):
    """Extract structured data from unstructured content."""

    async def __call__(
        self,
        content: str,
        schema: type[T],
        instructions: str | None = None,
    ) -> T:
        prompt = f"Extract the following information from this content:\n\n{content}"
        if instructions:
            prompt += f"\n\nExtraction guidance: {instructions}"
        return await self._complete_structured(prompt, schema)

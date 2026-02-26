"""EXTRACT verb: Extract structured data from content."""

from typing import TypeVar

from llm_saia.core.verb import Verb

T = TypeVar("T")


class Extract(Verb):
    """Extract structured data from unstructured content."""

    async def __call__(
        self,
        content: str,
        schema: type[T],
        instructions: str | None = None,
    ) -> T:
        """Extract structured data from content according to the schema."""
        prompt = f"Extract the following information from this content:\n\n{content}"
        if instructions:
            prompt += f"\n\nExtraction guidance: {instructions}"
        return await self._complete_structured(prompt, schema)

"""SYNTHESIZE verb: Combine multiple artifacts into structured output."""

from typing import Any, TypeVar

from llm_saia.core.verb import Verb

T = TypeVar("T")


class Synthesize(Verb):
    """Combine multiple artifacts into structured output."""

    async def __call__(self, artifacts: list[Any], schema: type[T]) -> T:
        arts = "\n---\n".join(str(a) for a in artifacts)
        prompt = f"Synthesize these artifacts into a combined output:\n\n{arts}"
        return await self._complete_structured(prompt, schema)

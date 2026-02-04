"""SYNTHESIZE verb: Combine multiple artifacts into structured output."""

from typing import Any, TypeVar

from llm_saia.core.types import LoopConfig
from llm_saia.verbs._base import _Verb

T = TypeVar("T")


class Synthesize(_Verb):
    """Combine multiple artifacts into structured output."""

    async def __call__(
        self, artifacts: list[Any], schema: type[T], loop: LoopConfig | None = None
    ) -> T:
        arts = "\n---\n".join(str(a) for a in artifacts)
        prompt = f"Synthesize these artifacts into a combined output:\n\n{arts}"
        return await self._complete_structured(prompt, schema, loop)

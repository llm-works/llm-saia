"""PARSE verb: Parse unstructured text into structured schema."""

from typing import TypeVar

from llm_saia.core.types import LoopConfig
from llm_saia.verbs._base import _Verb

T = TypeVar("T")


class Parse(_Verb):
    """Parse unstructured text into structured schema."""

    async def __call__(self, text: str, schema: type[T], loop: LoopConfig | None = None) -> T:
        prompt = f"Parse the following text into the requested format:\n\n{text}"
        return await self._complete_structured(prompt, schema, loop)

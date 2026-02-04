"""ASK verb: Query an artifact with a question."""

from typing import Any

from llm_saia.core.types import LoopConfig
from llm_saia.verbs._base import _Verb


class Ask(_Verb):
    """Query an artifact with a question."""

    async def __call__(self, artifact: Any, question: str, loop: LoopConfig | None = None) -> str:
        prompt = f"Given this artifact:\n{artifact}\n\nAnswer this question: {question}"
        return await self._complete(prompt, loop)

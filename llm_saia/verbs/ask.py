"""ASK verb: Query an artifact with a question."""

from typing import Any

from llm_saia.core.verb import Verb


class Ask(Verb):
    """Query an artifact with a question."""

    async def __call__(self, artifact: Any, question: str) -> str:
        """Query an artifact with a question and return the answer."""
        prompt = f"Given this artifact:\n{artifact}\n\nAnswer this question: {question}"
        return await self._complete(prompt)

"""DECOMPOSE verb: Break down task into subtasks."""

from dataclasses import dataclass

from llm_saia.core.verb import Verb


@dataclass
class DecomposeResult:
    """Internal schema for decompose structured output."""

    subtasks: list[str]


class Decompose(Verb):
    """Break down task into subtasks."""

    async def __call__(self, task: str) -> list[str]:
        prompt = f"Break down this task into subtasks:\n\n{task}"
        result = await self._complete_structured(prompt, DecomposeResult)
        return result.subtasks

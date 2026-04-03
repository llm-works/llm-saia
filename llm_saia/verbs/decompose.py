"""DECOMPOSE verb: Break down task into subtasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.conversation import ConversationLike


@dataclass
class DecomposeResult:
    """Internal schema for decompose structured output."""

    subtasks: list[str]


class Decompose(Verb):
    """Break down task into subtasks."""

    async def __call__(
        self, task: str, *, conversation: ConversationLike | None = None
    ) -> list[str]:
        """Break down a task into a list of subtasks."""
        prompt = f"Break down this task into subtasks:\n\n{task}"
        result = await self._complete_structured(prompt, DecomposeResult, conversation=conversation)
        return result.subtasks

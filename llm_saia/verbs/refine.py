"""REFINE verb: Improve artifact based on feedback."""

from typing import Any

from llm_saia.core.types import LoopConfig
from llm_saia.verbs._base import _Verb


class Refine(_Verb):
    """Improve artifact based on feedback."""

    async def __call__(self, artifact: Any, feedback: str, loop: LoopConfig | None = None) -> str:
        prompt = (
            f"Improve this artifact based on the feedback.\n\n"
            f"Artifact: {artifact}\n\nFeedback: {feedback}"
        )
        return await self._complete(prompt, loop)

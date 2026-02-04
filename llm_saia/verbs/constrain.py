"""CONSTRAIN verb: Enforce rules and boundaries on text."""

from llm_saia.core.types import LoopConfig
from llm_saia.verbs._base import _Verb


class Constrain(_Verb):
    """Enforce rules and boundaries on text."""

    async def __call__(self, text: str, rules: list[str], loop: LoopConfig | None = None) -> str:
        rules_str = "\n".join(f"- {r}" for r in rules)
        prompt = f"Rewrite this text to comply with these rules:\n{rules_str}\n\nText:\n{text}"
        return await self._complete(prompt, loop)

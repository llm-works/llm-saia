"""INSTRUCT verb: Give a directive and get a response."""

from llm_saia.verbs._base import _Verb


class Instruct(_Verb):
    """Give a directive and get a response."""

    async def __call__(self, directive: str, context: str | None = None) -> str:
        prompt = directive
        if context:
            prompt += f"\n\nContext: {context}"
        return await self._complete(prompt)

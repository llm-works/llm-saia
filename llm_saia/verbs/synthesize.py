"""SYNTHESIZE verb: Combine multiple artifacts into structured or text output."""

from typing import Any, TypeVar, overload

from llm_saia.core.verb import Verb

T = TypeVar("T")


class Synthesize(Verb):
    """Combine multiple artifacts into structured or text output."""

    @overload
    async def __call__(self, artifacts: list[Any], schema: type[T]) -> T: ...

    @overload
    async def __call__(self, artifacts: list[Any], *, goal: str) -> str: ...

    async def __call__(
        self,
        artifacts: list[Any],
        schema: type[T] | None = None,
        *,
        goal: str | None = None,
    ) -> T | str:
        """Combine multiple artifacts into a single output.

        Args:
            artifacts: List of artifacts to combine.
            schema: Optional type for structured output.
            goal: Optional goal description for text output.

        Returns:
            Structured output if schema provided, otherwise string.
        """
        if schema is not None and goal is not None:
            raise ValueError("Provide exactly one of schema or goal, not both")

        arts = "\n---\n".join(str(a) for a in artifacts)

        if goal is not None:
            prompt = (
                f"Synthesize these artifacts. Output ONLY the final result, no explanations.\n\n"
                f"Goal: {goal}\n\nArtifacts:\n{arts}"
            )
            return await self._complete(prompt)

        if schema is not None:
            prompt = f"Synthesize these artifacts into a combined output:\n\n{arts}"
            return await self._complete_structured(prompt, schema)

        raise ValueError("Either schema or goal must be provided")

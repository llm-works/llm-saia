"""SYNTHESIZE verb: Combine multiple artifacts into structured or text output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, overload

from ..core.types import VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.conversation import ConversationLike

T = TypeVar("T")


class Synthesize(Verb):
    """Combine multiple artifacts into structured or text output."""

    @overload
    async def __call__(
        self,
        artifacts: list[Any],
        schema: type[T],
        *,
        conversation: ConversationLike | None = None,
    ) -> VerbResult[T]: ...

    @overload
    async def __call__(
        self,
        artifacts: list[Any],
        *,
        goal: str,
        conversation: ConversationLike | None = None,
    ) -> VerbResult[str]: ...

    async def __call__(
        self,
        artifacts: list[Any],
        schema: type[T] | None = None,
        *,
        goal: str | None = None,
        conversation: ConversationLike | None = None,
    ) -> VerbResult[T] | VerbResult[str]:
        """Combine multiple artifacts into a single output.

        Args:
            artifacts: List of artifacts to combine.
            schema: Optional type for structured output.
            goal: Optional goal description for text output.
            conversation: Optional conversation object for message tracking.

        Returns:
            VerbResult wrapping structured output if schema provided, otherwise string.
        """
        if schema is not None and goal is not None:
            raise ValueError("Provide exactly one of schema or goal, not both")

        trace = self._init_verb_trace()
        try:
            arts = "\n---\n".join(str(a) for a in artifacts)

            if goal is not None:
                prompt = (
                    f"Synthesize these artifacts. Output ONLY the final result, "
                    f"no explanations.\n\nGoal: {goal}\n\nArtifacts:\n{arts}"
                )
                value = await self._complete(prompt, conversation=conversation, _trace=trace)
                return VerbResult(value=value, trace=trace)

            if schema is not None:
                prompt = f"Synthesize these artifacts into a combined output:\n\n{arts}"
                value = await self._complete_structured(
                    prompt,
                    schema,  # type: ignore[arg-type]
                    conversation=conversation,
                    _trace=trace,
                )
                return VerbResult(value=value, trace=trace)

            raise ValueError("Either schema or goal must be provided")
        finally:
            self._emit_verb_trace(trace)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""SYNTHESIZE verb: Combine multiple artifacts into structured or text output."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar, overload

from ..core.types import VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.backend import ChatResponse
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
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None = None,
        abort_signal: asyncio.Event | None = None,
        pause_check: Callable[[], Awaitable[bool]] | None = None,
        resume: bool = False,
    ) -> VerbResult[T]: ...

    @overload
    async def __call__(
        self,
        artifacts: list[Any],
        *,
        goal: str,
        conversation: ConversationLike | None = None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None = None,
        abort_signal: asyncio.Event | None = None,
        pause_check: Callable[[], Awaitable[bool]] | None = None,
        resume: bool = False,
    ) -> VerbResult[str]: ...

    async def __call__(
        self,
        artifacts: list[Any],
        schema: type[T] | None = None,
        *,
        goal: str | None = None,
        conversation: ConversationLike | None = None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None = None,
        abort_signal: asyncio.Event | None = None,
        pause_check: Callable[[], Awaitable[bool]] | None = None,
        resume: bool = False,
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
        if schema is None and goal is None:
            raise ValueError("Either schema or goal must be provided")
        trace = self._init_verb_trace()
        try:
            arts = "\n---\n".join(str(a) for a in artifacts)
            coop = (on_iteration, abort_signal, pause_check, resume)
            if goal is not None:
                value = await self._synthesize_text(arts, goal, conversation, trace, *coop)
                return VerbResult(value=value, trace=trace)
            assert schema is not None  # guarded above
            typed = await self._synthesize_typed(arts, schema, conversation, trace, *coop)
            return VerbResult(value=typed, trace=trace)
        finally:
            self._emit_verb_trace(trace)

    async def _synthesize_text(
        self,
        arts: str,
        goal: str,
        conversation: ConversationLike | None,
        trace: Any,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None,
        abort_signal: asyncio.Event | None,
        pause_check: Callable[[], Awaitable[bool]] | None,
        resume: bool,
    ) -> str:
        prompt = (
            f"Synthesize these artifacts. Output ONLY the final result, "
            f"no explanations.\n\nGoal: {goal}\n\nArtifacts:\n{arts}"
        )
        return await self._complete(
            prompt,
            conversation=conversation,
            _trace=trace,
            on_iteration=on_iteration,
            abort_signal=abort_signal,
            pause_check=pause_check,
            resume=resume,
        )

    async def _synthesize_typed(
        self,
        arts: str,
        schema: type[T],
        conversation: ConversationLike | None,
        trace: Any,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None,
        abort_signal: asyncio.Event | None,
        pause_check: Callable[[], Awaitable[bool]] | None,
        resume: bool,
    ) -> T:
        prompt = f"Synthesize these artifacts into a combined output:\n\n{arts}"
        return await self._complete_structured(
            prompt,
            schema,
            conversation=conversation,
            _trace=trace,
            on_iteration=on_iteration,
            abort_signal=abort_signal,
            pause_check=pause_check,
            resume=resume,
        )

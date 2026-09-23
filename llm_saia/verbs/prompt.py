# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Internal verb backing SAIA.complete_structured."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from ..core.types import VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.backend import ChatResponse
    from ..core.conversation import ConversationLike

T = TypeVar("T")


class _PromptVerb(Verb):
    """Backing verb for SAIA.complete_structured.

    Unlike the domain verbs (Extract, Verify, Classify, ...), this one adds no
    framing to the caller's prompt. It exists to publish
    :meth:`Verb._complete_structured` — the primitive every built-in verb
    already runs on — as a first-class SAIA method without exposing the
    ``Verb`` subclass API to callers.
    """

    async def __call__(
        self,
        prompt: str,
        schema: type[T],
        *,
        conversation: ConversationLike | None = None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None = None,
        abort_signal: asyncio.Event | None = None,
        pause_check: Callable[[], Awaitable[bool]] | None = None,
        resume: bool = False,
    ) -> VerbResult[T]:
        """Send prompt verbatim and parse the response against schema."""
        trace = self._init_verb_trace()
        try:
            value = await self._complete_structured(
                prompt,
                schema,
                conversation=conversation,
                _trace=trace,
                on_iteration=on_iteration,
                abort_signal=abort_signal,
                pause_check=pause_check,
                resume=resume,
            )
            return VerbResult(value=value, trace=trace)
        finally:
            self._emit_verb_trace(trace)

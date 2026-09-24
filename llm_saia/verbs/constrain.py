# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""CONSTRAIN verb: Enforce rules and boundaries on text."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from ..core.types import VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.backend import ChatResponse
    from ..core.conversation import ConversationLike


class Constrain(Verb):
    """Enforce rules and boundaries on text."""

    async def __call__(
        self,
        text: str,
        rules: list[str],
        *,
        conversation: ConversationLike | None = None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None = None,
        abort_signal: asyncio.Event | None = None,
        pause_check: Callable[[], Awaitable[bool]] | None = None,
        resume: bool = False,
    ) -> VerbResult[str]:
        """Rewrite text to comply with the specified rules."""
        trace = self._init_verb_trace()
        try:
            if not rules:
                return VerbResult(value=text, trace=trace)
            rules_str = "\n".join(f"- {r}" for r in rules)
            prompt = f"Rewrite this text to comply with these rules:\n{rules_str}\n\nText:\n{text}"
            value = await self._complete(
                prompt,
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

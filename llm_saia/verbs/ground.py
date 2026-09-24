# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""GROUND verb: Anchor artifact against sources for evidence."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from ..core.types import Evidence, VerbResult
from ..core.verb import Verb

if TYPE_CHECKING:
    from ..core.backend import ChatResponse
    from ..core.conversation import ConversationLike


class Ground(Verb):
    """Anchor artifact against sources for evidence."""

    async def __call__(
        self,
        artifact: Any,
        sources: list[Any],
        *,
        conversation: ConversationLike | None = None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None = None,
        abort_signal: asyncio.Event | None = None,
        pause_check: Callable[[], Awaitable[bool]] | None = None,
        resume: bool = False,
    ) -> VerbResult[list[Evidence]]:
        """Find evidence in sources that supports or refutes the artifact."""
        if resume:
            raise ValueError(
                "Ground does not support resume=True; each source needs its own prompt"
            )
        trace = self._init_verb_trace()
        try:
            # Snapshot conversation before the loop so each source starts from
            # the same baseline (no cross-source context leakage).
            baseline = self._fork_conversation(conversation)
            baseline_len = len(baseline.as_messages()) if baseline is not None else 0
            results: list[Evidence] = []
            for source in sources:
                results.append(
                    await self._ground_one_source(
                        artifact,
                        source,
                        baseline,
                        baseline_len,
                        conversation,
                        trace,
                        on_iteration,
                        abort_signal,
                        pause_check,
                        resume,
                    )
                )
            return VerbResult(value=results, trace=trace)
        finally:
            self._emit_verb_trace(trace)

    async def _ground_one_source(
        self,
        artifact: Any,
        source: Any,
        baseline: ConversationLike | None,
        baseline_len: int,
        outer_conv: ConversationLike | None,
        trace: Any,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None,
        abort_signal: asyncio.Event | None,
        pause_check: Callable[[], Awaitable[bool]] | None,
        resume: bool,
    ) -> Evidence:
        """Ground one source against the artifact on a fresh forked conversation."""
        prompt = (
            f"Find evidence in this source for the artifact.\n\n"
            f"Artifact: {artifact}\n\nSource: {source}"
        )
        source_conv = self._fork_conversation(baseline)
        result = await self._complete_structured(
            prompt,
            Evidence,
            conversation=source_conv,
            _trace=trace,
            on_iteration=on_iteration,
            abort_signal=abort_signal,
            pause_check=pause_check,
            resume=resume,
        )
        # Merge new messages from this source into the caller's conversation.
        # baseline_len (not target length) is the offset since each source_conv
        # is forked from the same baseline snapshot.
        if outer_conv is not None and source_conv is not None:
            for msg in source_conv.as_messages()[baseline_len:]:
                outer_conv.append(msg)
        return result

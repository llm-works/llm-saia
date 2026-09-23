# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Verb execution semantic invariants.

Pins the observable properties of verb execution that any change to the
underlying dispatch — direct call, tool loop, structured-output handler —
must preserve. Groups:

- Trace shape from structured output (Step.phase / parsed / parse_error).
- Per-attempt conversation isolation during parse retry.
- Structured-output routing when tools are configured.
- Observable equivalence of text verbs (Ask/Instruct) between the
  no-tools and tools-configured dispatch paths.
- StructuredOutputError field content on exhausted retries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from llm_saia.core.backend import ToolDef
from llm_saia.core.conversation import ListConversation, Message, Role
from llm_saia.core.errors import StructuredOutputError
from llm_saia.guards import schema_retry
from tests.unit.conftest import MockBackend, make_saia

pytestmark = pytest.mark.unit


@dataclass
class _Judgment:
    """Simple schema for structured-output pins."""

    verdict: str
    confidence: float


async def _noop_executor(name: str, args: dict[str, Any]) -> Any:
    """Executor that would run if the model called a tool. The pins do not
    exercise tool execution; they only assert that tool *configuration*
    routes verb dispatch through the loop path."""
    return f"result of {name}"


def _tool_def() -> ToolDef:
    return ToolDef(
        name="stub_tool",
        description="Stub tool present only to make _has_tools() true.",
        parameters={"type": "object", "properties": {}},
    )


# ---------------------------------------------------------------------------
# Trace shape from structured output
# ---------------------------------------------------------------------------


class TestStructuredOutputTraceShape:
    """Step attributes emitted on the structured-output path."""

    async def test_single_successful_attempt_emits_one_attempt_step(
        self, mock_backend: MockBackend
    ) -> None:
        saia = make_saia(mock_backend)
        mock_backend.set_structured_response(_Judgment, _Judgment("y", 0.9))

        result = await saia.complete_structured("Judge.", _Judgment)

        steps = result.trace.steps
        assert len(steps) == 1
        assert steps[0].phase == "attempt"
        assert steps[0].parsed is True
        assert steps[0].parse_error is None

    async def test_parse_retry_marks_failed_attempt_and_records_retry(
        self, mock_backend: MockBackend
    ) -> None:
        saia = make_saia(mock_backend).with_guard(schema_retry(max_retries=1))
        mock_backend.queue_raw_structured("not json at all")
        mock_backend.set_structured_response(_Judgment, _Judgment("y", 0.5))

        result = await saia.complete_structured("Judge.", _Judgment)

        steps = result.trace.steps
        assert len(steps) == 2, f"expected attempt + parse_retry, got {[s.phase for s in steps]}"
        # First attempt: failed parse.
        assert steps[0].phase == "attempt"
        assert steps[0].parsed is False
        assert steps[0].parse_error is not None
        # Second attempt: successful under parse_retry phase.
        assert steps[1].phase == "parse_retry"
        assert steps[1].parsed is True
        assert steps[1].parse_error is None

    async def test_no_retry_guard_surfaces_error_without_extra_steps(
        self, mock_backend: MockBackend
    ) -> None:
        """Parse failure with no schema_retry guard should raise on the first attempt."""
        saia = make_saia(mock_backend)
        mock_backend.queue_raw_structured("not json")

        with pytest.raises(StructuredOutputError):
            await saia.complete_structured("Judge.", _Judgment)


# ---------------------------------------------------------------------------
# Per-attempt conversation isolation
# ---------------------------------------------------------------------------


class TestStructuredOutputConversationIsolation:
    """Failed parse attempts must not leak into the outer conversation.

    Current implementation forks the conversation per attempt and merges
    only successful results back. Any dispatch change must preserve the
    property that outer callers never observe the failed-attempt messages.
    """

    async def test_failed_attempt_messages_absent_from_outer_conversation(
        self, mock_backend: MockBackend
    ) -> None:
        saia = make_saia(mock_backend).with_guard(schema_retry(max_retries=1))
        mock_backend.queue_raw_structured("not json")
        mock_backend.set_structured_response(_Judgment, _Judgment("y", 0.5))
        conv = ListConversation()

        await saia.complete_structured("Judge.", _Judgment, conversation=conv)

        msgs = conv.as_messages()
        # Two attempts ran but the outer conversation carries the caller's
        # original prompt + the final successful response only. The failed
        # exchange (bad response + retry framing) stays inside the strategy's
        # inner conversation and never leaks into the caller's history.
        assert len(msgs) == 2, (
            f"expected outer conv to hold only original + successful response, "
            f"got roles {[m.role for m in msgs]} and contents "
            f"{[m.content[:30] for m in msgs]}"
        )
        assert msgs[0].role == Role.USER
        assert msgs[0].content == "Judge."
        assert msgs[1].role == Role.ASSISTANT
        # Payload of the successful attempt (JSON-parseable), not the failed raw.
        parsed = json.loads(msgs[1].content)
        assert parsed == {"verdict": "y", "confidence": 0.5}

    async def test_prior_history_preserved_across_parse_retry(
        self, mock_backend: MockBackend
    ) -> None:
        """A parse retry should not clobber messages that pre-date the verb call."""
        saia = make_saia(mock_backend).with_guard(schema_retry(max_retries=1))
        mock_backend.queue_raw_structured("not json")
        mock_backend.set_structured_response(_Judgment, _Judgment("y", 0.5))
        conv = ListConversation()
        conv.append(Message(role=Role.USER, content="prior turn"))
        conv.append(Message(role=Role.ASSISTANT, content="prior response"))

        await saia.complete_structured("Judge.", _Judgment, conversation=conv)

        msgs = conv.as_messages()
        # Prior history intact at the front, followed by original prompt +
        # successful response only (isolation preserved; failed attempt and
        # retry framing not merged).
        assert len(msgs) == 4, [(m.role, m.content[:30]) for m in msgs]
        assert msgs[0].content == "prior turn"
        assert msgs[1].content == "prior response"
        assert msgs[2].role == Role.USER
        assert msgs[2].content == "Judge."
        assert msgs[3].role == Role.ASSISTANT


# ---------------------------------------------------------------------------
# Structured output when tools are configured
# ---------------------------------------------------------------------------


class TestStructuredOutputWithToolsRouting:
    """Typed verbs with tools configured must still honor the schema.

    Current dispatch routes through the tool loop when _has_tools() is true.
    Under any refactor, the observable outcome must remain: schema respected,
    parsed value returned in VerbResult, trace emitted.
    """

    async def test_typed_verb_with_tools_returns_parsed_value(
        self, mock_backend: MockBackend
    ) -> None:
        saia = make_saia(mock_backend, tools=[_tool_def()], executor=_noop_executor)
        mock_backend.set_structured_response(_Judgment, _Judgment("yes", 0.9))

        result = await saia.complete_structured("Judge.", _Judgment)

        assert result.value.verdict == "yes"
        assert result.value.confidence == 0.9
        assert result.trace.trace_id
        assert len(result.trace.steps) >= 1


# ---------------------------------------------------------------------------
# Ask/Instruct: direct vs loop dispatch
# ---------------------------------------------------------------------------


class TestTextVerbDispatchEquivalence:
    """Ask/Instruct produce equivalent observable content whether the dispatch
    goes through the direct-call path (no tools) or the loop path (tools
    configured), given identical backend responses.
    """

    async def test_ask_returns_same_content_across_dispatch_paths(self) -> None:
        direct_backend = MockBackend()
        direct_backend.set_complete_response("the answer")
        direct_saia = make_saia(direct_backend)

        loop_backend = MockBackend()
        loop_backend.set_complete_response("the answer")
        loop_saia = make_saia(loop_backend, tools=[_tool_def()], executor=_noop_executor)

        direct = await direct_saia.ask("artifact", "question?")
        looped = await loop_saia.ask("artifact", "question?")

        assert direct.value == looped.value == "the answer"

    async def test_instruct_returns_same_content_across_dispatch_paths(self) -> None:
        direct_backend = MockBackend()
        direct_backend.set_complete_response("done")
        direct_saia = make_saia(direct_backend)

        loop_backend = MockBackend()
        loop_backend.set_complete_response("done")
        loop_saia = make_saia(loop_backend, tools=[_tool_def()], executor=_noop_executor)

        direct = await direct_saia.instruct("do the thing")
        looped = await loop_saia.instruct("do the thing")

        assert direct.value == looped.value == "done"


# ---------------------------------------------------------------------------
# StructuredOutputError field content
# ---------------------------------------------------------------------------


class TestStructuredOutputErrorShape:
    """Public error fields must survive across dispatch refactors so callers
    that inspect them (schema_name, parse_error, raw_content) keep working."""

    async def test_error_carries_schema_name_and_raw_content(
        self, mock_backend: MockBackend
    ) -> None:
        saia = make_saia(mock_backend)
        mock_backend.queue_raw_structured("not json at all")

        with pytest.raises(StructuredOutputError) as exc_info:
            await saia.complete_structured("Judge.", _Judgment)

        err = exc_info.value
        assert err.schema_name == "_Judgment"
        assert err.raw_content == "not json at all"
        assert err.parse_error is not None

    async def test_error_after_exhausted_retries_carries_last_raw_content(
        self, mock_backend: MockBackend
    ) -> None:
        saia = make_saia(mock_backend).with_guard(schema_retry(max_retries=1))
        # Two consecutive parse failures — retries exhausted.
        mock_backend.queue_raw_structured("not json")
        mock_backend.queue_raw_structured("still not json")

        with pytest.raises(StructuredOutputError) as exc_info:
            await saia.complete_structured("Judge.", _Judgment)

        err = exc_info.value
        assert err.schema_name == "_Judgment"
        # The last attempt's raw content is what surfaces.
        assert err.raw_content == "still not json"
        assert err.parse_error is not None

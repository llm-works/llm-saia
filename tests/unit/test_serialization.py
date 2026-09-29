# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Round-trip tests for TaskResult and trace records through to_dict / from_dict."""

from __future__ import annotations

import asyncio
import json
from io import StringIO
from typing import Any

import pytest

from llm_saia.core.trace import GuardOutcome, LLMCall, Step, ToolOutcome, Tracer, VerbTrace
from llm_saia.core.types import (
    ChatResponse,
    LoopScore,
    Message,
    TaskResult,
    ToolCall,
    ToolDef,
)
from tests.unit.conftest import MockBackend, make_saia

pytestmark = pytest.mark.unit


_SEARCH = ToolDef(
    name="search",
    description="Search for information",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
)
_DONE = ToolDef(
    name="task_complete",
    description="Call when task is complete",
    parameters={"type": "object", "properties": {"summary": {"type": "string"}}},
)


async def _executor(name: str, args: dict[str, Any]) -> str:
    return f"Result of {name}"


def _tool_response(call_id: str, name: str, args: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        content=f"calling {name}",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
        finish_reason="tool_use",
    )


def _assert_round_trips(result: TaskResult) -> None:
    """from_dict(to_dict(r)) == r, and the dict survives a JSON round trip."""
    data = result.to_dict()
    assert TaskResult.from_dict(data) == result
    assert TaskResult.from_dict(json.loads(json.dumps(data))) == result


def _full_step() -> Step:
    """A Step with every nested record and optional field populated."""
    return Step(
        phase="guard_retry",
        ts=1727600000.123456,
        duration_ms=42,
        trace_id="abcd1234",
        verb="Complete",
        llm_call=LLMCall(
            call_id="c0ffee00",
            input_tokens=100,
            output_tokens=20,
            finish_reason="tool_use",
            duration_ms=40,
            model="m-1",
            llm_request_id="req-9",
        ),
        parsed=False,
        parse_error="bad json",
        guards=[GuardOutcome(name="g", passed=False, attempts=2, error="nope", blocking=False)],
        tools=[ToolOutcome(name="search", call_id="t1", success=False, error="boom")],
        action="execute_tools",
        reason="has_tool_calls",
        nudge_preview="try again",
        iterations_since_nudge=3,
        consecutive_degenerate=1,
        pending_terminal=True,
        classifier_called=True,
    )


class TestTaskResultRoundTrip:
    """TaskResult.from_dict(r.to_dict()) == r across result shapes."""

    def test_fully_populated(self) -> None:
        """Every field, nested record and optional value survives."""
        result = TaskResult(
            completed=True,
            output="done",
            iterations=2,
            history=[
                Message(role="user", content="go"),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="t1",
                            name="search",
                            arguments={"q": ["a", 1, None]},
                            extra_content={"sig": "x"},
                        )
                    ],
                ),
                Message(role="tool", content="r", tool_call_id="t1"),
            ],
            reason="completed",
            terminal_data={"summary": "ok", "n": 1.5},
            terminal_tool="task_complete",
            score=LoopScore(2, 2, 0, 0, 120, 0),
            trace=VerbTrace(
                verb="Complete",
                trace_id="abcd1234",
                ts=1727600000.5,
                duration_ms=99,
                request_id="r-1",
                steps=[_full_step(), Step(phase="iteration")],
                ok=False,
                error="x",
            ),
        )
        _assert_round_trips(result)

    def test_minimal(self) -> None:
        """Defaults (no score, empty trace, no terminal data) survive."""
        _assert_round_trips(TaskResult(completed=False, output="", iterations=0, history=[]))

    async def test_completed_run_with_tools_and_terminal(self, mock_backend: MockBackend) -> None:
        """A real completed complete() result round-trips."""
        saia = make_saia(
            mock_backend,
            tools=[_SEARCH, _DONE],
            executor=_executor,
            terminal_tool="task_complete",
        )
        mock_backend.queue_tool_response(_tool_response("c1", "search", {"query": "q"}))
        for call_id in ("c2", "c3"):  # terminal call + confirmation
            mock_backend.queue_tool_response(
                _tool_response(call_id, "task_complete", {"summary": "s"})
            )

        result = await saia.complete(task="Do work")

        assert result.completed is True
        assert result.score is not None
        assert result.trace.steps
        assert any(s.tools for s in result.trace.steps)
        _assert_round_trips(result)

    async def test_paused_run(self, mock_backend: MockBackend) -> None:
        """A real paused complete() result round-trips."""
        saia = make_saia(mock_backend, tools=[_SEARCH], executor=_executor)
        mock_backend.queue_tool_response(_tool_response("c1", "search", {"query": "q"}))
        signal = asyncio.Event()

        async def arm(iteration: int, response: ChatResponse) -> None:
            signal.set()

        result = await saia.complete(task="Do work", abort_signal=signal, on_iteration=arm)

        assert result.paused is True
        _assert_round_trips(result)


class TestTolerantFromDict:
    """from_dict loads dicts written by other SAIA versions."""

    def test_unknown_keys_ignored_at_every_level(self) -> None:
        """Extra keys on the result and nested records are dropped."""
        data = TaskResult(
            completed=True,
            output="o",
            iterations=1,
            history=[Message(role="user", content="hi")],
            score=LoopScore(1, 1, 0, 0, 10, 0),
            trace=VerbTrace(steps=[_full_step()]),
        ).to_dict()
        data["future_field"] = 1
        data["history"][0]["future_field"] = 1
        data["score"]["future_field"] = 1
        data["trace"]["future_field"] = 1
        step = data["trace"]["steps"][0]
        step["future_field"] = 1
        step["llm_call"]["future_field"] = 1
        step["guards"][0]["future_field"] = 1
        step["tools"][0]["future_field"] = 1

        restored = TaskResult.from_dict(data)

        assert restored.trace.steps[0] == _full_step()
        assert restored.score == LoopScore(1, 1, 0, 0, 10, 0)

    def test_missing_optional_keys_take_defaults(self) -> None:
        """Only the required keys are needed; everything else defaults."""
        restored = TaskResult.from_dict(
            {"completed": True, "output": "o", "iterations": 1, "history": []}
        )
        assert restored == TaskResult(completed=True, output="o", iterations=1, history=[])

    def test_missing_nested_optional_keys_take_defaults(self) -> None:
        """Sparse nested trace records fill in defaults."""
        trace = VerbTrace.from_dict({"steps": [{"phase": "attempt"}]})
        assert trace == VerbTrace(steps=[Step(phase="attempt")])


class TestTraceRecordDicts:
    """to_dict on trace records keeps the Tracer's JSONL record shape."""

    def test_step_to_dict_matches_tracer_record(self) -> None:
        """Step.to_dict() equals the record Tracer writes for the same Step."""
        buf = StringIO()
        Tracer(buf).write(_full_step())
        assert json.loads(buf.getvalue()) == _full_step().to_dict()

    def test_verb_trace_to_dict_matches_tracer_record(self) -> None:
        """VerbTrace.to_dict() equals the record Tracer writes for the same trace."""
        trace = VerbTrace(verb="Complete", steps=[_full_step()])
        buf = StringIO()
        Tracer(buf).write(trace)
        assert json.loads(buf.getvalue()) == trace.to_dict()

    @pytest.mark.parametrize(
        "record",
        [
            LLMCall(call_id="c", input_tokens=1, model="m"),
            GuardOutcome(name="g", passed=False, error="e"),
            ToolOutcome(name="t", call_id="c", success=False, error="e"),
            _full_step(),
        ],
        ids=["llm_call", "guard", "tool", "step"],
    )
    def test_record_round_trips(self, record: Any) -> None:
        """Each trace record type round-trips on its own."""
        assert type(record).from_dict(record.to_dict()) == record

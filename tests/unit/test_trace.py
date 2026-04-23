"""Tests for tree-structured trace: data model, Tracer, and verb integration."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from llm_saia.core.backend import ChatResponse, ToolDef
from llm_saia.core.conversation import ToolCall
from llm_saia.core.trace import (
    GuardOutcome,
    LLMCall,
    Step,
    ToolOutcome,
    Tracer,
    TracerFactory,
    VerbTrace,
    _generate_id,
    build_step_from_response,
)
from tests.unit.conftest import MockBackend, make_saia

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


class TestGenerateId:
    """Tests for trace ID generation."""

    def test_is_8_hex_chars(self) -> None:
        """Generated IDs are 8-character hex strings."""
        id1 = _generate_id()
        assert len(id1) == 8
        int(id1, 16)  # raises if not valid hex

    def test_is_unique(self) -> None:
        """Consecutive IDs are unique."""
        ids = {_generate_id() for _ in range(100)}
        assert len(ids) == 100

    async def test_chat_attaches_call_id(self, mock_backend: MockBackend) -> None:
        """_chat() attaches a call_id to the response."""
        saia = make_saia(mock_backend)
        verb = saia.complete
        from llm_saia.core.conversation import Message

        response = await verb._chat([Message(role="user", content="hi")], None)
        assert response.call_id
        assert len(response.call_id) == 8


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class TestLLMCall:
    """Tests for LLMCall dataclass."""

    def test_defaults(self) -> None:
        """LLMCall has sensible defaults."""
        call = LLMCall()
        assert call.call_id == ""
        assert call.input_tokens == 0
        assert call.output_tokens == 0
        assert call.finish_reason is None

    def test_construction(self) -> None:
        """LLMCall stores provided values."""
        call = LLMCall(call_id="abc", input_tokens=100, output_tokens=50, finish_reason="stop")
        assert call.call_id == "abc"
        assert call.input_tokens == 100


class TestGuardOutcome:
    """Tests for GuardOutcome dataclass."""

    def test_passed(self) -> None:
        """GuardOutcome for a passing guard."""
        g = GuardOutcome(name="max_length", passed=True, attempts=1)
        assert g.passed
        assert g.error is None

    def test_failed(self) -> None:
        """GuardOutcome for a failing guard."""
        g = GuardOutcome(name="max_length", passed=False, attempts=3, error="too long")
        assert not g.passed
        assert g.attempts == 3


class TestStep:
    """Tests for Step dataclass."""

    def test_defaults(self) -> None:
        """Step has sensible defaults."""
        s = Step()
        assert s.type == "step"
        assert s.phase == ""
        assert s.parsed is True
        assert s.guards == []
        assert s.tools == []
        assert s.action is None

    def test_attempt_step(self) -> None:
        """Step for an initial attempt."""
        s = Step(
            phase="attempt",
            trace_id="t1",
            verb="Extract",
            llm_call=LLMCall(call_id="c1", input_tokens=100, output_tokens=50),
        )
        assert s.phase == "attempt"
        assert s.llm_call.call_id == "c1"

    def test_parse_retry_step(self) -> None:
        """Step for a parse retry."""
        s = Step(phase="parse_retry", parsed=False, parse_error="invalid JSON")
        assert not s.parsed
        assert s.parse_error == "invalid JSON"

    def test_iteration_step_with_tools(self) -> None:
        """Step for a loop iteration with tool calls."""
        s = Step(
            phase="iteration",
            tools=[
                ToolOutcome(name="search", call_id="c1", success=True),
                ToolOutcome(name="read", call_id="c2", success=False, error="not found"),
            ],
            action="execute_tools",
            reason="has_tool_calls",
        )
        assert len(s.tools) == 2
        assert s.tools[1].success is False
        assert s.action == "execute_tools"


class TestVerbTrace:
    """Tests for VerbTrace dataclass and derived properties."""

    def _make_trace(self) -> VerbTrace:
        """Build a sample trace with 3 steps for testing aggregates."""
        return VerbTrace(
            verb="Extract",
            trace_id="t1",
            steps=[
                Step(
                    phase="attempt",
                    llm_call=LLMCall(input_tokens=100, output_tokens=30),
                    parsed=False,
                    parse_error="bad json",
                ),
                Step(
                    phase="parse_retry",
                    llm_call=LLMCall(input_tokens=150, output_tokens=40),
                    parsed=True,
                ),
                Step(
                    phase="guard_retry",
                    llm_call=LLMCall(input_tokens=160, output_tokens=25),
                    guards=[GuardOutcome(name="max_length", passed=True, attempts=2)],
                ),
            ],
        )

    def test_total_llm_calls(self) -> None:
        """total_llm_calls counts steps."""
        trace = self._make_trace()
        assert trace.total_llm_calls == 3

    def test_parse_retries(self) -> None:
        """parse_retries counts parse_retry steps."""
        trace = self._make_trace()
        assert trace.parse_retries == 1

    def test_guard_retries(self) -> None:
        """guard_retries counts guard_retry steps."""
        trace = self._make_trace()
        assert trace.guard_retries == 1

    def test_total_input_tokens(self) -> None:
        """total_input_tokens sums across steps."""
        trace = self._make_trace()
        assert trace.total_input_tokens == 410  # 100 + 150 + 160

    def test_total_output_tokens(self) -> None:
        """total_output_tokens sums across steps."""
        trace = self._make_trace()
        assert trace.total_output_tokens == 95  # 30 + 40 + 25

    def test_total_tokens(self) -> None:
        """total_tokens is input + output."""
        trace = self._make_trace()
        assert trace.total_tokens == 505

    def test_empty_trace(self) -> None:
        """Empty trace has zero aggregates."""
        trace = VerbTrace()
        assert trace.total_llm_calls == 0
        assert trace.parse_retries == 0
        assert trace.total_tokens == 0

    def test_add_step(self) -> None:
        """add_step appends to steps list."""
        trace = VerbTrace()
        step = Step(phase="attempt")
        trace.add_step(step)
        assert len(trace.steps) == 1
        assert trace.steps[0] is step


# ---------------------------------------------------------------------------
# Step builders
# ---------------------------------------------------------------------------


class TestBuildStepFromResponse:
    """Tests for build_step_from_response helper."""

    def test_builds_from_text_response(self) -> None:
        """Builds a Step from a plain text response."""
        response = ChatResponse(
            content="hello",
            tool_calls=[],
            input_tokens=50,
            output_tokens=10,
            finish_reason="end_turn",
            call_id="c1",
        )
        step = build_step_from_response(response, phase="attempt", trace_id="t1", verb="Ask")
        assert step.phase == "attempt"
        assert step.trace_id == "t1"
        assert step.verb == "Ask"
        assert step.llm_call.call_id == "c1"
        assert step.llm_call.input_tokens == 50
        assert step.llm_call.output_tokens == 10
        assert step.llm_call.finish_reason == "end_turn"
        assert step.tools == []

    def test_builds_from_tool_response(self) -> None:
        """Builds a Step from a response with tool calls."""
        response = ChatResponse(
            content="",
            tool_calls=[
                ToolCall(id="tc1", name="search", arguments={"q": "test"}),
                ToolCall(id="tc2", name="read", arguments={}),
            ],
            call_id="c2",
        )
        step = build_step_from_response(response, phase="iteration", trace_id="t2", verb="Ask")
        assert len(step.tools) == 2
        assert step.tools[0].name == "search"
        assert step.tools[0].call_id == "tc1"
        assert step.tools[1].name == "read"


# ---------------------------------------------------------------------------
# Tracer infrastructure
# ---------------------------------------------------------------------------


class TestTracer:
    """Tests for Tracer JSONL output."""

    def test_writes_step_as_jsonl(self, tmp_path: Path) -> None:
        """Tracer serializes a Step to JSONL."""
        path = tmp_path / "trace.jsonl"
        step = Step(
            phase="attempt",
            trace_id="t1",
            verb="Ask",
            llm_call=LLMCall(call_id="c1", input_tokens=100, output_tokens=20),
        )
        with TracerFactory.file(path) as tracer:
            tracer.write(step)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["type"] == "step"
        assert data["phase"] == "attempt"
        assert data["trace_id"] == "t1"
        assert data["llm_call"]["call_id"] == "c1"

    def test_writes_verb_trace_as_jsonl(self, tmp_path: Path) -> None:
        """Tracer serializes a VerbTrace to JSONL."""
        path = tmp_path / "trace.jsonl"
        trace = VerbTrace(
            verb="Extract",
            trace_id="t1",
            steps=[
                Step(phase="attempt", llm_call=LLMCall(call_id="c1")),
            ],
        )
        with TracerFactory.file(path) as tracer:
            tracer.write(trace)

        data = json.loads(path.read_text().strip())
        assert data["type"] == "verb_trace"
        assert data["verb"] == "Extract"
        assert len(data["steps"]) == 1

    def test_writes_metadata_header(self, tmp_path: Path) -> None:
        """Metadata is written as the first line via start()."""
        path = tmp_path / "trace.jsonl"
        step = Step(phase="attempt", trace_id="t1")
        with TracerFactory.file(path) as tracer:
            tracer.start({"trace_id": "t1", "request_id": "u1"})
            tracer.write(step)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        meta = json.loads(lines[0])
        assert meta["_meta"]["trace_id"] == "t1"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """TracerFactory.file creates parent directories if needed."""
        path = tmp_path / "sub" / "dir" / "trace.jsonl"
        step = Step(phase="attempt")
        with TracerFactory.file(path) as tracer:
            tracer.write(step)
        assert path.exists()

    def test_stream_tracer(self) -> None:
        """TracerFactory.stream writes to a caller-provided stream."""
        buf = StringIO()
        step = Step(phase="attempt", trace_id="t1")
        tracer = TracerFactory.stream(buf)
        tracer.write(step)

        data = json.loads(buf.getvalue().strip())
        assert data["trace_id"] == "t1"

    def test_callback_tracer(self) -> None:
        """TracerFactory.callback calls function with dict."""
        records: list[dict] = []
        tracer = TracerFactory.callback(records.append)
        step = Step(phase="attempt", trace_id="t1")
        tracer.write(step)

        assert len(records) == 1
        assert records[0]["type"] == "step"
        assert records[0]["trace_id"] == "t1"

    def test_callback_tracer_start(self) -> None:
        """Callback tracer forwards metadata."""
        records: list[dict] = []
        tracer = TracerFactory.callback(records.append)
        tracer.start({"trace_id": "t1"})
        assert records[0]["_meta"]["trace_id"] == "t1"

    def test_close_does_not_close_borrowed_writer(self) -> None:
        """Tracer with owns_writer=False does not close the writer."""
        buf = StringIO()
        tracer = Tracer(buf, owns_writer=False)
        tracer.close()
        assert not buf.closed

    def test_close_closes_owned_writer(self, tmp_path: Path) -> None:
        """Tracer with owns_writer=True closes the writer."""
        path = tmp_path / "owned.jsonl"
        tracer = TracerFactory.file(path)
        tracer.close()
        assert path.exists()


# ---------------------------------------------------------------------------
# Verb integration (basic — detailed integration tested after Step 2)
# ---------------------------------------------------------------------------


class TestCompleteTrace:
    """Tests for trace integration in the Complete verb."""

    async def test_complete_generates_trace_id(self, mock_backend: MockBackend) -> None:
        """Complete generates a trace_id on the result."""
        saia = make_saia(
            mock_backend,
            tools=[ToolDef(name="search", description="s", parameters={})],
            executor=lambda n, a: "result",
        )
        mock_backend.set_complete_response("done")
        result = await saia.complete("do something")
        assert result.trace.trace_id
        assert len(result.trace.trace_id) == 8

    async def test_complete_carries_request_id(self, mock_backend: MockBackend) -> None:
        """request_id from config is carried through to the result."""
        saia = make_saia(
            mock_backend,
            tools=[ToolDef(name="search", description="s", parameters={})],
            executor=lambda n, a: "result",
        )
        saia = saia.with_request_id("ext-984821")
        mock_backend.set_complete_response("done")
        result = await saia.complete("do something")
        assert result.trace.request_id == "ext-984821"


class TestWithRequestId:
    """Tests for SAIA.with_request_id() context pattern."""

    async def test_with_request_id_creates_new_saia(self, mock_backend: MockBackend) -> None:
        """with_request_id returns a new SAIA instance."""
        saia = make_saia(mock_backend)
        tagged = saia.with_request_id("req-1")
        assert tagged is not saia
        assert tagged.config.call.request_id == "req-1"
        assert saia.config.call.request_id is None

    async def test_with_request_id_shares_memory(self, mock_backend: MockBackend) -> None:
        """with_request_id shares memory with parent."""
        saia = make_saia(mock_backend)
        saia.store("key", "value")
        tagged = saia.with_request_id("req-1")
        assert tagged.recall("key") == ["value"]

    async def test_with_request_id_propagates_to_complete(self, mock_backend: MockBackend) -> None:
        """request_id from with_request_id reaches Complete result."""
        saia = make_saia(
            mock_backend,
            tools=[ToolDef(name="search", description="s", parameters={})],
            executor=lambda n, a: "result",
        )
        tagged = saia.with_request_id("ext-123")
        mock_backend.set_complete_response("done")
        result = await tagged.complete("do something")
        assert result.trace.request_id == "ext-123"

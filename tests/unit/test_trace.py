"""Tests for iteration trace: ID generation, Tracer, and Complete integration."""

from __future__ import annotations

import json
from dataclasses import replace
from io import StringIO
from pathlib import Path

import pytest

from llm_saia.core.backend import AgentResponse, ToolCall, ToolDef
from llm_saia.core.controller import Action, ActionType, Observation
from llm_saia.core.trace import Tracer, TracerFactory, build_base_trace, build_trace
from llm_saia.core.types import ClassifyResult, DecisionReason
from llm_saia.core.verb import Verb
from tests.unit.conftest import MockBackend, make_saia

pytestmark = pytest.mark.unit


class TestCallId:
    """Tests for call_id generation."""

    def test_generate_id_is_8_hex_chars(self) -> None:
        """Generated IDs are 8-character hex strings."""
        id1 = Verb._generate_id()
        assert len(id1) == 8
        int(id1, 16)  # raises if not valid hex

    def test_generate_id_is_unique(self) -> None:
        """Consecutive IDs are unique."""
        ids = {Verb._generate_id() for _ in range(100)}
        assert len(ids) == 100

    async def test_chat_attaches_call_id(self, mock_backend: MockBackend) -> None:
        """_chat() attaches a call_id to the response."""
        saia = make_saia(mock_backend)
        verb = saia.complete
        from llm_saia.core.backend import Message

        response = await verb._chat([Message(role="user", content="hi")], None)
        assert response.call_id
        assert len(response.call_id) == 8


class TestBuildTrace:
    """Tests for build_trace() helper."""

    def test_builds_from_observation_and_action(self) -> None:
        """build_trace produces an IterationTrace with correct fields."""
        response = AgentResponse(
            content="I'll use search",
            tool_calls=[],
            input_tokens=100,
            output_tokens=20,
            finish_reason="stop",
            call_id="abc12345",
        )
        obs = Observation(
            response=response,
            messages=[],
            iteration=3,
            task="do something",
            tool_names=["search"],
            terminal_tool=None,
        )
        action = Action(
            ActionType.INSTRUCT,
            DecisionReason.TEXT_TOOL_PATTERN,
            message="Please use tools",
        )

        trace = build_trace(
            obs,
            action,
            response,
            trace_id="task001",
            verb="Complete",
            phase="loop",
            request_id="user-req-1",
            classifier_called=False,
            iterations_since_nudge=2,
            consecutive_degenerate=1,
        )

        assert trace.trace_id == "task001"
        assert trace.call_id == "abc12345"
        assert trace.verb == "Complete"
        assert trace.phase == "loop"
        assert trace.request_id == "user-req-1"
        assert trace.iteration == 3
        assert trace.has_content is True
        assert trace.has_tool_calls is False
        assert trace.tool_call_count == 0
        assert trace.input_tokens == 100
        assert trace.output_tokens == 20
        assert trace.action == "instruct"
        assert trace.reason == "text_tool_pattern"
        assert trace.nudge_preview == "Please use tools"
        assert trace.iterations_since_nudge == 2
        assert trace.consecutive_degenerate == 1
        assert trace.classifier_called is False

    def test_captures_tool_names(self) -> None:
        """build_trace captures tool names from tool_calls."""
        response = AgentResponse(
            content="",
            tool_calls=[
                ToolCall(id="c1", name="search", arguments={}),
                ToolCall(id="c2", name="read_file", arguments={"path": "/tmp"}),
            ],
            call_id="def67890",
        )
        obs = Observation(
            response=response,
            messages=[],
            iteration=0,
            task="t",
            tool_names=["search", "read_file"],
            terminal_tool=None,
        )
        action = Action(ActionType.EXECUTE_TOOLS, DecisionReason.HAS_TOOL_CALLS)

        trace = build_trace(obs, action, response, trace_id="t1")
        assert trace.tool_names_used == ["search", "read_file"]
        assert trace.tool_call_count == 2
        assert trace.has_tool_calls is True

    def test_truncates_long_content(self) -> None:
        """Content preview is truncated to 200 chars."""
        long_content = "x" * 500
        response = AgentResponse(content=long_content, tool_calls=[], call_id="r1")
        obs = Observation(
            response=response,
            messages=[],
            iteration=0,
            task="t",
            tool_names=[],
            terminal_tool=None,
        )
        action = Action(ActionType.SKIP, DecisionReason.BACKOFF)

        trace = build_trace(obs, action, response, trace_id="t1")
        assert len(trace.content_preview) == 200


class TestBuildBaseTrace:
    """Tests for build_base_trace() helper."""

    def test_builds_complete_action_for_text_response(self) -> None:
        """build_base_trace sets action='complete' when no tool calls."""
        response = AgentResponse(
            content="hello",
            tool_calls=[],
            input_tokens=50,
            output_tokens=10,
            finish_reason="end_turn",
            call_id="c1",
        )
        trace = build_base_trace(
            response,
            trace_id="t1",
            verb="Ask",
            phase="direct",
            request_id="req-1",
        )
        assert trace.trace_id == "t1"
        assert trace.call_id == "c1"
        assert trace.verb == "Ask"
        assert trace.phase == "direct"
        assert trace.request_id == "req-1"
        assert trace.action == "complete"
        assert trace.reason == "base_trace"
        assert trace.has_content is True
        assert trace.has_tool_calls is False

    def test_builds_execute_tools_action_for_tool_response(self) -> None:
        """build_base_trace sets action='execute_tools' when tool calls present."""
        response = AgentResponse(
            content="",
            tool_calls=[ToolCall(id="c1", name="search", arguments={})],
            call_id="c2",
        )
        trace = build_base_trace(response, trace_id="t2", verb="Ask", phase="loop")
        assert trace.action == "execute_tools"
        assert trace.tool_names_used == ["search"]
        assert trace.tool_call_count == 1

    def test_iteration_tracking(self) -> None:
        """build_base_trace tracks iteration number."""
        response = AgentResponse(content="hi", tool_calls=[], call_id="c3")
        trace = build_base_trace(response, trace_id="t3", iteration=5, verb="Verify")
        assert trace.iteration == 5

    def test_no_controller_fields(self) -> None:
        """build_base_trace leaves controller internals as None."""
        response = AgentResponse(content="hi", tool_calls=[], call_id="c4")
        trace = build_base_trace(response, trace_id="t4", verb="Ask")
        assert trace.iterations_since_nudge is None
        assert trace.consecutive_degenerate is None
        assert trace.pending_terminal is None
        assert trace.classifier_called is False


class TestTracer:
    """Tests for Tracer JSONL output."""

    def test_writes_jsonl(self, tmp_path: Path) -> None:
        """Tracer produces valid JSONL via file factory."""
        path = tmp_path / "trace.jsonl"
        response = AgentResponse(content="hello", tool_calls=[], call_id="r1")
        obs = Observation(
            response=response,
            messages=[],
            iteration=0,
            task="t",
            tool_names=[],
            terminal_tool=None,
        )
        action = Action(ActionType.SKIP, DecisionReason.BACKOFF)
        record = build_trace(obs, action, response, trace_id="t1")

        with TracerFactory.file(path) as tracer:
            tracer.write(record)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["trace_id"] == "t1"
        assert data["call_id"] == "r1"
        assert data["action"] == "skip"

    def test_writes_metadata_header(self, tmp_path: Path) -> None:
        """Metadata is written as the first line via start()."""
        path = tmp_path / "trace.jsonl"
        response = AgentResponse(content="hi", tool_calls=[], call_id="r2")
        obs = Observation(
            response=response,
            messages=[],
            iteration=0,
            task="t",
            tool_names=[],
            terminal_tool=None,
        )
        action = Action(ActionType.COMPLETE, DecisionReason.CLASSIFIED_COMPLETE, output="result")
        record = build_trace(obs, action, response, trace_id="t2")

        with TracerFactory.file(path) as tracer:
            tracer.start({"trace_id": "t2", "request_id": "u1"})
            tracer.write(record)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        meta = json.loads(lines[0])
        assert meta["_meta"]["trace_id"] == "t2"
        assert meta["_meta"]["request_id"] == "u1"
        data = json.loads(lines[1])
        assert data["trace_id"] == "t2"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """TracerFactory.file creates parent directories if needed."""
        path = tmp_path / "sub" / "dir" / "trace.jsonl"
        response = AgentResponse(content="", tool_calls=[], call_id="r3")
        obs = Observation(
            response=response,
            messages=[],
            iteration=0,
            task="t",
            tool_names=[],
            terminal_tool=None,
        )
        action = Action(ActionType.SKIP, DecisionReason.BACKOFF)
        record = build_trace(obs, action, response, trace_id="t3")

        with TracerFactory.file(path) as tracer:
            tracer.write(record)

        assert path.exists()

    def test_stream_tracer(self) -> None:
        """TracerFactory.stream writes to a caller-provided stream."""
        buf = StringIO()
        response = AgentResponse(content="hi", tool_calls=[], call_id="r4")
        obs = Observation(
            response=response,
            messages=[],
            iteration=0,
            task="t",
            tool_names=[],
            terminal_tool=None,
        )
        action = Action(ActionType.SKIP, DecisionReason.BACKOFF)
        record = build_trace(obs, action, response, trace_id="t4")

        tracer = TracerFactory.stream(buf)
        tracer.write(record)

        data = json.loads(buf.getvalue().strip())
        assert data["trace_id"] == "t4"

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
        assert result.trace_id
        assert len(result.trace_id) == 8

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
        assert result.request_id == "ext-984821"

    async def test_complete_writes_trace(self, mock_backend: MockBackend, tmp_path: Path) -> None:
        """Complete writes JSONL trace when tracer is provided."""
        saia = make_saia(
            mock_backend,
            tools=[ToolDef(name="search", description="s", parameters={})],
            executor=lambda n, a: "result",
        )
        saia = saia.with_request_id("u1")
        mock_backend.set_complete_response("done")
        trace_path = tmp_path / "trace.jsonl"
        tracer = TracerFactory.file(trace_path)
        result = await saia.complete("do something", tracer=tracer)

        assert trace_path.exists()
        lines = trace_path.read_text().strip().split("\n")

        # First line is metadata
        meta = json.loads(lines[0])
        assert meta["_meta"]["trace_id"] == result.trace_id
        assert meta["_meta"]["request_id"] == "u1"

        # At least one iteration trace
        assert len(lines) >= 2
        data = json.loads(lines[1])
        assert data["trace_id"] == result.trace_id
        assert data["call_id"]  # should be set
        assert data["iteration"] == 0
        assert data["verb"] == "Complete"
        assert data["phase"] == "loop"
        assert data["request_id"] == "u1"

    async def test_complete_uses_config_tracer(
        self,
        mock_backend: MockBackend,
    ) -> None:
        """Complete uses config tracer when no per-call tracer given."""
        buf = StringIO()
        tracer = TracerFactory.stream(buf)
        saia = make_saia(
            mock_backend,
            tools=[ToolDef(name="search", description="s", parameters={})],
            executor=lambda n, a: "result",
        )
        saia._config = replace(saia._config, tracer=tracer)
        saia._init_verbs()
        mock_backend.set_complete_response("done")
        result = await saia.complete("do something")

        output = buf.getvalue().strip()
        assert output  # something was written
        lines = output.split("\n")
        meta = json.loads(lines[0])
        assert meta["_meta"]["trace_id"] == result.trace_id

    async def test_no_trace_without_tracer(self, mock_backend: MockBackend) -> None:
        """No trace written when no tracer configured."""
        saia = make_saia(
            mock_backend,
            tools=[ToolDef(name="search", description="s", parameters={})],
            executor=lambda n, a: "result",
        )
        mock_backend.set_structured_response(
            ClassifyResult, ClassifyResult(category="completed", confidence=0.9, reason="done")
        )
        mock_backend.set_complete_response("done")
        result = await saia.complete("do something")
        assert result.completed


class TestBaseVerbTrace:
    """Tests for universal tracing in non-Complete verbs."""

    async def test_ask_traces_with_config_tracer(self, mock_backend: MockBackend) -> None:
        """Ask verb writes trace records when config tracer is set."""
        buf = StringIO()
        tracer = TracerFactory.stream(buf)
        saia = make_saia(mock_backend)
        saia._config = replace(saia._config, tracer=tracer)
        saia._init_verbs()
        mock_backend.set_complete_response("The answer is 42")

        await saia.ask("question", "context")

        output = buf.getvalue().strip()
        assert output
        data = json.loads(output)
        assert data["verb"] == "Ask"
        assert data["phase"] == "direct"
        assert data["action"] == "complete"
        assert data["trace_id"]
        assert data["call_id"]

    async def test_verify_traces_with_config_tracer(self, mock_backend: MockBackend) -> None:
        """Verify verb writes trace records when config tracer is set."""
        buf = StringIO()
        tracer = TracerFactory.stream(buf)
        saia = make_saia(mock_backend)
        saia._config = replace(saia._config, tracer=tracer)
        saia._init_verbs()

        await saia.verify("claim", "criteria")

        output = buf.getvalue().strip()
        assert output
        data = json.loads(output)
        assert data["verb"] == "Verify"
        assert data["phase"] == "direct"

    async def test_no_trace_without_config_tracer(self, mock_backend: MockBackend) -> None:
        """Non-Complete verbs don't trace when no tracer configured."""
        saia = make_saia(mock_backend)
        mock_backend.set_complete_response("The answer")

        # Should not raise — no tracer to write to
        await saia.ask("question", "context")

    async def test_base_trace_includes_request_id(self, mock_backend: MockBackend) -> None:
        """Base verb trace includes request_id from config."""
        buf = StringIO()
        tracer = TracerFactory.stream(buf)
        saia = make_saia(mock_backend)
        # Update call options with request_id and add tracer to config
        new_call = replace(saia._config.call, request_id="req-42")
        saia._config = replace(saia._config, tracer=tracer, call=new_call)
        saia._init_verbs()
        mock_backend.set_complete_response("The answer")

        await saia.ask("question", "context")

        data = json.loads(buf.getvalue().strip())
        assert data["request_id"] == "req-42"

    async def test_loop_path_traces_each_iteration(self, mock_backend: MockBackend) -> None:
        """Verb _loop() traces each iteration when config tracer is set."""
        buf = StringIO()
        tracer = TracerFactory.stream(buf)
        saia = make_saia(
            mock_backend,
            tools=[ToolDef(name="search", description="s", parameters={})],
            executor=lambda n, a: "result",
        )
        saia._config = replace(saia._config, tracer=tracer)
        saia._init_verbs()

        # Queue: tool call response, then text response
        mock_backend.queue_response(
            AgentResponse(
                content="searching...",
                tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "test"})],
                finish_reason="tool_use",
            )
        )
        mock_backend.set_complete_response("found it")

        await saia.ask("find something", "context")

        lines = buf.getvalue().strip().split("\n")
        assert len(lines) >= 2  # At least 2 iterations
        first = json.loads(lines[0])
        assert first["phase"] == "loop"
        assert first["verb"] == "Ask"
        second = json.loads(lines[1])
        assert second["phase"] == "loop"


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
        assert result.request_id == "ext-123"

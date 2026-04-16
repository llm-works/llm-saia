"""Tests for IterationGuard feature and built-in iteration guards."""

from __future__ import annotations

from typing import Any

import pytest

from llm_saia import IterationContext, IterationGuard, OutputGuard
from llm_saia.core.backend import AgentResponse
from llm_saia.core.config import CallOptions, Config, TerminalConfig
from llm_saia.core.conversation import Message, ToolCall
from llm_saia.core.logger import NullLogger
from llm_saia.core.types import ToolDef
from llm_saia.guards import _ordinal, contradiction, terminal_schema, terminal_status
from llm_saia.verbs import Ask, Instruct
from llm_saia.verbs.complete import Complete
from tests.unit.conftest import MockBackend

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_response(content: str = "", tool_calls: list[ToolCall] | None = None) -> AgentResponse:
    """Create an AgentResponse with optional tool calls."""
    return AgentResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="tool_use" if tool_calls else "end_turn",
    )


def _make_tool_call(name: str = "search", args: dict[str, Any] | None = None) -> ToolCall:
    return ToolCall(id="tc_1", name=name, arguments=args or {})


SEARCH_TOOL = ToolDef(
    name="search",
    description="Search for something",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)

DONE_TOOL = ToolDef(
    name="done",
    description="Signal completion",
    parameters={
        "type": "object",
        "properties": {"output": {"type": "string"}, "status": {"type": "string"}},
    },
)


def _make_config_with_tools(
    backend: MockBackend,
    call: CallOptions | None = None,
    terminal_tool: str | None = None,
) -> Config:
    """Create a Config with search tool."""
    terminal = TerminalConfig(tool=terminal_tool) if terminal_tool else None
    tools = [SEARCH_TOOL]
    if terminal_tool:
        tools.append(DONE_TOOL)

    async def executor(name: str, args: dict[str, Any]) -> str:
        return f"result for {name}"

    return Config(
        lg=NullLogger(),
        backend=backend,
        tools=tools,
        executor=executor,
        call=call,
        terminal=terminal,
    )


# ---------------------------------------------------------------------------
# IterationGuard dataclass
# ---------------------------------------------------------------------------


class TestIterationGuardDataclass:
    """Tests for IterationGuard construction."""

    def test_create_guard(self) -> None:
        guard = IterationGuard(validator=lambda ctx: None, name="test")
        assert guard.name == "test"
        assert guard.validator is not None

    def test_guard_frozen(self) -> None:
        guard = IterationGuard(validator=lambda ctx: None)
        with pytest.raises(AttributeError):
            guard.name = "changed"  # type: ignore[misc]

    def test_guard_default_name_is_none(self) -> None:
        guard = IterationGuard(validator=lambda ctx: None)
        assert guard.name is None


# ---------------------------------------------------------------------------
# with_guard / with_guards dispatch
# ---------------------------------------------------------------------------


class TestGuardDispatch:
    """Tests that with_guard routes IterationGuard vs OutputGuard correctly."""

    def test_with_guard_routes_output_guard(self) -> None:
        backend = MockBackend()
        config = Config(lg=NullLogger(), backend=backend, tools=[], executor=None)
        verb = Ask(config)
        out_guard = OutputGuard(validator=lambda x: None, retry_instruction="Fix.")
        result = verb.with_guard(out_guard)
        assert out_guard in result._call.output_guards
        assert len(result._call.iteration_guards) == 0

    def test_with_guard_routes_iteration_guard(self) -> None:
        backend = MockBackend()
        config = Config(lg=NullLogger(), backend=backend, tools=[], executor=None)
        verb = Ask(config)
        iter_guard = IterationGuard(validator=lambda ctx: None, name="test")
        result = verb.with_guard(iter_guard)
        assert iter_guard in result._call.iteration_guards
        assert len(result._call.output_guards) == 0

    def test_with_guards_mixed(self) -> None:
        backend = MockBackend()
        config = Config(lg=NullLogger(), backend=backend, tools=[], executor=None)
        verb = Ask(config)
        out_guard = OutputGuard(validator=lambda x: None, retry_instruction="Fix.")
        iter_guard = IterationGuard(validator=lambda ctx: None, name="test")
        result = verb.with_guards(out_guard, iter_guard)
        assert out_guard in result._call.output_guards
        assert iter_guard in result._call.iteration_guards


# ---------------------------------------------------------------------------
# Iteration guard in _loop (base verb with tools)
# ---------------------------------------------------------------------------


class TestIterationGuardInLoop:
    """Tests for iteration guards in the base verb tool-calling loop."""

    async def test_guard_passes_no_feedback(self) -> None:
        """When guard returns None, loop proceeds normally."""
        backend = MockBackend()
        # First call: tool call, second call: final response
        backend.queue_response(
            _tool_response("I'll search", [_make_tool_call("search", {"q": "test"})])
        )
        backend.queue_response(_tool_response("Here are the results"))

        guard = IterationGuard(validator=lambda ctx: None, name="always_pass")
        call = CallOptions(iteration_guards=(guard,), max_iterations=5)
        config = _make_config_with_tools(backend, call=call)
        verb = Instruct(config)
        result = await verb("do something")
        assert result.value == "Here are the results"

    async def test_guard_injects_feedback(self) -> None:
        """When guard returns feedback, it's injected as a user message."""
        backend = MockBackend()
        call_count = 0
        captured_messages: list[list[Message]] = []

        original_chat = backend.chat

        async def tracking_chat(messages: list[Message], **kwargs: Any) -> AgentResponse:
            nonlocal call_count
            call_count += 1
            captured_messages.append(list(messages))
            return await original_chat(messages, **kwargs)

        backend.chat = tracking_chat  # type: ignore[assignment]

        # First: tool call without explanation → guard fires
        backend.queue_response(_tool_response("", [_make_tool_call("search", {"q": "test"})]))
        # Second: tool call with explanation → guard passes
        backend.queue_response(
            _tool_response(
                "Searching for test results",
                [_make_tool_call("search", {"q": "test2"})],
            )
        )
        # Third: final response
        backend.queue_response(_tool_response("Done"))

        def require_narrative(ctx: IterationContext) -> str | None:
            if ctx.response.tool_calls and not (ctx.response.content or "").strip():
                return "Explain what you're doing and why."
            return None

        guard = IterationGuard(validator=require_narrative, name="narrative")
        call = CallOptions(iteration_guards=(guard,), max_iterations=10)
        config = _make_config_with_tools(backend, call=call)
        verb = Instruct(config)
        result = await verb("do something")

        assert result.value == "Done"
        assert call_count == 3

        # The second call should have the guard feedback in messages
        second_call_msgs = captured_messages[1]
        feedback_msgs = [m for m in second_call_msgs if "Explain what" in m.content]
        assert len(feedback_msgs) == 1

    async def test_multiple_guards_combined_feedback(self) -> None:
        """Multiple failing guards combine feedback with double newline."""
        backend = MockBackend()
        # First: triggers both guards (empty content, has tool call)
        backend.queue_response(_tool_response("", [_make_tool_call("search")]))
        # Second (after guard feedback): passes both guards, has tool call
        backend.queue_response(
            _tool_response("Explained and detailed", [_make_tool_call("search")])
        )
        # Third: final response (no tools, passes both guards)
        backend.queue_response(_tool_response("All done with results"))

        guard1 = IterationGuard(
            validator=lambda ctx: (
                "Need explanation." if not (ctx.response.content or "").strip() else None
            ),
            name="explain",
        )
        guard2 = IterationGuard(
            validator=lambda ctx: "Be verbose." if len(ctx.response.content or "") < 5 else None,
            name="verbose",
        )

        call = CallOptions(iteration_guards=(guard1, guard2), max_iterations=10)
        config = _make_config_with_tools(backend, call=call)
        verb = Instruct(config)
        result = await verb("do something")
        assert result.value == "All done with results"

    async def test_guard_outcome_in_trace(self) -> None:
        """Guard outcomes are recorded in the trace Step."""
        backend = MockBackend()
        # Trigger guard, then succeed
        backend.queue_response(_tool_response("", [_make_tool_call("search")]))
        backend.queue_response(_tool_response("Done"))

        guard = IterationGuard(
            validator=lambda ctx: "Explain." if not (ctx.response.content or "").strip() else None,
            name="narrative",
        )
        call = CallOptions(iteration_guards=(guard,), max_iterations=10)
        config = _make_config_with_tools(backend, call=call)
        verb = Instruct(config)
        result = await verb("do something")

        # Check the trace has guard outcomes
        trace = result.trace
        assert trace is not None
        # First step should have guard outcome (failed)
        first_step = trace.steps[0]
        assert len(first_step.guards) > 0
        assert first_step.guards[0].name == "narrative"
        assert first_step.guards[0].passed is False
        assert first_step.guards[0].error == "Explain."

    async def test_validator_exception_becomes_feedback(self) -> None:
        """If a guard validator raises, the exception becomes feedback."""
        backend = MockBackend()
        backend.queue_response(_tool_response("content", [_make_tool_call("search")]))
        backend.queue_response(_tool_response("Done"))

        calls = 0

        def bad_once_validator(ctx: IterationContext) -> str | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("validator broke")
            return None

        guard = IterationGuard(validator=bad_once_validator, name="broken_once")
        call = CallOptions(iteration_guards=(guard,), max_iterations=10)
        config = _make_config_with_tools(backend, call=call)
        verb = Instruct(config)
        result = await verb("do something")
        assert result.value == "Done"
        assert calls == 2


# ---------------------------------------------------------------------------
# Iteration guard in Complete verb
# ---------------------------------------------------------------------------


class TestIterationGuardInComplete:
    """Tests for iteration guards in the Complete verb's loop."""

    async def test_guard_fires_in_complete(self) -> None:
        """Iteration guard fires and injects feedback in Complete verb."""
        backend = MockBackend()
        call_count = 0

        original_chat = backend.chat

        async def tracking_chat(messages: list[Message], **kwargs: Any) -> AgentResponse:
            nonlocal call_count
            call_count += 1
            return await original_chat(messages, **kwargs)

        backend.chat = tracking_chat  # type: ignore[assignment]

        # First: tool call without narrative → guard fires
        backend.queue_response(_tool_response("", [_make_tool_call("search", {"q": "test"})]))
        # Second: explains and calls done tool → controller asks for confirmation
        done_call = ToolCall(
            id="tc_done", name="done", arguments={"output": "found it", "status": "ok"}
        )
        backend.queue_response(_tool_response("Found what I needed", [done_call]))
        # Third: confirms done tool (controller needs confirmation)
        done_confirm = ToolCall(
            id="tc_done2", name="done", arguments={"output": "found it", "status": "ok"}
        )
        backend.queue_response(_tool_response("Confirming", [done_confirm]))

        def require_narrative(ctx: IterationContext) -> str | None:
            if ctx.response.tool_calls and not (ctx.response.content or "").strip():
                return "Explain."
            return None

        guard = IterationGuard(validator=require_narrative, name="narrative")
        call = CallOptions(iteration_guards=(guard,), max_iterations=10)
        config = _make_config_with_tools(backend, call=call, terminal_tool="done")
        verb = Complete(config)
        result = await verb("find something")

        assert result.completed
        assert call_count == 3

    async def test_guard_nudge_scored_as_non_productive(self) -> None:
        """Iteration guard nudges count as non-productive in LoopScore."""
        backend = MockBackend()
        # First: triggers guard
        backend.queue_response(_tool_response("", [_make_tool_call("search")]))
        # Second: passes guard, calls done
        done_call = ToolCall(id="tc_done", name="done", arguments={"output": "ok", "status": "ok"})
        backend.queue_response(_tool_response("Explained", [done_call]))

        guard = IterationGuard(
            validator=lambda ctx: "Explain." if not (ctx.response.content or "").strip() else None,
            name="narrative",
        )
        call = CallOptions(iteration_guards=(guard,), max_iterations=10)
        config = _make_config_with_tools(backend, call=call, terminal_tool="done")
        verb = Complete(config)
        result = await verb("do it")

        assert result.score is not None
        assert result.score.nudges >= 1

    async def test_guard_step_recorded_in_trace(self) -> None:
        """Guard nudge step is recorded in Complete's VerbTrace."""
        backend = MockBackend()
        # Trigger guard, then complete
        backend.queue_response(_tool_response("", [_make_tool_call("search")]))
        done_call = ToolCall(id="tc_done", name="done", arguments={"output": "ok", "status": "ok"})
        backend.queue_response(_tool_response("Explained", [done_call]))

        guard = IterationGuard(
            validator=lambda ctx: "Explain." if not (ctx.response.content or "").strip() else None,
            name="narrative",
        )
        call = CallOptions(iteration_guards=(guard,), max_iterations=10)
        config = _make_config_with_tools(backend, call=call, terminal_tool="done")
        verb = Complete(config)
        result = await verb("do it")

        assert result.trace is not None
        guard_steps = [s for s in result.trace.steps if s.reason == "iteration_guard"]
        assert len(guard_steps) >= 1
        assert guard_steps[0].nudge_preview == "Explain."


# ---------------------------------------------------------------------------
# Built-in iteration guard factories
# ---------------------------------------------------------------------------


def _make_response(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
) -> AgentResponse:
    """Create a minimal AgentResponse for testing."""
    return AgentResponse(
        content=content,
        tool_calls=tool_calls,
        input_tokens=10,
        output_tokens=10,
        call_id="test-call",
        finish_reason="stop",
    )


def _make_ctx(
    response: AgentResponse,
    iteration: int = 0,
    max_iterations: int = 10,
) -> IterationContext:
    """Wrap AgentResponse in IterationContext for guard testing."""
    return IterationContext(response=response, iteration=iteration, max_iterations=max_iterations)


class TestTerminalStatusGuard:
    """Tests for terminal_status guard factory."""

    def test_no_tool_calls_passes(self) -> None:
        guard = terminal_status("done", "status", ("stuck", "failed"))
        response = _make_response("Just some text")
        assert guard.validator(_make_ctx(response)) is None

    def test_non_terminal_tool_passes(self) -> None:
        guard = terminal_status("done", "status", ("stuck", "failed"))
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="other_tool", arguments={"status": "stuck"})]
        )
        assert guard.validator(_make_ctx(response)) is None

    def test_success_status_passes(self) -> None:
        guard = terminal_status("done", "status", ("stuck", "failed"))
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="done", arguments={"status": "complete"})]
        )
        assert guard.validator(_make_ctx(response)) is None

    def test_failure_status_returns_feedback(self) -> None:
        guard = terminal_status("done", "status", ("stuck", "failed"))
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="done", arguments={"status": "stuck"})]
        )
        feedback = guard.validator(_make_ctx(response))
        assert feedback is not None
        assert "stuck" in feedback

    def test_escalation_on_repeated_failures(self) -> None:
        guard = terminal_status("done", "status", ("stuck",), max_retries=3, escalate=True)
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="done", arguments={"status": "stuck"})]
        )
        # Use incrementing iterations to simulate consecutive failures within same task
        feedback1 = guard.validator(_make_ctx(response, iteration=0))
        assert feedback1 is not None
        assert "MUST" not in feedback1

        feedback2 = guard.validator(_make_ctx(response, iteration=1))
        assert feedback2 is not None
        assert "MUST" in feedback2

    def test_exhausted_retries_passes(self) -> None:
        guard = terminal_status("done", "status", ("stuck",), max_retries=2)
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="done", arguments={"status": "stuck"})]
        )
        # Use incrementing iterations to simulate consecutive failures within same task
        assert guard.validator(_make_ctx(response, iteration=0)) is not None
        assert guard.validator(_make_ctx(response, iteration=1)) is not None
        assert guard.validator(_make_ctx(response, iteration=2)) is None

    def test_state_resets_on_new_task(self) -> None:
        """Guard state resets when iteration=0 (new task starts)."""
        guard = terminal_status("done", "status", ("stuck",), max_retries=2)
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="done", arguments={"status": "stuck"})]
        )
        # First task: exhaust retries
        assert guard.validator(_make_ctx(response, iteration=0)) is not None
        assert guard.validator(_make_ctx(response, iteration=1)) is not None
        assert guard.validator(_make_ctx(response, iteration=2)) is None  # exhausted

        # Second task: iteration=0 resets state, guard fires again
        assert guard.validator(_make_ctx(response, iteration=0)) is not None
        assert guard.validator(_make_ctx(response, iteration=1)) is not None
        assert guard.validator(_make_ctx(response, iteration=2)) is None  # exhausted again

    def test_non_dict_arguments_passes(self) -> None:
        guard = terminal_status("done", "status", ("stuck",))
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="done", arguments="not a dict")]  # type: ignore
        )
        assert guard.validator(_make_ctx(response)) is None


class TestTerminalSchemaGuard:
    """Tests for terminal_schema guard factory."""

    @pytest.fixture
    def tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                name="report",
                description="Report findings",
                parameters={
                    "type": "object",
                    "required": ["summary", "score"],
                    "properties": {
                        "summary": {"type": "string"},
                        "score": {"type": "integer"},
                    },
                },
            ),
        ]

    def test_no_tool_calls_passes(self, tools: list[ToolDef]) -> None:
        guard = terminal_schema(tools, "report")
        response = _make_response("Just some text")
        assert guard.validator(_make_ctx(response)) is None

    def test_valid_arguments_passes(self, tools: list[ToolDef]) -> None:
        guard = terminal_schema(tools, "report")
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="report", arguments={"summary": "done", "score": 5})]
        )
        assert guard.validator(_make_ctx(response)) is None

    def test_missing_required_field_returns_feedback(self, tools: list[ToolDef]) -> None:
        guard = terminal_schema(tools, "report")
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="report", arguments={"summary": "done"})]
        )
        feedback = guard.validator(_make_ctx(response))
        assert feedback is not None
        assert "schema errors" in feedback
        assert "score" in feedback

    def test_wrong_type_returns_feedback(self, tools: list[ToolDef]) -> None:
        guard = terminal_schema(tools, "report")
        response = _make_response(
            tool_calls=[
                ToolCall(id="1", name="report", arguments={"summary": "done", "score": "five"})
            ]
        )
        feedback = guard.validator(_make_ctx(response))
        assert feedback is not None
        assert "expected integer" in feedback

    def test_escalation_on_repeated_failures(self, tools: list[ToolDef]) -> None:
        guard = terminal_schema(tools, "report", max_retries=3, escalate=True)
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="report", arguments={"summary": "done"})]
        )
        # Use incrementing iterations to simulate consecutive failures within same task
        feedback1 = guard.validator(_make_ctx(response, iteration=0))
        assert feedback1 is not None
        assert "STILL" not in feedback1

        feedback2 = guard.validator(_make_ctx(response, iteration=1))
        assert feedback2 is not None
        assert "STILL" in feedback2

    def test_exhausted_retries_passes(self, tools: list[ToolDef]) -> None:
        guard = terminal_schema(tools, "report", max_retries=2)
        response = _make_response(tool_calls=[ToolCall(id="1", name="report", arguments={})])
        # Use incrementing iterations to simulate consecutive failures within same task
        assert guard.validator(_make_ctx(response, iteration=0)) is not None
        assert guard.validator(_make_ctx(response, iteration=1)) is not None
        assert guard.validator(_make_ctx(response, iteration=2)) is None

    def test_unknown_tool_creates_noop_guard(self) -> None:
        guard = terminal_schema([], "nonexistent")
        response = _make_response(tool_calls=[ToolCall(id="1", name="nonexistent", arguments={})])
        assert guard.validator(_make_ctx(response)) is None


class TestContradictionGuard:
    """Tests for contradiction guard factory."""

    def test_no_tool_calls_passes(self) -> None:
        guard = contradiction("done")
        response = _make_response("I can't do this")
        assert guard.validator(_make_ctx(response)) is None

    def test_non_terminal_tool_passes(self) -> None:
        guard = contradiction("done")
        response = _make_response(
            content="However, I can't do this",
            tool_calls=[ToolCall(id="1", name="other_tool", arguments={})],
        )
        assert guard.validator(_make_ctx(response)) is None

    def test_terminal_without_contradiction_passes(self) -> None:
        guard = contradiction("done")
        response = _make_response(
            content="Task completed successfully!",
            tool_calls=[ToolCall(id="1", name="done", arguments={})],
        )
        assert guard.validator(_make_ctx(response)) is None

    def test_terminal_with_empty_content_passes(self) -> None:
        guard = contradiction("done")
        response = _make_response(
            content="",
            tool_calls=[ToolCall(id="1", name="done", arguments={})],
        )
        assert guard.validator(_make_ctx(response)) is None

    def test_contradiction_detected(self) -> None:
        guard = contradiction("done")
        response = _make_response(
            content="However, I wasn't able to complete everything",
            tool_calls=[ToolCall(id="1", name="done", arguments={})],
        )
        feedback = guard.validator(_make_ctx(response))
        assert feedback is not None
        assert "contradictory" in feedback

    def test_detects_various_signals(self) -> None:
        signals = ["unfortunately", "i can't", "i cannot", "not possible", "unable to"]
        for signal in signals:
            guard = contradiction("done")
            response = _make_response(
                content=f"Task done, {signal} verify everything",
                tool_calls=[ToolCall(id="1", name="done", arguments={})],
            )
            feedback = guard.validator(_make_ctx(response))
            assert feedback is not None, f"Should detect signal: {signal}"

    def test_escalation_on_repeated_contradictions(self) -> None:
        guard = contradiction("done", max_retries=3, escalate=True)
        response = _make_response(
            content="However, something went wrong",
            tool_calls=[ToolCall(id="1", name="done", arguments={})],
        )
        # Use incrementing iterations to simulate consecutive failures within same task
        feedback1 = guard.validator(_make_ctx(response, iteration=0))
        assert feedback1 is not None
        assert "2nd" not in feedback1

        feedback2 = guard.validator(_make_ctx(response, iteration=1))
        assert feedback2 is not None
        assert "2nd" in feedback2

    def test_exhausted_retries_passes(self) -> None:
        guard = contradiction("done", max_retries=2)
        response = _make_response(
            content="Unfortunately this failed",
            tool_calls=[ToolCall(id="1", name="done", arguments={})],
        )
        # Use incrementing iterations to simulate consecutive failures within same task
        assert guard.validator(_make_ctx(response, iteration=0)) is not None
        assert guard.validator(_make_ctx(response, iteration=1)) is not None
        assert guard.validator(_make_ctx(response, iteration=2)) is None


class TestOrdinalHelper:
    """Tests for _ordinal helper."""

    def test_ordinals(self) -> None:
        assert _ordinal(1) == "1st"
        assert _ordinal(2) == "2nd"
        assert _ordinal(3) == "3rd"
        assert _ordinal(4) == "4th"
        assert _ordinal(11) == "11th"
        assert _ordinal(12) == "12th"
        assert _ordinal(13) == "13th"
        assert _ordinal(21) == "21st"
        assert _ordinal(22) == "22nd"
        assert _ordinal(23) == "23rd"

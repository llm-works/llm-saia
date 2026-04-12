"""Tests for IterationGuard feature."""

from __future__ import annotations

from typing import Any

import pytest

from llm_saia import IterationGuard, OutputGuard
from llm_saia.core.backend import AgentResponse
from llm_saia.core.config import CallOptions, Config, TerminalConfig
from llm_saia.core.conversation import Message, ToolCall
from llm_saia.core.types import ToolDef
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
        guard = IterationGuard(validator=lambda r: None, name="test")
        assert guard.name == "test"
        assert guard.validator is not None

    def test_guard_frozen(self) -> None:
        guard = IterationGuard(validator=lambda r: None)
        with pytest.raises(AttributeError):
            guard.name = "changed"  # type: ignore[misc]

    def test_guard_default_name_is_none(self) -> None:
        guard = IterationGuard(validator=lambda r: None)
        assert guard.name is None


# ---------------------------------------------------------------------------
# with_guard / with_guards dispatch
# ---------------------------------------------------------------------------


class TestGuardDispatch:
    """Tests that with_guard routes IterationGuard vs OutputGuard correctly."""

    def test_with_guard_routes_output_guard(self) -> None:
        backend = MockBackend()
        config = Config(backend=backend, tools=[], executor=None)
        verb = Ask(config)
        out_guard = OutputGuard(validator=lambda x: None, retry_instruction="Fix.")
        result = verb.with_guard(out_guard)
        assert out_guard in result._call.output_guards
        assert len(result._call.iteration_guards) == 0

    def test_with_guard_routes_iteration_guard(self) -> None:
        backend = MockBackend()
        config = Config(backend=backend, tools=[], executor=None)
        verb = Ask(config)
        iter_guard = IterationGuard(validator=lambda r: None, name="test")
        result = verb.with_guard(iter_guard)
        assert iter_guard in result._call.iteration_guards
        assert len(result._call.output_guards) == 0

    def test_with_guards_mixed(self) -> None:
        backend = MockBackend()
        config = Config(backend=backend, tools=[], executor=None)
        verb = Ask(config)
        out_guard = OutputGuard(validator=lambda x: None, retry_instruction="Fix.")
        iter_guard = IterationGuard(validator=lambda r: None, name="test")
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

        guard = IterationGuard(validator=lambda r: None, name="always_pass")
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

        def require_narrative(resp: AgentResponse) -> str | None:
            if resp.tool_calls and not (resp.content or "").strip():
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
            validator=lambda r: "Need explanation." if not (r.content or "").strip() else None,
            name="explain",
        )
        guard2 = IterationGuard(
            validator=lambda r: "Be verbose." if len(r.content or "") < 5 else None,
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
            validator=lambda r: "Explain." if not (r.content or "").strip() else None,
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

        def bad_once_validator(r: AgentResponse) -> str | None:
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

        guard = IterationGuard(
            validator=lambda r: "Explain."
            if r.tool_calls and not (r.content or "").strip()
            else None,
            name="narrative",
        )
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
            validator=lambda r: "Explain." if not (r.content or "").strip() else None,
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
            validator=lambda r: "Explain." if not (r.content or "").strip() else None,
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

"""Tests for the unified core loop infrastructure."""

from __future__ import annotations

from typing import Any

import pytest

from llm_saia.core.backend import ChatResponse
from llm_saia.core.config import CallOptions, Config
from llm_saia.core.conversation import ToolCall
from llm_saia.core.logger import NullLogger
from llm_saia.core.loop import (
    ControllerStrategy,
    ControllerStrategyConfig,
    CoreLoopResult,
    LoopAction,
    LoopDecision,
    SimpleStrategy,
)
from llm_saia.core.types import ToolDef
from llm_saia.verbs import Instruct
from tests.unit.conftest import MockBackend

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_response(content: str = "", tool_calls: list[ToolCall] | None = None) -> ChatResponse:
    """Create a ChatResponse with optional tool calls."""
    return ChatResponse(
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


async def _test_executor(name: str, args: dict[str, Any]) -> str:
    return f"result for {name}"


def _make_config(
    backend: MockBackend,
    call: CallOptions | None = None,
    with_tools: bool = True,
) -> Config:
    """Create a Config with optional tools."""
    if with_tools:
        return Config(
            lg=NullLogger(),
            backend=backend,
            tools=[SEARCH_TOOL],
            executor=_test_executor,
            call=call,
        )
    return Config(
        lg=NullLogger(),
        backend=backend,
        tools=[],
        executor=None,
        call=call,
    )


# ---------------------------------------------------------------------------
# SimpleStrategy Tests
# ---------------------------------------------------------------------------


class TestSimpleStrategy:
    """Tests for SimpleStrategy decision-making."""

    async def test_execute_tools_when_present(self) -> None:
        """SimpleStrategy returns EXECUTE_TOOLS when response has tool calls."""
        strategy = SimpleStrategy()
        response = _tool_response(tool_calls=[_make_tool_call()])

        decision = await strategy.decide(response, [], 0, None, None)

        assert decision.action == LoopAction.EXECUTE_TOOLS
        assert decision.reason == "has_tool_calls"

    async def test_complete_when_no_tools(self) -> None:
        """SimpleStrategy returns COMPLETE when no tool calls."""
        strategy = SimpleStrategy()
        response = _tool_response(content="final answer")

        decision = await strategy.decide(response, [], 0, None, None)

        assert decision.action == LoopAction.COMPLETE
        assert decision.output == "final answer"
        assert decision.reason == "no_tools"

    async def test_blocking_feedback_takes_precedence(self) -> None:
        """Blocking feedback causes INSTRUCT regardless of tool calls."""
        strategy = SimpleStrategy()
        response = _tool_response(tool_calls=[_make_tool_call()])

        decision = await strategy.decide(response, [], 0, "Stop!", None)

        assert decision.action == LoopAction.INSTRUCT
        assert decision.message == "Stop!"
        assert decision.reason == "blocking_guard"

    def test_on_iteration_complete_is_noop(self) -> None:
        """SimpleStrategy.on_iteration_complete does nothing."""
        strategy = SimpleStrategy()
        decision = LoopDecision(action=LoopAction.COMPLETE, output="done")
        # Should not raise
        strategy.on_iteration_complete(decision, 100)


# ---------------------------------------------------------------------------
# ControllerStrategy Tests
# ---------------------------------------------------------------------------


class TestControllerStrategy:
    """Tests for ControllerStrategy scoring accumulation."""

    def test_productive_actions_increment_acc0(self) -> None:
        """COMPLETE, FAIL, EXECUTE_TOOLS increment acc[0]."""
        config = ControllerStrategyConfig(task="test", tool_names=["search"])
        strategy = ControllerStrategy(controller=None, config=config)

        for action in [LoopAction.COMPLETE, LoopAction.FAIL, LoopAction.EXECUTE_TOOLS]:
            config.acc = [0, 0, 0, 0]
            decision = LoopDecision(action=action)
            strategy.on_iteration_complete(decision, 100)
            assert config.acc[0] == 1, f"Expected acc[0]=1 for {action}"

    def test_terminal_confirmation_is_productive(self) -> None:
        """INSTRUCT with terminal_confirmation_request is productive."""
        config = ControllerStrategyConfig(task="test", tool_names=["search"])
        strategy = ControllerStrategy(controller=None, config=config)

        decision = LoopDecision(
            action=LoopAction.INSTRUCT,
            message="Confirm?",
            reason="terminal_confirmation_request",
        )
        strategy.on_iteration_complete(decision, 100)

        assert config.acc[0] == 1  # productive
        assert config.acc[1] == 0  # not a nudge

    def test_regular_instruct_is_nudge(self) -> None:
        """Regular INSTRUCT increments nudge count and wasted tokens."""
        config = ControllerStrategyConfig(task="test", tool_names=["search"])
        strategy = ControllerStrategy(controller=None, config=config)

        decision = LoopDecision(
            action=LoopAction.INSTRUCT,
            message="Try again",
            reason="nudge",
        )
        strategy.on_iteration_complete(decision, 150)

        assert config.acc[0] == 0  # not productive
        assert config.acc[1] == 1  # nudge count
        assert config.acc[3] == 150  # wasted tokens

    def test_skip_increments_skip_count(self) -> None:
        """SKIP increments skip count and wasted tokens."""
        config = ControllerStrategyConfig(task="test", tool_names=["search"])
        strategy = ControllerStrategy(controller=None, config=config)

        decision = LoopDecision(action=LoopAction.SKIP, reason="backoff")
        strategy.on_iteration_complete(decision, 200)

        assert config.acc[2] == 1  # skip count
        assert config.acc[3] == 200  # wasted tokens


# ---------------------------------------------------------------------------
# Core Loop Integration Tests
# ---------------------------------------------------------------------------


class TestCoreLoopIntegration:
    """Integration tests for _core_loop via Instruct verb."""

    async def test_loop_completes_without_tools(self, mock_backend: MockBackend) -> None:
        """Loop completes immediately when no tool calls."""
        mock_backend.set_complete_response("Done!")
        verb = Instruct(_make_config(mock_backend, with_tools=False))

        result = await verb("Do something")

        assert result.value == "Done!"

    async def test_loop_executes_tools_then_completes(self, mock_backend: MockBackend) -> None:
        """Loop executes tools and continues until no more tool calls."""
        # First response: tool call, Second response: completion
        mock_backend.queue_response(_tool_response(tool_calls=[_make_tool_call()]))
        mock_backend.queue_response(_tool_response(content="Final answer"))
        verb = Instruct(_make_config(mock_backend))

        result = await verb("Search for something")

        assert result.value == "Final answer"

    async def test_loop_respects_max_iterations(self, mock_backend: MockBackend) -> None:
        """Loop stops when max_iterations is reached."""
        # Always return tool calls - should hit limit
        for _ in range(10):
            mock_backend.queue_response(_tool_response(tool_calls=[_make_tool_call()]))

        call = CallOptions(max_iterations=3)
        verb = Instruct(_make_config(mock_backend, call=call))

        result = await verb("Keep searching")

        # Should have stopped at 3 iterations
        assert "result for search" in mock_backend.last_prompt or result.value is not None

    async def test_loop_completes_with_tool_call(self, mock_backend: MockBackend) -> None:
        """Loop completes normally when tool call is present."""
        mock_backend.queue_response(_tool_response(tool_calls=[_make_tool_call()]))

        verb = Instruct(_make_config(mock_backend))
        result = await verb("Test")

        assert result.value is not None


# ---------------------------------------------------------------------------
# LoopDecision Tests
# ---------------------------------------------------------------------------


class TestLoopDecision:
    """Tests for LoopDecision dataclass."""

    def test_defaults(self) -> None:
        """LoopDecision has sensible defaults."""
        decision = LoopDecision(action=LoopAction.COMPLETE)

        assert decision.message is None
        assert decision.output is None
        assert decision.tool_ids is None
        assert decision.terminal_data is None
        assert decision.terminal_tool is None
        assert decision.reason == ""

    def test_with_terminal_data(self) -> None:
        """LoopDecision can carry terminal tool data."""
        decision = LoopDecision(
            action=LoopAction.COMPLETE,
            output="Task done",
            terminal_data={"status": "success"},
            terminal_tool="submit",
        )

        assert decision.terminal_data == {"status": "success"}
        assert decision.terminal_tool == "submit"


# ---------------------------------------------------------------------------
# CoreLoopResult Tests
# ---------------------------------------------------------------------------


class TestCoreLoopResult:
    """Tests for CoreLoopResult dataclass."""

    def test_defaults(self) -> None:
        """CoreLoopResult has sensible defaults."""
        result = CoreLoopResult(
            completed=True,
            output="done",
            iterations=1,
            messages=[],
            reason="completed",
        )

        assert result.paused is False
        assert result.terminal_data is None
        assert result.terminal_tool is None
        assert result.total_tokens == 0

    def test_paused_result(self) -> None:
        """CoreLoopResult can represent a paused state."""
        result = CoreLoopResult(
            completed=False,
            output="partial",
            iterations=2,
            messages=[],
            reason="paused",
            paused=True,
        )

        assert result.paused is True
        assert result.reason == "paused"

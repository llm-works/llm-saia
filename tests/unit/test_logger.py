"""Tests for logger protocol and integration."""

from typing import Any

import pytest

from llm_saia import SAIA, NullLogger, SAIALogger
from llm_saia.core.types import AgentResponse, ToolCall, ToolDef
from tests.unit.conftest import MockBackend

pytestmark = pytest.mark.unit


class RecordingLogger:
    """Test logger that records all calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def trace(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self.calls.append(("trace", msg, extra))

    def debug(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self.calls.append(("debug", msg, extra))

    def info(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self.calls.append(("info", msg, extra))

    def warning(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self.calls.append(("warning", msg, extra))

    def error(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self.calls.append(("error", msg, extra))


class TestSAIALogger:
    def test_protocol_satisfaction(self) -> None:
        """RecordingLogger satisfies SAIALogger protocol."""
        logger = RecordingLogger()
        assert isinstance(logger, SAIALogger)

    def test_null_logger_satisfies_protocol(self) -> None:
        """NullLogger satisfies SAIALogger protocol."""
        logger = NullLogger()
        assert isinstance(logger, SAIALogger)

    def test_null_logger_no_op(self) -> None:
        """NullLogger does nothing."""
        logger = NullLogger()
        logger.trace("test")
        logger.debug("test")
        logger.info("test", extra={"key": "value"})
        logger.warning("test")
        logger.error("test")
        # No exception raised, no output


class TestLoggerIntegration:
    def test_saia_accepts_logger(self, mock_backend: MockBackend) -> None:
        """SAIA accepts a logger parameter."""
        logger = RecordingLogger()
        saia = SAIA(backend=mock_backend, lg=logger)
        assert saia._lg is logger

    def test_saia_without_logger(self, mock_backend: MockBackend) -> None:
        """SAIA works without a logger."""
        saia = SAIA(backend=mock_backend)
        assert saia._lg is None

    def test_logger_preserved_in_with_methods(self, mock_backend: MockBackend) -> None:
        """Logger is preserved through with_* methods."""
        logger = RecordingLogger()
        saia = SAIA(backend=mock_backend, lg=logger)

        saia2 = saia.with_max_iterations(5)
        assert saia2._lg is logger

        saia3 = saia.with_single_call()
        assert saia3._lg is logger

    async def test_logger_called_during_verb_with_tools(self, mock_backend: MockBackend) -> None:
        """Logger is called during verb execution when tools are configured."""
        logger = RecordingLogger()
        tools = [ToolDef(name="test_tool", description="test", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            return "result"

        saia = SAIA(backend=mock_backend, tools=tools, executor=executor, lg=logger)

        await saia.ask("artifact", "question")

        # Should have log calls from the loop
        assert len(logger.calls) > 0
        messages = [call[1] for call in logger.calls]
        assert "verb loop started" in messages

    async def test_loop_logs_start_and_response(self, mock_backend: MockBackend) -> None:
        """Loop logs start and LLM response."""
        logger = RecordingLogger()
        tools = [ToolDef(name="test_tool", description="test", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            return "result"

        saia = SAIA(backend=mock_backend, tools=tools, executor=executor, lg=logger)

        # Queue a tool call followed by completion
        mock_backend.queue_response(
            AgentResponse(
                content="calling tool",
                tool_calls=[ToolCall(id="1", name="test_tool", arguments={})],
                stop_reason="tool_use",
            )
        )
        mock_backend.queue_response(
            AgentResponse(content="done", tool_calls=[], stop_reason="end_turn")
        )

        await saia.instruct("do something")

        # Check for expected log messages
        messages = [call[1] for call in logger.calls]
        assert "verb loop started" in messages
        assert "llm response received" in messages
        assert "executing tool..." in messages

    async def test_loop_logs_limit_reached(self, mock_backend: MockBackend) -> None:
        """Loop logs when iteration limit is reached."""
        logger = RecordingLogger()
        tools = [ToolDef(name="test_tool", description="test", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            return "result"

        saia = SAIA(backend=mock_backend, tools=tools, executor=executor, lg=logger)

        # Queue multiple tool calls that exceed max_iterations
        for _ in range(5):
            mock_backend.queue_response(
                AgentResponse(
                    content="calling tool",
                    tool_calls=[ToolCall(id="1", name="test_tool", arguments={})],
                    stop_reason="tool_use",
                )
            )

        await saia.with_max_iterations(2).instruct("do something")

        # Check for limit warning
        messages = [call[1] for call in logger.calls]
        assert "verb loop limit reached" in messages

        # Check limit type is correct
        limit_call = [c for c in logger.calls if c[1] == "verb loop limit reached"][0]
        assert limit_call[2] is not None
        assert limit_call[2]["limit_type"] == "max_iterations"

    async def test_tool_error_logged(self, mock_backend: MockBackend) -> None:
        """Tool execution errors are logged."""
        logger = RecordingLogger()
        tools = [ToolDef(name="failing_tool", description="test", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            raise ValueError("Tool failed!")

        saia = SAIA(backend=mock_backend, tools=tools, executor=executor, lg=logger)

        mock_backend.queue_response(
            AgentResponse(
                content="calling tool",
                tool_calls=[ToolCall(id="1", name="failing_tool", arguments={})],
                stop_reason="tool_use",
            )
        )
        mock_backend.queue_response(
            AgentResponse(content="done", tool_calls=[], stop_reason="end_turn")
        )

        await saia.instruct("do something")

        # Check for error warning
        warning_calls = [c for c in logger.calls if c[0] == "warning"]
        assert len(warning_calls) > 0
        error_call = [c for c in warning_calls if c[1] == "tool execution failed"][0]
        assert error_call[2] is not None
        assert error_call[2]["tool"] == "failing_tool"
        assert isinstance(error_call[2]["exception"], ValueError)

    async def test_complete_verb_logs(self, mock_backend: MockBackend) -> None:
        """Complete verb logs start, responses, and limit reached."""
        logger = RecordingLogger()
        tools = [ToolDef(name="test_tool", description="test", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            return "result"

        saia = SAIA(backend=mock_backend, tools=tools, executor=executor, lg=logger)

        # Queue tool calls that exceed max_iterations (Complete has its own loop)
        for _ in range(5):
            mock_backend.queue_response(
                AgentResponse(
                    content="calling tool",
                    tool_calls=[ToolCall(id="1", name="test_tool", arguments={})],
                    stop_reason="tool_use",
                )
            )

        await saia.with_max_iterations(2).complete("do a task")

        # Verify Complete logged properly
        messages = [call[1] for call in logger.calls]
        assert "verb loop started" in messages
        assert "llm response received" in messages
        assert "verb loop limit reached" in messages

        # Check limit type
        limit_call = [c for c in logger.calls if c[1] == "verb loop limit reached"][0]
        assert limit_call[2] is not None
        assert limit_call[2]["limit_type"] == "max_iterations"

    async def test_complete_verb_logs_successful_completion(
        self, mock_backend: MockBackend
    ) -> None:
        """Complete verb logs when task completes successfully."""
        logger = RecordingLogger()
        tools = [ToolDef(name="test_tool", description="test", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            return "result"

        saia = SAIA(backend=mock_backend, tools=tools, executor=executor, lg=logger)

        # Queue: tool call -> completion response -> confirm says done
        mock_backend.queue_response(
            AgentResponse(
                content="calling tool",
                tool_calls=[ToolCall(id="1", name="test_tool", arguments={})],
                stop_reason="tool_use",
            )
        )
        mock_backend.queue_response(
            AgentResponse(content="task done", tool_calls=[], stop_reason="end_turn")
        )
        # ConfirmResult default is confirmed=True

        result = await saia.complete("do a task")

        assert result.completed is True
        messages = [call[1] for call in logger.calls]
        assert "verb loop started" in messages
        assert "llm response received" in messages
        # Should NOT have limit reached since it completed successfully
        assert "verb loop limit reached" not in messages

    async def test_warns_when_tools_not_used_but_json_in_response(
        self, mock_backend: MockBackend
    ) -> None:
        """Warns when LLM outputs tool-call JSON as text instead of using tool_calls."""
        logger = RecordingLogger()
        tools = [ToolDef(name="search", description="Search for info", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            return "result"

        saia = SAIA(backend=mock_backend, tools=tools, executor=executor, lg=logger)

        # LLM returns tool-call-like JSON in content instead of using tool_calls
        fake_tool_json = '{"name": "search", "arguments": {"query": "test"}}'
        mock_backend.queue_response(
            AgentResponse(
                content=f"I will search for that. {fake_tool_json}",
                tool_calls=[],
                stop_reason="end_turn",
            )
        )

        await saia.instruct("search for something")

        # Should have warning about tools not being used
        warning_calls = [c for c in logger.calls if c[0] == "warning"]
        assert len(warning_calls) > 0
        warn_messages = [c[1] for c in warning_calls]
        assert any("model may not support function calling" in msg for msg in warn_messages)

    async def test_no_warning_when_tools_used_correctly(self, mock_backend: MockBackend) -> None:
        """No warning when LLM uses tool_calls properly."""
        logger = RecordingLogger()
        tools = [ToolDef(name="search", description="Search for info", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            return "result"

        saia = SAIA(backend=mock_backend, tools=tools, executor=executor, lg=logger)

        # LLM properly uses tool_calls
        mock_backend.queue_response(
            AgentResponse(
                content="Searching...",
                tool_calls=[ToolCall(id="1", name="search", arguments={"query": "test"})],
                stop_reason="tool_use",
            )
        )
        mock_backend.queue_response(
            AgentResponse(content="Found results", tool_calls=[], stop_reason="end_turn")
        )

        await saia.instruct("search for something")

        # Should NOT have the function calling warning
        warning_calls = [c for c in logger.calls if c[0] == "warning"]
        warn_messages = [c[1] for c in warning_calls]
        assert not any("model may not support function calling" in msg for msg in warn_messages)

    async def test_warns_when_input_tokens_suspiciously_low(
        self, mock_backend: MockBackend
    ) -> None:
        """Warns when input tokens suggest server ignored tool definitions."""
        logger = RecordingLogger()
        tools = [ToolDef(name="search", description="Search for info", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            return "result"

        saia = SAIA(backend=mock_backend, tools=tools, executor=executor, lg=logger)

        # Very low input tokens (10) vs expected minimum (50 per tool)
        mock_backend.queue_response(
            AgentResponse(
                content="I don't have tools",
                tool_calls=[],
                stop_reason="end_turn",
                input_tokens=10,  # Suspiciously low for 1 tool (expected >= 50)
            )
        )

        await saia.instruct("search for something")

        warning_calls = [c for c in logger.calls if c[0] == "warning"]
        warn_messages = [c[1] for c in warning_calls]
        assert any("input tokens suspiciously low" in msg for msg in warn_messages)

    async def test_no_low_token_warning_when_tools_working(self, mock_backend: MockBackend) -> None:
        """No low token warning when tool_calls are present (tools working)."""
        logger = RecordingLogger()
        tools = [ToolDef(name="search", description="Search for info", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            return "result"

        saia = SAIA(backend=mock_backend, tools=tools, executor=executor, lg=logger)

        # Low input tokens but tools are working (tool_calls present)
        mock_backend.queue_response(
            AgentResponse(
                content="",
                tool_calls=[ToolCall(id="1", name="search", arguments={"query": "test"})],
                stop_reason="tool_use",
                input_tokens=10,  # Low, but tools work so no warning
            )
        )
        mock_backend.queue_response(
            AgentResponse(content="Found results", tool_calls=[], stop_reason="end_turn")
        )

        await saia.instruct("search for something")

        warning_calls = [c for c in logger.calls if c[0] == "warning"]
        warn_messages = [c[1] for c in warning_calls]
        assert not any("input tokens suspiciously low" in msg for msg in warn_messages)

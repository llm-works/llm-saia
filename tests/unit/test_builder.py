"""Tests for SAIABuilder."""

from typing import Any

import pytest

from llm_saia import SAIA, SAIABuilder
from llm_saia.core.types import ToolDef
from tests.unit.conftest import MockBackend

pytestmark = pytest.mark.unit


class TestSAIABuilder:
    def test_builder_returns_builder(self, mock_backend: MockBackend) -> None:
        """SAIA.builder() returns a SAIABuilder."""
        builder = SAIA.builder()
        assert isinstance(builder, SAIABuilder)

    def test_build_requires_backend(self) -> None:
        """build() raises if backend not set."""
        builder = SAIA.builder()
        with pytest.raises(ValueError, match="backend is required"):
            builder.build()

    def test_minimal_build(self, mock_backend: MockBackend) -> None:
        """Build with only backend."""
        saia = SAIA.builder().backend(mock_backend).build()

        assert saia._config.backend is mock_backend
        assert saia._config.tools == []
        assert saia._config.executor is None
        assert saia._config.system is None

    def test_build_with_tools(self, mock_backend: MockBackend) -> None:
        """Build with tools and executor."""
        tools = [ToolDef(name="test", description="test", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            return "result"

        saia = SAIA.builder().backend(mock_backend).tools(tools, executor).build()

        assert saia._config.tools == tools
        assert saia._config.executor is executor

    def test_build_with_system(self, mock_backend: MockBackend) -> None:
        """Build with system prompt."""
        saia = SAIA.builder().backend(mock_backend).system("You are helpful").build()

        assert saia._config.system == "You are helpful"

    def test_build_with_terminal_tool(self, mock_backend: MockBackend) -> None:
        """Build with terminal tool."""
        saia = SAIA.builder().backend(mock_backend).terminal_tool("finish").build()

        assert saia._config.terminal is not None
        assert saia._config.terminal.tool == "finish"

    def test_build_with_logger(self, mock_backend: MockBackend) -> None:
        """Build with logger."""
        from llm_saia import NullLogger

        lg = NullLogger()
        saia = SAIA.builder().backend(mock_backend).logger(lg).build()

        assert saia._config.lg is lg

    def test_build_with_warn_tool_support(self, mock_backend: MockBackend) -> None:
        """Build with warn_tool_support disabled."""
        saia = SAIA.builder().backend(mock_backend).warn_tool_support(False).build()

        assert saia._config.warn_tool_support is False

    def test_build_with_max_iterations(self, mock_backend: MockBackend) -> None:
        """Build with max_iterations."""
        saia = SAIA.builder().backend(mock_backend).max_iterations(10).build()

        assert saia.run_config.max_iterations == 10

    def test_build_with_max_call_tokens(self, mock_backend: MockBackend) -> None:
        """Build with max_call_tokens."""
        saia = SAIA.builder().backend(mock_backend).max_call_tokens(4096).build()

        assert saia.run_config.max_call_tokens == 4096

    def test_build_with_max_tokens(self, mock_backend: MockBackend) -> None:
        """Build with max_tokens (total budget)."""
        saia = SAIA.builder().backend(mock_backend).max_tokens(10000).build()

        assert saia.run_config.max_total_tokens == 10000

    def test_build_with_timeout(self, mock_backend: MockBackend) -> None:
        """Build with timeout."""
        saia = SAIA.builder().backend(mock_backend).timeout(30.0).build()

        assert saia.run_config.timeout_secs == 30.0

    def test_build_with_retries(self, mock_backend: MockBackend) -> None:
        """Build with retries."""
        saia = SAIA.builder().backend(mock_backend).retries(3, "Try harder").build()

        assert saia.run_config.max_retries == 3
        assert saia.run_config.retry_escalation == "Try harder"

    def test_fluent_chaining(self, mock_backend: MockBackend) -> None:
        """All methods return self for chaining."""
        tools = [ToolDef(name="test", description="test", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            return "result"

        saia = (
            SAIA.builder()
            .backend(mock_backend)
            .tools(tools, executor)
            .system("You are helpful")
            .terminal_tool("finish")
            .max_iterations(5)
            .max_call_tokens(2048)
            .max_tokens(8000)
            .timeout(60.0)
            .retries(2, "Be more careful")
            .build()
        )

        assert saia._config.backend is mock_backend
        assert saia._config.tools == tools
        assert saia._config.system == "You are helpful"
        assert saia._config.terminal is not None
        assert saia._config.terminal.tool == "finish"
        assert saia.run_config.max_iterations == 5
        assert saia.run_config.max_call_tokens == 2048
        assert saia.run_config.max_total_tokens == 8000
        assert saia.run_config.timeout_secs == 60.0
        assert saia.run_config.max_retries == 2
        assert saia.run_config.retry_escalation == "Be more careful"

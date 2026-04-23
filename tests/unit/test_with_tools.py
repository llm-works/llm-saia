"""Tests for with_tools() per-call tool override."""

from typing import Any

import pytest

from llm_saia.core.types import ChatResponse, ToolCall, ToolDef
from tests.unit.conftest import MockBackend, make_saia

pytestmark = pytest.mark.unit


def _make_tool(name: str) -> ToolDef:
    """Create a minimal tool definition."""
    return ToolDef(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {}},
    )


async def _original_executor(name: str, args: dict[str, Any]) -> str:
    return f"original:{name}"


async def _override_executor(name: str, args: dict[str, Any]) -> str:
    return f"override:{name}"


class TestWithTools:
    """Tests for Configurable.with_tools()."""

    def test_replaces_tools(self) -> None:
        """with_tools() replaces tool definitions on the cloned instance."""
        backend = MockBackend()
        original_tools = [_make_tool("search")]
        saia = make_saia(backend, tools=original_tools, executor=_original_executor)

        override_tools = [_make_tool("calculate"), _make_tool("translate")]
        cloned = saia.with_tools(override_tools)

        assert cloned._config.tools == override_tools
        assert saia._config.tools == original_tools  # original unchanged

    def test_preserves_executor_when_none(self) -> None:
        """with_tools() keeps existing executor when executor=None."""
        backend = MockBackend()
        saia = make_saia(backend, tools=[_make_tool("a")], executor=_original_executor)

        cloned = saia.with_tools([_make_tool("b")])

        assert cloned._config.executor is _original_executor

    def test_replaces_executor_when_provided(self) -> None:
        """with_tools() replaces executor when explicitly provided."""
        backend = MockBackend()
        saia = make_saia(backend, tools=[_make_tool("a")], executor=_original_executor)

        cloned = saia.with_tools([_make_tool("b")], executor=_override_executor)

        assert cloned._config.executor is _override_executor
        assert saia._config.executor is _original_executor  # original unchanged

    def test_original_unmodified(self) -> None:
        """with_tools() does not modify the original instance."""
        backend = MockBackend()
        original_tools = [_make_tool("search")]
        saia = make_saia(backend, tools=original_tools, executor=_original_executor)

        saia.with_tools([_make_tool("other")], executor=_override_executor)

        assert saia._config.tools == original_tools
        assert saia._config.executor is _original_executor

    def test_chains_with_other_overrides(self) -> None:
        """with_tools() composes with other with_*() methods."""
        backend = MockBackend()
        saia = make_saia(backend, tools=[_make_tool("a")], executor=_original_executor)

        chained = saia.with_tools([_make_tool("b")]).with_temperature(0.5).with_max_iterations(5)

        assert chained._config.tools == [_make_tool("b")]
        assert chained._config.call is not None
        assert chained._config.call.temperature == 0.5
        assert chained._config.call.max_iterations == 5

    @pytest.mark.asyncio
    async def test_tools_passed_to_backend(self) -> None:
        """Overridden tools are actually sent to the backend via ask()."""
        backend = MockBackend()
        saia = make_saia(backend, tools=[_make_tool("original")], executor=_original_executor)

        override_tools = [_make_tool("override_tool")]
        await saia.with_tools(override_tools).ask("artifact", "question")

        assert backend.last_tools == override_tools

    @pytest.mark.asyncio
    async def test_executor_used_on_tool_call(self) -> None:
        """Overridden executor is invoked when LLM makes a tool call."""
        backend = MockBackend()
        saia = make_saia(backend, tools=[_make_tool("calc")], executor=_original_executor)
        calls: list[tuple[str, dict[str, Any]]] = []

        async def tracking_executor(name: str, args: dict[str, Any]) -> str:
            calls.append((name, args))
            return f"override:{name}"

        # Queue a tool call response, then a final text response
        backend.queue_response(
            ChatResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="calc", arguments={"x": 1})],
                finish_reason="tool_use",
            )
        )
        backend.set_complete_response("done")

        await saia.with_tools([_make_tool("calc")], executor=tracking_executor).ask(
            "artifact", "question"
        )

        assert calls == [("calc", {"x": 1})]

    def test_empty_tools_clears(self) -> None:
        """with_tools([]) effectively removes tools."""
        backend = MockBackend()
        saia = make_saia(backend, tools=[_make_tool("a")], executor=_original_executor)

        cloned = saia.with_tools([])

        assert cloned._config.tools == []

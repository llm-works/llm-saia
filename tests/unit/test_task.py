"""Tests for the task execution loop."""

from typing import Any

import pytest

from llm_saia import SAIA
from llm_saia.core.types import AgentResponse, ConfirmResult, LoopConfig, ToolCall, ToolDef
from tests.unit.conftest import MockBackend

pytestmark = pytest.mark.unit


@pytest.fixture
def sample_tools() -> list[ToolDef]:
    """Sample tools for testing."""
    return [
        ToolDef(
            name="search",
            description="Search for information",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        ToolDef(
            name="read_file",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
    ]


async def dummy_executor(name: str, args: dict[str, Any]) -> str:
    """Default executor for tests."""
    return f"Executed {name}"


class TestTask:
    async def test_task_completes_without_tools(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Task completes when LLM returns text without tool calls."""
        saia = SAIA(backend=mock_backend, tools=sample_tools, executor=dummy_executor)

        mock_backend.queue_tool_response(
            AgentResponse(content="Task completed!", tool_calls=[], stop_reason="end_turn")
        )
        mock_backend.set_structured_response(
            ConfirmResult, ConfirmResult(confirmed=True, reason="Task is done")
        )

        result = await saia.complete(task="Do something simple")

        assert result.completed is True
        assert result.output == "Task completed!"
        assert result.iterations == 1
        assert len(result.history) >= 2

    async def test_task_executes_tools(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Task executes tools when LLM requests them."""
        executed_tools: list[tuple[str, dict[str, Any]]] = []

        async def tracking_executor(name: str, args: dict[str, Any]) -> str:
            executed_tools.append((name, args))
            return f"Results for {args.get('query', 'unknown')}"

        saia = SAIA(backend=mock_backend, tools=sample_tools, executor=tracking_executor)

        mock_backend.queue_tool_response(
            AgentResponse(
                content="Let me search for that.",
                tool_calls=[ToolCall(id="call_1", name="search", arguments={"query": "python"})],
                stop_reason="tool_use",
            )
        )
        mock_backend.queue_tool_response(
            AgentResponse(content="Found it. Done.", tool_calls=[], stop_reason="end_turn")
        )
        mock_backend.set_structured_response(
            ConfirmResult, ConfirmResult(confirmed=True, reason="Task done")
        )

        result = await saia.complete(task="Search for Python")

        assert result.completed is True
        assert result.iterations == 2
        assert len(executed_tools) == 1
        assert executed_tools[0] == ("search", {"query": "python"})

    async def test_task_injects_wrapup_when_not_complete(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Task injects wrap-up message when confirmation fails."""
        saia = SAIA(backend=mock_backend, tools=sample_tools, executor=dummy_executor)

        mock_backend.queue_tool_response(
            AgentResponse(content="I started...", tool_calls=[], stop_reason="end_turn")
        )
        mock_backend.queue_tool_response(
            AgentResponse(content="Now I finished!", tool_calls=[], stop_reason="end_turn")
        )

        # First call returns not confirmed, second returns confirmed
        confirm_calls = [
            ConfirmResult(confirmed=False, reason="Work not finished"),
            ConfirmResult(confirmed=True, reason="Done"),
        ]
        original = mock_backend.complete_structured

        async def patched(prompt: str, schema: type) -> Any:
            if schema == ConfirmResult and confirm_calls:
                return confirm_calls.pop(0)
            return await original(prompt, schema)

        mock_backend.complete_structured = patched  # type: ignore

        result = await saia.complete(task="Complete a task")

        assert result.completed is True
        assert result.iterations == 2
        wrap_up_messages = [m for m in result.history if "not yet complete" in m.content.lower()]
        assert len(wrap_up_messages) >= 1

    async def test_task_max_iterations(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Task stops after max iterations."""
        saia = SAIA(backend=mock_backend, tools=sample_tools, executor=dummy_executor)

        for _ in range(5):
            mock_backend.queue_tool_response(
                AgentResponse(content="Still working...", tool_calls=[], stop_reason="end_turn")
            )
        mock_backend.set_structured_response(
            ConfirmResult, ConfirmResult(confirmed=False, reason="Not done")
        )

        result = await saia.complete(task="Impossible task", loop=LoopConfig(max_iterations=3))

        assert result.completed is False
        assert result.iterations == 3

    async def test_task_with_on_iteration_callback(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Task calls on_iteration callback each iteration."""
        saia = SAIA(backend=mock_backend, tools=sample_tools, executor=dummy_executor)
        iterations_seen: list[int] = []

        mock_backend.queue_tool_response(
            AgentResponse(content="Done!", tool_calls=[], stop_reason="end_turn")
        )
        mock_backend.set_structured_response(
            ConfirmResult, ConfirmResult(confirmed=True, reason="Complete")
        )

        async def on_iteration(iteration: int, response: AgentResponse) -> None:
            iterations_seen.append(iteration)

        result = await saia.complete(task="Quick task", on_iteration=on_iteration)

        assert result.completed is True
        assert iterations_seen == [0]

    async def test_task_tool_execution_error_handled(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Task handles errors in tool execution gracefully."""

        async def failing_executor(name: str, args: dict[str, Any]) -> str:
            raise RuntimeError("Tool crashed!")

        saia = SAIA(backend=mock_backend, tools=sample_tools, executor=failing_executor)

        mock_backend.queue_tool_response(
            AgentResponse(
                content="Let me try this tool.",
                tool_calls=[ToolCall(id="call_1", name="search", arguments={"query": "test"})],
                stop_reason="tool_use",
            )
        )
        mock_backend.queue_tool_response(
            AgentResponse(content="Tool failed, but done.", tool_calls=[], stop_reason="end_turn")
        )
        mock_backend.set_structured_response(
            ConfirmResult, ConfirmResult(confirmed=True, reason="Done despite error")
        )

        result = await saia.complete(task="Try a failing tool")

        assert result.completed is True
        tool_results = [m for m in result.history if m.role == "tool_result"]
        assert len(tool_results) == 1
        assert "Error" in tool_results[0].content

    async def test_task_requires_tools(self, mock_backend: MockBackend) -> None:
        """Task raises error when no tools configured."""
        saia = SAIA(backend=mock_backend, executor=dummy_executor)

        with pytest.raises(ValueError, match="Complete requires tools and executor"):
            await saia.complete(task="Do something")

    async def test_task_requires_executor(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Task raises error when no executor configured."""
        saia = SAIA(backend=mock_backend, tools=sample_tools)

        with pytest.raises(ValueError, match="Complete requires tools and executor"):
            await saia.complete(task="Do something")

    async def test_task_timeout(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Task stops after soft timeout."""
        saia = SAIA(backend=mock_backend, tools=sample_tools, executor=dummy_executor)

        # Queue enough responses for multiple iterations
        for _ in range(10):
            mock_backend.queue_tool_response(
                AgentResponse(content="Working...", tool_calls=[], stop_reason="end_turn")
            )
        mock_backend.set_structured_response(
            ConfirmResult, ConfirmResult(confirmed=False, reason="Not done")
        )

        # Very short timeout to trigger quickly
        result = await saia.complete(task="Long task", loop=LoopConfig(timeout_secs=0.001))

        assert result.completed is False
        # Should have done at least 1 iteration before timeout
        assert result.iterations >= 1

    async def test_task_unlimited_iterations(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Task with max_iterations=0 runs until complete."""
        saia = SAIA(backend=mock_backend, tools=sample_tools, executor=dummy_executor)

        # Queue 5 "not done" responses, then one "done"
        for _ in range(5):
            mock_backend.queue_tool_response(
                AgentResponse(content="Working...", tool_calls=[], stop_reason="end_turn")
            )
        mock_backend.queue_tool_response(
            AgentResponse(content="Done!", tool_calls=[], stop_reason="end_turn")
        )

        # Start with "not confirmed", then flip to confirmed after 5 iterations
        call_count = 0

        def get_confirmation() -> ConfirmResult:
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                return ConfirmResult(confirmed=False, reason="Keep going")
            return ConfirmResult(confirmed=True, reason="Complete")

        # Monkey-patch to return different results
        original_method = mock_backend.complete_structured

        async def patched_complete_structured(prompt: str, schema: type) -> Any:
            if schema == ConfirmResult:
                return get_confirmation()
            return await original_method(prompt, schema)

        mock_backend.complete_structured = patched_complete_structured  # type: ignore

        result = await saia.complete(task="Long running task", loop=LoopConfig(max_iterations=0))

        assert result.completed is True
        assert result.iterations == 6  # 5 "not done" + 1 "done"

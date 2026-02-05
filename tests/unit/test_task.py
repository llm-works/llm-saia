"""Tests for the task execution loop."""

from typing import Any

import pytest

from llm_saia.core.types import AgentResponse, ConfirmResult, RunConfig, ToolCall, ToolDef
from tests.unit.conftest import MockBackend, make_saia

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
        saia = make_saia(mock_backend, tools=sample_tools, executor=dummy_executor)

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

        saia = make_saia(mock_backend, tools=sample_tools, executor=tracking_executor)

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
        saia = make_saia(mock_backend, tools=sample_tools, executor=dummy_executor)

        mock_backend.queue_tool_response(
            AgentResponse(content="I started...", tool_calls=[], stop_reason="end_turn")
        )
        mock_backend.queue_tool_response(
            AgentResponse(content="Now I finished!", tool_calls=[], stop_reason="end_turn")
        )

        # First call returns not confirmed, second returns confirmed
        mock_backend.queue_structured_response(
            ConfirmResult, ConfirmResult(confirmed=False, reason="Work not finished")
        )
        mock_backend.queue_structured_response(
            ConfirmResult, ConfirmResult(confirmed=True, reason="Done")
        )

        result = await saia.complete(task="Complete a task")

        assert result.completed is True
        assert result.iterations == 2
        wrap_up_messages = [m for m in result.history if "not yet complete" in m.content.lower()]
        assert len(wrap_up_messages) >= 1

    async def test_task_max_iterations(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Task stops after max iterations."""
        saia = make_saia(mock_backend, tools=sample_tools, executor=dummy_executor)
        saia = saia.with_max_iterations(3)

        for _ in range(5):
            mock_backend.queue_tool_response(
                AgentResponse(content="Still working...", tool_calls=[], stop_reason="end_turn")
            )
        mock_backend.set_structured_response(
            ConfirmResult, ConfirmResult(confirmed=False, reason="Not done")
        )

        result = await saia.complete(task="Impossible task")

        assert result.completed is False
        assert result.iterations == 3

    async def test_task_with_on_iteration_callback(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Task calls on_iteration callback each iteration."""
        saia = make_saia(mock_backend, tools=sample_tools, executor=dummy_executor)
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

        saia = make_saia(mock_backend, tools=sample_tools, executor=failing_executor)

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
        saia = make_saia(mock_backend, executor=dummy_executor)

        with pytest.raises(ValueError, match="Complete requires tools and executor"):
            await saia.complete(task="Do something")

    async def test_task_requires_executor(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Task raises error when no executor configured."""
        saia = make_saia(mock_backend, tools=sample_tools)

        with pytest.raises(ValueError, match="Complete requires tools and executor"):
            await saia.complete(task="Do something")

    async def test_task_timeout(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Task stops after soft timeout."""
        saia = make_saia(mock_backend, tools=sample_tools, executor=dummy_executor)
        saia = saia.with_timeout_secs(0.001)

        # Queue enough responses for multiple iterations
        for _ in range(10):
            mock_backend.queue_tool_response(
                AgentResponse(content="Working...", tool_calls=[], stop_reason="end_turn")
            )
        mock_backend.set_structured_response(
            ConfirmResult, ConfirmResult(confirmed=False, reason="Not done")
        )

        result = await saia.complete(task="Long task")

        assert result.completed is False
        # Should have done at least 1 iteration before timeout
        assert result.iterations >= 1

    async def test_task_unlimited_iterations(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Task with max_iterations=0 runs until complete."""
        saia = make_saia(mock_backend, tools=sample_tools, executor=dummy_executor)
        saia = saia.with_run_config(RunConfig(max_iterations=0))

        # Queue 5 "not done" responses, then one "done"
        for _ in range(5):
            mock_backend.queue_tool_response(
                AgentResponse(content="Working...", tool_calls=[], stop_reason="end_turn")
            )
        mock_backend.queue_tool_response(
            AgentResponse(content="Done!", tool_calls=[], stop_reason="end_turn")
        )

        # Queue 5 "not confirmed" responses, then one "confirmed"
        for _ in range(5):
            mock_backend.queue_structured_response(
                ConfirmResult, ConfirmResult(confirmed=False, reason="Keep going")
            )
        mock_backend.queue_structured_response(
            ConfirmResult, ConfirmResult(confirmed=True, reason="Complete")
        )

        result = await saia.complete(task="Long running task")

        assert result.completed is True
        assert result.iterations == 6  # 5 "not done" + 1 "done"

    async def test_task_terminal_tool_requires_confirmation(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Task completes when LLM calls terminal tool twice (initial + confirmation)."""
        # Add terminal tool to the tool list
        terminal_tool_def = ToolDef(
            name="task_complete",
            description="Call when task is complete",
            parameters={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        )
        tools_with_terminal = sample_tools + [terminal_tool_def]

        saia = make_saia(
            mock_backend,
            tools=tools_with_terminal,
            executor=dummy_executor,
            terminal_tool="task_complete",
        )

        # First: LLM calls terminal tool
        mock_backend.queue_tool_response(
            AgentResponse(
                content="Task finished!",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="task_complete",
                        arguments={"summary": "Successfully completed the search"},
                    )
                ],
                stop_reason="tool_use",
            )
        )
        # Second: LLM confirms by calling terminal tool again
        mock_backend.queue_tool_response(
            AgentResponse(
                content="Confirmed complete",
                tool_calls=[
                    ToolCall(
                        id="call_2",
                        name="task_complete",
                        arguments={"summary": "Successfully completed the search"},
                    )
                ],
                stop_reason="tool_use",
            )
        )

        result = await saia.complete(task="Do something")

        assert result.completed is True
        assert result.iterations == 2  # Initial call + confirmation
        assert result.terminal_tool == "task_complete"
        assert result.terminal_data == {"summary": "Successfully completed the search"}

    async def test_task_terminal_tool_not_executed_on_confirm(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Terminal tool itself is not executed - only signals completion."""
        executed_tools: list[str] = []

        async def tracking_executor(name: str, args: dict[str, Any]) -> str:
            executed_tools.append(name)
            return f"Executed {name}"

        terminal_tool_def = ToolDef(
            name="finish",
            description="Finish task",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        tools_with_terminal = sample_tools + [terminal_tool_def]

        saia = make_saia(
            mock_backend,
            tools=tools_with_terminal,
            executor=tracking_executor,
            terminal_tool="finish",
        )

        # First: LLM calls terminal tool
        mock_backend.queue_tool_response(
            AgentResponse(
                content="Done",
                tool_calls=[ToolCall(id="call_1", name="finish", arguments={})],
                stop_reason="tool_use",
            )
        )
        # Second: LLM confirms by calling terminal tool again
        mock_backend.queue_tool_response(
            AgentResponse(
                content="Confirmed",
                tool_calls=[ToolCall(id="call_2", name="finish", arguments={})],
                stop_reason="tool_use",
            )
        )

        result = await saia.complete(task="Do and finish")

        assert result.completed is True
        # Terminal tool is never executed - it only signals completion
        assert executed_tools == []
        assert result.terminal_tool == "finish"

    async def test_task_terminal_tool_in_batch_skips_other_tools(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """When terminal tool is in a batch with other tools, other tools are not executed."""
        executed_tools: list[str] = []

        async def tracking_executor(name: str, args: dict[str, Any]) -> str:
            executed_tools.append(name)
            return f"Executed {name}"

        terminal_tool_def = ToolDef(
            name="finish",
            description="Finish task",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        tools_with_terminal = sample_tools + [terminal_tool_def]

        saia = make_saia(
            mock_backend,
            tools=tools_with_terminal,
            executor=tracking_executor,
            terminal_tool="finish",
        )

        # LLM calls both a work tool and terminal tool in same batch
        mock_backend.queue_tool_response(
            AgentResponse(
                content="Done",
                tool_calls=[
                    ToolCall(id="call_1", name="search", arguments={"query": "test"}),
                    ToolCall(id="call_2", name="finish", arguments={}),
                ],
                stop_reason="tool_use",
            )
        )
        # LLM confirms terminal tool
        mock_backend.queue_tool_response(
            AgentResponse(
                content="Confirmed",
                tool_calls=[ToolCall(id="call_3", name="finish", arguments={})],
                stop_reason="tool_use",
            )
        )

        result = await saia.complete(task="Search and finish")

        assert result.completed is True
        # Neither tool was executed - terminal tool detection short-circuits
        assert executed_tools == []
        assert result.terminal_tool == "finish"

    async def test_task_terminal_tool_can_be_reconsidered(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """LLM can continue working instead of confirming terminal tool."""
        executed_tools: list[str] = []

        async def tracking_executor(name: str, args: dict[str, Any]) -> str:
            executed_tools.append(name)
            return f"Executed {name}"

        terminal_tool_def = ToolDef(
            name="finish",
            description="Finish task",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        tools_with_terminal = sample_tools + [terminal_tool_def]

        saia = make_saia(
            mock_backend,
            tools=tools_with_terminal,
            executor=tracking_executor,
            terminal_tool="finish",
        )

        # First: LLM prematurely calls terminal tool
        mock_backend.queue_tool_response(
            AgentResponse(
                content="Done",
                tool_calls=[ToolCall(id="call_1", name="finish", arguments={})],
                stop_reason="tool_use",
            )
        )
        # Second: After seeing confirmation prompt, LLM decides to continue working
        mock_backend.queue_tool_response(
            AgentResponse(
                content="Actually, let me search first",
                tool_calls=[ToolCall(id="call_2", name="search", arguments={"query": "more"})],
                stop_reason="tool_use",
            )
        )
        # Third: Now LLM calls terminal tool again
        mock_backend.queue_tool_response(
            AgentResponse(
                content="Now done",
                tool_calls=[ToolCall(id="call_3", name="finish", arguments={})],
                stop_reason="tool_use",
            )
        )
        # Fourth: LLM confirms
        mock_backend.queue_tool_response(
            AgentResponse(
                content="Confirmed",
                tool_calls=[ToolCall(id="call_4", name="finish", arguments={})],
                stop_reason="tool_use",
            )
        )

        result = await saia.complete(task="Search and finish")

        assert result.completed is True
        # The search tool was executed when LLM decided to continue
        assert executed_tools == ["search"]
        assert result.terminal_tool == "finish"

    async def test_task_without_terminal_tool_uses_confirm(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Without terminal_tool configured, task uses Confirm verb for completion."""
        saia = make_saia(
            mock_backend,
            tools=sample_tools,
            executor=dummy_executor,
            # No terminal_tool set
        )

        mock_backend.queue_tool_response(
            AgentResponse(content="Task done!", tool_calls=[], stop_reason="end_turn")
        )
        mock_backend.set_structured_response(
            ConfirmResult, ConfirmResult(confirmed=True, reason="Task is complete")
        )

        result = await saia.complete(task="Do something")

        assert result.completed is True
        assert result.terminal_tool is None
        assert result.terminal_data is None

    async def test_task_terminal_tool_handles_non_serializable_args(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Terminal tool confirmation handles non-JSON-serializable arguments gracefully."""
        from datetime import datetime

        terminal_tool_def = ToolDef(
            name="finish",
            description="Finish task",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        tools_with_terminal = sample_tools + [terminal_tool_def]

        saia = make_saia(
            mock_backend,
            tools=tools_with_terminal,
            executor=dummy_executor,
            terminal_tool="finish",
        )

        # Arguments contain a non-JSON-serializable object (datetime)
        non_serializable_args = {"timestamp": datetime.now(), "data": "test"}

        # First: LLM calls terminal tool with non-serializable args
        mock_backend.queue_tool_response(
            AgentResponse(
                content="Done",
                tool_calls=[ToolCall(id="call_1", name="finish", arguments=non_serializable_args)],
                stop_reason="tool_use",
            )
        )
        # Second: LLM confirms
        mock_backend.queue_tool_response(
            AgentResponse(
                content="Confirmed",
                tool_calls=[ToolCall(id="call_2", name="finish", arguments=non_serializable_args)],
                stop_reason="tool_use",
            )
        )

        # Should not crash - falls back to str() representation
        result = await saia.complete(task="Do something")

        assert result.completed is True
        assert result.terminal_tool == "finish"
        assert result.terminal_data == non_serializable_args

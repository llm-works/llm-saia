"""Tests for the task execution loop."""

from typing import Any

import pytest

from llm_saia.core.config import Config
from llm_saia.core.types import (
    AgentResponse,
    ClassifyResult,
    RunConfig,
    ToolCall,
    ToolDef,
)
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
            ClassifyResult, ClassifyResult(category="completed", confidence=1.0, reason="Done")
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
            ClassifyResult, ClassifyResult(category="completed", confidence=1.0, reason="Done")
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

        # First call returns wants_continue, second returns completed
        mock_backend.queue_structured_response(
            ClassifyResult,
            ClassifyResult(category="wants_continue", confidence=1.0, reason="Work not finished"),
        )
        mock_backend.queue_structured_response(
            ClassifyResult, ClassifyResult(category="completed", confidence=1.0, reason="Done")
        )

        result = await saia.complete(task="Complete a task")

        assert result.completed is True
        assert result.iterations == 2
        # Check for nudge message (now says "didn't use any tools" instead of "not yet complete")
        nudge_messages = [m for m in result.history if "didn't use any tools" in m.content.lower()]
        assert len(nudge_messages) >= 1

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
            ClassifyResult,
            ClassifyResult(category="wants_continue", confidence=1.0, reason="Not done"),
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
            ClassifyResult, ClassifyResult(category="completed", confidence=1.0, reason="Complete")
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
            ClassifyResult,
            ClassifyResult(category="completed", confidence=1.0, reason="Done despite error"),
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
            ClassifyResult,
            ClassifyResult(category="wants_continue", confidence=1.0, reason="Not done"),
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

        # Queue 5 "wants_continue" responses, then one "completed"
        for _ in range(5):
            mock_backend.queue_structured_response(
                ClassifyResult,
                ClassifyResult(category="wants_continue", confidence=1.0, reason="Keep going"),
            )
        mock_backend.queue_structured_response(
            ClassifyResult, ClassifyResult(category="completed", confidence=1.0, reason="Complete")
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
        # Output is from the confirmation response
        assert result.output == "Confirmed complete"

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

    async def test_task_terminal_tool_in_batch_executes_other_tools(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """When terminal tool is in a batch with other tools, other tools ARE executed."""
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
        # Non-terminal tools are executed before asking for confirmation
        assert executed_tools == ["search"]
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
        # Needs enough iterations for: 1st finish, reconsider, 2nd finish, confirm
        saia = saia.with_run_config(RunConfig(max_iterations=0))

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
        assert result.iterations == 4
        # The search tool was executed when LLM decided to continue
        assert executed_tools == ["search"]
        assert result.terminal_tool == "finish"
        assert result.output == "Confirmed"

    async def test_task_without_terminal_tool_uses_classifier(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Without terminal_tool configured, task uses classifier for completion."""
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
            ClassifyResult,
            ClassifyResult(category="completed", confidence=1.0, reason="Task is complete"),
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

    async def test_task_terminal_confirmation_produces_tool_results(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """INSTRUCT path for terminal confirmation adds tool_result for orphaned tool_calls."""
        terminal_tool_def = ToolDef(
            name="finish",
            description="Finish task",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        saia = make_saia(
            mock_backend,
            tools=sample_tools + [terminal_tool_def],
            executor=dummy_executor,
            terminal_tool="finish",
        )

        # First: LLM calls only the terminal tool (triggers INSTRUCT confirmation)
        mock_backend.queue_tool_response(
            AgentResponse(
                content="Done",
                tool_calls=[ToolCall(id="call_1", name="finish", arguments={"output": "ok"})],
                stop_reason="tool_use",
            )
        )
        # Second: LLM confirms
        mock_backend.queue_tool_response(
            AgentResponse(
                content="Confirmed",
                tool_calls=[ToolCall(id="call_2", name="finish", arguments={"output": "ok"})],
                stop_reason="tool_use",
            )
        )

        result = await saia.complete(task="Do something")

        assert result.completed is True
        # Verify that every non-final assistant tool_call has a matching tool_result.
        # The final assistant message (COMPLETE/FAIL) won't have tool_results since
        # no further _chat() call follows, which is correct.
        assistant_msgs = [
            (i, m) for i, m in enumerate(result.history) if m.role == "assistant" and m.tool_calls
        ]
        # Check all except the last assistant message (the confirmed terminal response)
        for idx, msg in assistant_msgs[:-1]:
            for tc in msg.tool_calls:
                matching = [
                    m
                    for m in result.history[idx + 1 :]
                    if m.role == "tool_result" and m.tool_call_id == tc.id
                ]
                assert matching, f"No tool_result for tool_call {tc.id} ({tc.name})"

    async def test_task_terminal_failure_preserves_terminal_data(
        self, mock_backend: MockBackend, sample_tools: list[ToolDef]
    ) -> None:
        """Terminal tool failure result preserves terminal_data and terminal_tool."""
        from llm_saia.core.config import TerminalConfig

        terminal_tool_def = ToolDef(
            name="finish",
            description="Finish task",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        config = Config(
            backend=mock_backend,
            tools=sample_tools + [terminal_tool_def],
            executor=dummy_executor,
            system=None,
            terminal=TerminalConfig(tool="finish", failure_values=("stuck", "failed")),
            run=RunConfig(max_retries=0),  # No failure retries
        )
        from llm_saia import SAIA

        saia = SAIA(config)

        # First: LLM calls terminal with failure status
        mock_backend.queue_tool_response(
            AgentResponse(
                content="I'm stuck",
                tool_calls=[
                    ToolCall(
                        id="c1", name="finish", arguments={"status": "stuck", "reason": "blocked"}
                    )
                ],
                stop_reason="tool_use",
            )
        )
        # Confirmation: same failure
        mock_backend.queue_tool_response(
            AgentResponse(
                content="Confirmed stuck",
                tool_calls=[
                    ToolCall(
                        id="c2", name="finish", arguments={"status": "stuck", "reason": "blocked"}
                    )
                ],
                stop_reason="tool_use",
            )
        )

        result = await saia.complete(task="Do something")

        assert result.completed is False
        assert result.terminal_tool == "finish"
        assert result.terminal_data == {"status": "stuck", "reason": "blocked"}


class TestDefaultController:
    """Tests for DefaultController."""

    def test_has_contradiction_detects_continuation_signals(
        self, mock_backend: MockBackend
    ) -> None:
        """Controller detects continuation signals in confirmation."""
        from llm_saia.core.controller import ControllerConfig, DefaultController

        config = Config(backend=mock_backend, tools=[], executor=None, system=None)
        controller = DefaultController(config=ControllerConfig(llm_config=config))

        # Should detect contradiction
        assert controller._has_contradiction("Let me check one more thing")
        assert controller._has_contradiction("I will continue")
        assert controller._has_contradiction("Let's proceed")

        # Clean confirmation - no contradiction
        assert not controller._has_contradiction("Confirmed")
        assert not controller._has_contradiction("Yes, done")
        assert not controller._has_contradiction("")

    def test_backoff_default_is_three(self, mock_backend: MockBackend) -> None:
        """Default backoff iterations is 3."""
        from llm_saia.core.controller import ControllerConfig

        config = Config(backend=mock_backend, tools=[], executor=None, system=None)
        ctrl_config = ControllerConfig(llm_config=config)
        assert ctrl_config.backoff_iterations == 3

    def test_is_empty_response(self, mock_backend: MockBackend) -> None:
        """Empty response detected when no content and no tool calls."""
        from llm_saia.core.controller import ControllerConfig, DefaultController

        config = Config(backend=mock_backend, tools=[], executor=None, system=None)
        controller = DefaultController(config=ControllerConfig(llm_config=config))

        # No content, no tool calls → empty
        assert controller._is_empty_response(
            AgentResponse(content="", tool_calls=[], output_tokens=0)
        )
        assert controller._is_empty_response(AgentResponse(content="", tool_calls=[]))
        # Has content → not empty
        assert not controller._is_empty_response(
            AgentResponse(content="hello", tool_calls=[], output_tokens=3)
        )
        # Has tool calls → not empty
        assert not controller._is_empty_response(
            AgentResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="search", arguments={})],
            )
        )

    def test_has_text_tool_pattern(self, mock_backend: MockBackend) -> None:
        """Text tool pattern detected when LLM writes tool names as text."""
        from llm_saia.core.controller import ControllerConfig, DefaultController

        config = Config(backend=mock_backend, tools=[], executor=None, system=None)
        controller = DefaultController(config=ControllerConfig(llm_config=config))

        tools = ["read_file", "run_command", "execute", "search"]
        # Matches actual tool names in content
        assert controller._has_text_tool_pattern("Let me read_file to check", tools)
        assert controller._has_text_tool_pattern("I'll run_command to see the output", tools)
        assert controller._has_text_tool_pattern("Using execute(ls)", tools)
        # No match — "shell" is not in tool_names
        assert not controller._has_text_tool_pattern("I need a shell script", tools)
        # No match — word-boundary prevents "search" matching inside "research"
        assert not controller._has_text_tool_pattern("I'll research the best approach", tools)
        # Clean content
        assert not controller._has_text_tool_pattern("The task is complete", tools)
        assert not controller._has_text_tool_pattern("", tools)
        # Falls back to static patterns when tool_names is empty
        assert controller._has_text_tool_pattern("Let me read_file to check", [])

    async def test_empty_response_bypasses_backoff(self, mock_backend: MockBackend) -> None:
        """Empty response sends immediate nudge, skipping classifier and backoff."""
        from llm_saia.core.controller import (
            ActionKind,
            ControllerConfig,
            DefaultController,
            Observation,
        )

        config = Config(backend=mock_backend, tools=[], executor=None, system=None)
        controller = DefaultController(config=ControllerConfig(llm_config=config))
        controller.reset()
        # Set last nudge to current iteration (normally would cause backoff)
        controller._last_nudge_iteration = 5

        obs = Observation(
            response=AgentResponse(content="", tool_calls=[], output_tokens=0),
            messages=[],
            iteration=6,  # Only 1 since last nudge — would normally be in backoff
            task="do something",
            tool_names=["search"],
            terminal_tool=None,
        )
        action = await controller.decide(obs)

        assert action.kind == ActionKind.INSTRUCT
        assert action.reason == "empty_response"
        assert "empty" in action.message.lower()

    async def test_text_tool_pattern_bypasses_backoff(self, mock_backend: MockBackend) -> None:
        """Text-tool-call pattern sends immediate nudge, skipping classifier and backoff."""
        from llm_saia.core.controller import (
            ActionKind,
            ControllerConfig,
            DefaultController,
            Observation,
        )

        config = Config(backend=mock_backend, tools=[], executor=None, system=None)
        controller = DefaultController(config=ControllerConfig(llm_config=config))
        controller.reset()
        controller._last_nudge_iteration = 5

        obs = Observation(
            response=AgentResponse(
                content="I'll use read_file to check the config",
                tool_calls=[],
                output_tokens=20,
            ),
            messages=[],
            iteration=6,  # Would normally be in backoff
            task="do something",
            tool_names=["read_file", "search"],
            terminal_tool=None,
        )
        action = await controller.decide(obs)

        assert action.kind == ActionKind.INSTRUCT
        assert action.reason == "text_tool_pattern"
        assert "text" in action.message.lower()

    async def test_degenerate_falls_through_after_limit(self, mock_backend: MockBackend) -> None:
        """After backoff_iterations consecutive degenerate nudges, falls through to classifier."""
        from llm_saia.core.controller import (
            ActionKind,
            ControllerConfig,
            DefaultController,
            Observation,
        )

        config = Config(backend=mock_backend, tools=[], executor=None, system=None)
        ctrl_config = ControllerConfig(llm_config=config, backoff_iterations=2)
        controller = DefaultController(config=ctrl_config)
        controller.reset()

        def make_obs(iteration: int) -> Observation:
            return Observation(
                response=AgentResponse(content="", tool_calls=[], output_tokens=0),
                messages=[],
                iteration=iteration,
                task="do something",
                tool_names=["search"],
                terminal_tool=None,
            )

        # First 2 degenerate responses → nudge (within limit)
        a1 = await controller.decide(make_obs(1))
        assert a1.kind == ActionKind.INSTRUCT
        assert a1.reason == "empty_response"

        a2 = await controller.decide(make_obs(2))
        assert a2.kind == ActionKind.INSTRUCT
        assert a2.reason == "empty_response"

        # 3rd consecutive → exceeds backoff_iterations=2, falls through to classifier.
        # Classifier returns WANTS_CONTINUE (default mock), and since the last nudge
        # was on iteration 2, backoff window still active → SKIP.
        a3 = await controller.decide(make_obs(3))
        assert a3.kind == ActionKind.SKIP
        assert "backoff" in a3.reason

    async def test_degenerate_counter_resets_on_tool_calls(self, mock_backend: MockBackend) -> None:
        """Consecutive degenerate counter resets when LLM makes real tool calls."""
        from llm_saia.core.controller import (
            ActionKind,
            ControllerConfig,
            DefaultController,
            Observation,
        )

        config = Config(backend=mock_backend, tools=[], executor=None, system=None)
        ctrl_config = ControllerConfig(llm_config=config, backoff_iterations=1)
        controller = DefaultController(config=ctrl_config)
        controller.reset()

        # 1 degenerate nudge (at the limit)
        obs_empty = Observation(
            response=AgentResponse(content="", tool_calls=[], output_tokens=0),
            messages=[],
            iteration=1,
            task="do something",
            tool_names=["search"],
            terminal_tool=None,
        )
        a1 = await controller.decide(obs_empty)
        assert a1.kind == ActionKind.INSTRUCT
        assert a1.reason == "empty_response"

        # LLM recovers — makes a tool call
        obs_tools = Observation(
            response=AgentResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="search", arguments={})],
                output_tokens=5,
            ),
            messages=[],
            iteration=2,
            task="do something",
            tool_names=["search"],
            terminal_tool=None,
        )
        a2 = await controller.decide(obs_tools)
        assert a2.kind == ActionKind.EXECUTE_TOOLS

        # Another degenerate — counter was reset, so should nudge again (not fall through)
        obs_empty2 = Observation(
            response=AgentResponse(content="", tool_calls=[], output_tokens=0),
            messages=[],
            iteration=3,
            task="do something",
            tool_names=["search"],
            terminal_tool=None,
        )
        a3 = await controller.decide(obs_empty2)
        assert a3.kind == ActionKind.INSTRUCT
        assert a3.reason == "empty_response"


class TestTaskStateClassifier:
    """Tests for TaskStateClassifier."""

    async def test_classifier_returns_completed_state(self, mock_backend: MockBackend) -> None:
        """Classifier returns COMPLETED when LLM classifies as completed."""
        from llm_saia.core.classifier import LLMTaskStateClassifier, TaskState

        config = Config(backend=mock_backend, tools=[], executor=None, system=None)
        classifier = LLMTaskStateClassifier(config)

        mock_backend.set_structured_response(
            ClassifyResult, ClassifyResult(category="completed", confidence=0.95, reason="Done")
        )

        result = await classifier.classify("do something", "Task done!", ["tool1", "tool2"])

        assert result.state == TaskState.COMPLETED
        assert result.confidence == 0.95
        assert result.reason == "Done"

    async def test_classifier_returns_stuck_state(self, mock_backend: MockBackend) -> None:
        """Classifier returns STUCK when LLM classifies as stuck."""
        from llm_saia.core.classifier import LLMTaskStateClassifier, TaskState

        config = Config(backend=mock_backend, tools=[], executor=None, system=None)
        classifier = LLMTaskStateClassifier(config)

        mock_backend.set_structured_response(
            ClassifyResult,
            ClassifyResult(category="stuck", confidence=0.8, reason="Cannot proceed"),
        )

        result = await classifier.classify("do something", "I'm stuck", ["tool1"])

        assert result.state == TaskState.STUCK

    async def test_classifier_falls_back_on_invalid_category(
        self, mock_backend: MockBackend
    ) -> None:
        """Classifier falls back to WANTS_CONTINUE on invalid category."""
        from llm_saia.core.classifier import LLMTaskStateClassifier, TaskState

        config = Config(backend=mock_backend, tools=[], executor=None, system=None)
        classifier = LLMTaskStateClassifier(config)

        mock_backend.set_structured_response(
            ClassifyResult,
            ClassifyResult(category="invalid_category", confidence=0.5, reason="Unknown"),
        )

        result = await classifier.classify("do something", "Response", [])

        assert result.state == TaskState.WANTS_CONTINUE

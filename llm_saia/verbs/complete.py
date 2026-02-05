"""COMPLETE verb: Execute a task with tool calling and completion confirmation."""

import json
import time
from collections.abc import Awaitable, Callable
from typing import Literal

from llm_saia.core.types import (
    AgentResponse,
    Message,
    RunConfig,
    TaskResult,
    ToolCall,
    VerbConfig,
)
from llm_saia.core.verb import Verb

# Default run config for complete (unlimited iterations)
DEFAULT_COMPLETE_RUN = RunConfig(max_iterations=0)


class Complete(Verb):
    """Execute a task with tool calling and completion confirmation."""

    async def __call__(
        self,
        task: str,
        on_iteration: Callable[[int, AgentResponse], Awaitable[None]] | None = None,
    ) -> TaskResult:
        """Execute a task using tools until completion or limit reached."""
        if not self._has_tools():
            raise ValueError("Complete requires tools and executor to be configured.")

        config = self._config.run or DEFAULT_COMPLETE_RUN
        messages: list[Message] = [Message(role="user", content=task)]
        start_time, iteration, total_tokens, last_content = time.monotonic(), 0, 0, ""
        self._log_loop_start(config)

        while not self._should_stop(config, iteration, start_time, total_tokens):
            response, tokens = await self._run_iteration(messages, config)
            total_tokens, last_content = total_tokens + tokens, response.content
            self._log_response(response, iteration, total_tokens)
            self._check_tool_support(response)
            if on_iteration:
                await on_iteration(iteration, response)

            result = await self._try_complete(task, response, messages, iteration)
            if result:
                self._log_loop_complete(
                    iteration, start_time, total_tokens, self._result_preview(result)
                )
                return result
            iteration += 1

        self._log_limit_reached(config, iteration, start_time, total_tokens)
        return TaskResult(False, last_content, iteration, messages)

    async def _try_complete(
        self, task: str, response: AgentResponse, messages: list[Message], iteration: int
    ) -> TaskResult | None:
        """Check if task completed via terminal tool or confirmation."""
        terminal_result = self._check_terminal_tool(task, response, messages, iteration)
        if terminal_result is False:
            # Terminal tool detected but awaiting confirmation - continue loop
            return None
        if terminal_result is not None:
            return terminal_result
        return await self._handle_response(task, response, messages, iteration)

    def _result_preview(self, result: TaskResult) -> str:
        """Get preview content from TaskResult for logging."""
        if result.output:
            return result.output
        if result.terminal_data:
            return self._safe_json_dumps(result.terminal_data)
        return ""

    def _safe_json_dumps(self, data: object, indent: int | None = None) -> str:
        """Serialize data to JSON, falling back to str() for non-serializable objects."""
        try:
            return json.dumps(data, indent=indent)
        except (TypeError, ValueError):
            return str(data)

    def _check_terminal_tool(
        self, task: str, response: AgentResponse, messages: list[Message], iteration: int
    ) -> TaskResult | Literal[False] | None:
        """Check if terminal tool was called.

        Returns:
            TaskResult: Terminal tool confirmed, task complete
            False: Terminal tool detected, confirmation injected, continue loop
            None: No terminal tool, proceed with normal handling
        """
        terminal_tool = self._config.terminal_tool
        if not terminal_tool or not response.tool_calls:
            return None

        terminal_call = next((tc for tc in response.tool_calls if tc.name == terminal_tool), None)
        if not terminal_call:
            return None

        messages.append(self._to_message(response))

        # Check if this is a confirmed terminal call (called twice in a row)
        if self._is_terminal_confirmed(messages, terminal_tool):
            return TaskResult(
                completed=True,
                output=response.content,
                iterations=iteration + 1,
                history=messages,
                terminal_data=terminal_call.arguments,
                terminal_tool=terminal_tool,
            )

        # First terminal call - ask for confirmation
        self._inject_terminal_confirmation(task, terminal_tool, terminal_call, messages)
        return False

    def _is_terminal_confirmed(self, messages: list[Message], terminal_tool: str) -> bool:
        """Check if this is a second terminal tool call (confirmation)."""
        # Look for a recent confirmation prompt in messages
        confirm_marker = f"call `{terminal_tool}` again to confirm"
        for msg in reversed(messages[:-1]):  # Exclude the message we just added
            if msg.role == "user" and confirm_marker in msg.content:
                return True
            # Only look back to the previous user message
            if msg.role == "user":
                break
        return False

    def _inject_terminal_confirmation(
        self, task: str, terminal_tool: str, terminal_call: ToolCall, messages: list[Message]
    ) -> None:
        """Inject confirmation prompt for terminal tool call."""
        # Add tool result acknowledging the call
        messages.append(
            Message(
                role="tool_result",
                content="Received. Please confirm this is your final response.",
                tool_call_id=terminal_call.id,
            )
        )
        # Add confirmation prompt
        data_preview = self._safe_json_dumps(terminal_call.arguments, indent=2)
        prompt = (
            f"You called `{terminal_tool}` to signal completion.\n\n"
            f"**Original task:** {task}\n\n"
            f"**Your response:**\n```json\n{data_preview}\n```\n\n"
            f"Is this your final response to the task?\n"
            f"- If YES, call `{terminal_tool}` again to confirm.\n"
            f"- If NO, continue working using the available tools."
        )
        messages.append(Message(role="user", content=prompt))

    async def _run_iteration(
        self, messages: list[Message], config: RunConfig
    ) -> tuple[AgentResponse, int]:
        """Run one LLM iteration and return response with token count."""
        max_tokens = config.max_call_tokens if config.max_call_tokens > 0 else None
        response = await self._chat(messages, max_tokens)
        return response, response.input_tokens + response.output_tokens

    async def _handle_response(
        self, task: str, response: AgentResponse, messages: list[Message], iteration: int
    ) -> TaskResult | None:
        """Handle LLM response - execute tools or check completion."""
        messages.append(self._to_message(response))

        if response.tool_calls:
            await self._execute_tools(response.tool_calls, messages)
            return None

        return await self._check_completion(task, response.content, messages, iteration)

    def _single_call_config(self) -> VerbConfig:
        """Create config for single-call verbs (no tools/looping)."""
        return VerbConfig(
            backend=self._config.backend,
            tools=[],
            executor=None,
            system=self._config.system,
            run=None,
            terminal_tool=None,
            lg=self._config.lg,
        )

    async def _check_completion(
        self, task: str, content: str, messages: list[Message], iteration: int
    ) -> TaskResult | None:
        """Check if task is complete. Returns TaskResult if done, None to continue."""
        from llm_saia.verbs.confirm import Confirm

        confirm = Confirm(self._single_call_config())
        confirmation = await confirm(
            claim="the task is complete based on the agent's response",
            context=f"Task: {task}\n\nAgent's response: {content}",
        )

        if confirmation.confirmed:
            return TaskResult(
                completed=True, output=content, iterations=iteration + 1, history=messages
            )

        wrap_up = (
            f"The task is not yet complete. Reason: {confirmation.reason}\n"
            "Please continue working on the task or use the available tools."
        )
        messages.append(Message(role="user", content=wrap_up))
        return None

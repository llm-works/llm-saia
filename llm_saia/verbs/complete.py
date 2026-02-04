"""COMPLETE verb: Execute a task with tool calling and completion confirmation."""

import time
from collections.abc import Awaitable, Callable

from llm_saia.core.types import (
    AgentResponse,
    Message,
    RunConfig,
    TaskResult,
)
from llm_saia.verbs._base import VerbConfig, _Verb

# Default run config for complete (unlimited iterations)
DEFAULT_COMPLETE_RUN = RunConfig(max_iterations=0)


class Complete(_Verb):
    """Execute a task with tool calling and completion confirmation."""

    async def __call__(
        self,
        task: str,
        on_iteration: Callable[[int, AgentResponse], Awaitable[None]] | None = None,
    ) -> TaskResult:
        if not self._has_tools():
            raise ValueError("Complete requires tools and executor to be configured.")

        config = self._config.run or DEFAULT_COMPLETE_RUN
        messages: list[Message] = [Message(role="user", content=task)]
        start_time, iteration, total_tokens, last_content = time.monotonic(), 0, 0, ""

        while not self._should_stop(config, iteration, start_time, total_tokens):
            response, tokens = await self._run_iteration(messages, config)
            total_tokens, last_content = total_tokens + tokens, response.content
            if on_iteration:
                await on_iteration(iteration, response)

            if result := self._check_terminal_tool(response, messages, iteration):
                return result
            if result := await self._handle_response(task, response, messages, iteration):
                return result
            iteration += 1

        return TaskResult(False, last_content, iteration, messages)

    def _check_terminal_tool(
        self, response: AgentResponse, messages: list[Message], iteration: int
    ) -> TaskResult | None:
        """Check if terminal tool was called. Returns TaskResult if so."""
        terminal_tool = self._config.terminal_tool
        if not terminal_tool or not response.tool_calls:
            return None

        terminal_call = next((tc for tc in response.tool_calls if tc.name == terminal_tool), None)
        if not terminal_call:
            return None

        messages.append(self._to_message(response))
        return TaskResult(
            completed=True,
            output=response.content,
            iterations=iteration + 1,
            history=messages,
            terminal_data=terminal_call.arguments,
            terminal_tool=terminal_tool,
        )

    async def _run_iteration(
        self, messages: list[Message], config: RunConfig
    ) -> tuple[AgentResponse, int]:
        """Run one LLM iteration and return response with token count."""
        self._backend.set_run_config(config)
        response = await self._backend.complete_with_tools(
            messages, self._config.tools, self._config.system
        )
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

    async def _check_completion(
        self, task: str, content: str, messages: list[Message], iteration: int
    ) -> TaskResult | None:
        """Check if task is complete. Returns TaskResult if done, None to continue."""
        # Use confirm verb for completion check with single-call config
        # to prevent nested loops with separate histories
        from llm_saia.verbs.confirm import Confirm

        single_call_config = VerbConfig(
            backend=self._config.backend,
            tools=[],  # No tools = no loop
            executor=None,
            system=self._config.system,
            run=None,
            terminal_tool=None,
        )
        confirm = Confirm(single_call_config)
        confirmation = await confirm(
            claim="the task is complete based on the agent's response",
            context=f"Task: {task}\n\nAgent's response: {content}",
        )

        if confirmation.confirmed:
            return TaskResult(
                completed=True, output=content, iterations=iteration + 1, history=messages
            )

        # Inject wrap-up prompt
        wrap_up = (
            f"The task is not yet complete. Reason: {confirmation.reason}\n"
            "Please continue working on the task or use the available tools."
        )
        messages.append(Message(role="user", content=wrap_up))
        return None

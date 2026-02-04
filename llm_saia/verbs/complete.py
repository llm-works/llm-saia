"""COMPLETE verb: Execute a task with tool calling and completion confirmation."""

import time
from collections.abc import Awaitable, Callable
from typing import Any

from llm_saia.core.types import (
    AgentResponse,
    LoopConfig,
    Message,
    TaskResult,
)
from llm_saia.verbs._base import _Verb

# Default loop config for complete (unlimited iterations)
DEFAULT_COMPLETE_LOOP = LoopConfig(max_iterations=0, timeout_secs=0)


class Complete(_Verb):
    """Execute a task with tool calling and completion confirmation."""

    async def __call__(
        self,
        task: str,
        loop: LoopConfig | None = None,
        on_iteration: Callable[[int, AgentResponse], Awaitable[None]] | None = None,
    ) -> TaskResult:
        if not self._has_tools():
            raise ValueError("Complete requires tools and executor to be configured.")

        config = loop or DEFAULT_COMPLETE_LOOP
        messages: list[Message] = [Message(role="user", content=task)]
        start_time, iteration, total_tokens, last_content = time.monotonic(), 0, 0, ""

        while not self._should_stop_complete(config, iteration, start_time, total_tokens):
            response, tokens = await self._run_iteration(messages, config)
            total_tokens, last_content = total_tokens + tokens, response.content
            if on_iteration:
                await on_iteration(iteration, response)
            result = await self._handle_response(task, response, messages, iteration)
            if result:
                return result
            iteration += 1

        return TaskResult(False, last_content, iteration, messages)

    async def _run_iteration(
        self, messages: list[Message], config: LoopConfig
    ) -> tuple[AgentResponse, int]:
        """Run one LLM iteration and return response with token count."""
        max_tokens = config.max_call_tokens if config.max_call_tokens > 0 else 4096
        response = await self._backend.complete_with_tools(
            messages, self._config.tools, self._config.system, max_tokens=max_tokens
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

    def _should_stop_complete(
        self, config: LoopConfig, iteration: int, start_time: float, total_tokens: int
    ) -> bool:
        """Check if complete loop should stop."""
        if config.max_iterations > 0 and iteration >= config.max_iterations:
            return True
        if config.timeout_secs > 0 and (time.monotonic() - start_time) >= config.timeout_secs:
            return True
        if config.max_total_tokens > 0 and total_tokens >= config.max_total_tokens:
            return True
        return False

    def _to_message(self, response: AgentResponse) -> Message:
        """Convert AgentResponse to Message."""
        return Message(
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls if response.tool_calls else None,
        )

    async def _check_completion(
        self, task: str, content: str, messages: list[Message], iteration: int
    ) -> TaskResult | None:
        """Check if task is complete. Returns TaskResult if done, None to continue."""
        # Use confirm verb for completion check
        from llm_saia.verbs.confirm import Confirm

        confirm = Confirm(self._config)
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

    async def _execute_tools(self, tool_calls: Any, messages: list[Message]) -> None:
        """Execute tool calls."""
        if not self._config.executor:
            return
        for tc in tool_calls:
            try:
                result = await self._config.executor(tc.name, tc.arguments)
            except Exception as e:
                result = f"Error: {e}"
            messages.append(Message(role="tool_result", content=str(result), tool_call_id=tc.id))

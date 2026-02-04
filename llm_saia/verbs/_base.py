"""Base class for verbs with shared loop logic."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from llm_saia.core.types import (
    AgentResponse,
    Message,
    RunConfig,
    ToolCall,
    ToolDef,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from llm_saia.core.protocols import SAIABackend

T = TypeVar("T")

# Default run config used when none provided
DEFAULT_RUN = RunConfig(max_iterations=3)


@dataclass
class VerbConfig:
    """Shared configuration for all verbs."""

    backend: SAIABackend
    tools: list[ToolDef]
    executor: Callable[[str, dict[str, Any]], Awaitable[Any]] | None
    system: str | None
    run: RunConfig | None = None
    terminal_tool: str | None = None


class _Verb(ABC):
    """Base class for all verbs. Provides shared loop functionality."""

    def __init__(self, config: VerbConfig):
        self._config = config

    @property
    def _backend(self) -> SAIABackend:
        return self._config.backend

    def _has_tools(self) -> bool:
        """Check if tools are configured."""
        return bool(self._config.tools and self._config.executor)

    def _get_run_config(self, run: RunConfig | None = None) -> RunConfig:
        """Get effective run config: call override > instance default > global default."""
        return run or self._config.run or DEFAULT_RUN

    async def _loop(
        self,
        prompt: str,
        run: RunConfig | None = None,
        schema: type[T] | None = None,
    ) -> tuple[str, T | None]:
        """Execute prompt with tool-calling loop."""
        config = self._get_run_config(run)
        messages: list[Message] = [Message(role="user", content=prompt)]
        start_time = time.monotonic()
        iteration = 0
        total_tokens = 0
        last_assistant_content = ""

        # Configure backend with run config
        self._backend.set_run_config(config)

        while not self._should_stop(config, iteration, start_time, total_tokens):
            response = await self._backend.complete_with_tools(
                messages, self._config.tools, self._config.system
            )
            total_tokens += response.input_tokens + response.output_tokens
            last_assistant_content = response.content
            messages.append(self._to_message(response))

            if response.tool_calls:
                await self._execute_tools(response.tool_calls, messages)
                iteration += 1
            else:
                return await self._finalize(prompt, response.content, schema)

        return await self._finalize(prompt, last_assistant_content, schema)

    def _should_stop(
        self, config: RunConfig, iteration: int, start_time: float, total_tokens: int
    ) -> bool:
        """Check if loop should stop."""
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

    async def _execute_tools(self, tool_calls: list[ToolCall], messages: list[Message]) -> None:
        """Execute tool calls and append results."""
        if not self._config.executor:
            return
        for tc in tool_calls:
            try:
                result = await self._config.executor(tc.name, tc.arguments)
            except Exception as e:
                result = f"Error: {e}"
            messages.append(Message(role="tool_result", content=str(result), tool_call_id=tc.id))

    async def _finalize(
        self, prompt: str, content: str, schema: type[T] | None
    ) -> tuple[str, T | None]:
        """Finalize result, optionally parsing structured output."""
        if schema:
            # Include gathered content so structured output has tool loop context
            structured_prompt = f"{prompt}\n\nBased on the following information:\n{content}"
            result = await self._backend.complete_structured(structured_prompt, schema)
            return content, result
        return content, None

    # --- High-level helpers for verbs ---

    async def _complete(self, prompt: str) -> str:
        """Complete with tools if available, otherwise direct."""
        if self._has_tools():
            content, _ = await self._loop(prompt)
            return content
        return await self._backend.complete(prompt)

    async def _complete_structured(self, prompt: str, schema: type[T]) -> T:
        """Complete structured with tools if available, otherwise direct."""
        if self._has_tools():
            _, result = await self._loop(prompt, schema=schema)
            if result is not None:
                return result
        return await self._backend.complete_structured(prompt, schema)

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the verb."""
        ...

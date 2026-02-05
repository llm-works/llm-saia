"""Base class for SAIA verbs."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from llm_saia.core.backend import AgentResponse, Message, ToolCall
from llm_saia.core.config import Config, RunConfig
from llm_saia.core.errors import StructuredOutputError, TruncatedResponseError
from llm_saia.core.schema import dataclass_to_json_schema, parse_json_to_dataclass

if TYPE_CHECKING:
    from llm_saia.core.backend import Backend
    from llm_saia.core.logger import Logger

T = TypeVar("T")

# Default run config used when none provided
DEFAULT_RUN = RunConfig(max_iterations=3)


class Verb(ABC):
    """Base class for all verbs. Subclass this to create custom verbs."""

    # Truncation limit for log previews
    _PREVIEW_LIMIT = 100

    def __init__(self, config: Config):
        """Initialize verb with configuration."""
        self._config = config

    @property
    def _backend(self) -> Backend:
        """Get the configured backend."""
        return self._config.backend

    @property
    def _lg(self) -> Logger | None:
        """Get the configured logger, if any."""
        return self._config.lg

    def _has_tools(self) -> bool:
        """Check if tools are configured."""
        return bool(self._config.tools and self._config.executor)

    def _get_run_config(self, run: RunConfig | None = None) -> RunConfig:
        """Get effective run config: call override > instance default > global default."""
        return run or self._config.run or DEFAULT_RUN

    def _log_loop_start(self, config: RunConfig) -> None:
        """Log verb loop start if logger available."""
        if self._lg:
            self._lg.debug(
                "verb loop started",
                extra={
                    "verb": self.__class__.__name__,
                    "max_iterations": config.max_iterations,
                    "timeout_secs": config.timeout_secs,
                    "max_total_tokens": config.max_total_tokens,
                },
            )

    def _log_response(self, response: AgentResponse, iteration: int, total_tokens: int) -> None:
        """Log LLM response if logger available."""
        if self._lg:
            self._lg.debug(
                "llm response received",
                extra={
                    "iters": iteration,
                    "tokens": {
                        "input": response.input_tokens,
                        "output": response.output_tokens,
                        "total": total_tokens,
                    },
                    "stop_reason": response.stop_reason,
                    "tool_calls": bool(response.tool_calls),
                    "preview": self._truncate(response.content, self._PREVIEW_LIMIT),
                },
            )
            self._lg.trace(
                "llm response content",
                extra={"content": response.content},
            )

    def _log_limit_reached(
        self, config: RunConfig, iteration: int, start_time: float, total_tokens: int
    ) -> None:
        """Log when loop limit is reached."""
        if self._lg:
            elapsed_secs = time.monotonic() - start_time
            self._lg.warning(
                "verb loop limit reached",
                extra={
                    "verb": self.__class__.__name__,
                    "iterations": iteration,
                    "total_tokens": total_tokens,
                    "elapsed_secs": int(elapsed_secs),
                    "limit_type": self._get_limit_type(
                        config, iteration, elapsed_secs, total_tokens
                    ),
                },
            )

    def _truncate(self, text: str, limit: int) -> str:
        """Truncate text with '... (N chars)' suffix if over limit."""
        if not text or len(text) <= limit:
            return text
        return f"{text[:limit]}... ({len(text)} chars)"

    # Tool-call JSON patterns that indicate LLM tried to call tools via text output.
    # These patterns are specific to common function-calling formats (OpenAI, Anthropic).
    # We require the pattern to look like a tool invocation structure, not just any JSON.
    _TOOL_CALL_PATTERNS = (
        '"function_call":',
        '"tool_calls":',
        '"tool_use":',
        '"name":',
    )

    # Minimum expected input tokens per tool definition (conservative estimate)
    _MIN_TOKENS_PER_TOOL = 50

    def _check_tool_support(self, response: AgentResponse) -> None:
        """Check for signs that the model may not natively support function calling."""
        if not self._config.warn_tool_support or not self._has_tools() or not self._lg:
            return
        self._warn_low_input_tokens(response)
        self._warn_tool_json_in_text(response)

    def _warn_low_input_tokens(self, response: AgentResponse) -> None:
        """Warn if input tokens suggest server ignored tool definitions."""
        if response.tool_calls:  # Tools working, no warning needed
            return
        tool_count = len(self._config.tools)
        min_expected = tool_count * self._MIN_TOKENS_PER_TOOL
        if response.input_tokens > 0 and response.input_tokens < min_expected:
            self._lg.warning(  # type: ignore[union-attr]
                "input tokens suspiciously low - server may be ignoring tool definitions",
                extra={
                    "input_tokens": response.input_tokens,
                    "tool_count": tool_count,
                    "min_expected": min_expected,
                },
            )

    def _warn_tool_json_in_text(self, response: AgentResponse) -> None:
        """Warn if LLM outputs tool-call JSON as text instead of using tool_calls."""
        if response.tool_calls or not response.content:
            return
        if any(pattern in response.content for pattern in self._TOOL_CALL_PATTERNS):
            self._lg.warning(  # type: ignore[union-attr]
                "tools configured but LLM returned text instead of tool_calls - "
                "model may not support function calling",
                extra={
                    "content_preview": self._truncate(response.content, self._PREVIEW_LIMIT),
                    "tool_count": len(self._config.tools),
                },
            )

    def _log_loop_complete(
        self, iteration: int, start_time: float, total_tokens: int, content: str
    ) -> None:
        """Log when loop completes normally."""
        if self._lg:
            self._lg.debug(
                "verb loop completed",
                extra={
                    "verb": self.__class__.__name__,
                    "iters": iteration + 1,
                    "total_tokens": total_tokens,
                    "elapsed_secs": int(time.monotonic() - start_time),
                    "preview": self._truncate(content, self._PREVIEW_LIMIT),
                },
            )

    async def _chat(self, messages: list[Message], max_tokens: int | None) -> AgentResponse:
        """Execute a single chat call."""
        if self._lg:
            last_msg = messages[-1] if messages else None
            self._lg.trace(
                "sending chat",
                extra={
                    "msg_count": len(messages),
                    "last_role": last_msg.role if last_msg else None,
                    "content": last_msg.content if last_msg else None,
                },
            )
        return await self._backend.chat(
            messages,
            system=self._config.system,
            tools=self._config.tools if self._config.tools else None,
            max_tokens=max_tokens,
        )

    async def _loop(
        self,
        prompt: str,
        run: RunConfig | None = None,
        schema: type[T] | None = None,
    ) -> tuple[str, T | None]:
        """Execute prompt with tool-calling loop."""
        config = self._get_run_config(run)
        messages: list[Message] = [Message(role="user", content=prompt)]
        start_time, iteration, total_tokens, last_content = time.monotonic(), 0, 0, ""

        self._log_loop_start(config)
        max_tokens = config.max_call_tokens if config.max_call_tokens > 0 else None

        while not self._should_stop(config, iteration, start_time, total_tokens):
            response = await self._chat(messages, max_tokens)
            total_tokens += response.input_tokens + response.output_tokens
            last_content = response.content
            messages.append(self._to_message(response))
            self._log_response(response, iteration, total_tokens)
            self._check_tool_support(response)

            if response.tool_calls:
                await self._execute_tools(response.tool_calls, messages)
                iteration += 1
            else:
                self._log_loop_complete(iteration, start_time, total_tokens, response.content)
                return await self._finalize(prompt, response.content, schema)

        self._log_limit_reached(config, iteration, start_time, total_tokens)
        return await self._finalize(prompt, last_content, schema)

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

    def _get_limit_type(
        self, config: RunConfig, iteration: int, elapsed_secs: float, total_tokens: int
    ) -> str:
        """Determine which limit caused the loop to stop."""
        if config.max_iterations > 0 and iteration >= config.max_iterations:
            return "max_iterations"
        if config.timeout_secs > 0 and elapsed_secs >= config.timeout_secs:
            return "timeout"
        if config.max_total_tokens > 0 and total_tokens >= config.max_total_tokens:
            return "max_tokens"
        return "unknown"

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
            if self._lg:
                self._lg.warning(
                    "tool calls received but no executor configured",
                    extra={"tool_count": len(tool_calls)},
                )
            return
        for tc in tool_calls:
            if self._lg:
                self._lg.trace(
                    "executing tool...",
                    extra={"tool": tc.name, "id": tc.id},
                )
            try:
                result = await self._config.executor(tc.name, tc.arguments)
            except Exception as e:
                if self._lg:
                    self._lg.warning(
                        "tool execution failed",
                        extra={"tool": tc.name, "id": tc.id, "exception": e},
                    )
                result = f"Error: {e}"
            else:
                if self._lg:
                    self._lg.trace(
                        "tool executed",
                        extra={"tool": tc.name, "id": tc.id, "result": str(result)},
                    )
            messages.append(Message(role="tool_result", content=str(result), tool_call_id=tc.id))

    async def _finalize(
        self, prompt: str, content: str, schema: type[T] | None
    ) -> tuple[str, T | None]:
        """Finalize result, optionally parsing structured output."""
        if schema:
            # Request structured output with schema
            structured_prompt = f"{prompt}\n\nBased on the following information:\n{content}"
            json_schema = dataclass_to_json_schema(schema)
            response = await self._backend.chat(
                [Message(role="user", content=structured_prompt)],
                system=self._config.system,
                response_schema=json_schema,
            )
            try:
                data = json.loads(response.content)
            except json.JSONDecodeError as e:
                if self._lg:
                    preview = self._truncate(response.content, self._PREVIEW_LIMIT)
                    self._lg.warning(
                        "json parse error in finalize",
                        extra={
                            "exception": e,
                            "content_preview": preview,
                            "schema": schema.__name__,
                        },
                    )
                raise ValueError(f"LLM returned invalid JSON for structured output: {e}") from e
            result = parse_json_to_dataclass(data, schema)
            return content, result
        return content, None

    # --- High-level helpers for verbs ---

    async def _complete(self, prompt: str) -> str:
        """Complete with tools if available, otherwise direct."""
        if self._has_tools():
            content, _ = await self._loop(prompt)
            return content
        response = await self._backend.chat(
            [Message(role="user", content=prompt)],
            system=self._config.system,
        )
        return response.content

    async def _complete_structured(self, prompt: str, schema: type[T]) -> T:
        """Complete structured with tools if available, otherwise direct."""
        if self._has_tools():
            _, result = await self._loop(prompt, schema=schema)
            if result is not None:
                return result
        # Direct structured completion
        json_schema = dataclass_to_json_schema(schema)
        response = await self._backend.chat(
            [Message(role="user", content=prompt)],
            system=self._config.system,
            response_schema=json_schema,
        )
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError as e:
            raise self._structured_output_error(e, response.content, schema.__name__) from e
        return parse_json_to_dataclass(data, schema)

    def _structured_output_error(
        self, error: json.JSONDecodeError, content: str, schema_name: str
    ) -> StructuredOutputError:
        """Create appropriate error for structured output parse failure."""
        error_msg = str(error)
        # Detect truncation patterns
        truncation_indicators = (
            "Unterminated string",
            "Unexpected end of JSON",
            "Expecting value",
            "Expecting ',' delimiter",
            "Expecting ':' delimiter",
        )
        is_truncated = any(indicator in error_msg for indicator in truncation_indicators)

        if is_truncated:
            return TruncatedResponseError(
                raw_content=content,
                schema_name=schema_name,
                parse_error=error_msg,
            )
        return StructuredOutputError(
            f"LLM returned invalid JSON for {schema_name}: {error_msg}",
            raw_content=content,
            schema_name=schema_name,
            parse_error=error_msg,
        )

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the verb."""
        ...

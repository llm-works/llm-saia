"""Logging and diagnostics mixin for verb classes.

Provides loop logging, response logging, tool-support warnings, and text truncation.
Separated from verb.py to keep the base class manageable.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .backend import ChatResponse
    from .config import CallOptions, Config
    from .logger import Logger


class VerbLoggingMixin:
    """Mixin providing logging and diagnostics for the verb loop.

    Expects the host class to provide:
      - ``_lg`` property -> Logger | None
      - ``_config`` -> Config
      - ``_PREVIEW_LIMIT`` class attribute (debug-level truncation)
      - ``_TRACE_LIMIT`` class attribute (trace-level truncation)
    """

    # Stubs for attributes/methods provided by the host class (Verb).

    _config: Config
    _lg: Logger
    _PREVIEW_LIMIT: int
    _TRACE_LIMIT: int

    def _has_tools(self) -> bool:
        raise NotImplementedError

    # -- Loop lifecycle logging --

    def _log_loop_start(self, config: CallOptions) -> None:
        """Log verb loop start."""
        self._lg.debug(
            "verb loop started",
            extra={
                "verb": self.__class__.__name__,
                "max_iterations": config.max_iterations,
                "timeout_secs": config.timeout_secs,
                "max_total_tokens": config.max_total_tokens,
            },
        )

    def _log_response(self, response: ChatResponse, iteration: int, total_tokens: int) -> None:
        """Log LLM response."""
        self._lg.debug(
            "llm response received",
            extra={
                "call_id": response.call_id,
                "iters": iteration,
                "tokens": {
                    "input": response.input_tokens,
                    "output": response.output_tokens,
                    "total": total_tokens,
                },
                "finish_reason": response.finish_reason,
                "tool_calls": len(response.tool_calls) if response.tool_calls else 0,
                "preview": self._truncate(response.content, self._PREVIEW_LIMIT),
            },
        )
        tools = (
            {
                str(i + 1): {"name": tc.name, "args": tc.arguments}
                for i, tc in enumerate(response.tool_calls)
            }
            if response.tool_calls
            else None
        )
        self._lg.trace(
            "llm response details",
            extra=OrderedDict([("tools", tools), ("content", response.content)]),
        )

    def _log_limit_reached(
        self, config: CallOptions, iteration: int, start_time: float, total_tokens: int
    ) -> None:
        """Log when loop limit is reached."""
        elapsed_secs = time.monotonic() - start_time
        self._lg.warning(
            "verb loop limit reached",
            extra={
                "verb": self.__class__.__name__,
                "iterations": iteration,
                "total_tokens": total_tokens,
                "elapsed_secs": int(elapsed_secs),
                "limit_type": self._get_limit_type(config, iteration, elapsed_secs, total_tokens),
            },
        )

    def _log_loop_complete(
        self, iteration: int, start_time: float, total_tokens: int, content: str
    ) -> None:
        """Log when loop completes normally."""
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

    # -- Limit type detection (used by _log_limit_reached) --

    def _get_limit_type(
        self,
        config: CallOptions,
        iteration: int,
        elapsed_secs: float,
        total_tokens: int,
    ) -> str:
        """Determine which limit triggered the stop."""
        if config.max_iterations > 0 and iteration >= config.max_iterations:
            return "max_iterations"
        if config.timeout_secs > 0 and elapsed_secs >= config.timeout_secs:
            return "timeout"
        if config.max_total_tokens > 0 and total_tokens >= config.max_total_tokens:
            return "max_tokens"
        return "unknown"

    # -- Message assembly logging (for debugging stuck loops) --

    def _log_message_assembly(self, call_id: str, messages: list[Any]) -> None:
        """Log the messages being sent to LLM (critical for debugging stuck loops)."""
        role_counts = self._count_roles(messages)
        last_user = self._find_last_user_msg(messages)
        tool_results = self._find_recent_tool_results(messages)

        self._lg.trace(
            "sending messages to llm",
            extra={
                "call_id": call_id,
                "msg_count": len(messages),
                "by_role": role_counts,
                "last_user_msg": last_user,
                "recent_tool_results": tool_results if tool_results else None,
            },
        )

    def _count_roles(self, messages: list[Any]) -> dict[str, int]:
        """Count messages by role."""
        counts: dict[str, int] = {}
        for msg in messages:
            role = msg.role if hasattr(msg, "role") else "unknown"
            counts[role] = counts.get(role, 0) + 1
        return counts

    def _find_last_user_msg(self, messages: list[Any]) -> str | None:
        """Find the last user message content (truncated for trace logging)."""
        for msg in reversed(messages):
            if hasattr(msg, "role") and msg.role == "user":
                return self._truncate(str(msg.content), self._TRACE_LIMIT)
        return None

    def _find_recent_tool_results(self, messages: list[Any]) -> list[str]:
        """Find recent tool results (last 2 from last 5 messages, truncated for trace logging)."""
        results = []
        for msg in reversed(messages[-5:]):
            if hasattr(msg, "role") and msg.role == "tool":
                results.append(self._truncate(str(msg.content), self._TRACE_LIMIT))
            if len(results) >= 2:
                break
        results.reverse()
        return results

    # -- Text truncation --

    def _truncate(self, text: str, limit: int) -> str:
        """Truncate text with '... (N chars)' suffix if over limit."""
        if not text or len(text) <= limit:
            return text
        return f"{text[:limit]}... ({len(text)} chars)"

    # -- Tool-support diagnostics --

    # Tool-call JSON patterns that indicate LLM tried to call tools via text output.
    _TOOL_CALL_PATTERNS = (
        '"function_call":',
        '"tool_calls":',
        '"tool_use":',
    )

    # Minimum expected input tokens per tool definition (conservative estimate)
    _MIN_TOKENS_PER_TOOL = 50

    def _check_tool_support(self, response: ChatResponse) -> None:
        """Check for signs that the model may not natively support function calling."""
        if not self._config.warn_tool_support or not self._has_tools():
            return
        self._warn_low_input_tokens(response)
        self._warn_tool_json_in_text(response)

    def _warn_low_input_tokens(self, response: ChatResponse) -> None:
        """Warn if input tokens suggest server ignored tool definitions."""
        if response.tool_calls:  # Tools working, no warning needed
            return
        tool_count = len(self._config.tools)
        min_expected = tool_count * self._MIN_TOKENS_PER_TOOL
        if response.input_tokens > 0 and response.input_tokens < min_expected:
            self._lg.warning(
                "input tokens suspiciously low - server may be ignoring tool definitions",
                extra={
                    "input_tokens": response.input_tokens,
                    "tool_count": tool_count,
                    "min_expected": min_expected,
                },
            )

    def _warn_tool_json_in_text(self, response: ChatResponse) -> None:
        """Warn if LLM outputs tool-call JSON as text instead of using tool_calls."""
        if response.tool_calls or not response.content:
            return
        if self._looks_like_tool_call_json(response.content):
            self._lg.warning(
                "tools configured but LLM returned text instead of tool_calls - "
                "model may not support function calling",
                extra={
                    "content_preview": self._truncate(response.content, self._PREVIEW_LIMIT),
                    "tool_count": len(self._config.tools),
                },
            )

    def _looks_like_tool_call_json(self, content: str) -> bool:
        """Check if content looks like tool-call JSON (not just any JSON with 'name')."""
        # Explicit tool-call patterns are definitive
        if any(pattern in content for pattern in self._TOOL_CALL_PATTERNS):
            return True
        # "name" alone is too broad; require it alongside "arguments" or "parameters"
        has_name = '"name":' in content
        has_args = '"arguments":' in content or '"parameters":' in content
        return has_name and has_args

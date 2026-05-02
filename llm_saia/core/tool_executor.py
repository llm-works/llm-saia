"""Tool execution for verbs.

Handles executing tool calls and appending results to messages.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

from .conversation import Message, Role
from .errors import PauseRequested

if TYPE_CHECKING:
    from .conversation import MessageAppendable, ToolCall


class _ToolHost(Protocol):
    """Protocol for capabilities the tool executor needs from its host."""

    _TRACE_LIMIT: int

    def _truncate(self, text: str, limit: int) -> str:
        """Truncate text to limit."""
        ...

    @staticmethod
    async def _append_msg(target: MessageAppendable, msg: Message) -> None:
        """Append message to target."""
        ...

    @property
    def _config(self) -> Any:
        """Get configuration."""
        ...

    @property
    def _lg(self) -> Any:
        """Logger instance."""
        ...


class _ToolExecutor:
    """Executes tool calls with logging and pause support."""

    def __init__(self, host: _ToolHost):
        """Initialize with host providing required capabilities."""
        self._host = host

    async def execute_tools(
        self,
        tool_calls: list[ToolCall],
        messages: MessageAppendable,
        pause_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        """Execute tool calls and append results.

        Args:
            tool_calls: Tool calls to execute.
            messages: Object supporting append() - list[Message] or ConversationLike.
            pause_check: Optional async callback checked between tool calls.
                If it returns True, remaining tools are acknowledged as "Paused."
                and ``PauseRequested`` is raised.
        """
        h = self._host
        if not h._config.executor:
            h._lg.warning(
                "tool calls received but no executor configured",
                extra={"tool_count": len(tool_calls)},
            )
            return
        for i, tc in enumerate(tool_calls):
            result = await self._execute_single(tc)
            await h._append_msg(
                messages, Message(role=Role.TOOL, content=str(result), tool_call_id=tc.id)
            )
            if pause_check is not None and i < len(tool_calls) - 1:
                if await pause_check():
                    await self._handle_pause(tool_calls[i + 1 :], messages, i + 1)

    async def _handle_pause(
        self, remaining: list[ToolCall], messages: MessageAppendable, executed: int
    ) -> None:
        """Handle pause by acknowledging remaining tools and raising."""
        h = self._host
        for rem_tc in remaining:
            msg = Message(role=Role.TOOL, content="Paused.", tool_call_id=rem_tc.id)
            await h._append_msg(messages, msg)
        h._lg.debug(
            "pause requested between tool calls",
            extra={"executed": executed, "remaining": len(remaining)},
        )
        raise PauseRequested()

    async def _execute_single(self, tc: ToolCall) -> str:
        """Execute a single tool call with logging."""
        self._log_start(tc)
        try:
            result = await self._host._config.executor(tc.name, tc.arguments)
        except Exception as e:
            self._log_error(tc, e)
            return f"Error: {e}"
        self._log_success(tc, result)
        return str(result)

    def _log_start(self, tc: ToolCall) -> None:
        """Log tool execution start."""
        self._host._lg.trace(
            "executing tool...",
            extra={"tool": tc.name, "id": tc.id, "tool_args": tc.arguments},
        )

    def _log_success(self, tc: ToolCall, result: Any) -> None:
        """Log successful tool execution with result."""
        h = self._host
        result_str = str(result)
        h._lg.trace(
            "tool result returned to llm",
            extra={
                "tool": tc.name,
                "id": tc.id,
                "result_len": len(result_str),
                "result": h._truncate(result_str, h._TRACE_LIMIT),
            },
        )

    def _log_error(self, tc: ToolCall, error: Exception) -> None:
        """Log failed tool execution."""
        self._host._lg.warning(
            "tool execution failed",
            extra={"tool": tc.name, "id": tc.id, "tool_args": tc.arguments, "exception": error},
        )

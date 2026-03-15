"""Configuration classes for SAIA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from llm_saia.core.backend import Backend, ToolDef
    from llm_saia.core.logger import Logger
    from llm_saia.core.trace import Tracer

__all__ = [
    "CallOptions",
    "TerminalConfig",
    "Config",
    "DEFAULT_CALL",
]


@dataclass
class CallOptions:
    """Per-call options that can vary between verb invocations.

    These settings can be modified via SAIA's with_*() methods to create
    new instances with different options per call.
    """

    # Prompt
    system: str | None = None  # System prompt

    # Sampling
    temperature: float | None = None  # Sampling temperature (None = backend default)

    # Limits
    max_call_tokens: int = 0  # Max tokens per LLM call (0 = backend default)
    max_total_tokens: int = 0  # Total token budget across loop (0 = unlimited)
    timeout_secs: float = 0  # Soft timeout in seconds (0 = no timeout)
    max_iterations: int = 3  # Max tool-calling rounds (0 = unlimited)

    # Retry behavior
    max_retries: int = 1  # Number of retry attempts (1 = no retry)
    retry_escalation: str | None = None  # Prompt added on retry attempts

    # Tracing
    request_id: str | None = None  # User-provided correlation ID


@dataclass
class TerminalConfig:
    """Configuration for terminal tool behavior.

    The terminal tool is a special tool that signals task completion.
    When the LLM calls this tool, the controller confirms and extracts the result.
    """

    tool: str  # Name of the terminal tool (e.g., "complete_task")
    output_field: str | None = None  # Field containing output (default: check common names)
    status_field: str | None = None  # Field containing status (default: "status")
    failure_values: tuple[str, ...] = ("stuck", "failed", "error")  # Status values = failure


@dataclass
class Config:
    """Immutable instance configuration for SAIA.

    These settings are fixed at construction time and cannot vary per-call.
    For per-call options, see CallOptions.
    """

    backend: Backend
    tools: list[ToolDef]
    executor: Callable[[str, dict[str, Any]], Awaitable[Any]] | None
    call: CallOptions | None = None  # Per-call options (defaults applied if None)
    terminal: TerminalConfig | None = None  # Terminal tool configuration
    lg: Logger | None = None
    tracer: Tracer | None = None  # Default tracer for iteration tracing
    warn_tool_support: bool = True


# Default call options
DEFAULT_CALL = CallOptions(max_iterations=3)

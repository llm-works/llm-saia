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
    "RunConfig",
    "TerminalConfig",
    "Config",
    "DEFAULT_RUN",
]


@dataclass
class RunConfig:
    """Configuration for verb execution.

    Controls limits, tool-calling iterations, and retry behavior.
    """

    max_call_tokens: int = 0  # Max tokens per LLM call (0 = backend default)
    max_total_tokens: int = 0  # Total token budget across loop (0 = unlimited)
    timeout_secs: float = 0  # Soft timeout in seconds (0 = no timeout)
    max_iterations: int = 3  # Max tool-calling rounds (0 = unlimited)
    max_retries: int = 1  # Number of retry attempts (1 = no retry)
    retry_escalation: str | None = None  # Prompt added on retry attempts

    def with_single_call(self) -> RunConfig:
        """Return a config for single LLM call (no looping)."""
        return RunConfig(
            max_call_tokens=self.max_call_tokens,
            max_total_tokens=self.max_total_tokens,
            timeout_secs=self.timeout_secs,
            max_iterations=1,
            max_retries=self.max_retries,
            retry_escalation=self.retry_escalation,
        )

    def with_retries(self, max_retries: int, escalation: str | None = None) -> RunConfig:
        """Return a config with retry settings."""
        return RunConfig(
            max_call_tokens=self.max_call_tokens,
            max_total_tokens=self.max_total_tokens,
            timeout_secs=self.timeout_secs,
            max_iterations=self.max_iterations,
            max_retries=max_retries,
            retry_escalation=escalation,
        )


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
    """Configuration for SAIA instances."""

    backend: Backend
    tools: list[ToolDef]
    executor: Callable[[str, dict[str, Any]], Awaitable[Any]] | None
    system: str | None
    run: RunConfig | None = None
    terminal: TerminalConfig | None = None  # Terminal tool configuration
    lg: Logger | None = None
    warn_tool_support: bool = True
    tracer: Tracer | None = None  # Default tracer for iteration tracing
    request_id: str | None = None  # User-provided correlation ID


# Default run config used when none provided
DEFAULT_RUN = RunConfig(max_iterations=3)

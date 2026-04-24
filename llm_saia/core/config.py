"""Configuration classes for SAIA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .logger import Logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .backend import Backend, ToolDef
    from .guard import IterationGuard, OutputGuard
    from .trace import Tracer

__all__ = [
    "CallOptions",
    "Config",
    "DEFAULT_CALL",
    "JsonParser",
    "TerminalConfig",
]


class JsonParser(Protocol):
    """Protocol for custom JSON parsers.

    Default is json.loads. Override to handle malformed JSON from some backends
    or to use alternative parsers (orjson, json-repair, etc.).
    """

    def __call__(self, content: str) -> Any:
        """Parse JSON string to Python object."""
        ...


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

    # Output guards (validators with retry)
    output_guards: tuple[OutputGuard, ...] = field(default_factory=tuple)

    # Iteration guards (behavioral constraints enforced each loop iteration)
    iteration_guards: tuple[IterationGuard, ...] = field(default_factory=tuple)

    # Tracing
    request_id: str | None = None  # User-provided correlation ID


@dataclass
class TerminalConfig:
    """Configuration for terminal tool behavior.

    The terminal tool is a special tool that signals task completion.
    When the LLM calls this tool, the controller confirms and extracts the result.

    Note:
        Many models respond to confirmation prompts with text instead of a second
        tool call, causing ``terminal_data`` to be ``None``. Set
        ``require_confirmation=False`` if you don't need explicit confirmation.
    """

    tool: str  # Name of the terminal tool (e.g., "complete_task")
    output_field: str | None = None  # Field containing output (default: check common names)
    status_field: str | None = None  # Field containing status (default: "status")
    failure_values: tuple[str, ...] = ("stuck", "failed", "error")  # Status values = failure
    require_confirmation: bool = True  # Require second call to confirm completion


@dataclass
class Config:
    """Immutable instance configuration for SAIA.

    These settings are fixed at construction time and cannot vary per-call.
    For per-call options, see CallOptions.
    """

    lg: Logger  # Logger is always first, never optional
    backend: Backend
    tools: list[ToolDef]
    executor: Callable[[str, dict[str, Any]], Awaitable[Any]] | None
    call: CallOptions | None = None  # Per-call options (defaults applied if None)
    terminal: TerminalConfig | None = None  # Terminal tool configuration
    tracer: Tracer | None = None  # Default tracer for iteration tracing
    warn_tool_support: bool = True
    json_parser: JsonParser | None = None


# Default call options
DEFAULT_CALL = CallOptions(max_iterations=3)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Configuration classes for SAIA."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .logger import Logger

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from .backend import Backend, ChatResponse, ToolDef
    from .conversation import Message
    from .guard import IterationGuard, OutputGuard
    from .trace import Tracer

__all__ = [
    "CallOptions",
    "Config",
    "DEFAULT_CALL",
    "JsonParser",
    "TerminalConfig",
    "ToolGate",
    "ToolGateContext",
    "ToolGateContextFactory",
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
    context: dict[str, Any] | None = None  # Passed to backend for callbacks


@dataclass(frozen=True)
class ToolGateContext:
    """Context passed to a :class:`ToolGate` callback.

    A :class:`ToolGate` is evaluated *before* each LLM call. When it blocks,
    SAIA strips the corresponding tool from the outbound tool schema for that
    call — the model never sees the tool and cannot spend output tokens
    generating a call to it.

    Attributes:
        iteration: Current 0-indexed loop iteration.
        tool_call_counts: Cumulative count of tool invocations observed in the
            prior transcript, keyed by tool name. Includes only tool calls
            actually issued by the assistant.
        last_response: Most recent :class:`ChatResponse`, or ``None`` on the
            first iteration.
        messages: Outbound transcript for the pending call. Read-only —
            mutating it will corrupt loop state.
        extra: Value returned by :data:`ToolGateContextFactory` when one is
            configured, otherwise ``None``. Use this to attach shared derived
            state that multiple gates read on the same iteration (avoids each
            gate re-walking the transcript).
    """

    iteration: int
    tool_call_counts: Mapping[str, int]
    last_response: ChatResponse | None
    messages: Sequence[Message]
    extra: Any = None


ToolGate = Callable[[ToolGateContext], "bool | str | None"]
"""Per-tool gate callback.

Return value semantics:

- ``True`` / ``None`` — allow (tool remains in the outbound schema).
- ``False`` — block silently.
- ``str`` — block; string is recorded in the trace/log as the reason.

Reasons are diagnostic for the operator, never surfaced to the model — the
whole point of schema-hiding is to prevent the model from thinking about a
tool we've decided to withhold.
"""

ToolGateContextFactory = Callable[[ToolGateContext], Any]
"""Optional factory that computes shared per-iteration state for gates.

Called once per LLM call with the base :class:`ToolGateContext`; its return
value is attached as ``ctx.extra`` for every gate on the same iteration. Use
when two or more gates need the same derived signal (e.g. a transcript walk
extracting some tool's arguments) — the walk happens once instead of once
per gate.
"""


@dataclass
class TerminalConfig:
    """Configuration for terminal tool behavior.

    The terminal tool is a special tool that signals task completion.
    When the LLM calls this tool, the controller confirms and extracts the result.

    ``min_iterations`` is a shortcut for the common "force N research
    iterations before self-termination" pattern — the terminal tool is
    hidden from the outbound schema until iteration ``>= min_iterations``.
    It lowers to an entry in :attr:`Config.tool_gates` keyed by
    :attr:`tool`; the runtime knows nothing about it directly. For richer
    gating logic — including gating tools other than the terminal one —
    register a :data:`ToolGate` via :attr:`Config.tool_gates` directly.

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
    min_iterations: int = 0  # Shortcut: hide terminal tool until iteration >= this

    def __post_init__(self) -> None:
        """Validate gating fields."""
        if self.min_iterations < 0:
            raise ValueError(f"min_iterations must be >= 0, got {self.min_iterations}")


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
    # Per-tool visibility gates evaluated before each LLM call. Keyed by
    # tool name; a blocked tool is filtered from that call's outbound schema
    # so the model never sees it and cannot spend tokens generating a call.
    tool_gates: Mapping[str, ToolGate] = field(default_factory=dict)
    # Optional factory computing shared per-iteration state for gates
    # (attached as ToolGateContext.extra). See :data:`ToolGateContextFactory`.
    tool_gate_context_factory: ToolGateContextFactory | None = None


# Default call options
DEFAULT_CALL = CallOptions(max_iterations=3)

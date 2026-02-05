"""Core data types for SAIA verb results and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Re-export backend types for convenience
from llm_saia.core.backend import AgentResponse, Message, ToolCall, ToolDef

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from llm_saia.core.backend import SAIABackend
    from llm_saia.core.logger import SAIALogger

__all__ = [
    # Backend types (re-exported)
    "AgentResponse",
    "Message",
    "ToolCall",
    "ToolDef",
    # Verb results
    "ChooseResult",
    "ClassifyResult",
    "ConfirmResult",
    "Critique",
    "Evidence",
    "VerbResult",
    "VerifyResult",
    # Task types
    "RunConfig",
    "TaskResult",
    # Configuration
    "VerbConfig",
]


@dataclass
class VerbResult:
    """Result from any verb execution."""

    value: Any
    verb: str
    success: bool
    error: str | None = None


@dataclass
class VerifyResult:
    """Result from VERIFY verb."""

    passed: bool
    reason: str


@dataclass
class Critique:
    """Result from CRITIQUE verb."""

    counter_argument: str
    weaknesses: list[str]
    strength: float  # 0.0 to 1.0


@dataclass
class Evidence:
    """Extracted evidence from a source."""

    content: str
    source: str
    direction: str  # "supports", "refutes", "neutral"
    strength: float  # 0.0 to 1.0


@dataclass
class ClassifyResult:
    """Result from CLASSIFY verb."""

    category: str
    confidence: float  # 0.0 to 1.0
    reason: str


@dataclass
class ConfirmResult:
    """Result from CONFIRM verb."""

    confirmed: bool
    reason: str


@dataclass
class ChooseResult:
    """Result from CHOOSE verb."""

    choice: str
    reason: str


@dataclass
class TaskResult:
    """Result from task execution."""

    completed: bool
    output: str
    iterations: int
    history: list[Message]
    terminal_data: dict[str, Any] | None = None
    terminal_tool: str | None = None


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
class VerbConfig:
    """Shared configuration for all verbs."""

    backend: SAIABackend
    tools: list[ToolDef]
    executor: Callable[[str, dict[str, Any]], Awaitable[Any]] | None
    system: str | None
    run: RunConfig | None = None
    terminal_tool: str | None = None
    lg: SAIALogger | None = None
    warn_tool_support: bool = True

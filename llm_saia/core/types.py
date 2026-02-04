"""Core data types for SAIA verb results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


# --- Tool calling types for task execution ---


@dataclass
class ToolDef:
    """Definition of a tool that can be called by the LLM.

    Example:
        ToolDef(
            name="search",
            description="Search for information",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        )
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass
class ToolCall:
    """A tool invocation from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """A message in the conversation history."""

    role: str  # "user", "assistant", "tool_result"
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool_result messages


@dataclass
class AgentResponse:
    """Response from LLM that may include tool calls."""

    content: str
    tool_calls: list[ToolCall]
    stop_reason: str | None = None  # "end_turn", "tool_use", etc.
    input_tokens: int = 0  # Tokens used for input/prompt
    output_tokens: int = 0  # Tokens used for response


@dataclass
class TaskResult:
    """Result from task execution."""

    completed: bool
    output: str
    iterations: int
    history: list[Message]
    terminal_data: dict[str, Any] | None = None  # Arguments from terminal tool
    terminal_tool: str | None = None  # Name of terminal tool that was called


@dataclass
class RunConfig:
    """Configuration for verb execution.

    Controls limits, tool-calling iterations, and retry behavior.

    Token limits are important for cost control with paid APIs.
    Set max_total_tokens=0 for unlimited (useful for local LLMs).
    """

    # Limits (apply to all calls)
    max_call_tokens: int = 0  # Max tokens per LLM call (0 = backend default)
    max_total_tokens: int = 0  # Total token budget across loop (0 = unlimited)
    timeout_secs: float = 0  # Soft timeout in seconds (0 = no timeout)

    # Loop (tool iterations)
    max_iterations: int = 3  # Max tool-calling rounds (0 = unlimited)

    # Retry (single-call re-attempts)
    max_retries: int = 1  # Number of retry attempts (1 = no retry)
    retry_escalation: str | None = None  # Prompt added on retry attempts

    def with_single_call(self) -> RunConfig:
        """Return a config for single LLM call (no looping).

        Preserves token limits but sets max_iterations=1.
        """
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

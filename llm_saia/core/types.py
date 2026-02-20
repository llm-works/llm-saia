"""Core data types for SAIA verb results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# Re-export backend types for convenience
from llm_saia.core.backend import AgentResponse, Message, ToolCall, ToolDef

# Re-export config types for backwards compatibility
from llm_saia.core.config import Config, RunConfig


class DecisionReason(Enum):
    """Reason codes for controller decisions.

    Used in Action.reason to indicate why the controller made a particular decision.
    Separated from ActionType to provide more granular insight into the decision-making
    process for debugging and analysis.
    """

    # EXECUTE_TOOLS reasons
    HAS_TOOL_CALLS = "has_tool_calls"  # LLM made tool calls
    TERMINAL_WITH_OTHER_TOOLS = "terminal_with_other_tools"  # Terminal + other tools in batch

    # INSTRUCT reasons
    EMPTY_RESPONSE = "empty_response"  # LLM returned empty response
    TEXT_TOOL_PATTERN = "text_tool_pattern"  # LLM wrote tool names as text
    TERMINAL_CONFIRMATION_REQUEST = "terminal_confirmation_request"  # Asking to confirm terminal
    TERMINAL_FAILURE_RETRY = "terminal_failure_retry"  # Retry after terminal failure
    CONTRADICTION_DETECTED = "contradiction_detected"  # LLM contradicted terminal confirmation
    NUDGE_CLASSIFIED = "nudge_classified"  # Classifier suggests nudge (wants_continue, stuck, etc)

    # SKIP reasons
    BACKOFF = "backoff"  # In backoff window after nudge

    # COMPLETE reasons
    TERMINAL_CONFIRMED = "terminal_confirmed"  # Terminal tool confirmed
    CLASSIFIED_COMPLETE = "classified_complete"  # Classifier detected completion

    # FAIL reasons
    TERMINAL_CONFIRMED_FAIL = "terminal_confirmed_fail"  # Terminal tool confirmed failure
    CONFIRMATION_RETRIES_EXCEEDED = (
        "confirmation_retries_exceeded"  # Too many confirmation attempts
    )


__all__ = [
    # Backend types (re-exported)
    "AgentResponse",
    "Message",
    "ToolCall",
    "ToolDef",
    # Config types (re-exported from config.py)
    "RunConfig",
    "Config",
    # Verb results
    "ChooseResult",
    "ClassifyResult",
    "Critique",
    "Evidence",
    "VerbResult",
    "VerifyResult",
    # Task types
    "LoopScore",
    "TaskResult",
    # Controller types
    "DecisionReason",
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
class ChooseResult:
    """Result from CHOOSE verb."""

    choice: str
    reason: str


@dataclass
class LoopScore:
    """Quality metrics for a Complete loop run.

    Two dimensions:
    - ``quality``: How well the LLM behaved (1.0 = no nudges/skips needed).
    - ``token_efficiency``: Fraction of tokens spent on productive work.
    """

    iterations: int  # Total iterations
    productive: int  # execute_tools + complete + fail + confirmation
    nudges: int  # instruct iterations (model needed correction)
    skips: int  # backoff/skip iterations
    total_tokens: int  # Total tokens across all iterations
    wasted_tokens: int  # Tokens spent on nudge/skip iterations

    @property
    def quality(self) -> float:
        """How well the LLM behaved (1.0 = no corrections needed)."""
        if self.iterations == 0:
            return 1.0
        return self.productive / self.iterations

    @property
    def token_efficiency(self) -> float:
        """Fraction of tokens spent on productive work (1.0 = no waste)."""
        if self.total_tokens == 0:
            return 1.0
        return 1.0 - (self.wasted_tokens / self.total_tokens)

    def __repr__(self) -> str:
        return f"quality={self.quality:.2f} token_eff={self.token_efficiency:.2f}"


@dataclass
class TaskResult:
    """Result from task execution."""

    completed: bool
    output: str
    iterations: int
    history: list[Message]
    terminal_data: dict[str, Any] | None = None
    terminal_tool: str | None = None
    trace_id: str = ""  # Constant across all LLM calls in one verb invocation
    request_id: str | None = None  # User-provided correlation ID
    score: LoopScore | None = None  # Loop quality metrics

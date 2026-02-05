"""Core data types for SAIA verb results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Re-export backend types for convenience
from llm_saia.core.backend import AgentResponse, Message, ToolCall, ToolDef

# Re-export config types for backwards compatibility
from llm_saia.core.config import Config, RunConfig

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
    "ConfirmResult",
    "Critique",
    "Evidence",
    "VerbResult",
    "VerifyResult",
    # Task types
    "TaskResult",
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

"""Core data types for SAIA verb results."""

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

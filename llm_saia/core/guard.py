"""Output guard for validating LLM responses with retry capability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["OutputGuard", "OutputGuardError"]


@dataclass(frozen=True)
class OutputGuard:
    """Validates output and retries with instruction on failure.

    Args:
        validator: Function returning None if valid, error string if invalid.
        retry_instruction: Instruction appended to prompt on retry.
        max_retries: Max retry attempts. Default 1.
        name: Optional name for logging/debugging.

    Example:
        >>> def is_short(text: str) -> str | None:
        ...     if len(text) > 100:
        ...         return f"Too long: {len(text)} chars"
        ...     return None
        >>> guard = OutputGuard(
        ...     validator=is_short,
        ...     retry_instruction="Keep response under 100 characters.",
        ...     name="length_check",
        ... )
    """

    validator: Callable[[Any], str | None]
    retry_instruction: str
    max_retries: int = 1
    name: str | None = None


class OutputGuardError(Exception):
    """Raised when output fails guard validation after all retries exhausted."""

    def __init__(self, guard_name: str | None, error: str, attempts: int) -> None:
        """Initialize OutputGuardError.

        Args:
            guard_name: Name of the guard that failed.
            error: The validation error message.
            attempts: Number of attempts made.
        """
        self.guard_name = guard_name
        self.error = error
        self.attempts = attempts
        name = guard_name or "guard"
        super().__init__(f"{name} failed after {attempts} attempts: {error}")

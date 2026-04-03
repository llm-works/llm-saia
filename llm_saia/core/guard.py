"""Output guard for validating LLM responses with retry capability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .errors import Error

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["Guarded", "OutputGuard", "OutputGuardError"]


@dataclass(frozen=True)
class OutputGuard:
    """Validates output and retries with instruction on failure.

    Args:
        validator: Function receiving the parsed result (Any type). Returns None
            if valid, error string if invalid. For text validation, use str(result).
        retry_instruction: Instruction appended to prompt on retry.
        max_retries: Max retry attempts (must be >= 0). Default 1.
        name: Optional name for logging/debugging.

    Example:
        >>> def is_short(result: Any) -> str | None:
        ...     text = str(result)
        ...     if len(text) > 100:
        ...         return f"Too long: {len(text)} chars"
        ...     return None
        >>> guard = OutputGuard(
        ...     validator=is_short,
        ...     retry_instruction="Keep response under 100 characters.",
        ...     name="length_check",
        ... )

    Raises:
        ValueError: If max_retries is negative.
    """

    validator: Callable[[Any], str | None]
    retry_instruction: str
    max_retries: int = 1
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate max_retries is non-negative."""
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")


class Guarded:
    """Marker for field-level guards in Annotated types.

    Use with typing.Annotated to specify guards for individual dataclass fields.
    These guards validate only the annotated field, not the entire result.

    Example:
        >>> from typing import Annotated
        >>> from dataclasses import dataclass
        >>> from llm_saia import Guarded
        >>> from llm_saia.guards import english_only, max_length
        >>>
        >>> @dataclass
        ... class Article:
        ...     title: Annotated[str, Guarded(english_only(), max_length(100))]
        ...     body: Annotated[str, Guarded(english_only())]
        ...     metadata: str  # Not guarded
    """

    __slots__ = ("guards",)

    def __init__(self, *guards: OutputGuard) -> None:
        """Initialize with one or more guards.

        Args:
            *guards: OutputGuard instances to apply to the annotated field.

        Raises:
            ValueError: If no guards are provided.
        """
        if not guards:
            raise ValueError("Guarded requires at least one guard")
        self.guards: tuple[OutputGuard, ...] = guards


class OutputGuardError(Error):
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

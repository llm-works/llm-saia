"""Output guard for validating LLM responses with retry capability."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .errors import Error

__all__ = [
    "Guarded",
    "IterationContext",
    "IterationGuard",
    "OutputGuard",
    "OutputGuardError",
    "UNLIMITED",
]


@dataclass(frozen=True)
class OutputGuard:
    """Validates output and retries with instruction on failure.

    Args:
        validator: Function receiving the parsed result (Any type). Returns None
            if valid, error string if invalid. For text validation, use str(result).
        retry_instruction: Static string or callable ``(attempt, result, error) -> str``
            appended to prompt on retry. Callables enable escalating retry tone.
        max_retries: Max retry attempts (must be >= 0). Default 1.
        name: Optional name for logging/debugging.

    Example:
        >>> guard = OutputGuard(
        ...     validator=lambda r: f"Too long: {len(str(r))}" if len(str(r)) > 100 else None,
        ...     retry_instruction="Keep response under 100 characters.",
        ...     name="length_check",
        ... )

    Raises:
        ValueError: If max_retries is negative.
    """

    validator: Callable[[Any], str | None]
    retry_instruction: str | Callable[[int, Any, str], str]
    max_retries: int = 1
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate max_retries is non-negative."""
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")

    def resolve_instruction(self, attempt: int, result: Any, error: str) -> str:
        """Resolve retry instruction for the given attempt.

        If ``retry_instruction`` is a string, returns it directly.
        If callable, calls it with ``(attempt, result, error)``.
        """
        if callable(self.retry_instruction):
            return self.retry_instruction(attempt, result, error)
        return self.retry_instruction


# Sentinel for unlimited iterations (avoids sys.maxsize import in hot path)
UNLIMITED = 2**63 - 1


@dataclass(frozen=True)
class IterationContext:
    """Context passed to iteration guard validators.

    Provides access to the current response and loop state, enabling guards
    that adapt behavior based on iteration progress (e.g., force terminal
    tool near the end of the loop).

    Attributes:
        response: The current LLM response to validate.
        iteration: Current iteration number (0-indexed).
        max_iterations: Maximum iterations configured for the loop.
    """

    response: Any  # AgentResponse, but Any to avoid circular import
    iteration: int
    max_iterations: int

    @property
    def remaining(self) -> int:
        """Iterations remaining (including current).

        Returns :const:`UNLIMITED` when ``max_iterations=0`` (unlimited).
        """
        if self.max_iterations == 0:
            return UNLIMITED
        return max(0, self.max_iterations - self.iteration)


@dataclass(frozen=True)
class IterationGuard:
    """Behavioral constraint enforced after each LLM response in a tool-calling loop.

    Unlike :class:`OutputGuard` (which validates the final result and retries the
    whole completion), an ``IterationGuard`` runs *during* the loop.  When its
    validator returns a feedback string the message is injected into the
    conversation and the loop continues — no retry, just a nudge.

    The validator receives an :class:`IterationContext` with the response and
    loop state, so it can inspect ``content``, ``tool_calls``, and adapt
    behavior based on iteration progress.

    Args:
        validator: Receives :class:`IterationContext`. Return ``None`` when
            the response is acceptable, or a feedback string to inject.
        name: Optional name for logging and trace records.

    Example:
        >>> guard = IterationGuard(
        ...     validator=lambda ctx: (
        ...         "Explain what you're doing and why."
        ...         if ctx.response.tool_calls and not (ctx.response.content or "").strip()
        ...         else None
        ...     ),
        ...     name="narrative",
        ... )

        >>> # Force terminal tool when iterations are running low
        >>> def force_terminal(ctx: IterationContext) -> str | None:
        ...     if ctx.remaining <= 3 and not calls_terminal(ctx.response):
        ...         return "You must call report_findings now."
        ...     return None
    """

    validator: Callable[[IterationContext], str | None]
    name: str | None = None


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

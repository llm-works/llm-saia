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

    This context is used in two scenarios:

    1. **Tool loop**: Guards run after each LLM response during tool calling.
       ``response`` contains the LLM response, ``parse_error`` is ``None``.

    2. **Parse retry**: Guards run when structured output parsing fails.
       ``response`` contains the response that failed to parse,
       ``parse_error`` contains the :class:`StructuredOutputError`.

    Guards can check ``parse_error`` to determine which scenario they're in.

    Attributes:
        response: The current LLM response to validate.
        iteration: Current iteration/attempt number (0-indexed).
        max_iterations: Maximum iterations/attempts configured.
        parse_error: If set, indicates this is a parse retry context.
    """

    response: Any  # ChatResponse, but Any to avoid circular import
    iteration: int
    max_iterations: int
    parse_error: Any = None  # StructuredOutputError, but Any to avoid circular import

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
    """Behavioral constraint enforced after each LLM response in a loop.

    Runs in two contexts:

    1. **Tool loop**: After each LLM response during tool calling. When the
       validator returns a feedback string, it's injected into the conversation
       and the loop continues.

    2. **Parse retry**: When structured output parsing fails (``ctx.parse_error``
       is set). Guards with ``parse_max_retries > 0`` participate in parse retry.
       Return feedback to retry, or ``None`` to stop retrying.

    The validator receives an :class:`IterationContext` with the response and
    loop state. Check ``ctx.parse_error`` to detect parse retry context.

    When multiple guards have ``parse_max_retries > 0``, their retry budgets are
    **summed** to determine total attempts. For example, two guards with
    ``parse_max_retries=2`` each allow up to 5 attempts (1 initial + 2 + 2).
    Each attempt evaluates all participating guards; their feedback is combined.

    Args:
        validator: Receives :class:`IterationContext`. Return ``None`` when
            the response is acceptable, or a feedback string to inject/retry.
        name: Optional name for logging and trace records.
        parse_max_retries: Retry budget for parse retry context. Guards with
            ``parse_max_retries > 0`` participate when structured output parsing
            fails. Default 0 (tool loop only, no parse retry).
        blocking: If ``True`` (default), tool calls are skipped when the guard
            fires - use for guards that reject bad tool calls (e.g., terminal
            tool with invalid status). If ``False``, tools execute first and
            feedback is injected afterward - use for advisory guards that want
            to shape behavior without blocking progress (e.g., narrative guards).

    Example:
        >>> # Tool loop guard - require explanation with tool calls (advisory)
        >>> guard = IterationGuard(
        ...     validator=lambda ctx: (
        ...         "Explain what you're doing."
        ...         if ctx.response.tool_calls and not ctx.parse_error
        ...         else None
        ...     ),
        ...     name="narrative",
        ...     blocking=False,  # Advisory: tools run, then feedback injected
        ... )

        >>> # Parse retry guard - retry on JSON errors
        >>> from llm_saia.guards import schema_retry
        >>> saia.with_guard(schema_retry(max_retries=2))
    """

    validator: Callable[[IterationContext], str | None]
    name: str | None = None
    parse_max_retries: int = 0  # >0 enables parse retry participation
    blocking: bool = True  # False = execute tools, then inject feedback (advisory mode)

    def __post_init__(self) -> None:
        """Validate parse_max_retries is non-negative."""
        if self.parse_max_retries < 0:
            raise ValueError(f"parse_max_retries must be >= 0, got {self.parse_max_retries}")


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

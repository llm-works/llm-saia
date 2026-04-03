"""Configurable interface for fluent per-call overrides."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from .config import CallOptions, Config
    from .guard import OutputGuard

__all__ = ["Configurable"]


class Configurable(ABC):
    """Interface for fluent per-call configuration overrides.

    Provides immutable `with_*()` methods that return new instances with
    modified configuration. All methods preserve shared state (like memory).

    Example:
        >>> result = await saia.with_temperature(1.0).verify(claim)
        >>> result = await saia.with_temperature(0.2).with_max_tokens(500).complete(task)
    """

    _config: Config
    _memory: dict[str, Any]

    @abstractmethod
    def _clone(self, config: Config) -> Self:
        """Create a new instance with the given config. Must preserve shared state."""
        ...

    def _with_config(self, **kwargs: Any) -> Self:
        """Return new instance with modified Config fields."""
        new_config = replace(self._config, **kwargs)
        return self._clone(new_config)

    def _with_call(self, **kwargs: Any) -> Self:
        """Return new instance with modified CallOptions fields."""
        from .config import DEFAULT_CALL

        base_call = self._config.call or DEFAULT_CALL
        new_call = replace(base_call, **kwargs)
        return self._with_config(call=new_call)

    # --- Call Options Overrides ---

    def with_call_options(self, call: CallOptions) -> Self:
        """Return new instance with different call options."""
        return self._with_config(call=call)

    def with_single_call(self) -> Self:
        """Return new instance for single LLM call (no looping)."""
        return self._with_call(max_iterations=1)

    def with_max_iterations(self, n: int) -> Self:
        """Return new instance with specified max iterations."""
        return self._with_call(max_iterations=n)

    def with_timeout(self, secs: float) -> Self:
        """Return new instance with specified timeout."""
        return self._with_call(timeout_secs=secs)

    def with_max_tokens(self, n: int) -> Self:
        """Return new instance with specified total token budget."""
        return self._with_call(max_total_tokens=n)

    def with_max_call_tokens(self, n: int) -> Self:
        """Return new instance with specified per-call token limit."""
        return self._with_call(max_call_tokens=n)

    def with_retries(self, max_retries: int, escalation: str | None = None) -> Self:
        """Return new instance with retry settings."""
        return self._with_call(max_retries=max_retries, retry_escalation=escalation)

    def with_temperature(self, temp: float | None) -> Self:
        """Return new instance with specified sampling temperature (None to clear)."""
        return self._with_call(temperature=temp)

    def with_request_id(self, request_id: str | None) -> Self:
        """Return new instance with a user-provided correlation ID (None to clear)."""
        return self._with_call(request_id=request_id)

    def with_system(self, system: str | None) -> Self:
        """Return new instance with different system prompt (None to clear)."""
        return self._with_call(system=system)

    def with_parse_retries(self, n: int) -> Self:
        """Return new instance with specified parse retry attempts.

        When structured output parsing fails (StructuredOutputError), SAIA will
        retry with feedback to the LLM about what went wrong. This is useful
        for local/smaller LLMs that may produce malformed JSON.

        Args:
            n: Number of retry attempts. 0 = no retry (default), 1 = one retry, etc.
        """
        return self._with_call(parse_retries=n)

    def with_guard(self, guard: OutputGuard) -> Self:
        """Add an output guard that validates output and retries if invalid.

        Guards are applied after completion. If validation fails, the request
        is retried with the guard's retry_instruction appended to the prompt.

        Guards work with both:
        - Text verbs (Ask, etc.) - validator receives the text string
        - Structured output verbs (Extract, etc.) - validator receives the parsed object

        Multiple guards can be chained and are applied in order. If a guard retry
        produces a different result, all guards are re-validated from the beginning.

        Example:
            >>> from llm_saia.guards import english_only, max_length
            >>> result = await (
            ...     saia
            ...     .with_guard(english_only())
            ...     .with_guard(max_length(500))
            ...     .ask(artifact, question)
            ... )

        Args:
            guard: OutputGuard with validator and retry instruction.
        """
        from .config import DEFAULT_CALL

        base_call = self._config.call or DEFAULT_CALL
        new_guards = base_call.output_guards + (guard,)
        return self._with_call(output_guards=new_guards)

    def with_guards(self, *guards: OutputGuard) -> Self:
        """Add multiple output guards at once.

        Convenience method equivalent to chaining multiple with_guard() calls.
        See with_guard() for details on guard behavior.

        Example:
            >>> from llm_saia.guards import english_only, max_length, no_preamble
            >>> result = await (
            ...     saia
            ...     .with_guards(english_only(), max_length(500), no_preamble())
            ...     .ask(artifact, question)
            ... )

        Args:
            *guards: OutputGuard instances to add.

        Raises:
            ValueError: If no guards are provided.
        """
        if not guards:
            raise ValueError("with_guards requires at least one guard")

        from .config import DEFAULT_CALL

        base_call = self._config.call or DEFAULT_CALL
        new_guards = base_call.output_guards + tuple(guards)
        return self._with_call(output_guards=new_guards)

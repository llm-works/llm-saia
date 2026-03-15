"""Configurable interface for fluent per-call overrides."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from llm_saia.core.config import CallOptions, Config

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
        new_call = replace(self._config.call, **kwargs)  # type: ignore[type-var]
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

    def with_temperature(self, temp: float) -> Self:
        """Return new instance with specified sampling temperature."""
        return self._with_call(temperature=temp)

    def with_request_id(self, request_id: str) -> Self:
        """Return new instance with a user-provided correlation ID."""
        return self._with_call(request_id=request_id)

    def with_system(self, system: str) -> Self:
        """Return new instance with different system prompt."""
        return self._with_call(system=system)

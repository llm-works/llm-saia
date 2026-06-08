"""Configurable interface for fluent per-call overrides."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .backend import ToolDef
    from .config import CallOptions, Config, JsonParser
    from .guard import IterationGuard, OutputGuard
    from .trace import Tracer

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

    # --- Config-Level Overrides ---

    def with_tracer(self, tracer: Tracer) -> Self:
        """Return new instance with specified tracer for iteration tracing."""
        return self._with_config(tracer=tracer)

    def with_tools(
        self,
        tools: list[ToolDef],
        executor: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
    ) -> Self:
        """Return new instance with different tool definitions.

        Useful for benchmarking scenarios where each test case defines its own
        function schemas (e.g., BFCL). The original instance is unmodified.

        Note:
            Like all ``with_*()`` methods, arguments are stored by reference
            (shallow ``dataclasses.replace``). Callers must not mutate the
            *tools* list after passing it.

        Args:
            tools: Tool definitions to use for this call.
            executor: Optional executor to replace the existing one. If None,
                keeps the current executor.
        """
        kwargs: dict[str, Any] = {"tools": tools}
        if executor is not None:
            kwargs["executor"] = executor
        return self._with_config(**kwargs)

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

    def with_temperature(self, temp: float | None) -> Self:
        """Return new instance with specified sampling temperature (None to clear)."""
        return self._with_call(temperature=temp)

    def with_request_id(self, request_id: str | None) -> Self:
        """Return new instance with a user-provided correlation ID (None to clear)."""
        return self._with_call(request_id=request_id)

    def with_context(self, context: dict[str, Any] | None) -> Self:
        """Return new instance with *context* merged onto the existing context.

        Merge rules (see :func:`llm_saia.core.context.merge_context`):

        - Two dict-valued nodes at the same key recurse to arbitrary depth;
          everything else replaces wholesale at its leaf. Last write wins.
        - Dict structure is independent of the parent: derivations from the
          same parent don't trample each other's namespaces.
        - Leaf values are shared by reference, so objects threaded through
          context (cost trackers, loggers, locks, clients) retain identity
          across derivations and backend callbacks. Mutating a mutable leaf
          (e.g. a list) after passing it in is visible on the merged result.

        Special inputs:

        - ``None`` clears the context entirely.
        - ``{}`` is a no-op and returns ``self`` unchanged.

        Args:
            context: Context to merge onto the existing context, or ``None``
                to clear.
        """
        if context is None:
            return self._with_call(context=None)
        if not context:
            return self
        from .config import DEFAULT_CALL
        from .context import merge_context

        base_call = self._config.call or DEFAULT_CALL
        merged = merge_context(base_call.context or {}, context)
        return self._with_call(context=merged)

    def with_system(self, system: str | None) -> Self:
        """Return new instance with different system prompt (None to clear)."""
        return self._with_call(system=system)

    def with_guard(self, guard: OutputGuard | IterationGuard) -> Self:
        """Add a guard to this instance.

        Accepts both guard types and routes them to the correct bucket:

        - :class:`OutputGuard` — validates the final result and retries if
          invalid (applied after completion).
        - :class:`IterationGuard` — enforces behavioral constraints after each
          LLM response in a loop. Runs during tool loops and parse retry loops.
          On failure the feedback string is injected and the loop continues.

        Multiple guards can be chained and are applied in order.

        Example:
            >>> from llm_saia.guards import english_only, schema_retry
            >>> # OutputGuard (english_only) applies to any completion
            >>> result = await saia.with_guard(english_only()).complete(task)
            >>> # IterationGuard with parse_max_retries applies to structured output
            >>> result = await saia.with_guard(schema_retry()).extract(Article, text)

        Args:
            guard: OutputGuard or IterationGuard instance.
        """
        from .config import DEFAULT_CALL
        from .guard import IterationGuard as _IG

        base_call = self._config.call or DEFAULT_CALL
        if isinstance(guard, _IG):
            new_iter_guards = base_call.iteration_guards + (guard,)
            return self._with_call(iteration_guards=new_iter_guards)
        new_guards = base_call.output_guards + (guard,)
        return self._with_call(output_guards=new_guards)

    def with_guards(self, *guards: OutputGuard | IterationGuard) -> Self:
        """Add multiple guards at once.

        Convenience method equivalent to chaining multiple with_guard() calls.
        Accepts a mix of :class:`OutputGuard` and :class:`IterationGuard`
        instances — each is routed to the correct bucket.

        See :meth:`with_guard` for details on guard behavior.

        Args:
            *guards: Guard instances to add.

        Raises:
            ValueError: If no guards are provided.
        """
        if not guards:
            raise ValueError("with_guards requires at least one guard")

        from .config import DEFAULT_CALL
        from .guard import IterationGuard as _IG

        base_call = self._config.call or DEFAULT_CALL
        new_output: tuple[OutputGuard, ...] = base_call.output_guards
        new_iter: tuple[IterationGuard, ...] = base_call.iteration_guards
        for g in guards:
            if isinstance(g, _IG):
                new_iter = new_iter + (g,)
            else:
                new_output = new_output + (g,)
        return self._with_call(output_guards=new_output, iteration_guards=new_iter)

    def with_json_parser(self, parser: JsonParser) -> Self:
        """Return new instance with custom JSON parser for structured output.

        Default is json.loads. Override to handle malformed JSON from some
        backends or to use alternative parsers (orjson, json-repair, etc.).

        Args:
            parser: Function that takes JSON string and returns parsed value.
        """
        return self._with_config(json_parser=parser)

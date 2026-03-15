"""Fluent builder for SAIA instances."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from llm_saia.core import trace
from llm_saia.core.backend import Backend, ToolDef
from llm_saia.core.config import CallOptions, Config, TerminalConfig
from llm_saia.core.logger import Logger

if TYPE_CHECKING:
    from llm_saia.saia import SAIA


class SAIABuilder:
    """Fluent builder for SAIA instances.

    Example:
        >>> saia = (SAIA.builder()
        ...     .backend(backend)
        ...     .tools(tools, executor)
        ...     .system("You are helpful")
        ...     .max_iterations(10)
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialize builder with defaults."""
        self._backend: Backend | None = None
        self._tools: list[ToolDef] = []
        self._executor: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None
        self._terminal: TerminalConfig | None = None
        self._lg: Logger | None = None
        self._warn_tool_support: bool = True
        self._tracer: trace.Tracer | None = None
        # CallOptions fields (defaults match CallOptions)
        self._system: str | None = None
        self._temperature: float | None = None
        self._max_iterations: int = 3
        self._max_call_tokens: int = 0
        self._max_total_tokens: int = 0
        self._timeout_secs: float = 0
        self._max_retries: int = 1
        self._retry_escalation: str | None = None
        self._request_id: str | None = None

    def backend(self, backend: Backend) -> SAIABuilder:
        """Set the LLM backend (required)."""
        self._backend = backend
        return self

    def tools(
        self,
        tools: list[ToolDef],
        executor: Callable[[str, dict[str, Any]], Awaitable[Any]],
    ) -> SAIABuilder:
        """Set tools and their executor."""
        self._tools = tools
        self._executor = executor
        return self

    def system(self, prompt: str) -> SAIABuilder:
        """Set system prompt."""
        self._system = prompt
        return self

    def terminal(
        self,
        tool: str,
        output_field: str | None = None,
        status_field: str | None = None,
        failure_values: tuple[str, ...] | None = None,
    ) -> SAIABuilder:
        """Set terminal tool configuration for task completion.

        Args:
            tool: Name of the terminal tool (e.g., "complete_task")
            output_field: Field in tool args containing output (default: check common names)
            status_field: Field in tool args containing status (default: "status")
            failure_values: Status values that indicate failure (default: stuck/failed/error)
        """
        self._terminal = TerminalConfig(
            tool=tool,
            output_field=output_field,
            status_field=status_field,
            failure_values=failure_values or ("stuck", "failed", "error"),
        )
        return self

    def terminal_tool(self, name: str) -> SAIABuilder:
        """Set terminal tool name for task completion (simple form).

        For more control, use .terminal() instead.
        """
        self._terminal = TerminalConfig(tool=name)
        return self

    def logger(self, lg: Logger) -> SAIABuilder:
        """Set logger for instrumentation."""
        self._lg = lg
        return self

    @property
    def tracing(self) -> trace.Builder[SAIABuilder]:
        """Configure iteration tracing destination."""
        return trace.Builder(self, self._set_tracer)

    def _set_tracer(self, tracer: trace.Tracer) -> None:
        """Callback for TracerBuilder to set the tracer."""
        self._tracer = tracer

    def warn_tool_support(self, enabled: bool = True) -> SAIABuilder:
        """Enable/disable tool support warnings."""
        self._warn_tool_support = enabled
        return self

    def max_iterations(self, n: int) -> SAIABuilder:
        """Set max tool-calling iterations (0 = unlimited)."""
        self._max_iterations = n
        return self

    def max_call_tokens(self, n: int) -> SAIABuilder:
        """Set max tokens per LLM call."""
        self._max_call_tokens = n
        return self

    def max_tokens(self, n: int) -> SAIABuilder:
        """Set total token budget across loop."""
        self._max_total_tokens = n
        return self

    def timeout(self, secs: float) -> SAIABuilder:
        """Set soft timeout in seconds."""
        self._timeout_secs = secs
        return self

    def retries(self, max_retries: int, escalation: str | None = None) -> SAIABuilder:
        """Set retry configuration."""
        self._max_retries = max_retries
        self._retry_escalation = escalation
        return self

    def request_id(self, request_id: str) -> SAIABuilder:
        """Set user-provided correlation ID."""
        self._request_id = request_id
        return self

    def temperature(self, temp: float) -> SAIABuilder:
        """Set sampling temperature."""
        self._temperature = temp
        return self

    def build(self) -> SAIA:
        """Build the SAIA instance.

        Raises:
            ValueError: If backend is not set.
        """
        if self._backend is None:
            raise ValueError("backend is required - call .backend() before .build()")

        from llm_saia.saia import SAIA

        call = CallOptions(
            system=self._system,
            temperature=self._temperature,
            max_call_tokens=self._max_call_tokens,
            max_total_tokens=self._max_total_tokens,
            timeout_secs=self._timeout_secs,
            max_iterations=self._max_iterations,
            max_retries=self._max_retries,
            retry_escalation=self._retry_escalation,
            request_id=self._request_id,
        )
        config = Config(
            backend=self._backend,
            tools=self._tools,
            executor=self._executor,
            call=call,
            terminal=self._terminal,
            lg=self._lg,
            tracer=self._tracer,
            warn_tool_support=self._warn_tool_support,
        )
        return SAIA(config)

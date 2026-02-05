"""Fluent builder for SAIA instances."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from llm_saia.core.backend import Backend, ToolDef
from llm_saia.core.config import Config, RunConfig
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
        self._system: str | None = None
        self._terminal_tool: str | None = None
        self._lg: Logger | None = None
        self._warn_tool_support: bool = True
        # RunConfig fields (defaults match RunConfig)
        self._max_iterations: int = 3
        self._max_call_tokens: int = 0
        self._max_total_tokens: int = 0
        self._timeout_secs: float = 0
        self._max_retries: int = 1
        self._retry_escalation: str | None = None

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

    def terminal_tool(self, name: str) -> SAIABuilder:
        """Set terminal tool name for task completion."""
        self._terminal_tool = name
        return self

    def logger(self, lg: Logger) -> SAIABuilder:
        """Set logger for instrumentation."""
        self._lg = lg
        return self

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

    def build(self) -> SAIA:
        """Build the SAIA instance.

        Raises:
            ValueError: If backend is not set.
        """
        if self._backend is None:
            raise ValueError("backend is required - call .backend() before .build()")

        from llm_saia.saia import SAIA

        run = RunConfig(
            max_iterations=self._max_iterations,
            max_call_tokens=self._max_call_tokens,
            max_total_tokens=self._max_total_tokens,
            timeout_secs=self._timeout_secs,
            max_retries=self._max_retries,
            retry_escalation=self._retry_escalation,
        )
        config = Config(
            backend=self._backend,
            tools=self._tools,
            executor=self._executor,
            system=self._system,
            run=run,
            terminal_tool=self._terminal_tool,
            lg=self._lg,
            warn_tool_support=self._warn_tool_support,
        )
        return SAIA(config)

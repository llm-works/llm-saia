"""SAIA class - the main interface for the framework."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from llm_saia.core.protocols import SAIABackend
from llm_saia.core.types import RunConfig, ToolDef
from llm_saia.verbs import (
    Ask,
    Choose,
    Classify,
    Complete,
    Confirm,
    Constrain,
    Critique_,
    Decompose,
    Extract,
    Ground,
    Instruct,
    Refine,
    Synthesize,
    VerbConfig,
    Verify,
)

# Default run config for SAIA
DEFAULT_RUN = RunConfig(max_iterations=3)


class SAIA:
    """Framework-agnostic verb vocabulary for LLM agents.

    Example:
        >>> saia = SAIA(backend=AnthropicBackend(), tools=tools, executor=exec)
        >>> result = await saia.verify(code, "compiles")
        >>> result = await saia.with_single_call().confirm(claim)
        >>> result = await saia.with_max_iterations(10).complete(task)
    """

    def __init__(
        self,
        backend: SAIABackend,
        tools: list[ToolDef] | None = None,
        executor: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
        system: str | None = None,
        run: RunConfig | None = None,
        terminal_tool: str | None = None,
        *,
        _memory: dict[str, Any] | None = None,
    ):
        """Initialize SAIA with a backend and optional tool configuration."""
        self._backend = backend
        self._tools = tools or []
        self._executor = executor
        self._system = system
        self._run = run or DEFAULT_RUN
        self._terminal_tool = terminal_tool
        self._memory = _memory if _memory is not None else {}
        self._init_verbs()

    def _init_verbs(self) -> None:
        """Initialize verb instances with current config."""
        config = VerbConfig(
            backend=self._backend,
            tools=self._tools,
            executor=self._executor,
            system=self._system,
            run=self._run,
            terminal_tool=self._terminal_tool,
        )
        self.ask = Ask(config)
        self.choose = Choose(config)
        self.classify = Classify(config)
        self.complete = Complete(config)
        self.confirm = Confirm(config)
        self.constrain = Constrain(config)
        self.critique = Critique_(config)
        self.decompose = Decompose(config)
        self.ground = Ground(config)
        self.instruct = Instruct(config)
        self.extract = Extract(config)
        self.refine = Refine(config)
        self.synthesize = Synthesize(config)
        self.verify = Verify(config)

    @property
    def run_config(self) -> RunConfig:
        """Current run configuration."""
        return self._run

    def _with_modified_run(self, **kwargs: Any) -> SAIA:
        """Return new SAIA with modified run config. Shares memory."""
        return SAIA(
            backend=self._backend,
            tools=self._tools,
            executor=self._executor,
            system=self._system,
            run=replace(self._run, **kwargs),
            terminal_tool=self._terminal_tool,
            _memory=self._memory,
        )

    def with_run_config(self, run: RunConfig) -> SAIA:
        """Return new SAIA with different run config. Shares memory."""
        return SAIA(
            backend=self._backend,
            tools=self._tools,
            executor=self._executor,
            system=self._system,
            run=run,
            terminal_tool=self._terminal_tool,
            _memory=self._memory,
        )

    def with_single_call(self) -> SAIA:
        """Return new SAIA for single LLM call (no looping). Shares memory."""
        return self._with_modified_run(max_iterations=1)

    def with_max_iterations(self, n: int) -> SAIA:
        """Return new SAIA with specified max iterations. Shares memory."""
        return self._with_modified_run(max_iterations=n)

    def with_timeout_secs(self, secs: float) -> SAIA:
        """Return new SAIA with specified timeout. Shares memory."""
        return self._with_modified_run(timeout_secs=secs)

    def with_max_tokens(self, n: int) -> SAIA:
        """Return new SAIA with specified total token budget. Shares memory."""
        return self._with_modified_run(max_total_tokens=n)

    def with_max_call_tokens(self, n: int) -> SAIA:
        """Return new SAIA with specified per-call token limit. Shares memory."""
        return self._with_modified_run(max_call_tokens=n)

    def with_retries(self, max_retries: int, escalation: str | None = None) -> SAIA:
        """Return new SAIA with retry settings. Shares memory."""
        return self._with_modified_run(max_retries=max_retries, retry_escalation=escalation)

    # --- Memory Verbs ---

    def recall(self, query: str) -> list[Any]:
        """RECALL: Retrieve values from memory matching query."""
        return [v for k, v in self._memory.items() if query.lower() in k.lower()]

    def store(self, key: str, value: Any) -> None:
        """STORE: Save a value to memory."""
        self._memory[key] = value

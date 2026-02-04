"""SAIA class - the main interface for the framework."""

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from llm_saia.core.protocols import SAIABackend
from llm_saia.core.types import LoopConfig, ToolDef
from llm_saia.verbs import (
    Ask,
    Choose,
    Classify,
    Complete,
    Confirm,
    Constrain,
    Critique_,
    Decompose,
    Ground,
    Instruct,
    Parse,
    Refine,
    Synthesize,
    VerbConfig,
    Verify,
)

# Default loop config for SAIA
DEFAULT_LOOP = LoopConfig(max_iterations=3, timeout_secs=0)


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
        loop: LoopConfig | None = None,
        *,
        _memory: dict[str, Any] | None = None,
    ):
        """Initialize SAIA with a backend and optional tool configuration."""
        self._backend = backend
        self._tools = tools or []
        self._executor = executor
        self._system = system
        self._loop = loop or DEFAULT_LOOP
        self._memory = _memory if _memory is not None else {}
        self._init_verbs()

    def _init_verbs(self) -> None:
        """Initialize verb instances with current config."""
        config = VerbConfig(
            backend=self._backend,
            tools=self._tools,
            executor=self._executor,
            system=self._system,
            loop=self._loop,
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
        self.parse = Parse(config)
        self.refine = Refine(config)
        self.synthesize = Synthesize(config)
        self.verify = Verify(config)

    @property
    def loop_config(self) -> LoopConfig:
        """Current loop configuration."""
        return self._loop

    def _with_modified_loop(self, **kwargs: Any) -> "SAIA":
        """Return new SAIA with modified loop config. Shares memory."""
        return SAIA(
            backend=self._backend,
            tools=self._tools,
            executor=self._executor,
            system=self._system,
            loop=replace(self._loop, **kwargs),
            _memory=self._memory,
        )

    def with_loop(self, loop: LoopConfig) -> "SAIA":
        """Return new SAIA with different loop config. Shares memory."""
        return SAIA(
            backend=self._backend,
            tools=self._tools,
            executor=self._executor,
            system=self._system,
            loop=loop,
            _memory=self._memory,
        )

    def with_single_call(self) -> "SAIA":
        """Return new SAIA for single LLM call (no looping). Shares memory."""
        return self._with_modified_loop(max_iterations=1)

    def with_max_iterations(self, n: int) -> "SAIA":
        """Return new SAIA with specified max iterations. Shares memory."""
        return self._with_modified_loop(max_iterations=n)

    def with_timeout_secs(self, secs: float) -> "SAIA":
        """Return new SAIA with specified timeout. Shares memory."""
        return self._with_modified_loop(timeout_secs=secs)

    def with_max_tokens(self, n: int) -> "SAIA":
        """Return new SAIA with specified total token budget. Shares memory."""
        return self._with_modified_loop(max_total_tokens=n)

    # --- Memory Verbs ---

    def recall(self, query: str) -> list[Any]:
        """RECALL: Retrieve values from memory matching query."""
        return [v for k, v in self._memory.items() if query.lower() in k.lower()]

    def store(self, key: str, value: Any) -> None:
        """STORE: Save a value to memory."""
        self._memory[key] = value

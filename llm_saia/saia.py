# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""SAIA class - the main interface for the framework."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Self, TypeVar

from .core.config import DEFAULT_CALL, CallOptions, Config
from .core.configurable import Configurable
from .core.types import VerbResult

if TYPE_CHECKING:
    from .builder import SAIABuilder
    from .core.conversation import ConversationLike
from .verbs import (
    Ask,
    Choose,
    Classify,
    Complete,
    Constrain,
    Critique_,
    Decompose,
    Extract,
    Find,
    Ground,
    Instruct,
    Refine,
    Synthesize,
    Verify,
)

T = TypeVar("T")


class SAIA(Configurable):
    """Framework-agnostic verb vocabulary for LLM agents.

    Example:
        >>> saia = (SAIA.builder()
        ...     .backend(backend)
        ...     .tools(tools, executor)
        ...     .max_iterations(10)
        ...     .build())
        >>> result = await saia.verify(code, "compiles")
        >>> result = await saia.with_single_call().verify(claim)
        >>> result = await saia.with_temperature(1.0).verify(claim)
    """

    @classmethod
    def builder(cls) -> SAIABuilder:
        """Create a fluent builder for SAIA."""
        from .builder import SAIABuilder

        return SAIABuilder()

    def __init__(self, config: Config, *, _memory: dict[str, Any] | None = None):
        """Initialize SAIA with configuration."""
        # Apply default call options if not specified
        if config.call is None:
            config = replace(config, call=DEFAULT_CALL)
        self._config = config
        self._memory = _memory if _memory is not None else {}
        self._init_verbs()

    def _clone(self, config: Config) -> Self:
        """Create a new instance with modified config. Preserves memory."""
        return self.__class__(config, _memory=self._memory)

    def _init_verbs(self) -> None:
        """Initialize verb instances with current config."""
        self.ask = Ask(self._config)
        self.choose = Choose(self._config)
        self.classify = Classify(self._config)
        self.complete = Complete(self._config)
        self.constrain = Constrain(self._config)
        self.critique = Critique_(self._config)
        self.decompose = Decompose(self._config)
        self.extract = Extract(self._config)
        self.find = Find(self._config)
        self.ground = Ground(self._config)
        self.instruct = Instruct(self._config)
        self.refine = Refine(self._config)
        self.synthesize = Synthesize(self._config)
        self.verify = Verify(self._config)

    @property
    def config(self) -> Config:
        """Current configuration."""
        return self._config

    @property
    def call_options(self) -> CallOptions:
        """Current call options."""
        return self._config.call  # type: ignore[return-value]

    # --- Structured Completion ---

    async def complete_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        conversation: ConversationLike | None = None,
    ) -> VerbResult[T]:
        """Send ``prompt`` verbatim and parse the response against ``schema``.

        Where :meth:`complete` returns raw text (and can drive a tool loop),
        ``complete_structured`` returns a typed value: it sends the prompt
        with no framing added, requests JSON output constrained by
        ``schema``, and returns the parsed instance.

        Use this when the caller has already composed the full prompt
        (graders, LLM judges, drift detectors, synthesis critics) and no
        domain verb fits — every domain verb (``extract``, ``verify``,
        ``classify``, ...) prepends its own framing.

        Args:
            prompt: Prompt sent verbatim to the backend.
            schema: The response shape. Either a stdlib ``@dataclass`` (the
                default; zero runtime dependency) or a
                :class:`pydantic.BaseModel` subclass. BaseModel support
                requires ``pip install 'llm-saia[pydantic]'`` and unlocks the
                full JSON-Schema vocabulary — ``Field(ge=..., le=...,
                pattern=..., max_length=..., discriminator=..., ...)`` —
                whose structural constraints are enforced by the backend's
                constrained decoder, and whose ``@field_validator`` /
                ``@model_validator`` checks run on parse and can trigger
                the same schema-retry loop as dataclass parse failures.
            conversation: Optional conversation to append messages to.

        Returns:
            A :class:`VerbResult` holding the parsed value and the emitted
            :class:`VerbTrace`.

        Guard semantics:
            ``with_guard(...)`` chaining applies here just as it does on the
            domain verbs.

            - Iteration guards' ``parse_max_retries`` budgets sum. A
              :class:`StructuredOutputError` (including its
              :class:`TruncatedResponseError` subclass) consumes budget until
              retries are exhausted or parsing succeeds.
            - ``output_guards`` fire after a successful parse and may force
              additional retries.
        """
        from .verbs.prompt import _PromptVerb

        return await _PromptVerb(self._config)(prompt, schema, conversation=conversation)

    # --- Memory Verbs ---

    def recall(self, query: str) -> list[Any]:
        """RECALL: Retrieve values from memory matching query."""
        return [v for k, v in self._memory.items() if query.lower() in k.lower()]

    def store(self, key: str, value: Any) -> None:
        """STORE: Save a value to memory."""
        self._memory[key] = value

    # --- Prompt Utilities ---

    def compose(self, *layers: str | None, separator: str = "\n\n") -> str:
        """COMPOSE: Build structured prompts from layers.

        Automatically filters out None and empty strings to simplify prompt
        composition when some layers may be optional.

        Args:
            *layers: Prompt layers to combine (None/empty are filtered out)
            separator: String to join layers with (default: double newline)

        Returns:
            Combined prompt string (may be empty if all layers filtered out)

        Example:
            >>> # Simple composition
            >>> prompt = saia.compose(
            ...     "You are a comedian",
            ...     "Past jokes: ...",
            ...     "Tell me a joke about cats"
            ... )
            >>> # -> "You are a comedian\\n\\nPast jokes: ...\\n\\nTell me a joke about cats"
            >>>
            >>> # With None/empty filtering (common pattern in agents)
            >>> context = None  # No context available
            >>> prompt = saia.compose("You are helpful", context, "Help me code")
            >>> # -> "You are helpful\\n\\nHelp me code"
            >>>
            >>> # Custom separator
            >>> prompt = saia.compose("Step 1", "Step 2", separator=" -> ")
            >>> # -> "Step 1 -> Step 2"
        """
        filtered = [layer for layer in layers if layer]
        return separator.join(filtered)

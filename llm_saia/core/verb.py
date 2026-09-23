# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Base class for SAIA verbs."""

from __future__ import annotations

import asyncio
import inspect
import time
from abc import abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Self, TypeVar

from .backend import ChatResponse
from .config import DEFAULT_CALL, CallOptions, Config
from .configurable import Configurable
from .conversation import (
    AsyncConversationLike,
    ConversationLike,
    ListConversation,
    Message,
    MessageAppendable,
    Role,
    ToolCall,
)
from .errors import StructuredOutputError, TruncatedResponseError
from .guard import IterationContext, IterationGuard
from .guard_eval import _GuardEvaluator
from .guards import OutputGuardMixin
from .loop import CoreLoopResult, LoopDecision, LoopStrategy, SimpleStrategy
from .loop_runner import _LoopRunner
from .schema_strategy import SchemaTerminatingStrategy
from .tool_executor import _ToolExecutor
from .tool_gate import apply_tool_gates

if TYPE_CHECKING:
    from .backend import Backend
    from .trace import GuardOutcome, Tracer, VerbTrace

T = TypeVar("T")

_SENTINEL: Any = object()  # default marker for _chat(tools=...)


class Verb(OutputGuardMixin, Configurable):
    """Base class for all verbs. Subclass this to create custom verbs."""

    # Truncation limit for log previews (debug level)
    _PREVIEW_LIMIT = 100
    # Truncation limit for trace logs (high ceiling to prevent pathological cases)
    _TRACE_LIMIT = 50_000
    # Patterns indicating truncated JSON response
    _TRUNCATION_INDICATORS: tuple[str, ...] = (
        "Unterminated string",
        "Unexpected end of JSON",
        "Expecting value",
        "Expecting ',' delimiter",
        "Expecting ':' delimiter",
    )

    def __init__(self, config: Config):
        """Initialize verb with configuration."""
        self._config = config
        self._memory: dict[str, Any] = {}  # Verbs don't use memory
        self._lg = config.lg

    def _clone(self, config: Config) -> Self:
        """Create a new instance with the given config."""
        return self.__class__(config)

    @property
    def _backend(self) -> Backend:
        """Get the configured backend."""
        return self._config.backend

    def _has_tools(self) -> bool:
        """Check if tools are configured."""
        return bool(self._config.tools and self._config.executor)

    @property
    def _call(self) -> CallOptions:
        """Get effective call options (instance default or global default)."""
        return self._config.call or DEFAULT_CALL

    def _get_call_options(self, override: CallOptions | None = None) -> CallOptions:
        """Get effective call options: override > instance default > global default."""
        return override or self._call

    def _structured_output_error(
        self, error: Exception, content: str, schema_name: str
    ) -> StructuredOutputError:
        """Create appropriate error for structured output parse failure."""
        error_msg = str(error)
        # Only apply truncation heuristic when parser provides a valid int position
        pos = getattr(error, "pos", None)
        if not isinstance(pos, int) or pos < 0 or pos > len(content):
            pos = None
        is_truncated = pos is not None and any(
            ind in error_msg for ind in self._TRUNCATION_INDICATORS
        )
        if is_truncated and content[pos:].strip():
            is_truncated = False

        if is_truncated:
            return TruncatedResponseError(
                raw_content=content,
                schema_name=schema_name,
                parse_error=error_msg,
            )
        return StructuredOutputError(
            f"LLM returned invalid JSON for {schema_name}: {error_msg}",
            raw_content=content,
            schema_name=schema_name,
            parse_error=error_msg,
        )

    @staticmethod
    def _generate_id() -> str:
        """Generate a short unique ID for tracing (8-char hex)."""
        from .trace import _generate_id

        return _generate_id()

    def _resolve_tracer(self, metadata: dict[str, Any]) -> Tracer | None:
        """Get config tracer and call start() if present."""
        tracer = self._config.tracer
        if tracer:
            tracer.start(metadata)
        return tracer

    def _init_verb_trace(self, trace_id: str = "") -> VerbTrace:
        """Create a new VerbTrace for this verb call."""
        from .trace import VerbTrace

        trace = VerbTrace(
            verb=self.__class__.__name__,
            trace_id=trace_id or self._generate_id(),
            ts=time.time(),
            request_id=self._call.request_id,
        )
        trace._mono_start = time.monotonic()  # type: ignore[attr-defined]
        self._lg.trace("verb started", extra={"verb": trace.verb, "trace_id": trace.trace_id})
        return trace

    def _record_step(
        self,
        response: ChatResponse,
        *,
        phase: str,
        _trace: VerbTrace | None = None,
    ) -> None:
        """Build a Step from response, append to trace, write to tracer."""
        from .trace import build_step_from_response

        step = build_step_from_response(
            response,
            phase=phase,
            trace_id=_trace.trace_id if _trace else "",
            verb=self.__class__.__name__,
        )
        if _trace is not None:
            _trace.add_step(step)
        tracer = self._config.tracer
        if tracer:
            tracer.write(step)

    def _emit_verb_trace(self, trace: VerbTrace, reason: str | None = None) -> None:
        """Finalize timing and write the full VerbTrace to the configured tracer."""
        mono_start = getattr(trace, "_mono_start", 0.0)
        trace.duration_ms = int((time.monotonic() - mono_start) * 1000) if mono_start else 0
        extra: dict[str, Any] = {
            "verb": trace.verb,
            "trace_id": trace.trace_id,
            "duration_ms": trace.duration_ms,
            "steps": len(trace.steps),
        }
        if reason:
            extra["reason"] = reason
        self._lg.trace("verb completed", extra=extra)
        tracer = self._config.tracer
        if tracer:
            tracer.write(trace)

    @staticmethod
    def _max_tokens(config: CallOptions) -> int | None:
        """Resolve max_call_tokens to None (no limit) or a positive int."""
        return config.max_call_tokens if config.max_call_tokens > 0 else None

    def _resolve_temperature(self, override: CallOptions | None) -> float | None:
        """Resolve temperature: override CallOptions > instance CallOptions."""
        if override is not None and override.temperature is not None:
            return override.temperature
        return self._call.temperature

    async def _chat(
        self,
        messages: list[Message],
        max_tokens: int | None,
        temperature: float | None = None,
        *,
        call: CallOptions | None = None,
        response_schema: dict[str, Any] | None = None,
        tools: list[Any] | None = _SENTINEL,
        abort_signal: asyncio.Event | None = None,
        iteration: int | None = None,
        last_response: ChatResponse | None = None,
    ) -> ChatResponse:
        """Execute a single chat call.

        Args:
            messages: Conversation messages.
            max_tokens: Token limit (None = unlimited).
            temperature: Sampling temperature.
            call: Per-invocation CallOptions override.
            response_schema: JSON schema for structured output.
            tools: Tool definitions. Default (sentinel) uses config tools;
                pass ``None`` or ``[]`` to suppress tools.
            abort_signal: Event that signals abort request. Backend may use
                streaming to enable fast abort between chunks.
            iteration: 0-indexed loop iteration. When provided and
                :attr:`Config.tool_gates` (or the
                :attr:`TerminalConfig.min_iterations` shortcut) are
                configured, gates are evaluated and blocked tools are
                stripped from the outbound schema for this call.
            last_response: Most recent :class:`ChatResponse` — exposed to
                gate callbacks via :attr:`ToolGateContext.last_response`.
        """
        call_id = self._generate_id()
        self._log_message_assembly(call_id, messages)
        resolved_tools = (
            (self._config.tools if self._config.tools else None)
            if tools is _SENTINEL
            else (tools or None)
        )
        if iteration is not None:
            resolved_tools = self._gate_tools(
                iteration, messages, last_response, resolved_tools, call_id
            )
        call_opts = self._get_call_options(call)
        t0 = time.monotonic()
        response = await self._backend.chat(
            messages,
            system=call_opts.system,
            tools=resolved_tools,
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=response_schema,
            context=call_opts.context,
            abort_signal=abort_signal,
        )
        response.call_id = call_id
        response._duration_ms = int((time.monotonic() - t0) * 1000)  # type: ignore[attr-defined]
        return response

    def _gate_tools(
        self,
        iteration: int,
        messages: list[Message],
        last_response: ChatResponse | None,
        resolved_tools: list[Any] | None,
        call_id: str,
    ) -> list[Any] | None:
        """Apply tool-visibility gates. Returns filtered tool list."""
        filtered, blocked = apply_tool_gates(
            self._config, iteration, messages, last_response, resolved_tools
        )
        if blocked:
            self._lg.debug(
                "tool-gate filtered tools",
                extra={
                    "call_id": call_id,
                    "iteration": iteration,
                    "blocked": blocked,
                },
            )
        return filtered if filtered else None

    async def _init_loop(
        self,
        prompt: str,
        run: CallOptions | None,
        conversation: ConversationLike | None,
        resume: bool = False,
    ) -> tuple[CallOptions, ConversationLike, int | None, float | None]:
        """Initialize loop state and return (config, conversation, max_tokens, temperature).

        Args:
            prompt: Task prompt (ignored when resuming).
            run: Call options override.
            conversation: External conversation (required when resuming).
            resume: If True, skip adding initial user message and continue from
                existing conversation state.

        Raises:
            ValueError: If resume=True but conversation is None.
        """
        if resume and conversation is None:
            raise ValueError("conversation is required when resume=True")
        config = self._get_call_options(run)
        conv = conversation if conversation is not None else ListConversation()
        if not resume:
            await self._append_msg(conv, Message(role=Role.USER, content=prompt))
        return config, conv, self._max_tokens(config), self._resolve_temperature(run)

    async def _loop(
        self,
        prompt: str,
        run: CallOptions | None = None,
        schema: type[T] | None = None,
        trace_id: str = "",
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None = None,
        resume: bool = False,
        abort_signal: asyncio.Event | None = None,
    ) -> tuple[str, T | None]:
        """Execute prompt with tool-calling loop.

        Args:
            prompt: Task prompt (ignored when resuming).
            run: Call options override.
            schema: Optional schema for structured finalization.
            trace_id: Trace correlation ID.
            conversation: External conversation for message management.
            _trace: Parent verb trace.
            on_iteration: Optional callback invoked each iteration. May raise
                ``PauseRequested`` to exit the loop early.
            resume: If True, continue from existing conversation state.
            abort_signal: Optional event for fast abort during LLM streaming.
                When set, backends that support streaming can abort within ~100ms.

        Returns:
            Tuple of (content, structured_result).

        Raises:
            PauseRequested: If on_iteration callback requests pause or abort_signal
                is set during an LLM call. The conversation is in a consistent
                state for later resumption.
        """
        config, conv, _max_tokens, temperature = await self._init_loop(
            prompt, run, conversation, resume=resume
        )
        trace_id = trace_id or self._generate_id()
        self._log_loop_start(config, abort_signal is not None, trace_id)

        # Build internal message list from conversation
        messages = list(conv.as_messages())

        result = await self._core_loop(
            messages=messages,
            config=config,
            strategy=SimpleStrategy(),
            conv=conv,
            abort_signal=abort_signal,
            on_iteration=on_iteration,
            trace=_trace,
        )

        if result.paused:
            from .errors import PauseRequested

            raise PauseRequested()

        return await self._finalize(
            prompt, result.output, schema, trace_id, temperature, run=config, _trace=_trace
        )

    async def _core_loop(
        self,
        messages: list[Message],
        config: CallOptions,
        strategy: LoopStrategy,
        *,
        conv: ConversationLike | None = None,
        abort_signal: asyncio.Event | None = None,
        pause_check: Callable[[], Awaitable[bool]] | None = None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None = None,
        on_decide: Callable[[ChatResponse, LoopDecision, int, list[Any]], None] | None = None,
        trace: VerbTrace | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> CoreLoopResult:
        """Unified loop with pluggable strategy. Delegates to _LoopRunner."""
        runner = _LoopRunner(self)
        return await runner.run(
            messages,
            config,
            strategy,
            conv=conv,
            abort_signal=abort_signal,
            pause_check=pause_check,
            on_iteration=on_iteration,
            on_decide=on_decide,
            trace=trace,
            response_schema=response_schema,
        )

    @staticmethod
    def _split_guard_feedback(
        outcomes: list[GuardOutcome],
    ) -> tuple[str | None, str | None]:
        """Split guard outcomes into blocking and advisory feedback strings."""
        return _GuardEvaluator.split_guard_feedback(outcomes)

    def _should_stop(
        self, config: CallOptions, iteration: int, start_time: float, total_tokens: int
    ) -> bool:
        """Check if loop should stop."""
        if config.max_iterations > 0 and iteration >= config.max_iterations:
            return True
        if config.timeout_secs > 0 and (time.monotonic() - start_time) >= config.timeout_secs:
            return True
        if config.max_total_tokens > 0 and total_tokens >= config.max_total_tokens:
            return True
        return False

    def _run_iteration_guards(
        self,
        guards: tuple[IterationGuard, ...],
        response: ChatResponse,
        iteration: int,
        max_iterations: int,
        _trace: VerbTrace | None = None,
    ) -> tuple[str | None, list[GuardOutcome]]:
        """Run iteration guards against the current response."""
        return _GuardEvaluator(self).run_iteration_guards(
            guards, response, iteration, max_iterations, _trace
        )

    def _eval_single_guard(self, guard: IterationGuard, ctx: IterationContext) -> str | None:
        """Evaluate a single guard, catching exceptions."""
        return _GuardEvaluator.eval_single_guard(guard, ctx)

    def _attach_guard_outcomes(
        self, _trace: VerbTrace | None, outcomes: list[GuardOutcome]
    ) -> None:
        """Attach outcomes to the most recent step if trace exists."""
        _GuardEvaluator.attach_guard_outcomes(_trace, outcomes)

    def _to_message(self, response: ChatResponse) -> Message:
        """Convert ChatResponse to Message."""
        return Message(
            role=Role.ASSISTANT,
            content=response.content,
            tool_calls=response.tool_calls if response.tool_calls else None,
        )

    @staticmethod
    async def _append_msg(target: MessageAppendable, msg: Message) -> None:
        """Append message, using async if target supports it."""
        if isinstance(target, AsyncConversationLike):
            result = target.append_async(msg)
            if not inspect.isawaitable(result):
                raise TypeError(
                    f"{type(target).__name__}.append_async() must be async (return awaitable), "
                    f"got {type(result).__name__}"
                )
            await result
        else:
            target.append(msg)

    @staticmethod
    def _fork_conversation(
        conversation: ConversationLike | None,
    ) -> ConversationLike | None:
        """Create a working copy of a conversation for isolated operations.

        Returns None when the caller did not provide a conversation (the
        downstream helpers will create a throwaway ListConversation).
        """
        if conversation is None:
            return None
        fork = ListConversation()
        for msg in conversation.as_messages():
            fork.append(msg)
        return fork

    @staticmethod
    async def _merge_conversation(
        target: ConversationLike | None,
        source: ConversationLike | None,
    ) -> None:
        """Append new messages from *source* back into *target*.

        Only messages added after the fork point (i.e. those beyond the
        original length of *target*) are copied.
        """
        if target is None or source is None:
            return
        base_len = len(target.as_messages())
        for msg in source.as_messages()[base_len:]:
            await Verb._append_msg(target, msg)

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        messages: MessageAppendable,
        pause_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        """Execute tool calls and append results."""
        await _ToolExecutor(self).execute_tools(tool_calls, messages, pause_check)

    async def _finalize(
        self,
        prompt: str,
        content: str,
        schema: type[T] | None,
        trace_id: str = "",
        temperature: float | None = None,
        run: CallOptions | None = None,
        _trace: VerbTrace | None = None,
    ) -> tuple[str, T | None]:
        """Finalize result, optionally parsing structured output.

        Called after a tool loop resolves. Poses the caller's prompt back with
        the loop's textual output as context, constrained to ``schema``. The
        finalize call is a single-shot parse (no parse-retry loop) — the tool
        loop already took the retries it was going to.
        """
        if schema is None:
            return content, None
        structured_prompt = f"{prompt}\n\nBased on the following information:\n{content}"
        parsed = await self._complete_structured_attempt(
            structured_prompt,
            schema,
            run=run,
            conversation=None,
            _trace=_trace,
            _phase="finalize",
        )
        return content, parsed

    # --- High-level helpers for verbs ---

    async def _complete(
        self,
        prompt: str,
        run: CallOptions | None = None,
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
    ) -> str:
        """Complete with tools if available, otherwise direct.

        Applies output guards if configured.
        """
        trace = _trace if _trace is not None else self._init_verb_trace()
        if self._has_tools():
            content, _ = await self._loop(prompt, run=run, conversation=conversation, _trace=trace)
        else:
            content = await self._complete_direct(prompt, run, conversation, trace)
        result = await self._apply_text_guards(
            prompt, content, run, conversation=conversation, _trace=trace
        )
        if _trace is None:
            self._emit_verb_trace(trace)
        return result

    async def _complete_direct(
        self,
        prompt: str,
        run: CallOptions | None,
        conversation: ConversationLike | None,
        trace: VerbTrace,
    ) -> str:
        """Direct (no-tool) text completion. Records step to trace."""
        config = self._get_call_options(run)
        conv = conversation if conversation is not None else ListConversation()
        await self._append_msg(conv, Message(role=Role.USER, content=prompt))
        response = await self._chat(
            conv.as_messages(),
            max_tokens=self._max_tokens(config),
            temperature=self._resolve_temperature(run),
            call=config,
            tools=[],
        )
        await self._append_msg(conv, self._to_message(response))
        self._record_step(response, phase="attempt", _trace=trace)
        return response.content

    async def _complete_text_attempt(
        self,
        prompt: str,
        run: CallOptions | None = None,
        phase: str = "direct",
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
    ) -> str:
        """Single attempt at text completion without applying guards.

        Used by guard retry logic to avoid recursion.
        """
        if self._has_tools():
            content, _ = await self._loop(prompt, run=run, conversation=conversation, _trace=_trace)
            return content
        config = self._get_call_options(run)
        conv = conversation if conversation is not None else ListConversation()
        await self._append_msg(conv, Message(role=Role.USER, content=prompt))
        response = await self._chat(
            conv.as_messages(),
            max_tokens=self._max_tokens(config),
            temperature=self._resolve_temperature(run),
            call=config,
            tools=[],
        )
        await self._append_msg(conv, self._to_message(response))
        self._record_step(response, phase=phase, _trace=_trace)
        return response.content

    async def _complete_structured(
        self,
        prompt: str,
        schema: type[T],
        run: CallOptions | None = None,
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
    ) -> T:
        """Complete structured with parse-retry and output guards.

        Drives ``_core_loop`` under :class:`SchemaTerminatingStrategy`. Parse
        retries (via ``schema_retry`` iteration guards) come for free from the
        strategy; output guards (instance + field) run over the parsed value
        once the strategy terminates.
        """
        trace = _trace if _trace is not None else self._init_verb_trace()
        try:
            result = await self._run_schema_loop(
                prompt,
                schema,
                run=run,
                conversation=conversation,
                trace=trace,
                phase="attempt",
                retry_on_parse_failure=True,
            )
        except StructuredOutputError:
            if _trace is None:
                self._emit_verb_trace(trace, reason="parse_error")
            raise
        result = await self._apply_guards(
            prompt, result, schema, run, conversation=conversation, _trace=trace
        )
        if _trace is None:
            self._emit_verb_trace(trace)
        return result

    async def _complete_structured_attempt(
        self,
        prompt: str,
        schema: type[T],
        run: CallOptions | None = None,
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
        _phase: str = "attempt",
    ) -> T:
        """Single structured attempt with parse-retry disabled.

        Used by output-guard field-level retries and the post-tool-loop
        finalize call. Bypasses ``schema_retry`` because guards run after a
        successful parse — JSON structure is expected to hold on retry — and
        finalize already sits downstream of a full tool loop.
        """
        trace = _trace if _trace is not None else self._init_verb_trace()
        try:
            return await self._run_schema_loop(
                prompt,
                schema,
                run=run,
                conversation=conversation,
                trace=trace,
                phase=_phase,
                retry_on_parse_failure=False,
            )
        finally:
            if _trace is None:
                self._emit_verb_trace(trace)

    async def _run_schema_loop(
        self,
        prompt: str,
        schema: type[T],
        *,
        run: CallOptions | None,
        conversation: ConversationLike | None,
        trace: VerbTrace,
        phase: str,
        retry_on_parse_failure: bool,
    ) -> T:
        """Drive ``_core_loop`` with :class:`SchemaTerminatingStrategy`.

        Runs the loop on a forked inner conversation and merges only the
        original prompt + final successful response back to the caller's
        conversation, so failed parse attempts stay isolated from callers'
        durable history.
        """
        from .schema import to_json_schema

        config = self._get_call_options(run)
        strategy = SchemaTerminatingStrategy(
            schema, self, config, retry_on_parse_failure=retry_on_parse_failure
        )
        config = self._bump_iterations_for_parse_budget(config, strategy)
        inner_conv, prior_len, messages = await self._seed_inner_conv(conversation, prompt)

        result = await self._core_loop(
            messages=messages,
            config=config,
            strategy=strategy,
            conv=inner_conv,
            on_decide=self._make_schema_on_decide(strategy, trace, phase),
            trace=trace,
            response_schema=to_json_schema(schema),
        )

        if not result.completed:
            self._raise_schema_loop_failure(result, strategy, schema)
        await self._merge_successful_exchange(conversation, inner_conv, prior_len)
        assert strategy.parsed_value is not None
        return strategy.parsed_value

    async def _merge_successful_exchange(
        self,
        outer: ConversationLike | None,
        inner: ConversationLike,
        prior_len: int,
    ) -> None:
        """Copy the original prompt + final response from ``inner`` to ``outer``.

        Intermediate parse-retry framing and failed responses stay inside the
        inner conversation so the caller's durable history reads as
        "asked X, got Y", not the full attempt-by-attempt exchange.
        """
        if outer is None:
            return
        new_msgs = inner.as_messages()[prior_len:]
        if not new_msgs:
            return
        await self._append_msg(outer, new_msgs[0])
        if len(new_msgs) >= 2:
            await self._append_msg(outer, new_msgs[-1])

    @staticmethod
    def _bump_iterations_for_parse_budget(
        config: CallOptions, strategy: SchemaTerminatingStrategy[Any]
    ) -> CallOptions:
        """Ensure the loop budget covers the parse-retry budget.

        Parse retries share the loop's iteration counter under the unified
        engine. Consumers who configure ``schema_retry(max_retries=N)`` still
        expect N retries even when ``max_iterations`` is lower. Zero
        (unlimited) is left untouched.
        """
        from dataclasses import replace

        if config.max_iterations > 0 and strategy.parse_budget > config.max_iterations:
            return replace(config, max_iterations=strategy.parse_budget)
        return config

    async def _seed_inner_conv(
        self, outer: ConversationLike | None, prompt: str
    ) -> tuple[ConversationLike, int, list[Message]]:
        """Create the strategy's inner conversation seeded from the outer one.

        Returns ``(inner_conv, prior_len, messages)``. ``prior_len`` marks the
        boundary between messages inherited from the outer conversation and
        those added during the strategy run — used to slice out the newly
        produced exchange for the outer merge.
        """
        inner: ConversationLike = ListConversation()
        if outer is not None:
            for msg in outer.as_messages():
                inner.append(msg)
        prior_len = len(inner.as_messages())
        await self._append_msg(inner, Message(role=Role.USER, content=prompt))
        return inner, prior_len, list(inner.as_messages())

    def _make_schema_on_decide(
        self,
        strategy: SchemaTerminatingStrategy[Any],
        trace: VerbTrace,
        phase: str,
    ) -> Callable[[ChatResponse, LoopDecision, int, list[GuardOutcome]], None]:
        """Build the ``on_decide`` callback that stamps parse outcome onto steps."""

        def on_decide(
            response: ChatResponse,
            decision: LoopDecision,
            iteration: int,
            outcomes: list[GuardOutcome],
        ) -> None:
            step_phase = phase if iteration == 0 else "parse_retry"
            self._record_step(response, phase=step_phase, _trace=trace)
            if strategy.last_parse_error is not None and trace.steps:
                trace.steps[-1].parsed = False
                trace.steps[-1].parse_error = strategy.last_parse_error.parse_error
            self._attach_guard_outcomes(trace, outcomes)

        return on_decide

    @staticmethod
    def _raise_schema_loop_failure(
        result: CoreLoopResult,
        strategy: SchemaTerminatingStrategy[Any],
        schema: type[T],
    ) -> None:
        """Translate a non-completed loop result into the right exception."""
        if strategy.last_parse_error is not None:
            raise strategy.last_parse_error
        if result.paused:
            from .errors import PauseRequested

            raise PauseRequested()
        raise StructuredOutputError(
            f"Loop terminated without a successful parse of {schema.__name__}",
            schema_name=schema.__name__,
        )

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the verb."""
        ...

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
from .structured_output import _StructuredOutputHandler
from .tool_executor import _ToolExecutor

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
        """
        call_id = self._generate_id()
        self._log_message_assembly(call_id, messages)
        resolved_tools = (
            (self._config.tools if self._config.tools else None)
            if tools is _SENTINEL
            else (tools or None)
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
        try:
            return guard.validator(ctx)
        except Exception as e:
            return f"Validator raised {type(e).__name__}: {e}"

    def _attach_guard_outcomes(
        self, _trace: VerbTrace | None, outcomes: list[GuardOutcome]
    ) -> None:
        """Attach outcomes to the most recent step if trace exists."""
        if _trace and _trace.steps:
            _trace.steps[-1].guards = outcomes

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
        """Finalize result, optionally parsing structured output."""
        return await _StructuredOutputHandler(self).finalize(
            prompt, content, schema, trace_id, temperature, run, _trace
        )

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
        """Complete structured with iteration guards and output guards."""
        return await _StructuredOutputHandler(self).complete_structured(
            prompt, schema, run, conversation, _trace
        )

    async def _complete_structured_attempt(
        self,
        prompt: str,
        schema: type[T],
        run: CallOptions | None = None,
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
        _phase: str = "attempt",
    ) -> T:
        """Single attempt at structured completion (used by guard retry)."""
        return await _StructuredOutputHandler(self)._complete_attempt(
            prompt, schema, run, conversation, _trace, _phase
        )

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the verb."""
        ...

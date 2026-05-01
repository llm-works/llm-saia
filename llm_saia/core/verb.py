"""Base class for SAIA verbs."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from abc import abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Self, TypedDict, TypeVar

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
from .errors import PauseRequested, StructuredOutputError, TruncatedResponseError
from .guard import IterationContext, IterationGuard
from .guards import OutputGuardMixin
from .schema import dataclass_to_json_schema, parse_json_to_dataclass

if TYPE_CHECKING:
    from .backend import Backend
    from .trace import GuardOutcome, Tracer, VerbTrace

T = TypeVar("T")

_SENTINEL: Any = object()  # default marker for _chat(tools=...)


class _ParseRetryState(TypedDict):
    """State for parse retry loop (avoids many parameters)."""

    error: StructuredOutputError | None
    feedback: str | None


class Verb(OutputGuardMixin, Configurable):
    """Base class for all verbs. Subclass this to create custom verbs."""

    # Truncation limit for log previews (debug level)
    _PREVIEW_LIMIT = 100
    # Truncation limit for trace logs (high ceiling to prevent pathological cases)
    _TRACE_LIMIT = 50_000
    # Patterns indicating truncated JSON response
    _TRUNCATION_INDICATORS = (
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

    async def _loop_iteration(
        self,
        conv: ConversationLike,
        config: CallOptions,
        max_tokens: int | None,
        temperature: float | None,
        iteration: int,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None,
        _trace: VerbTrace | None,
        abort_signal: asyncio.Event | None = None,
    ) -> tuple[ChatResponse, bool]:
        """Execute one loop iteration. Returns (response, should_continue)."""
        response = await self._chat(
            conv.as_messages(), max_tokens, temperature, call=config, abort_signal=abort_signal
        )
        self._log_response(response, iteration, response.input_tokens + response.output_tokens)
        self._check_tool_support(response)

        # Call on_iteration BEFORE appending to conv - if PauseRequested is raised,
        # nothing is persisted and resume will re-issue the same request cleanly.
        if on_iteration is not None:
            await on_iteration(iteration, response)

        await self._append_msg(conv, self._to_message(response))
        self._record_step(response, phase="iteration", _trace=_trace)

        should_continue = await self._apply_iteration_guards_or_tools(
            config.iteration_guards, response, conv, iteration, config.max_iterations, _trace
        )
        return response, should_continue

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
        config, conv, max_tokens, temperature = await self._init_loop(
            prompt, run, conversation, resume=resume
        )
        self._log_loop_start(config, abort_signal is not None, trace_id)
        start_time, iteration, total_tokens, last_content = time.monotonic(), 0, 0, ""
        trace_id = trace_id or self._generate_id()

        while not self._should_stop(config, iteration, start_time, total_tokens):
            response, should_continue = await self._loop_iteration(
                conv, config, max_tokens, temperature, iteration, on_iteration, _trace, abort_signal
            )
            total_tokens += response.input_tokens + response.output_tokens
            last_content = response.content
            if should_continue:
                iteration += 1
                continue
            self._log_loop_complete(iteration, start_time, total_tokens, response.content)
            return await self._finalize(
                prompt, response.content, schema, trace_id, temperature, run=config, _trace=_trace
            )

        self._log_limit_reached(config, iteration, start_time, total_tokens)
        return await self._finalize(
            prompt, last_content, schema, trace_id, temperature, run=config, _trace=_trace
        )

    async def _apply_iteration_guards_or_tools(
        self,
        iter_guards: tuple[IterationGuard, ...],
        response: ChatResponse,
        conv: ConversationLike,
        iteration: int,
        max_iterations: int,
        _trace: VerbTrace | None,
    ) -> bool:
        """Check iteration guards and execute tools. Returns True if loop should continue."""
        _feedback, outcomes = self._run_iteration_guards(
            iter_guards, response, iteration, max_iterations, _trace
        )
        blocking_fb, advisory_fb = self._split_guard_feedback(outcomes)

        if blocking_fb:
            await self._apply_blocking_feedback(response, conv, blocking_fb)
            return True

        if response.tool_calls:
            await self._execute_tools(response.tool_calls, conv)

        if advisory_fb:
            await self._apply_advisory_feedback(conv, advisory_fb)
            return True

        return bool(response.tool_calls)

    async def _apply_blocking_feedback(
        self, response: ChatResponse, conv: ConversationLike, feedback: str
    ) -> None:
        """ACK tool calls and inject blocking guard feedback."""
        for tc in response.tool_calls or []:
            await self._append_msg(
                conv, Message(role=Role.TOOL, content="Acknowledged.", tool_call_id=tc.id)
            )
        await self._append_msg(conv, Message(role=Role.USER, content=feedback))
        self._lg.trace(
            "blocking guard feedback injected",
            extra={
                "feedback_len": len(feedback),
                "feedback": self._truncate(feedback, self._TRACE_LIMIT),
                "acked_tools": [tc.name for tc in (response.tool_calls or [])],
            },
        )

    async def _apply_advisory_feedback(self, conv: ConversationLike, feedback: str) -> None:
        """Inject advisory guard feedback after tool execution."""
        await self._append_msg(conv, Message(role=Role.USER, content=feedback))
        self._lg.trace(
            "advisory guard feedback injected",
            extra={
                "feedback_len": len(feedback),
                "feedback": self._truncate(feedback, self._TRACE_LIMIT),
            },
        )

    @staticmethod
    def _split_guard_feedback(
        outcomes: list[GuardOutcome],
    ) -> tuple[str | None, str | None]:
        """Split guard outcomes into blocking and advisory feedback strings."""
        blocking_parts: list[str] = []
        advisory_parts: list[str] = []
        for o in outcomes:
            if not o.passed and o.error:
                if o.blocking:
                    blocking_parts.append(o.error)
                else:
                    advisory_parts.append(o.error)
        blocking = "\n\n".join(blocking_parts) if blocking_parts else None
        advisory = "\n\n".join(advisory_parts) if advisory_parts else None
        return blocking, advisory

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
        """Run iteration guards against the current response.

        Returns ``(feedback, outcomes)`` — *feedback* is the combined feedback
        string if any guard fires (or ``None`` when all pass), and *outcomes*
        is the list of per-guard results for trace recording.
        """
        from .trace import GuardOutcome

        if not guards:
            return None, []

        ctx = IterationContext(
            response=response, iteration=iteration, max_iterations=max_iterations
        )
        self._lg.trace("running iteration guards", extra={"guards": [g.name for g in guards]})
        feedback_parts, outcomes = self._eval_guards(guards, ctx, GuardOutcome)
        self._attach_guard_outcomes(_trace, outcomes)
        return self._finalize_guard_result(feedback_parts, outcomes)

    def _eval_guards(
        self,
        guards: tuple[IterationGuard, ...],
        ctx: IterationContext,
        outcome_cls: type[GuardOutcome],
    ) -> tuple[list[str], list[GuardOutcome]]:
        """Evaluate each guard and collect feedback and outcomes."""
        feedback_parts: list[str] = []
        outcomes: list[GuardOutcome] = []
        for guard in guards:
            result = self._eval_single_guard(guard, ctx)
            passed = result is None
            outcomes.append(
                outcome_cls(
                    name=guard.name,
                    passed=passed,
                    error=result if not passed else None,
                    blocking=guard.blocking,
                )
            )
            if not passed:
                self._log_iteration_guard_fired(guard.name, result)
                feedback_parts.append(result)  # type: ignore[arg-type]
        return feedback_parts, outcomes

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

    def _finalize_guard_result(
        self, feedback_parts: list[str], outcomes: list[GuardOutcome]
    ) -> tuple[str | None, list[GuardOutcome]]:
        """Combine feedback and log the result."""
        if feedback_parts:
            combined = "\n\n".join(feedback_parts)
            self._lg.trace(
                "iteration guards triggered feedback",
                extra={
                    "guards_fired": [o.name for o in outcomes if not o.passed],
                    "feedback": self._truncate(combined, self._TRACE_LIMIT),
                },
            )
            return combined, outcomes
        self._lg.trace("all iteration guards passed")
        return None, outcomes

    def _log_iteration_guard_fired(self, name: str | None, feedback: str | None) -> None:
        """Log that an iteration guard fired."""
        self._lg.debug(
            "iteration guard fired",
            extra={"guard": name, "feedback": feedback},
        )

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
        """Execute tool calls and append results.

        Args:
            tool_calls: Tool calls to execute.
            messages: Object supporting append() - list[Message] or ConversationLike.
            pause_check: Optional async callback checked between tool calls.
                If it returns True, remaining tools are acknowledged as "Paused."
                and ``PauseRequested`` is raised.
        """
        if not self._config.executor:
            self._lg.warning(
                "tool calls received but no executor configured",
                extra={"tool_count": len(tool_calls)},
            )
            return
        for i, tc in enumerate(tool_calls):
            result = await self._execute_single_tool(tc)
            await self._append_msg(
                messages, Message(role=Role.TOOL, content=str(result), tool_call_id=tc.id)
            )
            # Check for pause after each tool (except the last one)
            if pause_check is not None and i < len(tool_calls) - 1:
                if await pause_check():
                    remaining = tool_calls[i + 1 :]
                    for rem_tc in remaining:
                        msg = Message(role=Role.TOOL, content="Paused.", tool_call_id=rem_tc.id)
                        await self._append_msg(messages, msg)
                    self._lg.debug(
                        "pause requested between tool calls",
                        extra={"executed": i + 1, "remaining": len(remaining)},
                    )
                    raise PauseRequested()

    async def _execute_single_tool(self, tc: ToolCall) -> str:
        """Execute a single tool call with logging."""
        self._log_tool_start(tc)
        try:
            result = await self._config.executor(tc.name, tc.arguments)  # type: ignore[misc]
        except Exception as e:
            self._log_tool_error(tc, e)
            return f"Error: {e}"
        self._log_tool_success(tc, result)
        return str(result)

    def _log_tool_start(self, tc: ToolCall) -> None:
        """Log tool execution start."""
        self._lg.trace(
            "executing tool...",
            extra={"tool": tc.name, "id": tc.id, "tool_args": tc.arguments},
        )

    def _log_tool_success(self, tc: ToolCall, result: Any) -> None:
        """Log successful tool execution with result."""
        result_str = str(result)
        self._lg.trace(
            "tool result returned to llm",
            extra={
                "tool": tc.name,
                "id": tc.id,
                "result_len": len(result_str),
                "result": self._truncate(result_str, self._TRACE_LIMIT),
            },
        )

    def _log_tool_error(self, tc: ToolCall, error: Exception) -> None:
        """Log failed tool execution."""
        self._lg.warning(
            "tool execution failed",
            extra={"tool": tc.name, "id": tc.id, "tool_args": tc.arguments, "exception": error},
        )

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
        if schema:
            structured_prompt = f"{prompt}\n\nBased on the following information:\n{content}"
            json_schema = dataclass_to_json_schema(schema)
            response = await self._chat(
                [Message(role=Role.USER, content=structured_prompt)],
                max_tokens=None,
                temperature=temperature,
                call=run,
                response_schema=json_schema,
                tools=[],
            )
            self._record_step(response, phase="finalize", _trace=_trace)
            parser = self._config.json_parser or json.loads
            try:
                data = parser(response.content)
            except Exception as e:
                self._log_finalize_parse_error(e, response.content, schema.__name__)
                raise self._structured_output_error(e, response.content, schema.__name__) from e
            try:
                result = parse_json_to_dataclass(data, schema)
            except (TypeError, ValueError) as e:
                raise StructuredOutputError(
                    f"Response does not match {schema.__name__}: {e}",
                    raw_content=response.content,
                    schema_name=schema.__name__,
                    parse_error=str(e),
                ) from e
            return content, result
        return content, None

    def _log_finalize_parse_error(self, error: Exception, content: str, schema_name: str) -> None:
        """Log a JSON parse error during finalize phase."""
        self._lg.warning(
            "json parse error in finalize",
            extra={
                "exception": error,
                "content_preview": self._truncate(content, self._PREVIEW_LIMIT),
                "schema": schema_name,
            },
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
        """Complete structured with iteration guards (for parse retry) and output guards."""
        trace = _trace if _trace is not None else self._init_verb_trace()
        config = self._get_call_options(run)
        # Guards with parse_max_retries > 0 participate in parse retry
        parse_retry_guards = tuple(g for g in config.iteration_guards if g.parse_max_retries > 0)
        max_attempts = 1 + sum(g.parse_max_retries for g in parse_retry_guards)
        state: _ParseRetryState = {"error": None, "feedback": None}

        for attempt in range(max_attempts):
            try:
                return await self._structured_attempt_with_guards(
                    prompt, schema, run, conversation, state, attempt, trace, _trace
                )
            except StructuredOutputError as e:
                should_retry = self._handle_parse_error(
                    e, parse_retry_guards, attempt, max_attempts, trace, state
                )
                if should_retry:
                    continue
                self._emit_verb_trace_if_root(trace, _trace)
                raise

        raise state["error"]  # type: ignore[misc]

    async def _structured_attempt_with_guards(
        self,
        prompt: str,
        schema: type[T],
        run: CallOptions | None,
        conversation: ConversationLike | None,
        state: _ParseRetryState,
        attempt: int,
        trace: VerbTrace,
        parent_trace: VerbTrace | None,
    ) -> T:
        """Single structured attempt with output guards applied."""
        phase = "attempt" if attempt == 0 else "parse_retry"
        step_num_before = len(trace.steps)
        result = await self._structured_attempt_cycle(
            prompt,
            schema,
            run,
            conversation,
            state["error"],
            state["feedback"],
            attempt,
            phase,
            trace,
        )
        # Log successful parse attempt (step was just added)
        if trace.steps and len(trace.steps) > step_num_before:
            self._log_structured_attempt(trace.steps[-1], len(trace.steps))
        result = await self._apply_guards(
            prompt, result, schema, run, conversation=conversation, _trace=trace
        )
        self._emit_verb_trace_if_root(trace, parent_trace)
        return result

    def _emit_verb_trace_if_root(
        self, trace: VerbTrace, parent: VerbTrace | None, reason: str | None = None
    ) -> None:
        """Emit verb trace only if this is the root trace (not nested)."""
        if parent is None:
            self._emit_verb_trace(trace, reason)

    def _handle_parse_error(
        self,
        error: StructuredOutputError,
        guards: tuple[IterationGuard, ...],
        attempt: int,
        max_attempts: int,
        trace: VerbTrace,
        state: _ParseRetryState,
    ) -> bool:
        """Handle parse error: mark trace, evaluate guards, update state. Returns True to retry."""
        self._mark_last_step_parse_failure(trace, error)
        # Log the failed attempt with error details
        if trace.steps:
            self._log_structured_attempt(
                trace.steps[-1],
                len(trace.steps),
                error=error.parse_error,
                raw_content=error.raw_content,
            )
        # Create a minimal response for the context
        response = ChatResponse(content=error.raw_content or "", tool_calls=[])
        feedback = self._eval_parse_retry_guards(
            guards, response, error, attempt, max_attempts, trace
        )
        state["error"] = error
        state["feedback"] = feedback
        if feedback and attempt < max_attempts - 1:
            schema_name = error.schema_name or "unknown"
            self._log_parse_retry(schema_name, attempt + 1, max_attempts, error)
            return True
        return False

    def _eval_parse_retry_guards(
        self,
        guards: tuple[IterationGuard, ...],
        response: ChatResponse,
        error: StructuredOutputError,
        attempt: int,
        max_attempts: int,
        trace: VerbTrace,
    ) -> str | None:
        """Evaluate iteration guards in parse retry context.

        Like tool loop guards, evaluates all guards and combines feedback.
        Uses _eval_single_guard for consistent exception handling.
        """
        if not guards:
            return None
        ctx = IterationContext(
            response=response,
            iteration=attempt,
            max_iterations=max_attempts,
            parse_error=error,
        )
        feedback_parts: list[str] = []
        for guard in guards:
            feedback = self._eval_single_guard(guard, ctx)
            if feedback is not None:
                self._log_parse_guard_trigger(guard.name, attempt, feedback)
                feedback_parts.append(feedback)
        return "\n\n".join(feedback_parts) if feedback_parts else None

    def _log_parse_guard_trigger(self, name: str | None, attempt: int, feedback: str) -> None:
        """Log when a parse guard triggers a retry."""
        self._lg.debug(
            "parse guard triggered retry",
            extra={"guard": name or "guard", "attempt": attempt, "feedback": feedback[:100]},
        )

    async def _structured_attempt_cycle(
        self,
        prompt: str,
        schema: type[T],
        run: CallOptions | None,
        conversation: ConversationLike | None,
        last_error: StructuredOutputError | None,
        last_feedback: str | None,
        attempt: int,
        phase: str,
        trace: VerbTrace,
    ) -> T:
        """Run one parse attempt: build prompt, fork conv, call backend."""
        use_prompt = self._retry_prompt_or_original(
            prompt, schema, last_error, last_feedback, attempt
        )
        attempt_conv = self._fork_conversation(conversation)
        result = await self._complete_structured_attempt(
            use_prompt, schema, run, conversation=attempt_conv, _trace=trace, _phase=phase
        )
        await self._merge_conversation(conversation, attempt_conv)
        return result

    @staticmethod
    def _mark_last_step_parse_failure(trace: VerbTrace, error: StructuredOutputError) -> None:
        """Mark the last step in the trace as a parse failure."""
        if trace.steps:
            trace.steps[-1].parsed = False
            trace.steps[-1].parse_error = error.parse_error

    async def _complete_structured_attempt(
        self,
        prompt: str,
        schema: type[T],
        run: CallOptions | None = None,
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
        _phase: str = "attempt",
    ) -> T:
        """Single attempt at structured completion."""
        if self._has_tools():
            _, result = await self._loop(
                prompt, run=run, schema=schema, conversation=conversation, _trace=_trace
            )
            if result is not None:
                return result
        # Direct structured completion
        conv = conversation if conversation is not None else ListConversation()
        await self._append_msg(conv, Message(role=Role.USER, content=prompt))
        json_schema = dataclass_to_json_schema(schema)
        config = self._get_call_options(run)
        response = await self._chat(
            conv.as_messages(),
            max_tokens=self._max_tokens(config),
            temperature=self._resolve_temperature(run),
            call=config,
            response_schema=json_schema,
            tools=[],
        )
        await self._append_msg(conv, self._to_message(response))
        self._record_step(response, phase=_phase, _trace=_trace)
        return self._parse_structured_response(response.content, schema)

    def _parse_structured_response(self, content: str, schema: type[T]) -> T:
        """Parse JSON response into schema, raising StructuredOutputError on failure."""
        parser = self._config.json_parser or json.loads
        try:
            data = parser(content)
        except Exception as e:
            raise self._structured_output_error(e, content, schema.__name__) from e
        try:
            return parse_json_to_dataclass(data, schema)
        except (TypeError, ValueError) as e:
            raise StructuredOutputError(
                f"Response does not match {schema.__name__}: {e}",
                raw_content=content,
                schema_name=schema.__name__,
                parse_error=str(e),
            ) from e

    def _retry_prompt_or_original(
        self,
        prompt: str,
        schema: type[T],
        last_error: StructuredOutputError | None,
        last_feedback: str | None,
        attempt: int,
    ) -> str:
        """Return original prompt on first attempt, retry prompt on subsequent ones."""
        if attempt == 0:
            return prompt
        # Use guard feedback if provided, otherwise build from error
        if last_feedback:
            return self._build_parse_retry_prompt_with_feedback(prompt, last_error, last_feedback)
        return self._build_parse_retry_prompt(
            prompt,
            schema,
            last_error,  # type: ignore[arg-type]
        )

    def _build_parse_retry_prompt(
        self, original_prompt: str, schema: type[T], error: StructuredOutputError
    ) -> str:
        """Build a retry prompt with feedback about the parse failure."""
        parts = [original_prompt, "\n\n---\n\nYour previous response could not be parsed."]

        if error.parse_error:
            parts.append(f"\n\nParse error: {error.parse_error}")

        if error.raw_content:
            # Truncate raw content to avoid token bloat
            raw = error.raw_content
            if len(raw) > 500:
                raw = raw[:500] + "... (truncated)"
            parts.append(f"\n\nYour response was:\n```\n{raw}\n```")

        parts.append(
            f"\n\nPlease provide a valid JSON response matching the {schema.__name__} schema."
        )
        return "".join(parts)

    def _build_parse_retry_prompt_with_feedback(
        self, original_prompt: str, error: StructuredOutputError | None, feedback: str
    ) -> str:
        """Build a retry prompt using guard-provided feedback."""
        parts = [original_prompt, "\n\n---\n\n", feedback]
        if error and error.raw_content:
            raw = error.raw_content
            if len(raw) > 500:
                raw = raw[:500] + "... (truncated)"
            parts.append(f"\n\nYour response was:\n```\n{raw}\n```")
        return "".join(parts)

    def _log_parse_retry(
        self, schema_name: str, attempt: int, max_attempts: int, error: StructuredOutputError
    ) -> None:
        """Log parse retry attempt."""
        self._lg.debug(
            "parse retry",
            extra={
                "schema": schema_name,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "parse_error": error.parse_error,
            },
        )

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the verb."""
        ...

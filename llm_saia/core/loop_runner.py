"""Loop execution engine for verbs.

Executes the unified loop on behalf of a Verb, delegating decisions
to a LoopStrategy.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

from .loop import CoreLoopResult, LoopAction, LoopDecision, LoopStrategy

if TYPE_CHECKING:
    from .backend import ChatResponse
    from .config import CallOptions
    from .conversation import ConversationLike, Message, ToolCall
    from .guard import IterationGuard
    from .trace import GuardOutcome, VerbTrace


class _LoopHost(Protocol):
    """Protocol for capabilities the loop runner needs from its host (Verb)."""

    async def _chat(
        self,
        messages: list[Message],
        max_tokens: int | None,
        temperature: float | None,
        *,
        call: CallOptions | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> ChatResponse:
        """Call the LLM backend."""
        ...

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        messages: list[Message],
        pause_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        """Execute tool calls and append results to messages."""
        ...

    def _to_message(self, response: ChatResponse) -> Message:
        """Convert ChatResponse to Message."""
        ...

    @staticmethod
    async def _append_msg(conv: ConversationLike, msg: Message) -> None:
        """Append message to conversation."""
        ...

    def _should_stop(
        self, config: CallOptions, iteration: int, start_time: float, total_tokens: int
    ) -> bool:
        """Check if loop should stop due to limits."""
        ...

    def _run_iteration_guards(
        self,
        guards: tuple[IterationGuard, ...],
        response: ChatResponse,
        iteration: int,
        max_iterations: int,
        trace: VerbTrace | None = None,
    ) -> tuple[str | None, list[GuardOutcome]]:
        """Run iteration guards and return feedback."""
        ...

    def _split_guard_feedback(self, outcomes: list[GuardOutcome]) -> tuple[str | None, str | None]:
        """Split guard outcomes into blocking and advisory feedback."""
        ...

    def _record_step(
        self, response: ChatResponse, *, phase: str, _trace: VerbTrace | None = None
    ) -> None:
        """Record a step to the trace."""
        ...

    def _attach_guard_outcomes(self, trace: VerbTrace | None, outcomes: list[GuardOutcome]) -> None:
        """Attach guard outcomes to the current step."""
        ...

    def _log_response(self, response: ChatResponse, iteration: int, tokens: int) -> None:
        """Log LLM response details."""
        ...

    def _log_loop_complete(
        self, iteration: int, start_time: float, total_tokens: int, output: str
    ) -> None:
        """Log loop completion."""
        ...

    def _log_limit_reached(
        self, config: CallOptions, iteration: int, start_time: float, total_tokens: int
    ) -> None:
        """Log when loop limit is reached."""
        ...

    def _max_tokens(self, config: CallOptions) -> int | None:
        """Get max tokens from config."""
        ...

    def _resolve_temperature(self, override: CallOptions | None) -> float | None:
        """Resolve temperature from override or config."""
        ...

    def _check_tool_support(self, response: ChatResponse) -> None:
        """Check if backend supports tool calls."""
        ...


class _LoopRunner:
    """Executes the unified loop on behalf of a Verb."""

    def __init__(self, host: _LoopHost):
        """Initialize with host providing required capabilities."""
        self._host = host

    async def run(
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
        """Run the loop until completion, failure, pause, or limit."""
        from .errors import PauseRequested

        h = self._host
        start_time, iteration, total_tokens, last_content = time.monotonic(), 0, 0, ""
        max_tokens, temperature = h._max_tokens(config), h._resolve_temperature(config)

        try:
            while not h._should_stop(config, iteration, start_time, total_tokens):
                result, tokens, content = await self._run_iteration(
                    messages,
                    config,
                    strategy,
                    conv,
                    abort_signal,
                    pause_check,
                    on_iteration,
                    on_decide,
                    trace,
                    iteration,
                    max_tokens,
                    temperature,
                )
                total_tokens, last_content = total_tokens + tokens, content
                if result is not None:
                    return self._finalize_complete(result, iteration, start_time, total_tokens)
                iteration += 1
            h._log_limit_reached(config, iteration, start_time, total_tokens)
            return self._incomplete_result(messages, iteration, total_tokens, last_content, False)
        except PauseRequested:
            return self._incomplete_result(messages, iteration, total_tokens, last_content, True)

    async def _run_iteration(
        self,
        messages: list[Message],
        config: CallOptions,
        strategy: LoopStrategy,
        conv: ConversationLike | None,
        abort_signal: asyncio.Event | None,
        pause_check: Callable[[], Awaitable[bool]] | None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None,
        on_decide: Callable[[ChatResponse, LoopDecision, int, list[Any]], None] | None,
        trace: VerbTrace | None,
        iteration: int,
        max_tokens: int | None,
        temperature: float | None,
    ) -> tuple[CoreLoopResult | None, int, str]:
        """Run one loop iteration. Returns (result, tokens, content)."""
        h = self._host
        llm_messages = conv.as_messages() if conv else messages
        response = await h._chat(
            llm_messages, max_tokens, temperature, call=config, abort_signal=abort_signal
        )
        tokens = response.input_tokens + response.output_tokens
        h._log_response(response, iteration, tokens)
        h._check_tool_support(response)

        _, outcomes = h._run_iteration_guards(
            config.iteration_guards, response, iteration, config.max_iterations
        )
        blocking_fb, advisory_fb = h._split_guard_feedback(outcomes)

        if on_iteration:
            await on_iteration(iteration, response)

        decision = await strategy.decide(response, messages, iteration, blocking_fb, advisory_fb)

        if on_decide:
            on_decide(response, decision, iteration, outcomes)
        else:
            h._record_step(response, phase="iteration", _trace=trace)
            h._attach_guard_outcomes(trace, outcomes)

        result = await self._execute_decision(
            decision, response, messages, conv, pause_check, advisory_fb
        )
        strategy.on_iteration_complete(decision, tokens)
        return result, tokens, response.content

    async def _execute_decision(
        self,
        decision: LoopDecision,
        response: ChatResponse,
        messages: list[Message],
        conv: ConversationLike | None,
        pause_check: Callable[[], Awaitable[bool]] | None,
        advisory_fb: str | None,
    ) -> CoreLoopResult | None:
        """Execute a loop decision. Returns result if loop should exit."""
        match decision.action:
            case LoopAction.EXECUTE_TOOLS:
                await self._do_execute_tools(
                    decision, response, messages, conv, pause_check, advisory_fb
                )
                return None
            case LoopAction.INSTRUCT:
                await self._do_ack_and_inject(response, messages, conv, decision.message)
                return None
            case LoopAction.SKIP:
                await self._do_ack_and_inject(response, messages, conv, "Continue.")
                return None
            case LoopAction.COMPLETE:
                await self._do_ack_and_inject(response, messages, conv, None)
                return self._make_loop_result(True, decision, response, messages)
            case LoopAction.FAIL:
                await self._do_ack_and_inject(response, messages, conv, None)
                return self._make_loop_result(False, decision, response, messages)
        return None

    async def _do_execute_tools(
        self,
        decision: LoopDecision,
        response: ChatResponse,
        messages: list[Message],
        conv: ConversationLike | None,
        pause_check: Callable[[], Awaitable[bool]] | None,
        advisory_fb: str | None,
    ) -> None:
        """Handle EXECUTE_TOOLS: filter, ack skipped, execute, sync."""
        h = self._host
        messages.append(h._to_message(response))
        if conv:
            await h._append_msg(conv, h._to_message(response))

        tool_calls = response.tool_calls or []
        if decision.tool_ids is not None:
            execute_ids = set(decision.tool_ids)
            for tc in tool_calls:
                if tc.id not in execute_ids:
                    await self._ack_tool_to(tc, messages, conv)
            tool_calls = [tc for tc in tool_calls if tc.id in execute_ids]

        pre_len = len(messages)
        try:
            await h._execute_tools(tool_calls, messages, pause_check)
        finally:
            if conv:
                for msg in messages[pre_len:]:
                    await h._append_msg(conv, msg)

        if advisory_fb:
            msg = self._make_user_message(advisory_fb)
            messages.append(msg)
            if conv:
                await h._append_msg(conv, msg)

    async def _do_ack_and_inject(
        self,
        response: ChatResponse,
        messages: list[Message],
        conv: ConversationLike | None,
        inject: str | None,
    ) -> None:
        """Append response, ack all tools, optionally inject user message."""
        h = self._host
        messages.append(h._to_message(response))
        if conv:
            await h._append_msg(conv, h._to_message(response))
        for tc in response.tool_calls or []:
            await self._ack_tool_to(tc, messages, conv)
        if inject:
            msg = self._make_user_message(inject)
            messages.append(msg)
            if conv:
                await h._append_msg(conv, msg)

    async def _ack_tool_to(
        self, tc: ToolCall, messages: list[Message], conv: ConversationLike | None
    ) -> None:
        """Acknowledge a tool call to messages and optionally conv."""
        msg = self._make_tool_ack(tc.id)
        messages.append(msg)
        if conv:
            await self._host._append_msg(conv, msg)

    def _finalize_complete(
        self, result: CoreLoopResult, iteration: int, start_time: float, total_tokens: int
    ) -> CoreLoopResult:
        """Finalize a complete result with iteration stats and logging."""
        result.iterations, result.total_tokens = iteration + 1, total_tokens
        self._host._log_loop_complete(iteration, start_time, total_tokens, result.output)
        return result

    @staticmethod
    def _incomplete_result(
        messages: list[Message], iteration: int, total_tokens: int, content: str, paused: bool
    ) -> CoreLoopResult:
        """Build CoreLoopResult for incomplete loop (limit or pause)."""
        return CoreLoopResult(
            completed=False,
            output=content,
            iterations=iteration,
            messages=messages,
            reason="paused" if paused else "limit_reached",
            paused=paused,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _make_user_message(content: str) -> Message:
        """Create a user message."""
        from .conversation import Message, Role

        return Message(role=Role.USER, content=content)

    @staticmethod
    def _make_tool_ack(tool_call_id: str) -> Message:
        """Create a tool acknowledgment message."""
        from .conversation import Message, Role

        return Message(role=Role.TOOL, content="Acknowledged.", tool_call_id=tool_call_id)

    @staticmethod
    def _make_loop_result(
        completed: bool, decision: LoopDecision, response: ChatResponse, messages: list[Message]
    ) -> CoreLoopResult:
        """Build CoreLoopResult for COMPLETE/FAIL actions."""
        return CoreLoopResult(
            completed=completed,
            output=decision.output or response.content,
            iterations=0,
            messages=messages,
            reason="completed" if completed else "failed",
            terminal_data=decision.terminal_data,
            terminal_tool=decision.terminal_tool,
        )

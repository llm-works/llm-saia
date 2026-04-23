"""COMPLETE verb: Execute a task with tool calling and completion confirmation."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..core.backend import ChatResponse
from ..core.config import CallOptions
from ..core.controller import (
    Action,
    ActionType,
    ControllerConfig,
    DefaultController,
    LoopController,
    Observation,
)
from ..core.conversation import ConversationLike, Message, Role, ToolCall
from ..core.guard import IterationGuard
from ..core.trace import GuardOutcome, LLMCall, Step, ToolOutcome, Tracer, VerbTrace
from ..core.types import DecisionReason, LoopScore, TaskResult
from ..core.verb import Verb

# Default call options for complete (unlimited iterations)
DEFAULT_COMPLETE_CALL = CallOptions(max_iterations=0)

# Reasons that count as productive despite being INSTRUCT
_PRODUCTIVE_INSTRUCT_REASONS = frozenset({DecisionReason.TERMINAL_CONFIRMATION_REQUEST})


@dataclass
class _LoopCtx:
    """Mutable loop context bundling iteration state and scoring.

    Manages dual message storage: internal ``messages`` (always complete) and
    optional external ``conv`` (may compact). Use ``append()`` for writes and
    ``llm_messages()`` for LLM calls.
    """

    task: str
    trace_id: str
    ctrl: LoopController
    tracer: Tracer | None
    on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None
    call_options: CallOptions
    messages: list[Message]  # Complete internal history
    tool_names: list[str]
    iteration_guards: tuple[IterationGuard, ...] = ()
    verb_trace: VerbTrace | None = None
    acc: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    conv: ConversationLike | None = None  # External conversation (may compact)

    def llm_messages(self) -> list[Message]:
        """Messages for LLM calls - uses conv (possibly compacted) if available."""
        return self.conv.as_messages() if self.conv is not None else self.messages

    async def append(self, msg: Message) -> None:
        """Append to internal history and external conv (if present).

        Uses async append when conversation supports it (e.g., for non-blocking
        compaction with LLM-based summarization).
        """
        self.messages.append(msg)
        if self.conv is not None:
            await Verb._append_msg(self.conv, msg)


class Complete(Verb):
    """Execute a task with tool calling and completion confirmation."""

    async def __call__(
        self,
        task: str,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None = None,
        controller: LoopController | None = None,
        conversation: ConversationLike | None = None,
    ) -> TaskResult:
        """Execute a task using tools until completion or limit reached.

        Args:
            task: The task description / prompt.
            on_iteration: Optional async callback invoked each iteration.
            controller: Custom loop controller (uses default if None).
            conversation: Optional external conversation for message management.
                If provided, messages are appended to both an internal history
                (returned in ``TaskResult.history``) and this conversation.
                The LLM sees ``conversation.as_messages()`` which may be compacted.
        """
        if not self._has_tools():
            raise ValueError("Complete requires tools and executor to be configured.")

        verb_trace = self._init_verb_trace()
        trace_id = verb_trace.trace_id
        ctrl = controller or self._default_controller()
        ctrl.reset()

        # Tracer is from config - caller owns it and is responsible for closing.
        tracer = self._resolve_tracer(
            {"trace_id": trace_id, "request_id": verb_trace.request_id, "task": task[:200]},
        )

        try:
            result = await self._run_loop(
                task, trace_id, ctrl, tracer, on_iteration, verb_trace, conversation
            )
            return self._tag_result(result, verb_trace)
        finally:
            self._emit_verb_trace(verb_trace)

    @staticmethod
    def _score_action(acc: list[int], action: Action, tokens: int) -> None:
        """Accumulate scoring stats. acc = [productive, nudges, skips, wasted_tokens]."""
        is_productive_instruct = action.reason in _PRODUCTIVE_INSTRUCT_REASONS
        if action.kind in (ActionType.EXECUTE_TOOLS, ActionType.COMPLETE, ActionType.FAIL):
            acc[0] += 1
        elif action.kind == ActionType.INSTRUCT and is_productive_instruct:
            acc[0] += 1
        elif action.kind == ActionType.INSTRUCT:
            acc[1] += 1
            acc[3] += tokens
        elif action.kind == ActionType.SKIP:
            acc[2] += 1
            acc[3] += tokens

    @staticmethod
    def _build_score(iters: int, total_tokens: int, acc: list[int]) -> LoopScore:
        """Build LoopScore from accumulated stats."""
        return LoopScore(iters, acc[0], acc[1], acc[2], total_tokens, acc[3])

    async def _init_loop_ctx(
        self,
        task: str,
        trace_id: str,
        ctrl: LoopController,
        tracer: Tracer | None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None,
        verb_trace: VerbTrace | None,
        conversation: ConversationLike | None,
    ) -> _LoopCtx:
        """Initialize loop context with dual message storage."""
        call_options = self._config.call or DEFAULT_COMPLETE_CALL
        initial_msg = Message(role=Role.USER, content=task)
        ctx = _LoopCtx(
            task=task,
            trace_id=trace_id,
            ctrl=ctrl,
            tracer=tracer,
            on_iteration=on_iteration,
            call_options=call_options,
            messages=[initial_msg],
            tool_names=[t.name for t in (self._config.tools or [])],
            iteration_guards=call_options.iteration_guards,
            verb_trace=verb_trace,
            conv=conversation,
        )
        # Sync initial message to external conversation if provided
        if conversation is not None:
            await self._append_msg(conversation, initial_msg)
        return ctx

    async def _run_loop(
        self,
        task: str,
        trace_id: str,
        ctrl: LoopController,
        tracer: Tracer | None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None,
        verb_trace: VerbTrace | None = None,
        conversation: ConversationLike | None = None,
    ) -> TaskResult:
        """Execute the main tool-calling loop."""
        ctx = await self._init_loop_ctx(
            task, trace_id, ctrl, tracer, on_iteration, verb_trace, conversation
        )
        self._log_loop_start(ctx.call_options)
        start_time, iteration, total_tokens, last_content = time.monotonic(), 0, 0, ""

        while not self._should_stop(ctx.call_options, iteration, start_time, total_tokens):
            result, tokens, last_content = await self._run_one_iteration(ctx, iteration)
            total_tokens += tokens
            if result:
                self._log_loop_complete(iteration, start_time, total_tokens, result.output or "")
                result.score = self._build_score(iteration + 1, total_tokens, ctx.acc)
                return result
            iteration += 1

        self._log_limit_reached(ctx.call_options, iteration, start_time, total_tokens)
        result = TaskResult(False, last_content, iteration, ctx.messages)
        result.score = self._build_score(iteration, total_tokens, ctx.acc)
        return result

    async def _run_one_iteration(
        self,
        ctx: _LoopCtx,
        iteration: int,
    ) -> tuple[TaskResult | None, int, str]:
        """Run one loop iteration. Returns (result, tokens, last_content)."""
        response, tokens = await self._run_iteration(ctx.llm_messages(), ctx.call_options)
        self._log_response(response, iteration, tokens)

        # Run iteration guards before controller decides.
        # Don't pass verb_trace here — Complete builds its own Step later and
        # attaches outcomes explicitly (avoids overwriting the previous step).
        feedback, outcomes = self._run_iteration_guards(
            ctx.iteration_guards, response, iteration, ctx.call_options.max_iterations
        )

        # Split feedback by blocking mode
        blocking_fb, advisory_fb = self._split_guard_feedback(outcomes)

        # Blocking guards fire → skip tool execution, inject feedback
        if blocking_fb:
            await self._apply_guard_nudge(ctx, response, blocking_fb, outcomes, iteration, tokens)
            return None, tokens, response.content

        # No blocking guards, process iteration (may execute tools)
        action, result = await self._process_iteration(ctx, response, iteration, outcomes)
        self._score_action(ctx.acc, action, tokens)

        # Advisory guards → inject feedback after tool execution (skip if task completed)
        if advisory_fb and result is None:
            await ctx.append(Message(role=Role.USER, content=advisory_fb))
            self._log_advisory_feedback(iteration, outcomes, advisory_fb)

        return result, tokens, response.content

    def _log_advisory_feedback(
        self, iteration: int, outcomes: list[GuardOutcome], feedback: str
    ) -> None:
        """Log advisory guard feedback injection."""
        self._lg.trace(
            "advisory guard feedback injected after tool execution",
            extra={
                "iteration": iteration,
                "guards_fired": [o.name for o in outcomes if not o.passed and not o.blocking],
                "feedback_len": len(feedback),
                "feedback": self._truncate(feedback, self._TRACE_LIMIT),
            },
        )

    async def _apply_guard_nudge(
        self,
        ctx: _LoopCtx,
        response: ChatResponse,
        feedback: str,
        outcomes: list[GuardOutcome],
        iteration: int,
        tokens: int,
    ) -> None:
        """Inject iteration-guard feedback into conversation and record the step."""
        if ctx.on_iteration:
            await ctx.on_iteration(iteration, response)
        await ctx.append(self._to_message(response))
        await self._ack_response_tools(response, ctx)
        await ctx.append(Message(role=Role.USER, content=feedback))
        # Log guard feedback injection (critical for debugging stuck loops)
        self._lg.trace(
            "guard feedback injected into conversation",
            extra={
                "iteration": iteration,
                "guards_fired": [o.name for o in outcomes if not o.passed],
                "feedback_len": len(feedback),
                "feedback": self._truncate(feedback, self._TRACE_LIMIT),
                "acked_tools": [tc.name for tc in (response.tool_calls or [])],
            },
        )
        step = self._build_guard_nudge_step(response, ctx.trace_id, feedback)
        step.guards = outcomes
        if ctx.verb_trace is not None:
            ctx.verb_trace.add_step(step)
        if ctx.tracer:
            ctx.tracer.write(step)
        nudge_action = Action(
            kind=ActionType.INSTRUCT,
            reason=DecisionReason.ITERATION_GUARD,
        )
        self._score_action(ctx.acc, nudge_action, tokens)

    async def _process_iteration(
        self,
        ctx: _LoopCtx,
        response: ChatResponse,
        iteration: int,
        guard_outcomes: list[GuardOutcome] | None = None,
    ) -> tuple[Action, TaskResult | None]:
        """Process a single iteration: callback, decide, execute, trace."""
        self._check_tool_support(response)

        if ctx.on_iteration:
            await ctx.on_iteration(iteration, response)

        terminal = self._config.terminal
        obs = Observation(
            response=response,
            messages=ctx.messages,  # Controller sees full history for decisions
            iteration=iteration,
            task=ctx.task,
            tool_names=ctx.tool_names,
            terminal_tool=terminal.tool if terminal else None,
        )
        action = await ctx.ctrl.decide(obs)
        self._log_action(action)

        step = self._build_iteration_step(obs, action, response, ctx.ctrl, ctx.trace_id)
        if guard_outcomes:
            step.guards = guard_outcomes
        if ctx.verb_trace is not None:
            ctx.verb_trace.add_step(step)
        if ctx.tracer:
            ctx.tracer.write(step)

        result = await self._execute_action(action, response, ctx, iteration)
        return action, result

    def _default_controller(self) -> DefaultController:
        """Create default controller with config from this verb."""
        from ..core.config import Config

        # Controller needs a config for classifier calls (no tools).
        # Copy call options with system and temperature from current config.
        call_options = CallOptions(
            system=self._call.system,
            temperature=self._call.temperature,
        )
        llm_config = Config(
            lg=self._config.lg,
            backend=self._config.backend,
            tools=[],
            executor=None,
            call=call_options,
            terminal=None,
            warn_tool_support=self._config.warn_tool_support,
        )
        return DefaultController(
            config=ControllerConfig(
                llm_config=llm_config,
                terminal=self._config.terminal,
            ),
        )

    async def _run_iteration(
        self, messages: list[Message], config: CallOptions
    ) -> tuple[ChatResponse, int]:
        """Run one LLM iteration and return response with token count."""
        max_tokens = config.max_call_tokens if config.max_call_tokens > 0 else None
        temperature = self._resolve_temperature(config)
        response = await self._chat(messages, max_tokens, temperature)
        return response, response.input_tokens + response.output_tokens

    async def _execute_action(
        self,
        action: Action,
        response: ChatResponse,
        ctx: _LoopCtx,
        iteration: int,
    ) -> TaskResult | None:
        """Execute the action decided by the controller."""
        match action.kind:
            case ActionType.EXECUTE_TOOLS:
                await self._execute_tool_action(action, response, ctx)
                return None

            case ActionType.INSTRUCT:
                await self._add_response_if_needed(ctx, response)
                await self._ack_response_tools(response, ctx)
                if action.message:
                    await ctx.append(Message(role=Role.USER, content=action.message))
                return None

            case ActionType.SKIP:
                await self._add_response_if_needed(ctx, response)
                await self._ack_response_tools(response, ctx)
                await ctx.append(Message(role=Role.USER, content="Continue."))
                return None

            case ActionType.COMPLETE:
                await self._add_response_if_needed(ctx, response)
                await self._ack_response_tools(response, ctx)
                return self._make_result(True, action, response, ctx.messages, iteration)

            case ActionType.FAIL:
                await self._add_response_if_needed(ctx, response)
                await self._ack_response_tools(response, ctx)
                return self._make_result(False, action, response, ctx.messages, iteration)

        return None

    async def _execute_tool_action(
        self, action: Action, response: ChatResponse, ctx: _LoopCtx
    ) -> None:
        """Handle EXECUTE_TOOLS action: add response, ack skipped, execute."""
        await ctx.append(self._to_message(response))
        if response.tool_calls:
            calls = self._filter_tool_calls(response.tool_calls, action.tool_ids_to_execute)
            await self._ack_skipped_tools(
                response.tool_calls, action.tool_ids_to_execute, ctx, confirmation_pending=True
            )
            # Execute tools - pass internal list, then sync to conv
            pre_len = len(ctx.messages)
            await self._execute_tools(calls, ctx.messages)
            await self._sync_tool_results_to_conv(ctx, pre_len)

    async def _sync_tool_results_to_conv(self, ctx: _LoopCtx, pre_len: int) -> None:
        """Sync tool results from internal list to external conversation."""
        if ctx.conv is None:
            return
        for msg in ctx.messages[pre_len:]:
            await self._append_msg(ctx.conv, msg)

    def _filter_tool_calls(
        self, tool_calls: list[ToolCall], tool_ids: list[str] | None
    ) -> list[ToolCall]:
        """Filter tool calls by ID. Returns all if tool_ids is None."""
        if tool_ids is None:
            return tool_calls
        return [c for c in tool_calls if c.id in tool_ids]

    async def _ack_skipped_tools(
        self,
        all_calls: list[ToolCall],
        execute_ids: list[str] | None,
        ctx: _LoopCtx,
        *,
        confirmation_pending: bool = False,
    ) -> None:
        """Add synthetic tool results for tool calls that won't be executed.

        LLM APIs require every tool_call in an assistant message to have a
        matching tool result. When we skip executing a tool (e.g., the terminal
        tool during confirmation), we still need to provide a result.

        Args:
            confirmation_pending: If True, use "Awaiting confirmation." message
                (for terminal tool confirmation flow). Otherwise use neutral
                "Acknowledged." (for COMPLETE/FAIL/INSTRUCT/SKIP paths).
        """
        if execute_ids is None:
            return
        content = (
            "Acknowledged. Awaiting confirmation." if confirmation_pending else "Acknowledged."
        )
        skip_ids = {c.id for c in all_calls} - set(execute_ids)
        for call in all_calls:
            if call.id in skip_ids:
                await ctx.append(Message(role=Role.TOOL, content=content, tool_call_id=call.id))

    async def _ack_response_tools(self, response: ChatResponse, ctx: _LoopCtx) -> None:
        """Acknowledge all tool_calls in a response that won't be executed.

        Must be called after _add_response_if_needed for INSTRUCT/SKIP/COMPLETE/FAIL
        paths where the assistant message contains tool_calls but no tools are executed.
        """
        if response.tool_calls:
            await self._ack_skipped_tools(response.tool_calls, [], ctx)

    async def _add_response_if_needed(self, ctx: _LoopCtx, response: ChatResponse) -> None:
        """Add response to messages if not already added."""
        if ctx.messages:
            last = ctx.messages[-1]
            if (
                last.role == Role.ASSISTANT
                and last.content == response.content
                and last.tool_calls == (response.tool_calls or None)
            ):
                return
        await ctx.append(self._to_message(response))

    def _make_result(
        self,
        completed: bool,
        action: Action,
        response: ChatResponse,
        messages: list[Message],
        iteration: int,
    ) -> TaskResult:
        """Build a TaskResult from action and response."""
        return TaskResult(
            completed=completed,
            output=action.output or response.content,
            iterations=iteration + 1,
            history=messages,
            terminal_data=action.terminal_data,
            terminal_tool=action.terminal_tool,
        )

    @staticmethod
    def _tag_result(result: TaskResult, trace: VerbTrace) -> TaskResult:
        """Attach VerbTrace to a TaskResult."""
        result.trace = trace
        return result

    def _log_action(self, action: Action) -> None:
        """Log the controller's decision."""
        self._lg.debug(
            "controller_action",
            extra={"kind": action.kind.value, "reason": action.reason.value},
        )
        # Detailed trace with all action fields
        self._lg.trace(
            "controller decision details",
            extra={
                "action": action.kind.value,
                "reason": action.reason.value,
                "nudge": action.message[:200] if action.message else None,
                "output": action.output[:200] if action.output else None,
                "terminal_tool": action.terminal_tool,
                "terminal_data": action.terminal_data,
                "tool_ids": action.tool_ids_to_execute,
            },
        )

    # --- Trace helpers ---

    def _build_iteration_step(
        self,
        obs: Observation,
        action: Action,
        response: ChatResponse,
        ctrl: LoopController,
        trace_id: str,
    ) -> Step:
        """Build a Step from Complete verb iteration state."""
        step = Step(
            phase="iteration",
            ts=time.time(),
            trace_id=trace_id,
            verb="Complete",
            llm_call=LLMCall(
                call_id=response.call_id,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                finish_reason=response.finish_reason,
                model=response.model,
            ),
            tools=[ToolOutcome(name=tc.name, call_id=tc.id) for tc in (response.tool_calls or [])],
            action=action.kind.value,
            reason=action.reason.value,
            nudge_preview=action.message[:200] if action.message else None,
            classifier_called=action.reason
            in (DecisionReason.CLASSIFIED_COMPLETE, DecisionReason.NUDGE_CLASSIFIED),
        )
        self._attach_controller_internals(step, obs, ctrl)
        return step

    @staticmethod
    def _build_guard_nudge_step(response: ChatResponse, trace_id: str, feedback: str) -> Step:
        """Build a Step for an iteration guard nudge."""
        return Step(
            phase="iteration",
            ts=time.time(),
            trace_id=trace_id,
            verb="Complete",
            llm_call=LLMCall(
                call_id=response.call_id,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                finish_reason=response.finish_reason,
                model=response.model,
            ),
            tools=[ToolOutcome(name=tc.name, call_id=tc.id) for tc in (response.tool_calls or [])],
            action=ActionType.INSTRUCT.value,
            reason=DecisionReason.ITERATION_GUARD.value,
            nudge_preview=feedback[:200] if feedback else None,
        )

    @staticmethod
    def _attach_controller_internals(step: Step, obs: Observation, ctrl: LoopController) -> None:
        """Attach DefaultController internals to a Step, if available."""
        if isinstance(ctrl, DefaultController):
            step.iterations_since_nudge = obs.iteration - ctrl.iterations_since_last_nudge
            step.consecutive_degenerate = ctrl.consecutive_degenerate
            step.pending_terminal = ctrl.has_pending_terminal

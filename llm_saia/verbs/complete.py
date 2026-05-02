"""COMPLETE verb: Execute a task with tool calling and completion confirmation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from ..core.backend import ChatResponse
from ..core.config import CallOptions
from ..core.controller import ControllerConfig, DefaultController, LoopController
from ..core.conversation import ConversationLike, Message, Role
from ..core.loop import ControllerStrategy, ControllerStrategyConfig, CoreLoopResult, LoopDecision
from ..core.trace import GuardOutcome, LLMCall, Step, ToolOutcome, Tracer, VerbTrace
from ..core.types import LoopScore, TaskResult
from ..core.verb import Verb

# Default call options for complete (unlimited iterations)
DEFAULT_COMPLETE_CALL = CallOptions(max_iterations=0)


class Complete(Verb):
    """Execute a task with tool calling and completion confirmation."""

    def _validate_call(self, resume: bool, conversation: ConversationLike | None) -> None:
        """Validate Complete call parameters."""
        if not self._has_tools():
            raise ValueError("Complete requires tools and executor to be configured.")
        if resume and conversation is None:
            raise ValueError("conversation is required when resume=True")

    async def __call__(
        self,
        task: str,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None = None,
        controller: LoopController | None = None,
        conversation: ConversationLike | None = None,
        resume: bool = False,
        pause_check: Callable[[], Awaitable[bool]] | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> TaskResult:
        """Execute a task using tools until completion or limit reached.

        Args:
            task: The task description / prompt (ignored when resuming).
            on_iteration: Optional async callback invoked each iteration. May raise
                ``PauseRequested`` to exit the loop early with ``paused=True``.
            controller: Custom loop controller (uses default if None).
            conversation: Optional external conversation for message management.
                If provided, messages are appended to both an internal history
                (returned in ``TaskResult.history``) and this conversation.
                The LLM sees ``conversation.as_messages()`` which may be compacted.
                Required when resuming.
            resume: If True, continue from existing conversation state instead of
                starting fresh. The conversation must contain the prior history.
            pause_check: Optional async callback checked between tool calls in a
                batch. Return True to pause after the current tool completes.
                Remaining tools are acknowledged but not executed.
            abort_signal: Optional event for fast abort during LLM streaming.
                Set this event to abort the current LLM call within ~100ms
                (requires backend support). Raises ``PauseRequested`` on abort.
        """
        self._validate_call(resume, conversation)
        verb_trace = self._init_verb_trace()
        ctrl = controller or self._default_controller()
        ctrl.reset()
        tid, rid = verb_trace.trace_id, verb_trace.request_id
        tracer = self._resolve_tracer({"trace_id": tid, "request_id": rid, "task": task[:200]})
        reason: str | None = None
        try:
            result = await self._run_loop(
                task,
                verb_trace.trace_id,
                ctrl,
                tracer,
                on_iteration,
                verb_trace,
                conversation,
                resume,
                pause_check,
                abort_signal,
            )
            reason = self._result_reason(result)
            return self._tag_result(result, verb_trace)
        finally:
            self._emit_verb_trace(verb_trace, reason)

    @staticmethod
    def _result_reason(result: TaskResult) -> str:
        """Get completion reason from TaskResult."""
        return result.reason

    @staticmethod
    def _build_score(iters: int, total_tokens: int, acc: list[int]) -> LoopScore:
        """Build LoopScore from accumulated stats."""
        return LoopScore(iters, acc[0], acc[1], acc[2], total_tokens, acc[3])

    async def _run_loop(
        self,
        task: str,
        trace_id: str,
        ctrl: LoopController,
        tracer: Tracer | None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None,
        verb_trace: VerbTrace | None = None,
        conversation: ConversationLike | None = None,
        resume: bool = False,
        pause_check: Callable[[], Awaitable[bool]] | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> TaskResult:
        """Execute the main tool-calling loop using unified core loop."""
        call_options = self._config.call or DEFAULT_COMPLETE_CALL
        messages = await self._init_messages(task, conversation, resume)
        acc, strategy = self._create_strategy(task, ctrl)

        def on_decide(
            response: ChatResponse,
            decision: LoopDecision,
            iteration: int,
            outcomes: list[GuardOutcome],
        ) -> None:
            self._record_decision(
                response, decision, iteration, outcomes, ctrl, trace_id, verb_trace, tracer
            )

        self._log_loop_start(call_options, abort_signal is not None, trace_id)
        result = await self._core_loop(
            messages=messages,
            config=call_options,
            strategy=strategy,
            conv=conversation,
            abort_signal=abort_signal,
            pause_check=pause_check,
            on_iteration=on_iteration,
            on_decide=on_decide,
        )
        return self._core_result_to_task_result(result, messages, acc)

    async def _init_messages(
        self, task: str, conversation: ConversationLike | None, resume: bool
    ) -> list[Message]:
        """Initialize message list for loop."""
        if resume and conversation is not None:
            return list(conversation.as_messages())
        initial_msg = Message(role=Role.USER, content=task)
        if conversation is not None:
            await self._append_msg(conversation, initial_msg)
        return [initial_msg]

    def _create_strategy(
        self, task: str, ctrl: LoopController
    ) -> tuple[list[int], ControllerStrategy]:
        """Create strategy with scoring accumulator."""
        tool_names = [t.name for t in (self._config.tools or [])]
        terminal = self._config.terminal
        acc: list[int] = [0, 0, 0, 0]
        strategy = ControllerStrategy(
            ctrl,
            ControllerStrategyConfig(
                task=task,
                tool_names=tool_names,
                terminal_tool=terminal.tool if terminal else None,
                acc=acc,
            ),
        )
        return acc, strategy

    def _record_decision(
        self,
        response: ChatResponse,
        decision: LoopDecision,
        iteration: int,
        outcomes: list[GuardOutcome],
        ctrl: LoopController,
        trace_id: str,
        verb_trace: VerbTrace | None,
        tracer: Tracer | None,
    ) -> None:
        """Record decision step to trace."""
        step = self._build_step_from_decision(response, decision, ctrl, trace_id, iteration)
        if outcomes:
            step.guards = outcomes
        if verb_trace is not None:
            verb_trace.add_step(step)
        if tracer:
            tracer.write(step)

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

    @staticmethod
    def _tag_result(result: TaskResult, trace: VerbTrace) -> TaskResult:
        """Attach VerbTrace to a TaskResult."""
        result.trace = trace
        return result

    def _build_step_from_decision(
        self,
        response: ChatResponse,
        decision: LoopDecision,
        ctrl: LoopController,
        trace_id: str,
        iteration: int,
    ) -> Step:
        """Build a Step from LoopDecision (used by unified loop)."""
        from ..core.loop import LoopAction

        action_map = {
            LoopAction.EXECUTE_TOOLS: "execute_tools",
            LoopAction.INSTRUCT: "instruct",
            LoopAction.SKIP: "skip",
            LoopAction.COMPLETE: "complete",
            LoopAction.FAIL: "fail",
        }
        step = Step(
            phase="iteration",
            ts=time.time(),
            trace_id=trace_id,
            verb="Complete",
            llm_call=self._response_to_llm_call(response),
            tools=[ToolOutcome(name=tc.name, call_id=tc.id) for tc in (response.tool_calls or [])],
            action=action_map.get(decision.action, "unknown"),
            reason=decision.reason,
            nudge_preview=decision.message[:200] if decision.message else None,
            classifier_called=decision.reason in ("classified_complete", "nudge_classified"),
        )
        if isinstance(ctrl, DefaultController):
            step.iterations_since_nudge = iteration - ctrl.iterations_since_last_nudge
            step.consecutive_degenerate = ctrl.consecutive_degenerate
            step.pending_terminal = ctrl.has_pending_terminal
        return step

    def _response_to_llm_call(self, response: ChatResponse) -> LLMCall:
        return LLMCall(
            call_id=response.call_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            finish_reason=response.finish_reason,
            model=response.model,
        )

    def _core_result_to_task_result(
        self,
        result: CoreLoopResult,
        messages: list[Message],
        acc: list[int],
    ) -> TaskResult:
        """Convert CoreLoopResult to TaskResult with scoring."""
        task_result = TaskResult(
            completed=result.completed,
            output=result.output,
            iterations=result.iterations,
            history=messages,
            reason=result.reason,
            paused=result.paused,
            terminal_data=result.terminal_data,
            terminal_tool=result.terminal_tool,
        )
        task_result.score = self._build_score(result.iterations, result.total_tokens, acc)
        return task_result

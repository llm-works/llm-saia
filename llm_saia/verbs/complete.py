"""COMPLETE verb: Execute a task with tool calling and completion confirmation."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from llm_saia.core.backend import AgentResponse, Message, ToolCall
from llm_saia.core.config import RunConfig
from llm_saia.core.controller import (
    Action,
    ActionType,
    ControllerConfig,
    DefaultController,
    LoopController,
    Observation,
)
from llm_saia.core.trace import Tracer, build_trace
from llm_saia.core.types import DecisionReason, LoopScore, TaskResult
from llm_saia.core.verb import Verb

# Default run config for complete (unlimited iterations)
DEFAULT_COMPLETE_RUN = RunConfig(max_iterations=0)

# Reasons that count as productive despite being INSTRUCT
_PRODUCTIVE_INSTRUCT_REASONS = frozenset({DecisionReason.TERMINAL_CONFIRMATION_REQUEST})


@dataclass
class _LoopCtx:
    """Mutable loop context bundling iteration state and scoring."""

    task: str
    trace_id: str
    ctrl: LoopController
    tracer: Tracer | None
    on_iteration: Callable[[int, AgentResponse], Awaitable[None]] | None
    run_config: RunConfig
    messages: list[Message]
    tool_names: list[str]
    acc: list[int] = field(default_factory=lambda: [0, 0, 0, 0])


class Complete(Verb):
    """Execute a task with tool calling and completion confirmation."""

    async def __call__(
        self,
        task: str,
        on_iteration: Callable[[int, AgentResponse], Awaitable[None]] | None = None,
        controller: LoopController | None = None,
        tracer: Tracer | None = None,
    ) -> TaskResult:
        """Execute a task using tools until completion or limit reached.

        Args:
            task: The task description / prompt.
            on_iteration: Optional async callback invoked each iteration.
            controller: Custom loop controller (uses default if None).
            tracer: Per-call tracer (closed on completion). Falls back to
                config tracer if not provided.
        """
        if not self._has_tools():
            raise ValueError("Complete requires tools and executor to be configured.")

        trace_id = self._generate_id()
        request_id = self._config.request_id
        ctrl = controller or self._default_controller()
        ctrl.reset()

        owns_tracer, active_tracer = self._resolve_tracer(
            tracer,
            {"trace_id": trace_id, "request_id": request_id, "task": task[:200]},
        )

        try:
            result = await self._run_loop(task, trace_id, ctrl, active_tracer, on_iteration)
            return self._tag_result(result, trace_id, request_id)
        finally:
            if active_tracer and owns_tracer:
                active_tracer.close()

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

    async def _run_loop(
        self,
        task: str,
        trace_id: str,
        ctrl: LoopController,
        tracer: Tracer | None,
        on_iteration: Callable[[int, AgentResponse], Awaitable[None]] | None,
    ) -> TaskResult:
        """Execute the main tool-calling loop."""
        run_config = self._config.run or DEFAULT_COMPLETE_RUN
        ctx = _LoopCtx(
            task=task,
            trace_id=trace_id,
            ctrl=ctrl,
            tracer=tracer,
            on_iteration=on_iteration,
            run_config=run_config,
            messages=[Message(role="user", content=task)],
            tool_names=[t.name for t in (self._config.tools or [])],
        )
        self._log_loop_start(run_config)
        start_time, iteration, total_tokens, last_content = time.monotonic(), 0, 0, ""

        while not self._should_stop(run_config, iteration, start_time, total_tokens):
            result, tokens, last_content = await self._run_one_iteration(ctx, iteration)
            total_tokens += tokens
            if result:
                self._log_loop_complete(iteration, start_time, total_tokens, result.output or "")
                result.score = self._build_score(iteration + 1, total_tokens, ctx.acc)
                return result
            iteration += 1

        self._log_limit_reached(run_config, iteration, start_time, total_tokens)
        result = TaskResult(False, last_content, iteration, ctx.messages)
        result.score = self._build_score(iteration, total_tokens, ctx.acc)
        return result

    async def _run_one_iteration(
        self,
        ctx: _LoopCtx,
        iteration: int,
    ) -> tuple[TaskResult | None, int, str]:
        """Run one loop iteration. Returns (result, tokens, last_content)."""
        response, tokens = await self._run_iteration(ctx.messages, ctx.run_config)
        self._log_response(response, iteration, tokens)
        action, result = await self._process_iteration(
            response,
            ctx.messages,
            iteration,
            ctx.task,
            ctx.tool_names,
            ctx.ctrl,
            ctx.on_iteration,
            ctx.tracer,
            ctx.trace_id,
        )
        self._score_action(ctx.acc, action, tokens)
        return result, tokens, response.content

    async def _process_iteration(
        self,
        response: AgentResponse,
        messages: list[Message],
        iteration: int,
        task: str,
        tool_names: list[str],
        ctrl: LoopController,
        on_iteration: Callable[[int, AgentResponse], Awaitable[None]] | None,
        tracer: Tracer | None = None,
        trace_id: str = "",
    ) -> tuple[Action, TaskResult | None]:
        """Process a single iteration: callback, decide, execute, trace."""
        self._check_tool_support(response)

        if on_iteration:
            await on_iteration(iteration, response)

        terminal = self._config.terminal
        obs = Observation(
            response=response,
            messages=messages,
            iteration=iteration,
            task=task,
            tool_names=tool_names,
            terminal_tool=terminal.tool if terminal else None,
        )
        action = await ctrl.decide(obs)
        self._log_action(action)

        if tracer:
            self._write_trace(tracer, obs, action, response, ctrl, trace_id)

        result = await self._execute_action(action, response, messages, iteration)
        return action, result

    def _default_controller(self) -> DefaultController:
        """Create default controller with config from this verb."""
        from llm_saia.core.config import Config

        # Controller needs a config for classifier calls (no tools)
        llm_config = Config(
            backend=self._config.backend,
            tools=[],
            executor=None,
            system=self._config.system,
            run=None,
            terminal=None,
            lg=self._config.lg,
            warn_tool_support=self._config.warn_tool_support,
            temperature=self._config.temperature,
        )
        run = self._config.run or DEFAULT_COMPLETE_RUN
        return DefaultController(
            config=ControllerConfig(
                llm_config=llm_config,
                terminal=self._config.terminal,
                max_failure_retries=run.max_retries,
            ),
        )

    async def _run_iteration(
        self, messages: list[Message], config: RunConfig
    ) -> tuple[AgentResponse, int]:
        """Run one LLM iteration and return response with token count."""
        max_tokens = config.max_call_tokens if config.max_call_tokens > 0 else None
        temperature = self._resolve_temperature(config)
        response = await self._chat(messages, max_tokens, temperature)
        return response, response.input_tokens + response.output_tokens

    async def _execute_action(
        self,
        action: Action,
        response: AgentResponse,
        messages: list[Message],
        iteration: int,
    ) -> TaskResult | None:
        """Execute the action decided by the controller."""
        match action.kind:
            case ActionType.EXECUTE_TOOLS:
                await self._execute_tool_action(action, response, messages)
                return None

            case ActionType.INSTRUCT:
                self._add_response_if_needed(messages, response)
                self._ack_response_tools(response, messages)
                if action.message:
                    messages.append(Message(role="user", content=action.message))
                return None

            case ActionType.SKIP:
                self._add_response_if_needed(messages, response)
                self._ack_response_tools(response, messages)
                messages.append(Message(role="user", content="Continue."))
                return None

            case ActionType.COMPLETE:
                self._add_response_if_needed(messages, response)
                return self._make_result(True, action, response, messages, iteration)

            case ActionType.FAIL:
                self._add_response_if_needed(messages, response)
                return self._make_result(False, action, response, messages, iteration)

        return None

    async def _execute_tool_action(
        self, action: Action, response: AgentResponse, messages: list[Message]
    ) -> None:
        """Handle EXECUTE_TOOLS action: add response, ack skipped, execute."""
        messages.append(self._to_message(response))
        if response.tool_calls:
            calls = self._filter_tool_calls(response.tool_calls, action.tool_ids_to_execute)
            self._ack_skipped_tools(response.tool_calls, action.tool_ids_to_execute, messages)
            await self._execute_tools(calls, messages)

    def _filter_tool_calls(
        self, tool_calls: list[ToolCall], tool_ids: list[str] | None
    ) -> list[ToolCall]:
        """Filter tool calls by ID. Returns all if tool_ids is None."""
        if tool_ids is None:
            return tool_calls
        return [c for c in tool_calls if c.id in tool_ids]

    def _ack_skipped_tools(
        self,
        all_calls: list[ToolCall],
        execute_ids: list[str] | None,
        messages: list[Message],
    ) -> None:
        """Add synthetic tool_results for tool calls that won't be executed.

        LLM APIs require every tool_call in an assistant message to have a
        matching tool_result. When we skip executing a tool (e.g., the terminal
        tool during confirmation), we still need to provide a result.
        """
        if execute_ids is None:
            return
        skip_ids = {c.id for c in all_calls} - set(execute_ids)
        for call in all_calls:
            if call.id in skip_ids:
                messages.append(
                    Message(
                        role="tool_result",
                        content="Acknowledged. Awaiting confirmation.",
                        tool_call_id=call.id,
                    )
                )

    def _ack_response_tools(self, response: AgentResponse, messages: list[Message]) -> None:
        """Acknowledge all tool_calls in a response that won't be executed.

        Must be called after _add_response_if_needed for INSTRUCT/SKIP paths
        where the assistant message contains tool_calls but no tools are executed.
        """
        if response.tool_calls:
            self._ack_skipped_tools(response.tool_calls, [], messages)

    def _add_response_if_needed(self, messages: list[Message], response: AgentResponse) -> None:
        """Add response to messages if not already added."""
        if messages:
            last = messages[-1]
            if (
                last.role == "assistant"
                and last.content == response.content
                and last.tool_calls == (response.tool_calls or None)
            ):
                return
        messages.append(self._to_message(response))

    def _make_result(
        self,
        completed: bool,
        action: Action,
        response: AgentResponse,
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
    def _tag_result(result: TaskResult, trace_id: str, request_id: str | None) -> TaskResult:
        """Attach tracing IDs to a TaskResult."""
        result.trace_id = trace_id
        result.request_id = request_id
        return result

    def _log_action(self, action: Action) -> None:
        """Log the controller's decision."""
        if self._lg:
            self._lg.debug(
                "controller_action",
                extra={"kind": action.kind.value, "reason": action.reason.value},
            )

    # --- Trace helpers ---

    def _write_trace(
        self,
        tracer: Tracer,
        obs: Observation,
        action: Action,
        response: AgentResponse,
        ctrl: LoopController,
        trace_id: str,
    ) -> None:
        """Write one iteration trace record."""
        # Extract controller internals if available (DefaultController exposes these)
        iterations_since_nudge = None
        consecutive_degenerate = None
        pending_terminal = None
        if isinstance(ctrl, DefaultController):
            iterations_since_nudge = obs.iteration - ctrl.iterations_since_last_nudge
            consecutive_degenerate = ctrl.consecutive_degenerate
            pending_terminal = ctrl.has_pending_terminal

        # Detect if classifier was called
        classifier_called = action.reason in (
            DecisionReason.CLASSIFIED_COMPLETE,
            DecisionReason.NUDGE_CLASSIFIED,
        )

        record = build_trace(
            obs,
            action,
            response,
            trace_id=trace_id,
            verb="Complete",
            phase="loop",
            request_id=self._config.request_id,
            classifier_called=classifier_called,
            iterations_since_nudge=iterations_since_nudge,
            consecutive_degenerate=consecutive_degenerate,
            pending_terminal=pending_terminal,
        )
        tracer.write(record)

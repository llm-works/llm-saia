"""COMPLETE verb: Execute a task with tool calling and completion confirmation."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from llm_saia.core.backend import AgentResponse, Message, ToolCall
from llm_saia.core.config import RunConfig
from llm_saia.core.controller import (
    Action,
    ActionKind,
    ControllerConfig,
    DefaultController,
    LoopController,
    Observation,
)
from llm_saia.core.types import TaskResult
from llm_saia.core.verb import Verb

# Default run config for complete (unlimited iterations)
DEFAULT_COMPLETE_RUN = RunConfig(max_iterations=0)


class Complete(Verb):
    """Execute a task with tool calling and completion confirmation."""

    async def __call__(
        self,
        task: str,
        on_iteration: Callable[[int, AgentResponse], Awaitable[None]] | None = None,
        controller: LoopController | None = None,
    ) -> TaskResult:
        """Execute a task using tools until completion or limit reached."""
        if not self._has_tools():
            raise ValueError("Complete requires tools and executor to be configured.")

        ctrl = controller or self._default_controller()
        ctrl.reset()

        run_config = self._config.run or DEFAULT_COMPLETE_RUN
        messages: list[Message] = [Message(role="user", content=task)]
        start_time, iteration, total_tokens, last_content = time.monotonic(), 0, 0, ""
        tool_names = [t.name for t in (self._config.tools or [])]
        self._log_loop_start(run_config)

        while not self._should_stop(run_config, iteration, start_time, total_tokens):
            response, tokens = await self._run_iteration(messages, run_config)
            total_tokens += tokens
            last_content = response.content

            result = await self._process_iteration(
                response, messages, iteration, total_tokens, task, tool_names, ctrl, on_iteration
            )
            if result:
                self._log_loop_complete(iteration, start_time, total_tokens, result.output or "")
                return result
            iteration += 1

        self._log_limit_reached(run_config, iteration, start_time, total_tokens)
        return TaskResult(False, last_content, iteration, messages)

    async def _process_iteration(
        self,
        response: AgentResponse,
        messages: list[Message],
        iteration: int,
        total_tokens: int,
        task: str,
        tool_names: list[str],
        ctrl: LoopController,
        on_iteration: Callable[[int, AgentResponse], Awaitable[None]] | None,
    ) -> TaskResult | None:
        """Process a single iteration: log, callback, decide, execute."""
        self._log_response(response, iteration, total_tokens)
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

        return await self._execute_action(action, response, messages, iteration)

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
        response = await self._chat(messages, max_tokens)
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
            case ActionKind.EXECUTE_TOOLS:
                await self._execute_tool_action(action, response, messages)
                return None

            case ActionKind.INSTRUCT:
                self._add_response_if_needed(messages, response)
                if action.message:
                    messages.append(Message(role="user", content=action.message))
                return None

            case ActionKind.SKIP:
                self._add_response_if_needed(messages, response)
                messages.append(Message(role="user", content="Continue."))
                return None

            case ActionKind.COMPLETE:
                self._add_response_if_needed(messages, response)
                return self._make_result(True, action, response, messages, iteration)

            case ActionKind.FAIL:
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
            terminal_data=action.terminal_data if completed else None,
            terminal_tool=action.terminal_tool if completed else None,
        )

    def _log_action(self, action: Action) -> None:
        """Log the controller's decision."""
        if self._lg:
            self._lg.debug(
                "controller_action",
                extra={"kind": action.kind.value, "reason": action.reason},
            )

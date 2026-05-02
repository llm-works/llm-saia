"""Unified loop infrastructure with pluggable strategies.

Provides core loop mechanics shared by all verbs, with decision-making
delegated to LoopStrategy implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .backend import ChatResponse
    from .conversation import Message


class LoopAction(Enum):
    """Actions the loop can take after a decision."""

    EXECUTE_TOOLS = "execute_tools"
    INSTRUCT = "instruct"
    SKIP = "skip"
    COMPLETE = "complete"
    FAIL = "fail"


@dataclass
class LoopDecision:
    """Decision from LoopStrategy."""

    action: LoopAction
    message: str | None = None  # For INSTRUCT
    output: str | None = None  # For COMPLETE/FAIL
    tool_ids: list[str] | None = None  # Which tools to execute (None = all)
    terminal_data: dict[str, Any] | None = None
    terminal_tool: str | None = None
    reason: str = ""


@dataclass
class CoreLoopResult:
    """Result from the core loop."""

    completed: bool
    output: str
    iterations: int
    messages: list[Message]
    reason: str  # "completed", "failed", "paused", "limit_reached"
    paused: bool = False
    terminal_data: dict[str, Any] | None = None
    terminal_tool: str | None = None
    total_tokens: int = 0


@runtime_checkable
class LoopStrategy(Protocol):
    """Strategy for loop decision-making."""

    async def decide(
        self,
        response: ChatResponse,
        messages: list[Message],
        iteration: int,
        blocking_feedback: str | None,
        advisory_feedback: str | None,
    ) -> LoopDecision:
        """Decide what to do after an LLM response."""
        ...

    def on_iteration_complete(self, decision: LoopDecision, tokens: int) -> None:
        """Called after each iteration for scoring/tracing."""
        ...


class SimpleStrategy:
    """Simple strategy: execute tools if present, else complete."""

    async def decide(
        self,
        response: ChatResponse,
        messages: list[Message],
        iteration: int,
        blocking_feedback: str | None,
        advisory_feedback: str | None,
    ) -> LoopDecision:
        if blocking_feedback:
            return LoopDecision(
                action=LoopAction.INSTRUCT, message=blocking_feedback, reason="blocking_guard"
            )
        if response.tool_calls:
            return LoopDecision(action=LoopAction.EXECUTE_TOOLS, reason="has_tool_calls")
        return LoopDecision(action=LoopAction.COMPLETE, output=response.content, reason="no_tools")

    def on_iteration_complete(self, decision: LoopDecision, tokens: int) -> None:
        pass


@dataclass
class ControllerStrategyConfig:
    """Configuration for ControllerStrategy."""

    task: str
    tool_names: list[str]
    terminal_tool: str | None = None
    acc: list[int] = field(default_factory=lambda: [0, 0, 0, 0])


class ControllerStrategy:
    """Strategy using LoopController for decisions."""

    def __init__(self, controller: Any, config: ControllerStrategyConfig):
        self.controller = controller
        self.config = config

    async def decide(
        self,
        response: ChatResponse,
        messages: list[Message],
        iteration: int,
        blocking_feedback: str | None,
        advisory_feedback: str | None,
    ) -> LoopDecision:
        if blocking_feedback:
            return LoopDecision(
                action=LoopAction.INSTRUCT, message=blocking_feedback, reason="iteration_guard"
            )
        from .controller import Observation

        obs = Observation(
            response=response,
            messages=messages,
            iteration=iteration,
            task=self.config.task,
            tool_names=self.config.tool_names,
            terminal_tool=self.config.terminal_tool,
        )
        action = await self.controller.decide(obs)
        return self._to_decision(action)

    def _to_decision(self, action: Any) -> LoopDecision:
        from .controller import ActionType

        reason = action.reason.value if action.reason else ""
        match action.kind:
            case ActionType.EXECUTE_TOOLS:
                return LoopDecision(
                    action=LoopAction.EXECUTE_TOOLS,
                    tool_ids=action.tool_ids_to_execute,
                    reason=reason,
                )
            case ActionType.INSTRUCT:
                return LoopDecision(
                    action=LoopAction.INSTRUCT, message=action.message or "Continue.", reason=reason
                )
            case ActionType.SKIP:
                return LoopDecision(action=LoopAction.SKIP, reason=reason)
            case ActionType.COMPLETE:
                return self._terminal_decision(LoopAction.COMPLETE, action, reason)
            case ActionType.FAIL:
                return self._terminal_decision(LoopAction.FAIL, action, reason)
        return LoopDecision(action=LoopAction.COMPLETE, output="", reason="unknown")

    def _terminal_decision(self, loop_action: LoopAction, action: Any, reason: str) -> LoopDecision:
        return LoopDecision(
            action=loop_action,
            output=action.output,
            terminal_data=action.terminal_data,
            terminal_tool=action.terminal_tool,
            reason=reason,
        )

    def on_iteration_complete(self, decision: LoopDecision, tokens: int) -> None:
        acc = self.config.acc
        if decision.action in (LoopAction.COMPLETE, LoopAction.FAIL, LoopAction.EXECUTE_TOOLS):
            acc[0] += 1
        elif decision.action == LoopAction.INSTRUCT:
            if decision.reason == "terminal_confirmation_request":
                acc[0] += 1
            else:
                acc[1] += 1
                acc[3] += tokens
        elif decision.action == LoopAction.SKIP:
            acc[2] += 1
            acc[3] += tokens

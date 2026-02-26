"""Loop controllers for the Complete verb.

Controllers decide what action to take at each iteration of the agent loop.
They observe the current state and return an action (execute tools, instruct,
skip, complete, or fail).

Different controllers can be used for different models - weaker models may
need gentler, less frequent nudges while stronger models can be pushed harder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from llm_saia.core.classifier import LLMTaskStateClassifier, TaskState
from llm_saia.core.types import DecisionReason

if TYPE_CHECKING:
    from llm_saia.core.backend import AgentResponse, Message, ToolCall
    from llm_saia.core.config import Config, TerminalConfig


@dataclass
class Observation:
    """What the controller observes at each iteration."""

    response: AgentResponse
    messages: list[Message]
    iteration: int
    task: str
    tool_names: list[str]
    terminal_tool: str | None


class ActionType(Enum):
    """Types of actions the controller can decide."""

    EXECUTE_TOOLS = "execute_tools"  # Normal flow - execute tool calls
    INSTRUCT = "instruct"  # Send a nudge/instruction message
    SKIP = "skip"  # Do nothing, let the loop continue
    COMPLETE = "complete"  # Task is done
    FAIL = "fail"  # Give up


@dataclass
class Action:
    """Action decided by the controller."""

    kind: ActionType
    reason: DecisionReason  # Why this action was chosen
    message: str | None = None  # For INSTRUCT
    output: str | None = None  # For COMPLETE/FAIL
    terminal_data: Any = None  # For COMPLETE via terminal tool
    terminal_tool: str | None = None  # Name of terminal tool if used
    tool_ids_to_execute: list[str] | None = None  # For EXECUTE_TOOLS - None means all
    reason_details: str | None = None  # Additional context (e.g., backoff count, classifier output)


class LoopController(Protocol):
    """Protocol for loop controllers.

    Controllers observe the loop state and decide what action to take.
    This separates the decision logic from the loop mechanics.
    """

    async def decide(self, obs: Observation) -> Action:
        """Decide what action to take based on observation.

        Args:
            obs: Current observation (response, messages, iteration, etc.)

        Returns:
            Action to take (execute tools, instruct, skip, complete, fail)
        """
        ...

    def reset(self) -> None:
        """Reset controller state for a new task."""
        ...


@dataclass
class ControllerConfig:
    """Configuration for the default controller."""

    llm_config: Config  # Config for classifier LLM calls
    terminal: TerminalConfig | None = None  # Terminal tool configuration
    backoff_iterations: int = 3  # Iterations to wait after nudging
    max_confirmation_retries: int = 3  # Max retries for terminal tool confirmation
    max_failure_retries: int = 1  # Max retries when terminal tool indicates failure


# Default nudge messages for each state
DEFAULT_NUDGES: dict[TaskState, str] = {
    TaskState.STUCK: (
        "You indicated you're stuck, but please continue working on the task. "
        "Try a different approach if needed. Use the available tools to proceed."
    ),
    TaskState.WANTS_CONTINUE: (
        "You indicated you want to continue but didn't use any tools. "
        "Please use the available tools to proceed with the task."
    ),
    TaskState.ASKING: "Yes, please proceed with the task using the available tools.",
}

DEFAULT_NUDGE_FALLBACK = "Please continue working on the task using the available tools."

# Nudges for degenerate states (bypass backoff and classifier)
_EMPTY_RESPONSE_NUDGE = (
    "Your last response was empty. Please use the available tools to continue working on the task."
)
_TEXT_TOOL_NUDGE = (
    "You appear to be writing tool calls as text instead of actually calling them. "
    "Please use the available tools directly - do not write tool names in your response text."
)


@dataclass
class DefaultController:
    """Default controller with backoff and terminal tool support.

    This controller implements:
    - Terminal tool confirmation flow
    - State classification via LLM
    - Nudge backoff to avoid overwhelming the model
    - Contradiction detection in confirmations
    """

    config: ControllerConfig
    nudges: dict[TaskState, str] = field(default_factory=lambda: DEFAULT_NUDGES.copy())

    # Internal state
    _classifier: LLMTaskStateClassifier = field(init=False, repr=False)
    _last_nudge_iteration: int = field(default=-100, init=False, repr=False)
    _pending_terminal: ToolCall | None = field(default=None, init=False, repr=False)
    _confirmation_retries: int = field(default=0, init=False, repr=False)
    _failure_retries: int = field(default=0, init=False, repr=False)
    _consecutive_degenerate: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._classifier = LLMTaskStateClassifier(self.config.llm_config)

    def reset(self) -> None:
        """Reset state for a new task."""
        self._last_nudge_iteration = -self.config.backoff_iterations
        self._pending_terminal = None
        self._confirmation_retries = 0
        self._failure_retries = 0
        self._consecutive_degenerate = 0

    @property
    def iterations_since_last_nudge(self) -> int:
        """Number of iterations since the last nudge was sent."""
        return self._last_nudge_iteration

    @property
    def consecutive_degenerate(self) -> int:
        """Number of consecutive degenerate responses (empty or text-tool patterns)."""
        return self._consecutive_degenerate

    @property
    def has_pending_terminal(self) -> bool:
        """Whether a terminal tool call is pending confirmation."""
        return self._pending_terminal is not None

    async def decide(self, obs: Observation) -> Action:
        """Decide what action to take."""
        # 1. Check for terminal tool
        terminal_action = self._check_terminal_tool(obs)
        if terminal_action:
            return terminal_action

        # 2. Has tool calls → execute them (reset degenerate counter)
        if obs.response.tool_calls:
            self._consecutive_degenerate = 0
            return Action(ActionType.EXECUTE_TOOLS, DecisionReason.HAS_TOOL_CALLS)

        # 3. No tool calls → classify and decide
        return await self._handle_no_tools(obs)

    def _check_terminal_tool(self, obs: Observation) -> Action | None:
        """Handle terminal tool flow (first call, confirmation, etc.)."""
        if not obs.terminal_tool:
            return None

        terminal_call = self._find_terminal_call(obs.response.tool_calls, obs.terminal_tool)

        # Check if this is a confirmation of pending terminal
        if self._pending_terminal and terminal_call:
            return self._handle_confirmation(terminal_call, obs)

        # First terminal call - request confirmation
        if terminal_call:
            return self._request_confirmation(terminal_call, obs)

        # LLM reconsidered - clear pending state so next terminal call is fresh
        if self._pending_terminal:
            self._pending_terminal = None

        return None

    def _find_terminal_call(
        self, tool_calls: list[ToolCall] | None, terminal_tool: str
    ) -> ToolCall | None:
        """Find terminal tool call in the list."""
        if not tool_calls:
            return None
        for call in tool_calls:
            if call.name == terminal_tool:
                return call
        return None

    def _request_confirmation(self, call: ToolCall, obs: Observation) -> Action:
        """Request confirmation for terminal tool."""
        self._pending_terminal = call
        self._confirmation_retries = 0
        self._failure_retries = 0

        # Execute other tools if any (but not the terminal tool)
        other_calls = [c for c in (obs.response.tool_calls or []) if c.name != obs.terminal_tool]
        if other_calls:
            return Action(
                ActionType.EXECUTE_TOOLS,
                DecisionReason.TERMINAL_WITH_OTHER_TOOLS,
                tool_ids_to_execute=[c.id for c in other_calls],
            )

        # Only terminal call - send confirmation request
        msg = (
            f"You called `{obs.terminal_tool}` to indicate task completion. "
            f"Please call `{obs.terminal_tool}` again to confirm this is your final answer, "
            "or continue working if you have more to do."
        )
        return Action(
            ActionType.INSTRUCT, DecisionReason.TERMINAL_CONFIRMATION_REQUEST, message=msg
        )

    def _handle_confirmation(self, call: ToolCall, obs: Observation) -> Action:
        """Handle confirmation of terminal tool."""
        if self._has_contradiction(obs.response.content, obs.tool_names):
            return self._handle_contradiction(obs.response.content)

        output = self._extract_terminal_output(call.arguments, obs.response.content)
        is_failure = self._is_terminal_failure(call.arguments)

        if is_failure and self._failure_retries < self.config.max_failure_retries:
            return self._handle_failure_retry(call.arguments)

        self._pending_terminal = None
        return Action(
            ActionType.FAIL if is_failure else ActionType.COMPLETE,
            DecisionReason.TERMINAL_CONFIRMED_FAIL
            if is_failure
            else DecisionReason.TERMINAL_CONFIRMED,
            output=output,
            terminal_data=call.arguments,
            terminal_tool=call.name,
        )

    def _handle_contradiction(self, content: str) -> Action:
        """Handle contradictory confirmation (continuation signals detected)."""
        self._confirmation_retries += 1
        if self._confirmation_retries >= self.config.max_confirmation_retries:
            self._pending_terminal = None
            return Action(
                ActionType.FAIL,
                DecisionReason.CONFIRMATION_RETRIES_EXCEEDED,
                output=content,
            )
        msg = (
            "Your response contains contradictory signals - you confirmed completion "
            "but also indicated you want to continue. Please either continue working "
            "using the available tools, or call the terminal tool with a clear final answer."
        )
        return Action(ActionType.INSTRUCT, DecisionReason.CONTRADICTION_DETECTED, message=msg)

    def _handle_failure_retry(self, arguments: dict[str, Any]) -> Action:
        """Push back on failure status and let the agent retry."""
        self._failure_retries += 1
        self._pending_terminal = None
        status = arguments.get("status", "unknown")
        msg = (
            f"You indicated status '{status}' but the task is not complete.\n"
            f"Please continue working on the task using the available tools. "
            f"Do not give up - try a different approach if needed."
        )
        return Action(ActionType.INSTRUCT, DecisionReason.TERMINAL_FAILURE_RETRY, message=msg)

    def _is_terminal_failure(self, arguments: dict[str, Any]) -> bool:
        """Check if terminal tool arguments indicate failure.

        Uses configured status_field if set, otherwise checks "status".
        Returns True if status value exactly matches any failure value (case-insensitive).
        """
        terminal = self.config.terminal
        # Get status field name (default: "status")
        status_field = (terminal.status_field if terminal else None) or "status"
        status_value = arguments.get(status_field)

        if status_value is None:
            return False

        # Get failure values (default if not configured)
        failure_values = terminal.failure_values if terminal else ("stuck", "failed", "error")

        # Exact match, case-insensitive
        status_lower = str(status_value).lower()
        return status_lower in {v.lower() for v in failure_values}

    def _extract_terminal_output(self, arguments: dict[str, Any], fallback: str) -> str:
        """Extract output from terminal tool arguments.

        Uses the configured output_field if set.
        Falls back to checking common field names, then response content.
        """
        terminal = self.config.terminal
        # Use configured field if set
        field_name = terminal.output_field if terminal else None
        if field_name and field_name in arguments and arguments[field_name] is not None:
            return str(arguments[field_name])

        # Fallback: check common field names
        common_fields = ("conclusion", "output", "result", "answer", "response")
        for name in common_fields:
            if name in arguments and arguments[name] is not None:
                return str(arguments[name])

        return fallback

    # Phrases indicating the LLM wants to continue (contradicts completion confirmation)
    _CONTINUATION_SIGNALS = (
        # Intent to act
        "let me ",
        "i will ",
        "i'll ",
        "let's ",
        "now i'll",
        "now i will",
        "next, i will",
        "next i will",
        "i'm going to use",
        "i am going to use",
        # Asking for permission
        "shall i ",
        "should i ",
        "would you like me to",
        "would you like to proceed",
        "do you want me to",
        "do you want to proceed",
        "want me to continue",
    )

    # Patterns that look like tool invocations written as text (tool access lost)
    _TEXT_TOOL_PATTERNS = (
        "read_file",
        "shell ",
        "execute(",
        "run_command",
        "search_files",
        "list_files",
    )

    def _has_contradiction(self, content: str, tool_names: list[str] | None = None) -> bool:
        """Check if content has continuation signals (contradiction).

        Detects two categories:
        1. Continuation phrases (e.g., "let me check", "I will continue")
        2. Text tool patterns (e.g., LLM writes "read_file" as text instead of calling it)
        """
        if not content:
            return False
        content_lower = content.lower()
        if any(s in content_lower for s in self._CONTINUATION_SIGNALS):
            return True
        return self._has_text_tool_pattern(content, tool_names or [])

    @staticmethod
    def _is_empty_response(response: AgentResponse) -> bool:
        """Check if the LLM produced an empty response (no content and no tool calls)."""
        return not response.content and not response.tool_calls

    def _has_text_tool_pattern(self, content: str, tool_names: list[str]) -> bool:
        """Check if content contains tool names written as text (tool access lost).

        Uses word-boundary matching against actual tool names to avoid false
        positives (e.g. ``"search"`` inside ``"research"``).  Falls back to the
        static ``_TEXT_TOOL_PATTERNS`` (which have their own delimiters) when no
        tool names are available.
        """
        if not content:
            return False
        content_lower = content.lower()
        if tool_names:
            return any(
                re.search(r"\b" + re.escape(t.lower()) + r"\b", content_lower) for t in tool_names
            )
        return any(p in content_lower for p in self._TEXT_TOOL_PATTERNS)

    async def _handle_no_tools(self, obs: Observation) -> Action:
        """Handle response with no tool calls.

        Checks for degenerate states (empty response, text-tool patterns) first,
        bypassing backoff and classifier. After ``backoff_iterations`` consecutive
        degenerate nudges, falls through to classify-and-nudge to avoid infinite
        loops when the LLM genuinely cannot make tool calls.
        """
        is_degenerate = self._is_empty_response(obs.response) or self._has_text_tool_pattern(
            obs.response.content, obs.tool_names
        )

        if is_degenerate:
            self._consecutive_degenerate += 1
        else:
            self._consecutive_degenerate = 0
            return await self._classify_and_nudge(obs)

        # After enough consecutive degenerate nudges, fall through to classifier
        # so backoff kicks in and the loop can terminate naturally.
        if self._consecutive_degenerate > self.config.backoff_iterations:
            return await self._classify_and_nudge(obs)

        is_empty = self._is_empty_response(obs.response)
        reason = DecisionReason.EMPTY_RESPONSE if is_empty else DecisionReason.TEXT_TOOL_PATTERN
        message = _EMPTY_RESPONSE_NUDGE if is_empty else _TEXT_TOOL_NUDGE
        self._last_nudge_iteration = obs.iteration
        return Action(ActionType.INSTRUCT, reason, message=message)

    async def _classify_and_nudge(self, obs: Observation) -> Action:
        """Classify the response state and decide whether to nudge or backoff."""
        result = await self._classifier.classify(obs.task, obs.response.content, obs.tool_names)

        if result.state == TaskState.COMPLETED:
            return Action(
                ActionType.COMPLETE,
                DecisionReason.CLASSIFIED_COMPLETE,
                output=obs.response.content,
            )

        # Check backoff
        iterations_since_nudge = obs.iteration - self._last_nudge_iteration
        if iterations_since_nudge < self.config.backoff_iterations:
            return Action(
                ActionType.SKIP,
                DecisionReason.BACKOFF,
                reason_details=f"{iterations_since_nudge}/{self.config.backoff_iterations}",
            )

        # Send nudge
        self._last_nudge_iteration = obs.iteration
        nudge = self.nudges.get(result.state, DEFAULT_NUDGE_FALLBACK)
        return Action(
            ActionType.INSTRUCT,
            DecisionReason.NUDGE_CLASSIFIED,
            message=nudge,
            reason_details=result.state.value,
        )

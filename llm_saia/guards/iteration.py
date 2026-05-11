"""Iteration guards for controlling agent tool loops.

Iteration guards run during tool calling loops and can inject feedback
to shape agent behavior, validate terminal tool calls, or enforce
constraints on the loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.guard import IterationContext, IterationGuard
from ._helpers import GuardState, ordinal, validate_schema

if TYPE_CHECKING:
    from ..core.backend import ChatResponse
    from ..core.types import ToolDef

__all__ = [
    "contradiction",
    "narrative",
    "terminal_compliance",
    "terminal_deadline",
    "terminal_schema",
    "terminal_status",
]


# ---------------------------------------------------------------------------
# Terminal tool validation guards
# ---------------------------------------------------------------------------

# Phrases that indicate the LLM is hedging or unable to complete
_CONTINUATION_SIGNALS = frozenset(
    {
        "however",
        "but",
        "although",
        "unfortunately",
        "i can't",
        "i cannot",
        "i'm unable",
        "i am unable",
        "not possible",
        "isn't possible",
        "unable to",
        "cannot complete",
        "can't complete",
    }
)


def terminal_status(
    tool: str,
    status_field: str,
    failure_values: tuple[str, ...],
    max_retries: int = 3,
    *,
    escalate: bool = True,
) -> IterationGuard:
    """Reject terminal tool calls with failure status and ask to retry.

    Args:
        tool: Name of the terminal tool.
        status_field: Field in tool arguments containing the status.
        failure_values: Status values that indicate failure.
        max_retries: Max retry attempts before giving up. Default 3.
        escalate: Use increasingly forceful retry instructions. Default True.
    """
    failure_set = frozenset(failure_values)
    state = GuardState(max_retries)

    def check(ctx: IterationContext) -> str | None:
        state.reset_if_new(ctx.iteration)
        status = _find_failure_status(ctx.response, tool, status_field, failure_set)
        if status is None:
            return None
        return state.feedback(_status_feedback, escalate, status=status)

    return IterationGuard(validator=check, name="terminal_status")


def terminal_schema(
    tools: list[ToolDef],
    terminal_tool: str,
    max_retries: int = 2,
    *,
    escalate: bool = True,
) -> IterationGuard:
    """Validate terminal tool arguments against its JSON schema.

    Args:
        tools: List of tool definitions (used to find the terminal tool's schema).
        terminal_tool: Name of the terminal tool.
        max_retries: Max retry attempts before giving up. Default 2.
        escalate: Use increasingly forceful retry instructions. Default True.
    """
    schema = _find_tool_schema(tools, terminal_tool)
    if schema is None:
        return IterationGuard(validator=lambda ctx: None, name="terminal_schema")

    state = GuardState(max_retries)

    def check(ctx: IterationContext) -> str | None:
        state.reset_if_new(ctx.iteration)
        errors = _find_schema_errors(ctx.response, terminal_tool, schema)
        if not errors:
            return None
        return state.feedback(_schema_feedback, escalate, tool=terminal_tool, errors=errors)

    return IterationGuard(validator=check, name="terminal_schema")


def contradiction(
    terminal_tool: str,
    max_retries: int = 2,
    *,
    escalate: bool = True,
) -> IterationGuard:
    """Detect when the LLM contradicts itself after calling the terminal tool.

    Detects when the LLM calls the terminal tool but includes hedging language
    (e.g., "however", "I can't", "unfortunately").

    Args:
        terminal_tool: Name of the terminal tool.
        max_retries: Max retry attempts before giving up. Default 2.
        escalate: Use increasingly forceful retry instructions. Default True.
    """
    state = GuardState(max_retries)

    def check(ctx: IterationContext) -> str | None:
        state.reset_if_new(ctx.iteration)
        signal = _find_contradiction(ctx.response, terminal_tool)
        if signal is None:
            return None
        return state.feedback(_contradiction_feedback, escalate, tool=terminal_tool, signal=signal)

    return IterationGuard(validator=check, name="contradiction")


# ---------------------------------------------------------------------------
# Behavioral iteration guards
# ---------------------------------------------------------------------------


def narrative(
    terminal_tool: str,
    *,
    max_retries: int = 2,
    escalate: bool = True,
) -> IterationGuard:
    """Nudge the LLM to explain what it's doing when making tool calls.

    Fires when the LLM makes tool calls without any narrative text.
    Skips check for terminal tool (don't demand explanation at finish).

    This is an advisory guard (blocking=False) - tools execute first,
    then feedback is injected afterward.

    Args:
        terminal_tool: Name of the terminal tool. Calls to this tool
            do not trigger the narrative requirement.
        max_retries: Max retry attempts before giving up. Default 2.
        escalate: Use increasingly forceful retry instructions. Default True.
    """
    state = GuardState(max_retries)

    def check(ctx: IterationContext) -> str | None:
        state.reset_if_new(ctx.iteration)
        response = ctx.response
        if not response.tool_calls:
            return None
        # Don't demand narrative for terminal tool - let it finish
        if any(tc.name == terminal_tool for tc in response.tool_calls):
            return None
        # Only nudge if there are tool calls but no explanation
        if not (response.content or "").strip():
            tool_names = ", ".join(tc.name for tc in response.tool_calls)
            return state.feedback(_narrative_feedback, escalate, tools=tool_names)
        return None

    return IterationGuard(validator=check, name="narrative", blocking=False)


def terminal_deadline(
    terminal_tool: str,
    *,
    threshold: int = 3,
    max_retries: int = 3,
    escalate: bool = True,
) -> IterationGuard:
    """Enforce terminal tool call when iterations are running low.

    When iterations remaining drops to the threshold or below, requires the
    agent to call ONLY the terminal tool - no mixing with other tools.
    Prevents agents from continuing to explore when they should wrap up.

    Args:
        terminal_tool: Name of the terminal tool that signals completion.
        threshold: Fire when remaining iterations <= this value. Default 3.
        max_retries: Max retry attempts before giving up. Default 3.
        escalate: Use increasingly forceful retry instructions. Default True.
    """
    state = GuardState(max_retries)

    def check(ctx: IterationContext) -> str | None:
        state.reset_if_new(ctx.iteration)
        tool_calls = ctx.response.tool_calls or []
        tool_names = {tc.name for tc in tool_calls}
        calls_terminal = terminal_tool in tool_names

        # When low on iterations, require ONLY the terminal tool
        if ctx.remaining <= threshold:
            if not calls_terminal:
                return state.feedback(
                    _deadline_feedback,
                    escalate,
                    tool=terminal_tool,
                    remaining=ctx.remaining,
                    issue="Do not call any other tool.",
                )
            if len(tool_names) > 1:
                other_tools = ", ".join(tool_names - {terminal_tool})
                return state.feedback(
                    _deadline_feedback,
                    escalate,
                    tool=terminal_tool,
                    remaining=ctx.remaining,
                    issue=f"Do not mix with other tools ({other_tools}).",
                )
        return None

    return IterationGuard(validator=check, name="terminal_deadline")


def terminal_compliance(
    terminal_tool: str,
    *,
    threshold: int = 2,
    max_retries: int = 2,
    escalate: bool = True,
) -> IterationGuard:
    """Catch LLM saying it will call terminal tool but not actually doing it.

    Detects the "said but didn't call" pattern where the LLM says something
    like "Calling report_findings now" in text but doesn't include the actual
    tool call. Only fires when iterations remaining is at or below the threshold.

    Args:
        terminal_tool: Name of the terminal tool.
        threshold: Fire when remaining iterations <= this value. Default 2.
        max_retries: Max retry attempts before giving up. Default 2.
        escalate: Use increasingly forceful retry instructions. Default True.
    """
    state = GuardState(max_retries)

    def check(ctx: IterationContext) -> str | None:
        state.reset_if_new(ctx.iteration)
        content = (ctx.response.content or "").lower()
        tool_calls = ctx.response.tool_calls or []
        calls_terminal = any(tc.name == terminal_tool for tc in tool_calls)

        # Detect: mentions terminal tool in text but didn't call it
        mentions_terminal = terminal_tool.lower() in content
        said_but_didnt = mentions_terminal and not calls_terminal and len(tool_calls) == 0

        if said_but_didnt and ctx.remaining <= threshold:
            return state.feedback(_compliance_feedback, escalate, tool=terminal_tool)
        return None

    return IterationGuard(validator=check, name="terminal_compliance")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_failure_status(
    response: ChatResponse, tool: str, field: str, failures: frozenset[str]
) -> str | None:
    """Find failure status in terminal tool call, or None."""
    for tc in response.tool_calls or []:
        if tc.name == tool and isinstance(tc.arguments, dict):
            status = tc.arguments.get(field)
            if status in failures:
                return str(status)
    return None


def _status_feedback(count: int, *, status: str) -> tuple[str, str]:
    """Generate feedback for terminal status failure."""
    base = f"You reported status='{status}' but the task is not complete. Try a different approach."
    forceful = (
        f"You have reported '{status}' {count} times. "
        f"You MUST complete the task. Try a completely different approach."
    )
    return base, forceful


def _find_tool_schema(tools: list[ToolDef], name: str) -> dict[str, Any] | None:
    """Find schema for a tool by name."""
    for t in tools:
        if t.name == name:
            return t.parameters
    return None


def _find_schema_errors(response: ChatResponse, tool: str, schema: dict[str, Any]) -> list[str]:
    """Find schema errors in terminal tool call."""
    for tc in response.tool_calls or []:
        if tc.name == tool:
            return validate_schema(tc.arguments, schema)
    return []


def _schema_feedback(count: int, *, tool: str, errors: list[str]) -> tuple[str, str]:
    """Generate feedback for schema validation failure."""
    error_list = "\n".join(f"- {e}" for e in errors)
    base = (
        f"Your `{tool}` call has schema errors:\n{error_list}\n\n"
        f"Please try again with valid arguments."
    )
    forceful = (
        f"Your `{tool}` call STILL has schema errors (attempt {count}):\n{error_list}\n\n"
        f"You MUST call `{tool}` with VALID arguments."
    )
    return base, forceful


def _find_contradiction(response: ChatResponse, tool: str) -> str | None:
    """Find contradiction signal when terminal tool is called."""
    has_terminal = any(tc.name == tool for tc in (response.tool_calls or []))
    if not has_terminal:
        return None
    content = (response.content or "").lower()
    for signal in _CONTINUATION_SIGNALS:
        if signal in content:
            return signal
    return None


def _contradiction_feedback(count: int, *, tool: str, signal: str) -> tuple[str, str]:
    """Generate feedback for contradiction detection."""
    base = (
        f"You called `{tool}` but your response contains contradictory language ('{signal}'). "
        f"Complete the task confidently or do not call `{tool}`."
    )
    forceful = (
        f"You called `{tool}` but your response suggests you cannot complete ('{signal}'). "
        f"This is the {ordinal(count)} contradiction. Complete confidently or explain the blocker."
    )
    return base, forceful


def _narrative_feedback(attempt: int, *, tools: str) -> tuple[str, str]:
    """Generate narrative feedback messages."""
    base = (
        f"You called {tools} without explaining why. "
        "In one sentence, explain what you're doing and why."
    )
    forceful = (
        f"REQUIRED: You called {tools} without explanation. "
        "You MUST state what you're doing and why. One sentence."
    )
    return base, forceful


def _deadline_feedback(attempt: int, *, tool: str, remaining: int, issue: str) -> tuple[str, str]:
    """Generate terminal deadline feedback messages."""
    base = (
        f"Only {remaining} iteration(s) remaining. "
        f"You MUST call {tool} now with your findings. {issue}"
    )
    forceful = (
        f"CRITICAL: {remaining} iteration(s) left. STOP exploring. Call {tool} IMMEDIATELY. {issue}"
    )
    return base, forceful


def _compliance_feedback(attempt: int, *, tool: str) -> tuple[str, str]:
    """Generate terminal compliance feedback messages."""
    base = (
        f"You said you would call {tool} but you DID NOT include the tool call. "
        f"You MUST include the actual {tool} tool call. "
        f"Do not just say you will - actually call the tool."
    )
    forceful = (
        f"FAILURE: You mentioned {tool} but DID NOT CALL IT. "
        f"This is your LAST chance. Include the {tool} tool call NOW or the task fails."
    )
    return base, forceful

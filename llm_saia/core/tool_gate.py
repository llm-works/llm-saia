"""Per-iteration tool-visibility gate evaluation.

Evaluates :data:`~llm_saia.ToolGate` callbacks registered in
:attr:`Config.tool_gates` before each LLM call. Blocked tools are filtered
from the outbound schema so the model never sees them and cannot spend
output tokens generating a call.

The reason strings returned by gates are diagnostic only — they are recorded
in the trace/log, never injected back into the conversation. Injecting them
would cue the model to think about the tool we intentionally hid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from .config import Config, ToolGate, ToolGateContext

if TYPE_CHECKING:
    from .backend import ChatResponse, ToolDef
    from .conversation import Message


__all__ = ["apply_tool_gates"]


def _compute_tool_call_counts(messages: Sequence[Message]) -> dict[str, int]:
    """Walk the transcript and count assistant-issued tool calls by name."""
    counts: dict[str, int] = {}
    for msg in messages:
        for tc in msg.tool_calls or []:
            counts[tc.name] = counts.get(tc.name, 0) + 1
    return counts


def _min_iterations_gate(n: int) -> ToolGate:
    """Shortcut gate produced by :attr:`TerminalConfig.min_iterations`."""
    reason = f"below TerminalConfig.min_iterations ({n})"

    def gate(ctx: ToolGateContext) -> bool | str | None:
        return True if ctx.iteration >= n else reason

    return gate


def _effective_gates(config: Config) -> Mapping[str, ToolGate]:
    """Merge explicit ``Config.tool_gates`` with the TerminalConfig shortcut.

    An explicit entry in ``Config.tool_gates`` for the terminal tool wins over
    the ``min_iterations`` shortcut — users who register their own gate opt
    out of the built-in on that tool.
    """
    gates: dict[str, ToolGate] = dict(config.tool_gates)
    terminal = config.terminal
    if terminal is not None and terminal.min_iterations > 0 and terminal.tool not in gates:
        gates[terminal.tool] = _min_iterations_gate(terminal.min_iterations)
    return gates


def _run_gate(gate: ToolGate, ctx: ToolGateContext, tool_name: str) -> str | None:
    """Evaluate a single gate; return reason string if blocked, else None.

    A blocking ``False`` return produces a synthetic reason so the trace/log
    always has *something* to attribute the hiding to. Gate exceptions are
    caught and converted to blocking reasons (matching IterationGuard behavior).
    """
    try:
        verdict = gate(ctx)
    except Exception as exc:  # noqa: BLE001 - isolate user callback failures
        return f"gate for {tool_name!r} raised {type(exc).__name__}: {exc}"
    if verdict is True or verdict is None:
        return None
    if verdict is False:
        return f"gate for {tool_name!r} returned False"
    return str(verdict)


def _build_context(
    config: Config,
    iteration: int,
    messages: Sequence[Message],
    last_response: ChatResponse | None,
) -> ToolGateContext:
    """Build the per-iteration :class:`ToolGateContext`, running the factory if configured."""
    base_ctx = ToolGateContext(
        iteration=iteration,
        tool_call_counts=_compute_tool_call_counts(messages),
        last_response=last_response,
        messages=messages,
    )
    factory = config.tool_gate_context_factory
    if factory is None:
        return base_ctx
    return ToolGateContext(
        iteration=base_ctx.iteration,
        tool_call_counts=base_ctx.tool_call_counts,
        last_response=base_ctx.last_response,
        messages=base_ctx.messages,
        extra=factory(base_ctx),
    )


def apply_tool_gates(
    config: Config,
    iteration: int,
    messages: Sequence[Message],
    last_response: ChatResponse | None,
    tools: Sequence[ToolDef] | None,
) -> tuple[list[ToolDef] | None, dict[str, str]]:
    """Filter ``tools`` by evaluating registered gates.

    Returns:
        Tuple of ``(filtered_tools, blocked_reasons)``.
        ``filtered_tools`` is ``None`` when the input was ``None`` (no
        change); otherwise a new list with blocked tools removed.
        ``blocked_reasons`` maps tool name → reason string for every tool
        that was filtered out this call. Empty when nothing was gated.
    """
    gates = _effective_gates(config)
    if not gates or not tools:
        return (None if tools is None else list(tools)), {}

    ctx = _build_context(config, iteration, messages, last_response)
    kept: list[ToolDef] = []
    blocked: dict[str, str] = {}
    for tool in tools:
        gate = gates.get(tool.name)
        reason = _run_gate(gate, ctx, tool.name) if gate is not None else None
        if reason is None:
            kept.append(tool)
        else:
            blocked[tool.name] = reason
    return kept, blocked

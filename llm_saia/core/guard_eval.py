"""Guard evaluation for verbs.

Handles iteration guard evaluation and feedback collection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from .guard import IterationContext, IterationGuard

if TYPE_CHECKING:
    from .backend import ChatResponse
    from .trace import GuardOutcome, VerbTrace


class _GuardHost(Protocol):
    """Protocol for capabilities the guard evaluator needs from its host."""

    _TRACE_LIMIT: int

    def _truncate(self, text: str, limit: int) -> str:
        """Truncate text to limit."""
        ...

    @property
    def _lg(self) -> Any:
        """Logger instance."""
        ...


class _GuardEvaluator:
    """Evaluates iteration guards and collects feedback."""

    def __init__(self, host: _GuardHost):
        """Initialize with host providing required capabilities."""
        self._host = host

    def run_iteration_guards(
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
        self._host._lg.trace("running iteration guards", extra={"guards": [g.name for g in guards]})
        feedback_parts, outcomes = self._eval_guards(guards, ctx, GuardOutcome)
        self.attach_guard_outcomes(_trace, outcomes)
        return self._finalize_result(feedback_parts, outcomes)

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
            result = self.eval_single_guard(guard, ctx)
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
                self._log_guard_fired(guard.name, result)
                feedback_parts.append(result)  # type: ignore[arg-type]
        return feedback_parts, outcomes

    @staticmethod
    def eval_single_guard(guard: IterationGuard, ctx: IterationContext) -> str | None:
        """Evaluate a single guard, catching exceptions."""
        try:
            return guard.validator(ctx)
        except Exception as e:
            return f"Validator raised {type(e).__name__}: {e}"

    @staticmethod
    def attach_guard_outcomes(_trace: VerbTrace | None, outcomes: list[GuardOutcome]) -> None:
        """Attach outcomes to the most recent step if trace exists."""
        if _trace and _trace.steps:
            _trace.steps[-1].guards = outcomes

    def _finalize_result(
        self, feedback_parts: list[str], outcomes: list[GuardOutcome]
    ) -> tuple[str | None, list[GuardOutcome]]:
        """Combine feedback and log the result."""
        h = self._host
        if feedback_parts:
            combined = "\n\n".join(feedback_parts)
            h._lg.trace(
                "iteration guards triggered feedback",
                extra={
                    "guards_fired": [o.name for o in outcomes if not o.passed],
                    "feedback": h._truncate(combined, h._TRACE_LIMIT),
                },
            )
            return combined, outcomes
        h._lg.trace("all iteration guards passed")
        return None, outcomes

    def _log_guard_fired(self, name: str | None, feedback: str | None) -> None:
        """Log that an iteration guard fired."""
        self._host._lg.debug(
            "iteration guard fired",
            extra={"guard": name, "feedback": feedback},
        )

    @staticmethod
    def split_guard_feedback(
        outcomes: list[GuardOutcome],
    ) -> tuple[str | None, str | None]:
        """Split guard outcomes into blocking and advisory feedback.

        Returns (blocking_feedback, advisory_feedback).
        """
        blocking_parts: list[str] = []
        advisory_parts: list[str] = []
        for outcome in outcomes:
            if not outcome.passed and outcome.error:
                if outcome.blocking:
                    blocking_parts.append(outcome.error)
                else:
                    advisory_parts.append(outcome.error)
        blocking = "\n\n".join(blocking_parts) if blocking_parts else None
        advisory = "\n\n".join(advisory_parts) if advisory_parts else None
        return blocking, advisory

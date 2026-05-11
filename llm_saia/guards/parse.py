"""Parse guards for handling structured output parsing failures.

Parse guards run when structured output parsing fails and can provide
feedback to help the LLM produce valid output.
"""

from __future__ import annotations

from ..core.guard import IterationContext, IterationGuard

__all__ = ["schema_retry"]


def schema_retry(max_retries: int = 2, *, escalate: bool = True) -> IterationGuard:
    """Retry when structured output parsing fails (malformed JSON, schema mismatch).

    This guard only activates in parse retry context (when ``ctx.parse_error`` is set).
    It does not affect tool loop behavior.

    Args:
        max_retries: Max retry attempts. Default 2.
        escalate: Use increasingly forceful retry instructions. Default True.

    Example:
        >>> from llm_saia.guards import schema_retry
        >>> result = await saia.with_guard(schema_retry()).extract(Article, text)
    """

    def check(ctx: IterationContext) -> str | None:
        if ctx.parse_error is None:
            return None  # Not a parse error context (tool loop)
        return _schema_retry_feedback(ctx, escalate)

    return IterationGuard(validator=check, name="schema_retry", parse_max_retries=max_retries)


def _schema_retry_feedback(ctx: IterationContext, escalate: bool) -> str:
    """Generate feedback message for schema retry."""
    parse_err = getattr(ctx.parse_error, "parse_error", None)
    if ctx.iteration == 0 or not escalate:
        msg = "Your response was not valid JSON. Please respond with valid JSON."
        return f"{msg}\n\nParse error: {parse_err}" if parse_err else msg
    return (
        f"YOU HAVE FAILED TO PRODUCE VALID JSON {ctx.iteration + 1} TIMES. "
        f"Parse error: {parse_err or 'unknown'}. "
        f"Respond with VALID JSON ONLY. No markdown, no explanation, just the JSON."
    )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Parse guards for handling structured output parsing failures.

Parse guards run when structured output parsing fails and can provide
feedback to help the LLM produce valid output.
"""

from __future__ import annotations

from ..core.guard import IterationContext, IterationGuard
from ..core.parse_failure import ParseFailure, ParseFailureKind

__all__ = ["schema_retry"]


def schema_retry(max_retries: int = 2, *, escalate: bool = True) -> IterationGuard:
    """Retry when structured output parsing fails (malformed JSON, schema mismatch).

    The retry restarts from the original prompt, without the failed answer,
    and the feedback says what was wrong with it: cut off at the token limit
    (and whether it ended repeating itself), empty, wrapped in a code block,
    surrounded by other text, malformed at a quoted spot, or valid JSON with a
    named field that does not fit the schema.

    This guard only activates in parse retry context (when ``ctx.parse_error`` is set).
    It does not affect tool loop behavior.

    Args:
        max_retries: Max retry attempts. Default 2.
        escalate: From the second retry on, add that earlier attempts failed
            too. Default True.

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
    error = ctx.parse_error
    failure = getattr(error, "failure", None)
    if failure is None:
        detail = getattr(error, "parse_error", None) or str(error)
        failure = ParseFailure(ParseFailureKind.OTHER, detail)
    feedback = _describe_failure(failure)
    if escalate and ctx.iteration > 0:
        feedback += (
            f"\n\nEarlier attempts failed as well ({ctx.iteration + 1} so far). "
            "Follow the instructions above exactly."
        )
    return feedback


# Feedback per failure kind; {detail} and {excerpt} come from the ParseFailure.
_FEEDBACK: dict[ParseFailureKind, str] = {
    ParseFailureKind.EMPTY: "Your previous answer was empty. Respond with the JSON object.",
    ParseFailureKind.CODE_FENCE: (
        "Your previous answer wrapped the JSON in a markdown code block. "
        "Respond with the JSON object only: no code fences, no text around it."
    ),
    ParseFailureKind.TEXT_BEFORE: (
        "Your previous answer started with text instead of JSON: «{excerpt}». "
        "Respond with the JSON object only, starting with {{."
    ),
    ParseFailureKind.TEXT_AFTER: (
        "Your previous answer had text after the JSON: «{excerpt}». "
        "Respond with the JSON object only."
    ),
    ParseFailureKind.SYNTAX: (
        "Your previous answer is not valid JSON: {detail}:\n{excerpt}\n"
        "Fix this and send the complete JSON again."
    ),
    ParseFailureKind.SCHEMA: (
        "Your previous answer was valid JSON but does not match the required structure: "
        "{detail}. Send the complete JSON again with this fixed."
    ),
    ParseFailureKind.OTHER: (
        "Your previous answer could not be parsed: {detail}. Respond with valid JSON."
    ),
}


def _describe_failure(failure: ParseFailure) -> str:
    """Feedback to the model on why its previous answer failed to parse."""
    if failure.kind is ParseFailureKind.TRUNCATED:
        return _describe_cut_off(failure)
    template = _FEEDBACK.get(failure.kind, _FEEDBACK[ParseFailureKind.OTHER])
    return template.format(detail=failure.detail, excerpt=failure.excerpt)


def _describe_cut_off(failure: ParseFailure) -> str:
    """Feedback for an answer cut off before the JSON was complete."""
    limit = f" ({failure.output_tokens} tokens)" if failure.output_tokens else ""
    head = (
        f"Your previous answer was cut off at the output limit{limit} before the JSON was complete."
    )
    if failure.repeated:
        return (
            f"{head} It ended by repeating the same text over and over: «{failure.repeated}». "
            "Write each value once, keep string values short, and close every string, "
            "array and object."
        )
    # Never ask for fewer items: a shortened list parses and loses content silently.
    return f"{head} Keep every item, but write shorter string values so the complete JSON fits."

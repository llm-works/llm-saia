# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""What went wrong when a structured answer failed to parse.

A :class:`ParseFailure` is built once, when parsing fails, from everything
known at that point: the answer, the parser's exception, and the backend's
``finish_reason`` and output token count. Retry guards such as
:func:`~llm_saia.guards.schema_retry` turn it into feedback for the model.
The model never sees its failed answer again (each retry restarts from the
original prompt), so the diagnosis carries the parts of the answer the
feedback needs: an excerpt around a syntax error, the text that preceded or
followed the JSON, or the text a cut-off answer kept repeating.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = ["ParseFailure", "ParseFailureKind", "diagnose_json_error", "diagnose_schema_error"]

# finish_reason values for an answer cut off at the output token limit
# (OpenAI-compatible backends report "length", Anthropic "max_tokens").
_CUT_OFF_REASONS = frozenset({"length", "max_tokens"})

# Decoder messages that mean the document ended early; used to spot a cut-off
# answer only when the backend reports no finish_reason.
_EARLY_END_MESSAGES: tuple[str, ...] = (
    "Unterminated string",
    "Unexpected end of JSON",
    "Expecting value",
    "Expecting ',' delimiter",
    "Expecting ':' delimiter",
)

_EXCERPT_RADIUS = 60  # characters shown on each side of a syntax error
_QUOTE_LIMIT = 80  # longest quoted piece of the answer (leading/trailing text)
_REPEAT_TAIL = 2000  # how much of a cut-off answer's end is checked for repetition
_REPEAT_MAX_UNIT = 200  # longest repeated unit looked for
_REPEAT_MIN_SPAN = 200  # the repetition must cover at least this many characters
_REPEAT_MIN_COUNT = 4  # ... and repeat the unit at least this many times
_MAX_SCHEMA_ERRORS = 5  # schema errors listed from one pydantic ValidationError


class ParseFailureKind(StrEnum):
    """Why a structured answer failed to parse."""

    TRUNCATED = "truncated"  # cut off at the output token limit
    EMPTY = "empty"  # no content
    CODE_FENCE = "code_fence"  # JSON wrapped in a markdown code block
    TEXT_BEFORE = "text_before"  # starts with text instead of JSON
    TEXT_AFTER = "text_after"  # text follows a complete JSON value
    SYNTAX = "syntax"  # malformed JSON
    SCHEMA = "schema"  # valid JSON that does not fit the schema
    OTHER = "other"  # error from a custom json_parser


@dataclass(frozen=True)
class ParseFailure:
    """Diagnosis of a structured answer that failed to parse.

    Attributes:
        kind: Why parsing failed.
        detail: The parser's or validator's message; schema errors start with
            the field path (``items[2].kind: ...``).
        excerpt: SYNTAX: the answer around the error, with a caret line under
            the error position. TEXT_BEFORE / TEXT_AFTER: the extra text.
        repeated: TRUNCATED: the text the answer kept repeating up to the
            cut, or ``""`` when the end of the answer does not repeat.
        output_tokens: TRUNCATED: tokens generated before the cut, ``0`` when
            the backend does not report it.
    """

    kind: ParseFailureKind
    detail: str
    excerpt: str = ""
    repeated: str = ""
    output_tokens: int = 0


def diagnose_json_error(
    content: str, error: Exception, finish_reason: str | None, output_tokens: int
) -> ParseFailure:
    """Diagnose an answer the JSON parser rejected."""
    detail = str(error)
    if _is_cut_off(content, error, finish_reason):
        return ParseFailure(
            ParseFailureKind.TRUNCATED,
            detail,
            repeated=_repeating_tail(content),
            output_tokens=output_tokens,
        )
    stripped = content.strip()
    if not stripped:
        return ParseFailure(ParseFailureKind.EMPTY, detail)
    if not isinstance(error, json.JSONDecodeError):
        return ParseFailure(ParseFailureKind.OTHER, detail)
    return _diagnose_decode_error(content, stripped, error)


def diagnose_schema_error(error: Exception) -> ParseFailure:
    """Diagnose valid JSON that does not fit the schema."""
    errors = getattr(error, "errors", None)
    if callable(errors):  # pydantic ValidationError
        return ParseFailure(ParseFailureKind.SCHEMA, _pydantic_detail(errors()))
    return ParseFailure(ParseFailureKind.SCHEMA, str(error))


def _is_cut_off(content: str, error: Exception, finish_reason: str | None) -> bool:
    """True for an answer cut off at the token limit.

    ``finish_reason`` decides when the backend reports one. Without it, a
    decoder error at the very end of the content stands in for it.
    """
    if finish_reason is not None:
        return finish_reason in _CUT_OFF_REASONS
    if not isinstance(error, json.JSONDecodeError) or not content.strip():
        return False
    early_end = any(error.msg.startswith(m) for m in _EARLY_END_MESSAGES)
    return early_end and not content[error.pos :].strip()


def _diagnose_decode_error(
    content: str, stripped: str, error: json.JSONDecodeError
) -> ParseFailure:
    """Tell wrapped or surrounded JSON apart from malformed JSON."""
    detail = str(error)
    if stripped.startswith("```"):
        return ParseFailure(ParseFailureKind.CODE_FENCE, detail)
    if error.msg == "Extra data":
        extra = _quote(content[error.pos :].strip())
        return ParseFailure(ParseFailureKind.TEXT_AFTER, detail, excerpt=extra)
    if stripped[0] not in "{[":
        return ParseFailure(ParseFailureKind.TEXT_BEFORE, detail, excerpt=_quote(stripped))
    where = f"{error.msg} at line {error.lineno}, column {error.colno}"
    return ParseFailure(ParseFailureKind.SYNTAX, where, excerpt=_excerpt(content, error.pos))


def _excerpt(content: str, pos: int) -> str:
    """One line of the answer around ``pos`` and a caret line under it."""
    start = max(0, pos - _EXCERPT_RADIUS)
    end = min(len(content), pos + _EXCERPT_RADIUS)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(content) else ""
    line = prefix + _one_line(content[start:end]) + suffix
    caret = " " * (len(prefix) + pos - start) + "^"
    return f"{line}\n{caret}"


def _one_line(text: str) -> str:
    """Show line breaks and tabs as single visible characters, keeping positions."""
    return text.replace("\r", "⏎").replace("\n", "⏎").replace("\t", " ")


def _quote(text: str) -> str:
    """A short single-line quote of ``text``."""
    if len(text) > _QUOTE_LIMIT:
        text = text[:_QUOTE_LIMIT] + "…"
    return _one_line(text)


def _repeating_tail(content: str) -> str:
    """The unit the end of ``content`` repeats, or ``""`` when it doesn't.

    Checks units of 1 to ``_REPEAT_MAX_UNIT`` characters, shortest first, for
    one that the last ``_REPEAT_TAIL`` characters end by repeating over at
    least ``_REPEAT_MIN_SPAN`` characters and ``_REPEAT_MIN_COUNT`` times.
    """
    tail = content[-_REPEAT_TAIL:]
    n = len(tail)
    for unit in range(1, min(_REPEAT_MAX_UNIT, n // _REPEAT_MIN_COUNT) + 1):
        span = unit
        while span < n and tail[n - span - 1] == tail[n - span - 1 + unit]:
            span += 1
        if span >= max(_REPEAT_MIN_SPAN, unit * _REPEAT_MIN_COUNT):
            return _quote(tail[n - unit :])
    return ""


def _pydantic_detail(errors: list[dict[str, Any]]) -> str:
    """Pydantic validation errors as ``path: message`` entries."""
    parts = [_pydantic_entry(e) for e in errors[:_MAX_SCHEMA_ERRORS]]
    if len(errors) > _MAX_SCHEMA_ERRORS:
        parts.append(f"and {len(errors) - _MAX_SCHEMA_ERRORS} more")
    return "; ".join(parts)


def _pydantic_entry(error: dict[str, Any]) -> str:
    """One pydantic error; scalar inputs are quoted so the model sees the bad value."""
    path = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in error.get("loc", ()))
    entry = f"{path.lstrip('.')}: {error.get('msg', 'invalid')}" if path else error.get("msg", "")
    value = error.get("input")
    if isinstance(value, str | int | float | bool) and len(repr(value)) <= _QUOTE_LIMIT:
        entry += f" (got {value!r})"
    return entry

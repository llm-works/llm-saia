"""Tree-structured tracing for verb execution observability.

Every verb call produces a :class:`VerbTrace` — a tree with :class:`Step`
children.  Each Step captures one LLM call and what happened because of it
(parse result, guard outcomes, tool executions, controller decisions).

The ``Tracer`` class serializes trace records to JSONL and writes them to a
pluggable writer (any ``IO[str]``).  Use the factory functions or the builder
to create a tracer with the desired destination.

Usage via builder::

    saia = (SAIA.builder()
        .backend(backend)
        .tracing.file("/tmp/trace.jsonl")
        .build())

Usage via factory::

    tracer = TracerFactory.file("/tmp/trace.jsonl")
    result = await saia.complete("task", tracer=tracer)
"""

from __future__ import annotations

import json
import secrets
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any, Generic, TypeVar

_P = TypeVar("_P")
_CONTENT_PREVIEW_LIMIT = 200


def _generate_id() -> str:
    """Generate an 8-character hex ID for trace/call correlation."""
    return secrets.token_hex(4)


# ---------------------------------------------------------------------------
# Trace data model
# ---------------------------------------------------------------------------


@dataclass
class LLMCall:
    """Token and identity data for a single backend.chat() invocation."""

    call_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    duration_ms: int = 0


@dataclass
class GuardOutcome:
    """Result of applying one output guard."""

    name: str | None = None
    passed: bool = True
    attempts: int = 1
    error: str | None = None


@dataclass
class ToolOutcome:
    """Result of executing one tool call."""

    name: str = ""
    call_id: str = ""
    success: bool = True
    error: str | None = None


@dataclass
class Step:
    """One logical step in a verb's execution — one LLM call + its outcome.

    The ``phase`` field discriminates the kind of step:

    - ``"attempt"`` — initial LLM call (text or structured)
    - ``"parse_retry"`` — retry after StructuredOutputError
    - ``"guard_retry"`` — retry after guard validation failure
    - ``"iteration"`` — one iteration of a tool-calling loop
    - ``"finalize"`` — structured extraction after tool loop completes
    """

    type: str = "step"
    phase: str = ""
    ts: float = 0.0
    duration_ms: int = 0
    trace_id: str = ""
    verb: str = ""

    # LLM call data
    llm_call: LLMCall = field(default_factory=LLMCall)

    # Parse outcome (structured verbs)
    parsed: bool = True
    parse_error: str | None = None

    # Guard outcomes
    guards: list[GuardOutcome] = field(default_factory=list)

    # Tool outcomes
    tools: list[ToolOutcome] = field(default_factory=list)

    # Controller decision (Complete verb)
    action: str | None = None
    reason: str | None = None
    nudge_preview: str | None = None

    # Controller internals (Complete verb, DefaultController)
    iterations_since_nudge: int | None = None
    consecutive_degenerate: int | None = None
    pending_terminal: bool | None = None
    classifier_called: bool = False


@dataclass
class VerbTrace:
    """Tree-structured trace for a single verb invocation.

    Contains an ordered list of :class:`Step` records — one per LLM call.
    Summary properties are derived from the steps (no separate counters).
    """

    type: str = "verb_trace"
    verb: str = ""
    trace_id: str = ""
    ts: float = 0.0
    duration_ms: int = 0
    request_id: str | None = None
    steps: list[Step] = field(default_factory=list)
    ok: bool = True
    error: str | None = None

    # --- Derived aggregates ---

    @property
    def total_llm_calls(self) -> int:
        """Number of LLM calls made during this verb invocation."""
        return len(self.steps)

    @property
    def parse_retries(self) -> int:
        """Number of parse retry attempts."""
        return sum(1 for s in self.steps if s.phase == "parse_retry")

    @property
    def guard_retries(self) -> int:
        """Number of guard retry attempts."""
        return sum(1 for s in self.steps if s.phase == "guard_retry")

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all LLM calls."""
        return sum(s.llm_call.input_tokens for s in self.steps)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all LLM calls."""
        return sum(s.llm_call.output_tokens for s in self.steps)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) across all LLM calls."""
        return self.total_input_tokens + self.total_output_tokens

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict (including all steps)."""
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to a JSON string. Accepts ``json.dumps`` keyword args."""
        return json.dumps(self.to_dict(), **kwargs)

    # --- Mutation helpers (used internally during verb execution) ---

    def add_step(self, step: Step) -> None:
        """Append a step."""
        self.steps.append(step)


# ---------------------------------------------------------------------------
# Step builders
# ---------------------------------------------------------------------------


def build_step_from_response(
    response: Any,
    *,
    phase: str = "attempt",
    trace_id: str = "",
    verb: str = "",
) -> Step:
    """Build a Step from an ChatResponse (non-Complete verbs).

    Args:
        response: ChatResponse from backend.chat().
        phase: Step phase (attempt, parse_retry, guard_retry, iteration, finalize).
        trace_id: Trace ID for correlation.
        verb: Verb class name.
    """
    call_duration = getattr(response, "_duration_ms", 0)
    return Step(
        phase=phase,
        ts=time.time(),
        duration_ms=call_duration,
        trace_id=trace_id,
        verb=verb,
        llm_call=LLMCall(
            call_id=getattr(response, "call_id", ""),
            input_tokens=getattr(response, "input_tokens", 0),
            output_tokens=getattr(response, "output_tokens", 0),
            finish_reason=getattr(response, "finish_reason", None),
            duration_ms=call_duration,
        ),
        tools=[ToolOutcome(name=tc.name, call_id=tc.id) for tc in (response.tool_calls or [])],
    )


# ---------------------------------------------------------------------------
# Tracer: JSONL serialization with pluggable writer
# ---------------------------------------------------------------------------


class Tracer:
    """Writes JSONL trace records to a pluggable writer.

    The ``Tracer`` handles JSONL serialization.  The *writer* is any writable
    text stream (``IO[str]``) — file, stdout, ``StringIO``, etc.

    For custom destinations (database, socket), either wrap them as an
    ``IO[str]`` or subclass ``Tracer`` and override :meth:`write`.

    Args:
        writer: Writable text stream.
        owns_writer: If True, close the writer on :meth:`close`.
    """

    def __init__(self, writer: IO[str], *, owns_writer: bool = False) -> None:
        self._writer = writer
        self._owns_writer = owns_writer

    def start(self, metadata: dict[str, Any]) -> None:
        """Write metadata header line."""
        self._writer.write(json.dumps({"_meta": metadata}) + "\n")
        self._writer.flush()

    def write(self, record: Step | VerbTrace) -> None:
        """Write one JSONL record (Step or VerbTrace)."""
        self._writer.write(json.dumps(asdict(record)) + "\n")
        self._writer.flush()

    def close(self) -> None:
        """Close the writer if owned."""
        if self._owns_writer:
            self._writer.close()

    def __enter__(self) -> Tracer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class CallbackTracer(Tracer):
    """Tracer that calls a user-provided function for each record.

    The callback receives a plain dict (the serialized trace record).
    """

    def __init__(self, callback: Callable[[dict[str, Any]], None]) -> None:
        super().__init__(sys.stdout, owns_writer=False)  # unused writer
        self._callback = callback

    def start(self, metadata: dict[str, Any]) -> None:
        """Forward metadata to callback."""
        self._callback({"_meta": metadata})

    def write(self, record: Step | VerbTrace) -> None:
        """Forward serialized record to callback."""
        self._callback(asdict(record))

    def close(self) -> None:
        """No-op — callback tracers have nothing to close."""


# ---------------------------------------------------------------------------
# TracerFactory: create Tracer instances with different writers
# ---------------------------------------------------------------------------


class TracerFactory:
    """Factory for creating :class:`Tracer` instances with different writers.

    Example::

        tracer = TracerFactory.file("/tmp/trace.jsonl")
        tracer = TracerFactory.console()
        tracer = TracerFactory.stream(my_stream)
    """

    @staticmethod
    def file(path: str | Path) -> Tracer:
        """Create a tracer that writes JSONL to a file.

        Creates parent directories if needed.

        Args:
            path: Output file path.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return Tracer(p.open("w"), owns_writer=True)

    @staticmethod
    def console() -> Tracer:
        """Create a tracer that writes JSONL to stdout."""
        return Tracer(sys.stdout, owns_writer=False)

    @staticmethod
    def callback(fn: Callable[[dict[str, Any]], None]) -> CallbackTracer:
        """Create a tracer that calls a function for each record.

        The callback receives a plain dict per trace record.

        Args:
            fn: Callable receiving a dict per trace record.
        """
        return CallbackTracer(fn)

    @staticmethod
    def stream(writer: IO[str]) -> Tracer:
        """Create a tracer that writes JSONL to a caller-provided stream.

        The caller retains ownership — :meth:`close` will not close the stream.

        Args:
            writer: Any writable text stream.
        """
        return Tracer(writer, owns_writer=False)


# ---------------------------------------------------------------------------
# TracerBuilder: fluent sub-builder for tracer configuration
# ---------------------------------------------------------------------------


class Builder(Generic[_P]):
    """Fluent sub-builder for configuring a :class:`Tracer`.

    Generic over the parent builder type so it can be embedded in any
    builder without circular imports.  Each method sets the tracer via the
    *on_tracer* callback and returns the parent for continued chaining.

    Example (inside SAIABuilder)::

        @property
        def tracing(self) -> trace.Builder[SAIABuilder]:
            return trace.Builder(self, self._set_tracer)
    """

    def __init__(self, parent: _P, on_tracer: Callable[[Tracer], None]) -> None:
        self._parent = parent
        self._on_tracer = on_tracer

    def file(self, path: str) -> _P:
        """Write JSONL traces to a file. Creates parent dirs if needed."""
        self._on_tracer(TracerFactory.file(path))
        return self._parent

    def console(self) -> _P:
        """Write JSONL traces to stdout."""
        self._on_tracer(TracerFactory.console())
        return self._parent

    def callback(self, fn: Callable[[dict[str, Any]], None]) -> _P:
        """Call a function for each trace record."""
        self._on_tracer(TracerFactory.callback(fn))
        return self._parent

    def stream(self, writer: IO[str]) -> _P:
        """Write JSONL traces to a caller-provided stream."""
        self._on_tracer(TracerFactory.stream(writer))
        return self._parent

    def custom(self, tracer: Tracer) -> _P:
        """Use a pre-built or custom tracer instance."""
        self._on_tracer(tracer)
        return self._parent

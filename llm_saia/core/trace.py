"""Iteration trace for automated analysis of controller decision-making.

Writes one JSONL record per iteration, capturing the observation (what the
controller saw) and the decision (what it did).  Designed for machine
consumption — load with ``pd.read_json(path, lines=True)`` or feed to an
LLM for self-analysis.

The ``Tracer`` class serializes traces to JSONL and writes them to a
pluggable writer (any ``IO[str]``).  Use the factory functions or the
builder to create a tracer with the desired destination.

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
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any, Generic, TypeVar

from llm_saia.core.backend import AgentResponse
from llm_saia.core.controller import Action, Observation

_P = TypeVar("_P")
_CONTENT_PREVIEW_LIMIT = 200


@dataclass
class IterationTrace:
    """Single iteration record for the decision trace."""

    # Identity
    trace_id: str  # Constant across all LLM calls in one verb invocation
    call_id: str  # Unique per _chat() invocation
    iteration: int
    ts: float  # epoch seconds

    # Context
    verb: str  # Verb class name (e.g. "Complete", "Ask")
    phase: str  # "loop", "direct", "finalize"
    request_id: str | None  # User-provided correlation ID

    # Observation — what the controller saw
    has_content: bool
    has_tool_calls: bool
    tool_call_count: int
    tool_names_used: list[str]
    input_tokens: int
    output_tokens: int
    finish_reason: str | None
    content_preview: str

    # Decision — what the controller did
    action: str  # ActionType value
    reason: str
    nudge_preview: str | None

    # Controller internals (None when not using DefaultController)
    iterations_since_nudge: int | None = None
    consecutive_degenerate: int | None = None
    pending_terminal: bool | None = None

    # Cost tracking
    classifier_called: bool = False


def build_trace(
    obs: Observation,
    action: Action,
    response: AgentResponse,
    *,
    trace_id: str = "",
    verb: str = "",
    phase: str = "loop",
    request_id: str | None = None,
    classifier_called: bool = False,
    iterations_since_nudge: int | None = None,
    consecutive_degenerate: int | None = None,
    pending_terminal: bool | None = None,
) -> IterationTrace:
    """Build an ``IterationTrace`` from loop state (Complete verb)."""
    content = response.content or ""
    preview = content[:_CONTENT_PREVIEW_LIMIT] if content else ""
    nudge = action.message[:_CONTENT_PREVIEW_LIMIT] if action.message else None
    tool_names = [tc.name for tc in response.tool_calls] if response.tool_calls else []

    return IterationTrace(
        trace_id=trace_id,
        call_id=response.call_id,
        iteration=obs.iteration,
        ts=time.time(),
        verb=verb,
        phase=phase,
        request_id=request_id,
        has_content=bool(content),
        has_tool_calls=bool(response.tool_calls),
        tool_call_count=len(response.tool_calls) if response.tool_calls else 0,
        tool_names_used=tool_names,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        finish_reason=response.finish_reason,
        content_preview=preview,
        action=action.kind.value,
        reason=action.reason.value,
        nudge_preview=nudge,
        iterations_since_nudge=iterations_since_nudge,
        consecutive_degenerate=consecutive_degenerate,
        pending_terminal=pending_terminal,
        classifier_called=classifier_called,
    )


def build_base_trace(
    response: AgentResponse,
    *,
    trace_id: str = "",
    iteration: int = 0,
    verb: str = "",
    phase: str = "loop",
    request_id: str | None = None,
) -> IterationTrace:
    """Build a simple ``IterationTrace`` for non-Complete verbs.

    Unlike :func:`build_trace`, this does not require an :class:`Observation`
    or :class:`Action` — it infers the action from the response directly.
    """
    content = response.content or ""
    preview = content[:_CONTENT_PREVIEW_LIMIT] if content else ""
    tool_names = [tc.name for tc in response.tool_calls] if response.tool_calls else []
    action = "execute_tools" if response.tool_calls else "complete"

    return IterationTrace(
        trace_id=trace_id,
        call_id=response.call_id,
        iteration=iteration,
        ts=time.time(),
        verb=verb,
        phase=phase,
        request_id=request_id,
        has_content=bool(content),
        has_tool_calls=bool(response.tool_calls),
        tool_call_count=len(response.tool_calls) if response.tool_calls else 0,
        tool_names_used=tool_names,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        finish_reason=response.finish_reason,
        content_preview=preview,
        action=action,
        reason="base_trace",
        nudge_preview=None,
    )


# ---------------------------------------------------------------------------
# Tracer: JSONL serialization with pluggable writer
# ---------------------------------------------------------------------------


class Tracer:
    """Writes JSONL iteration traces to a pluggable writer.

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

    def write(self, trace: IterationTrace) -> None:
        """Write one JSONL record."""
        self._writer.write(json.dumps(asdict(trace)) + "\n")
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

    The callback receives a plain dict (the serialized ``IterationTrace``).
    """

    def __init__(self, callback: Callable[[dict[str, Any]], None]) -> None:
        super().__init__(sys.stdout, owns_writer=False)  # unused writer
        self._callback = callback

    def start(self, metadata: dict[str, Any]) -> None:
        self._callback({"_meta": metadata})

    def write(self, trace: IterationTrace) -> None:
        self._callback(asdict(trace))

    def close(self) -> None:
        pass


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

        The callback receives a plain dict (the serialized ``IterationTrace``).

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

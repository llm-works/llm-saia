# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Loop strategy that terminates on a successful schema parse.

Structured-output verbs run through :class:`_LoopRunner` under this strategy
instead of a parallel handler. Each LLM response is parsed against the schema;
success terminates the loop with a typed value, failure delegates to the
existing :func:`schema_retry` iteration-guard family for retry feedback and
issues INSTRUCT until parse succeeds or the retry budget is exhausted.

A retry restarts from the point before the first failed answer: the failed
answer is dropped and only the feedback is added, so every attempt sends the
original prompt plus one feedback message. Resending a failed answer would
grow each attempt by its full length (a cut-off answer is as long as the
token limit) and anchor the model on the text that failed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from .errors import StructuredOutputError, TruncatedResponseError
from .guard import IterationContext, IterationGuard
from .loop import LoopAction, LoopDecision
from .parse_failure import (
    ParseFailure,
    ParseFailureKind,
    diagnose_json_error,
    diagnose_schema_error,
)
from .schema import parse as parse_schema

if TYPE_CHECKING:
    from .backend import ChatResponse
    from .config import CallOptions, Config
    from .conversation import Message

T = TypeVar("T")


class _StrategyHost(Protocol):
    """Verb capabilities the strategy needs.

    Kept narrow so the strategy stays testable and the coupling stays honest.
    """

    _config: Config

    def _eval_single_guard(self, guard: IterationGuard, ctx: IterationContext) -> str | None: ...


class SchemaTerminatingStrategy(Generic[T]):
    """Terminate the loop when the LLM response parses to ``schema``.

    Loop mechanics stay in :class:`_LoopRunner`; this strategy owns the parse
    attempt and the parse-retry decision. On success, the loop halts with the
    parsed value on :attr:`parsed_value`. On failure, ``schema_retry``-family
    iteration guards produce feedback and the strategy issues INSTRUCT.

    Args:
        schema: Dataclass or Pydantic model the response must parse to.
        host: Verb providing config, guard-eval, and error-formatting hooks.
        call: Effective :class:`CallOptions` for this invocation.
        retry_on_parse_failure: When False, a parse failure yields FAIL on the
            first miss (used by output-guard retries and post-tool finalize,
            which must not re-drive the parse-retry loop).
    """

    def __init__(
        self,
        schema: type[T],
        host: _StrategyHost,
        call: CallOptions,
        *,
        retry_on_parse_failure: bool = True,
    ) -> None:
        self._schema = schema
        self._host = host
        self._retry = retry_on_parse_failure
        self._parse_guards: tuple[IterationGuard, ...] = (
            tuple(g for g in call.iteration_guards if g.parse_max_retries > 0)
            if retry_on_parse_failure
            else ()
        )
        self._parse_budget = 1 + sum(g.parse_max_retries for g in self._parse_guards)

        self.parsed_value: T | None = None
        self.last_parse_error: StructuredOutputError | None = None
        self.parse_attempts: int = 0
        # Message count before the first failed answer; every retry rewinds here.
        self._retry_from: int | None = None

    @property
    def parse_budget(self) -> int:
        """Total attempts allowed: 1 initial + sum of guard retries."""
        return self._parse_budget

    async def decide(
        self,
        response: ChatResponse,
        messages: list[Message],
        iteration: int,
        blocking_feedback: str | None,
        advisory_feedback: str | None,
    ) -> LoopDecision:
        """Route each iteration to tools, retry, success, or fail."""
        if blocking_feedback:
            return LoopDecision(
                action=LoopAction.INSTRUCT,
                message=blocking_feedback,
                reason="blocking_guard",
            )
        # Tools compose with schema termination: iterate tool calls, then parse
        # once the model returns a candidate answer.
        if response.tool_calls:
            return LoopDecision(action=LoopAction.EXECUTE_TOOLS, reason="has_tool_calls")
        return self._decide_after_parse(response, len(messages))

    def _decide_after_parse(self, response: ChatResponse, message_count: int) -> LoopDecision:
        """Attempt to parse the response and translate the outcome to a decision.

        ``message_count`` is the loop's message count before ``response``.
        """
        self.parse_attempts += 1
        try:
            self.parsed_value = self._parse(response)
        except StructuredOutputError as err:
            self.last_parse_error = err
            if self._retry_from is None:
                self._retry_from = message_count
            return self._decide_after_parse_failure(response)
        self.last_parse_error = None
        return LoopDecision(
            action=LoopAction.COMPLETE, output=response.content, reason="schema_parsed"
        )

    def _decide_after_parse_failure(self, response: ChatResponse) -> LoopDecision:
        """Choose between fail-fast, exhausted-budget FAIL, or retry INSTRUCT."""
        if not self._retry or self.parse_attempts >= self._parse_budget:
            reason = "parse_error_exhausted" if self._retry else "parse_error"
            return LoopDecision(action=LoopAction.FAIL, output=response.content, reason=reason)
        feedback = self._parse_retry_feedback(response)
        if feedback is None:
            return LoopDecision(
                action=LoopAction.FAIL,
                output=response.content,
                reason="parse_error_no_feedback",
            )
        return LoopDecision(
            action=LoopAction.INSTRUCT,
            message=feedback,
            reason="parse_retry",
            rewind_to=self._retry_from,
        )

    def on_iteration_complete(self, decision: LoopDecision, tokens: int) -> None:
        """Strategy has no per-iteration scoring; hook kept for protocol compliance."""

    # -- internals --

    def _parse(self, response: ChatResponse) -> T:
        content = response.content
        parser = self._host._config.json_parser or json.loads
        try:
            data = parser(content)
        except Exception as e:
            failure = diagnose_json_error(
                content, e, response.finish_reason, response.output_tokens
            )
            raise self._error(failure, content, e) from e
        try:
            return parse_schema(data, self._schema)
        except (TypeError, ValueError) as e:
            raise self._error(diagnose_schema_error(e), content, e) from e

    def _error(
        self, failure: ParseFailure, content: str, cause: Exception
    ) -> StructuredOutputError:
        """Build the error raised for ``failure``; a cut-off answer gets its own type."""
        name = self._schema.__name__
        error_type = StructuredOutputError
        if failure.kind is ParseFailureKind.TRUNCATED:
            error_type = TruncatedResponseError
            message = (
                f"LLM response for {name} was cut off before the JSON was complete. "
                "Consider increasing max_call_tokens in your config."
            )
        elif failure.kind is ParseFailureKind.SCHEMA:
            message = f"Response does not match {name}: {failure.detail}"
        else:
            message = f"LLM returned invalid JSON for {name}: {cause}"
        return error_type(
            message,
            raw_content=content,
            schema_name=name,
            parse_error=str(cause),
            failure=failure,
        )

    def _parse_retry_feedback(self, response: ChatResponse) -> str | None:
        """Build parse-retry feedback by evaluating schema_retry-family guards."""
        if not self._parse_guards:
            return None
        ctx = IterationContext(
            response=response,
            iteration=self.parse_attempts - 1,
            max_iterations=self._parse_budget,
            parse_error=self.last_parse_error,
        )
        parts: list[str] = []
        for guard in self._parse_guards:
            fb = self._host._eval_single_guard(guard, ctx)
            if fb is not None:
                parts.append(fb)
        return "\n\n".join(parts) if parts else None


__all__ = ["SchemaTerminatingStrategy"]

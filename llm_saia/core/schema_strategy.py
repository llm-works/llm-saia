# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Loop strategy that terminates on a successful schema parse.

Structured-output verbs run through :class:`_LoopRunner` under this strategy
instead of a parallel handler. Each LLM response is parsed against the schema;
success terminates the loop with a typed value, failure delegates to the
existing :func:`schema_retry` iteration-guard family for retry feedback and
issues INSTRUCT until parse succeeds or the retry budget is exhausted.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from .errors import StructuredOutputError
from .guard import IterationContext, IterationGuard
from .loop import LoopAction, LoopDecision
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

    def _structured_output_error(
        self, error: Exception, content: str, schema_name: str
    ) -> StructuredOutputError: ...


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
        return self._decide_after_parse(response)

    def _decide_after_parse(self, response: ChatResponse) -> LoopDecision:
        """Attempt to parse the response and translate the outcome to a decision."""
        self.parse_attempts += 1
        try:
            self.parsed_value = self._parse(response.content)
        except StructuredOutputError as err:
            self.last_parse_error = err
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
        return LoopDecision(action=LoopAction.INSTRUCT, message=feedback, reason="parse_retry")

    def on_iteration_complete(self, decision: LoopDecision, tokens: int) -> None:
        """Strategy has no per-iteration scoring; hook kept for protocol compliance."""
        return None

    # -- internals --

    def _parse(self, content: str) -> T:
        parser = self._host._config.json_parser or json.loads
        try:
            data = parser(content)
        except Exception as e:
            raise self._host._structured_output_error(e, content, self._schema.__name__) from e
        try:
            return parse_schema(data, self._schema)
        except (TypeError, ValueError) as e:
            raise StructuredOutputError(
                f"Response does not match {self._schema.__name__}: {e}",
                raw_content=content,
                schema_name=self._schema.__name__,
                parse_error=str(e),
            ) from e

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

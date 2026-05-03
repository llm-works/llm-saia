"""Structured output handling for verbs.

Handles JSON schema-based structured output completion, including
parse retry logic with guard support.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, TypeVar

from .backend import ChatResponse
from .conversation import ListConversation, Message, Role
from .errors import StructuredOutputError
from .guard import IterationContext, IterationGuard
from .schema import dataclass_to_json_schema, parse_json_to_dataclass

if TYPE_CHECKING:
    from .config import CallOptions
    from .conversation import ConversationLike
    from .trace import Step, VerbTrace

T = TypeVar("T")


class _ParseRetryState(TypedDict):
    """State for parse retry loop (avoids many parameters)."""

    error: StructuredOutputError | None
    feedback: str | None


class _StructuredHost(Protocol):
    """Protocol for capabilities the structured output handler needs from its host."""

    _PREVIEW_LIMIT: int
    _TRACE_LIMIT: int
    _TRUNCATION_INDICATORS: tuple[str, ...]

    def _has_tools(self) -> bool:
        """Check if tools are configured."""
        ...

    def _get_call_options(self, override: CallOptions | None) -> CallOptions:
        """Get effective call options."""
        ...

    def _max_tokens(self, config: CallOptions) -> int | None:
        """Get max tokens from config."""
        ...

    def _resolve_temperature(self, override: CallOptions | None) -> float | None:
        """Resolve temperature from override or config."""
        ...

    async def _chat(
        self,
        messages: list[Message],
        max_tokens: int | None,
        temperature: float | None,
        *,
        call: CallOptions | None = None,
        response_schema: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
    ) -> ChatResponse:
        """Call the LLM backend."""
        ...

    async def _loop(
        self,
        prompt: str,
        run: CallOptions | None = None,
        schema: type[T] | None = None,
        trace_id: str = "",
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
        on_iteration: Callable[[int, ChatResponse], Awaitable[None]] | None = None,
        resume: bool = False,
        abort_signal: asyncio.Event | None = None,
    ) -> tuple[str, T | None]:
        """Run the tool loop."""
        ...

    def _record_step(
        self, response: ChatResponse, *, phase: str, _trace: VerbTrace | None = None
    ) -> None:
        """Record a step to the trace."""
        ...

    def _to_message(self, response: ChatResponse) -> Message:
        """Convert ChatResponse to Message."""
        ...

    @staticmethod
    async def _append_msg(target: Any, msg: Message) -> None:
        """Append message to conversation."""
        ...

    @staticmethod
    def _fork_conversation(conv: ConversationLike | None) -> ConversationLike | None:
        """Fork conversation for isolated attempt."""
        ...

    @staticmethod
    async def _merge_conversation(
        target: ConversationLike | None, source: ConversationLike | None
    ) -> None:
        """Merge forked conversation back to target."""
        ...

    def _structured_output_error(
        self, error: Exception, content: str, schema_name: str
    ) -> StructuredOutputError:
        """Create structured output error."""
        ...

    def _init_verb_trace(self, trace_id: str = "") -> VerbTrace:
        """Initialize a new verb trace."""
        ...

    def _emit_verb_trace(self, trace: VerbTrace, reason: str | None = None) -> None:
        """Emit completed verb trace."""
        ...

    def _eval_single_guard(self, guard: IterationGuard, ctx: IterationContext) -> str | None:
        """Evaluate a single guard."""
        ...

    async def _apply_guards(
        self,
        prompt: str,
        result: T,
        schema: type[T],
        run: CallOptions | None,
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
    ) -> T:
        """Apply output guards to result."""
        ...

    def _log_structured_attempt(
        self,
        step: Step,
        step_num: int,
        *,
        error: str | None = None,
        guard: str | None = None,
        raw_content: str | None = None,
    ) -> None:
        """Log structured output attempt."""
        ...

    @property
    def _config(self) -> Any:
        """Get configuration."""
        ...

    @property
    def _lg(self) -> Any:
        """Logger instance."""
        ...


class _StructuredOutputHandler:
    """Handles structured output completion with parse retry support."""

    def __init__(self, host: _StructuredHost):
        """Initialize with host providing required capabilities."""
        self._host = host

    async def finalize(
        self,
        prompt: str,
        content: str,
        schema: type[T] | None,
        trace_id: str = "",
        temperature: float | None = None,
        run: CallOptions | None = None,
        _trace: VerbTrace | None = None,
    ) -> tuple[str, T | None]:
        """Finalize result, optionally parsing structured output."""
        if not schema:
            return content, None
        structured_prompt = f"{prompt}\n\nBased on the following information:\n{content}"
        json_schema = dataclass_to_json_schema(schema)
        response = await self._host._chat(
            [Message(role=Role.USER, content=structured_prompt)],
            max_tokens=None,
            temperature=temperature,
            call=run,
            response_schema=json_schema,
            tools=[],
        )
        self._host._record_step(response, phase="finalize", _trace=_trace)
        return content, self._parse_finalize_response(response.content, schema)

    def _parse_finalize_response(self, content: str, schema: type[T]) -> T:
        """Parse JSON response from finalize phase into schema."""
        h = self._host
        parser = h._config.json_parser or json.loads
        try:
            data = parser(content)
        except Exception as e:
            self._log_finalize_parse_error(e, content, schema.__name__)
            raise h._structured_output_error(e, content, schema.__name__) from e
        try:
            return parse_json_to_dataclass(data, schema)
        except (TypeError, ValueError) as e:
            raise StructuredOutputError(
                f"Response does not match {schema.__name__}: {e}",
                raw_content=content,
                schema_name=schema.__name__,
                parse_error=str(e),
            ) from e

    def _log_finalize_parse_error(self, error: Exception, content: str, schema_name: str) -> None:
        """Log a JSON parse error during finalize phase."""
        h = self._host
        h._lg.warning(
            "json parse error in finalize",
            extra={
                "exception": error,
                "content_preview": content[: h._PREVIEW_LIMIT],
                "schema": schema_name,
            },
        )

    async def complete_structured(
        self,
        prompt: str,
        schema: type[T],
        run: CallOptions | None = None,
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
    ) -> T:
        """Complete structured with iteration guards and output guards."""
        h = self._host
        trace = _trace if _trace is not None else h._init_verb_trace()
        config = h._get_call_options(run)
        parse_retry_guards = tuple(g for g in config.iteration_guards if g.parse_max_retries > 0)
        max_attempts = 1 + sum(g.parse_max_retries for g in parse_retry_guards)
        state: _ParseRetryState = {"error": None, "feedback": None}

        for attempt in range(max_attempts):
            try:
                return await self._attempt_with_guards(
                    prompt, schema, run, conversation, state, attempt, trace, _trace
                )
            except StructuredOutputError as e:
                should_retry = self._handle_parse_error(
                    e, parse_retry_guards, attempt, max_attempts, trace, state
                )
                if should_retry:
                    continue
                self._emit_trace_if_root(trace, _trace)
                raise

        raise state["error"]  # type: ignore[misc]

    async def _attempt_with_guards(
        self,
        prompt: str,
        schema: type[T],
        run: CallOptions | None,
        conversation: ConversationLike | None,
        state: _ParseRetryState,
        attempt: int,
        trace: VerbTrace,
        parent_trace: VerbTrace | None,
    ) -> T:
        """Single structured attempt with output guards applied."""
        h = self._host
        phase = "attempt" if attempt == 0 else "parse_retry"
        step_num_before = len(trace.steps)
        result = await self._attempt_cycle(
            prompt,
            schema,
            run,
            conversation,
            state["error"],
            state["feedback"],
            attempt,
            phase,
            trace,
        )
        if trace.steps and len(trace.steps) > step_num_before:
            h._log_structured_attempt(trace.steps[-1], len(trace.steps))
        result = await h._apply_guards(
            prompt, result, schema, run, conversation=conversation, _trace=trace
        )
        self._emit_trace_if_root(trace, parent_trace)
        return result

    def _emit_trace_if_root(
        self, trace: VerbTrace, parent: VerbTrace | None, reason: str | None = None
    ) -> None:
        """Emit verb trace only if this is the root trace."""
        if parent is None:
            self._host._emit_verb_trace(trace, reason)

    def _handle_parse_error(
        self,
        error: StructuredOutputError,
        guards: tuple[IterationGuard, ...],
        attempt: int,
        max_attempts: int,
        trace: VerbTrace,
        state: _ParseRetryState,
    ) -> bool:
        """Handle parse error. Returns True to retry."""
        h = self._host
        self._mark_last_step_parse_failure(trace, error)
        if trace.steps:
            h._log_structured_attempt(
                trace.steps[-1],
                len(trace.steps),
                error=error.parse_error,
                raw_content=error.raw_content,
            )
        response = ChatResponse(content=error.raw_content or "", tool_calls=[])
        feedback = self._eval_parse_retry_guards(
            guards, response, error, attempt, max_attempts, trace
        )
        state["error"] = error
        state["feedback"] = feedback
        if feedback and attempt < max_attempts - 1:
            schema_name = error.schema_name or "unknown"
            self._log_parse_retry(schema_name, attempt + 1, max_attempts, error)
            return True
        return False

    def _eval_parse_retry_guards(
        self,
        guards: tuple[IterationGuard, ...],
        response: ChatResponse,
        error: StructuredOutputError,
        attempt: int,
        max_attempts: int,
        trace: VerbTrace,
    ) -> str | None:
        """Evaluate iteration guards in parse retry context."""
        if not guards:
            return None
        h = self._host
        ctx = IterationContext(
            response=response, iteration=attempt, max_iterations=max_attempts, parse_error=error
        )
        feedback_parts: list[str] = []
        for guard in guards:
            feedback = h._eval_single_guard(guard, ctx)
            if feedback is not None:
                self._log_parse_guard_trigger(guard.name, attempt, feedback)
                feedback_parts.append(feedback)
        return "\n\n".join(feedback_parts) if feedback_parts else None

    def _log_parse_guard_trigger(self, name: str | None, attempt: int, feedback: str) -> None:
        """Log when a parse guard triggers a retry."""
        self._host._lg.debug(
            "parse guard triggered retry",
            extra={"guard": name or "guard", "attempt": attempt, "feedback": feedback[:100]},
        )

    async def _attempt_cycle(
        self,
        prompt: str,
        schema: type[T],
        run: CallOptions | None,
        conversation: ConversationLike | None,
        last_error: StructuredOutputError | None,
        last_feedback: str | None,
        attempt: int,
        phase: str,
        trace: VerbTrace,
    ) -> T:
        """Run one parse attempt: build prompt, fork conv, call backend."""
        h = self._host
        use_prompt = self._retry_prompt_or_original(
            prompt, schema, last_error, last_feedback, attempt
        )
        attempt_conv = h._fork_conversation(conversation)
        result = await self._complete_attempt(
            use_prompt, schema, run, conversation=attempt_conv, _trace=trace, _phase=phase
        )
        await h._merge_conversation(conversation, attempt_conv)
        return result

    @staticmethod
    def _mark_last_step_parse_failure(trace: VerbTrace, error: StructuredOutputError) -> None:
        """Mark the last step in the trace as a parse failure."""
        if trace.steps:
            trace.steps[-1].parsed = False
            trace.steps[-1].parse_error = error.parse_error

    async def _complete_attempt(
        self,
        prompt: str,
        schema: type[T],
        run: CallOptions | None = None,
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
        _phase: str = "attempt",
    ) -> T:
        """Single attempt at structured completion."""
        h = self._host
        if h._has_tools():
            _, result = await h._loop(
                prompt, run=run, schema=schema, conversation=conversation, _trace=_trace
            )
            if result is not None:
                return result
        conv = conversation if conversation is not None else ListConversation()
        await h._append_msg(conv, Message(role=Role.USER, content=prompt))
        json_schema = dataclass_to_json_schema(schema)
        config = h._get_call_options(run)
        response = await h._chat(
            conv.as_messages(),
            max_tokens=h._max_tokens(config),
            temperature=h._resolve_temperature(run),
            call=config,
            response_schema=json_schema,
            tools=[],
        )
        await h._append_msg(conv, h._to_message(response))
        h._record_step(response, phase=_phase, _trace=_trace)
        return self._parse_response(response.content, schema)

    def _parse_response(self, content: str, schema: type[T]) -> T:
        """Parse JSON response into schema."""
        h = self._host
        parser = h._config.json_parser or json.loads
        try:
            data = parser(content)
        except Exception as e:
            raise h._structured_output_error(e, content, schema.__name__) from e
        try:
            return parse_json_to_dataclass(data, schema)
        except (TypeError, ValueError) as e:
            raise StructuredOutputError(
                f"Response does not match {schema.__name__}: {e}",
                raw_content=content,
                schema_name=schema.__name__,
                parse_error=str(e),
            ) from e

    def _retry_prompt_or_original(
        self,
        prompt: str,
        schema: type[T],
        last_error: StructuredOutputError | None,
        last_feedback: str | None,
        attempt: int,
    ) -> str:
        """Return original prompt on first attempt, retry prompt on subsequent ones."""
        if attempt == 0:
            return prompt
        if last_feedback:
            return self._build_retry_prompt_with_feedback(prompt, last_error, last_feedback)
        return self._build_retry_prompt(prompt, schema, last_error)  # type: ignore[arg-type]

    def _build_retry_prompt(
        self, original_prompt: str, schema: type[T], error: StructuredOutputError
    ) -> str:
        """Build a retry prompt with feedback about the parse failure."""
        parts = [original_prompt, "\n\n---\n\nYour previous response could not be parsed."]
        if error.parse_error:
            parts.append(f"\n\nParse error: {error.parse_error}")
        if error.raw_content:
            raw = error.raw_content
            if len(raw) > 500:
                raw = raw[:500] + "... (truncated)"
            parts.append(f"\n\nYour response was:\n```\n{raw}\n```")
        parts.append(
            f"\n\nPlease provide a valid JSON response matching the {schema.__name__} schema."
        )
        return "".join(parts)

    def _build_retry_prompt_with_feedback(
        self, original_prompt: str, error: StructuredOutputError | None, feedback: str
    ) -> str:
        """Build a retry prompt using guard-provided feedback."""
        parts = [original_prompt, "\n\n---\n\n", feedback]
        if error and error.raw_content:
            raw = error.raw_content
            if len(raw) > 500:
                raw = raw[:500] + "... (truncated)"
            parts.append(f"\n\nYour response was:\n```\n{raw}\n```")
        return "".join(parts)

    def _log_parse_retry(
        self, schema_name: str, attempt: int, max_attempts: int, error: StructuredOutputError
    ) -> None:
        """Log parse retry attempt."""
        self._host._lg.debug(
            "parse retry",
            extra={
                "schema": schema_name,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "parse_error": error.parse_error,
            },
        )

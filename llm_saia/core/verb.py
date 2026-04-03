"""Base class for SAIA verbs."""

from __future__ import annotations

import json
import time
import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Self, TypeVar

from llm_saia.core.backend import AgentResponse
from llm_saia.core.config import DEFAULT_CALL, CallOptions, Config
from llm_saia.core.configurable import Configurable
from llm_saia.core.conversation import (
    ConversationLike,
    ListConversation,
    Message,
    MessageAppendable,
    Role,
    ToolCall,
)
from llm_saia.core.errors import StructuredOutputError
from llm_saia.core.guards import OutputGuardMixin
from llm_saia.core.logging import VerbLoggingMixin
from llm_saia.core.schema import dataclass_to_json_schema, parse_json_to_dataclass

if TYPE_CHECKING:
    from llm_saia.core.backend import Backend
    from llm_saia.core.trace import Tracer

T = TypeVar("T")


class Verb(OutputGuardMixin, VerbLoggingMixin, Configurable):
    """Base class for all verbs. Subclass this to create custom verbs."""

    # Truncation limit for log previews
    _PREVIEW_LIMIT = 100

    def __init__(self, config: Config):
        """Initialize verb with configuration."""
        self._config = config
        self._memory: dict[str, Any] = {}  # Verbs don't use memory
        self._lg = config.lg

    def _clone(self, config: Config) -> Self:
        """Create a new instance with the given config."""
        return self.__class__(config)

    @property
    def _backend(self) -> Backend:
        """Get the configured backend."""
        return self._config.backend

    def _has_tools(self) -> bool:
        """Check if tools are configured."""
        return bool(self._config.tools and self._config.executor)

    @property
    def _call(self) -> CallOptions:
        """Get effective call options (instance default or global default)."""
        return self._config.call or DEFAULT_CALL

    def _get_call_options(self, override: CallOptions | None = None) -> CallOptions:
        """Get effective call options: override > instance default > global default."""
        return override or self._call

    @staticmethod
    def _generate_id() -> str:
        """Generate a short unique ID for tracing (8-char hex)."""
        return uuid.uuid4().hex[:8]

    def _resolve_tracer(
        self,
        tracer: Tracer | None,
        metadata: dict[str, Any],
    ) -> tuple[bool, Tracer | None]:
        """Resolve per-call vs config tracer and call start().

        Returns ``(owns_tracer, active_tracer)``.  A per-call tracer is
        *owned* (the caller is responsible for closing it); the config
        tracer is *borrowed* (shared across calls, never closed by a verb).
        """
        owns = tracer is not None
        active = tracer or self._config.tracer
        if active:
            active.start(metadata)
        return owns, active

    def _write_base_trace(
        self,
        response: AgentResponse,
        *,
        trace_id: str,
        iteration: int = 0,
        phase: str = "loop",
    ) -> None:
        """Write a base trace record if a tracer is configured."""
        tracer = self._config.tracer
        if not tracer:
            return
        from llm_saia.core.trace import build_base_trace

        record = build_base_trace(
            response,
            trace_id=trace_id,
            iteration=iteration,
            verb=self.__class__.__name__,
            phase=phase,
            request_id=self._call.request_id,
        )
        tracer.write(record)

    def _resolve_temperature(self, override: CallOptions | None) -> float | None:
        """Resolve temperature: override CallOptions > instance CallOptions."""
        if override is not None and override.temperature is not None:
            return override.temperature
        return self._call.temperature

    async def _chat(
        self,
        messages: list[Message],
        max_tokens: int | None,
        temperature: float | None = None,
    ) -> AgentResponse:
        """Execute a single chat call."""
        call_id = self._generate_id()
        if self._lg:
            last_msg = messages[-1] if messages else None
            self._lg.trace(
                "sending chat",
                extra={
                    "call_id": call_id,
                    "msg_count": len(messages),
                    "last_role": last_msg.role if last_msg else None,
                    "content": last_msg.content if last_msg else None,
                },
            )
        response = await self._backend.chat(
            messages,
            system=self._call.system,
            tools=self._config.tools if self._config.tools else None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response.call_id = call_id
        return response

    async def _loop(
        self,
        prompt: str,
        run: CallOptions | None = None,
        schema: type[T] | None = None,
        trace_id: str = "",
        conversation: ConversationLike | None = None,
    ) -> tuple[str, T | None]:
        """Execute prompt with tool-calling loop.

        Args:
            prompt: The user prompt to process.
            run: Optional call options override.
            schema: Optional schema for structured output parsing.
            trace_id: Optional trace ID for correlation.
            conversation: Optional conversation object for message management.
                If provided, messages are appended to it and read via as_messages().
                Enables external systems (e.g., kelt) to handle compaction and persistence.
                If None, uses a simple internal list (original behavior).
        """
        config = self._get_call_options(run)
        conv = conversation if conversation is not None else ListConversation()
        conv.append(Message(role=Role.USER, content=prompt))
        start_time, iteration, total_tokens, last_content = time.monotonic(), 0, 0, ""
        trace_id = trace_id or self._generate_id()

        self._log_loop_start(config)
        max_tokens = config.max_call_tokens if config.max_call_tokens > 0 else None
        temperature = self._resolve_temperature(run)

        while not self._should_stop(config, iteration, start_time, total_tokens):
            response = await self._chat(conv.as_messages(), max_tokens, temperature)
            total_tokens += response.input_tokens + response.output_tokens
            last_content = response.content
            conv.append(self._to_message(response))
            self._log_response(response, iteration, total_tokens)
            self._check_tool_support(response)
            self._write_base_trace(response, trace_id=trace_id, iteration=iteration, phase="loop")

            if response.tool_calls:
                await self._execute_tools(response.tool_calls, conv)
                iteration += 1
            else:
                self._log_loop_complete(iteration, start_time, total_tokens, response.content)
                return await self._finalize(prompt, response.content, schema, trace_id, temperature)

        self._log_limit_reached(config, iteration, start_time, total_tokens)
        return await self._finalize(prompt, last_content, schema, trace_id, temperature)

    def _should_stop(
        self, config: CallOptions, iteration: int, start_time: float, total_tokens: int
    ) -> bool:
        """Check if loop should stop."""
        if config.max_iterations > 0 and iteration >= config.max_iterations:
            return True
        if config.timeout_secs > 0 and (time.monotonic() - start_time) >= config.timeout_secs:
            return True
        if config.max_total_tokens > 0 and total_tokens >= config.max_total_tokens:
            return True
        return False

    def _to_message(self, response: AgentResponse) -> Message:
        """Convert AgentResponse to Message."""
        return Message(
            role=Role.ASSISTANT,
            content=response.content,
            tool_calls=response.tool_calls if response.tool_calls else None,
        )

    @staticmethod
    def _fork_conversation(
        conversation: ConversationLike | None,
    ) -> ConversationLike | None:
        """Create a working copy of a conversation for isolated operations.

        Returns None when the caller did not provide a conversation (the
        downstream helpers will create a throwaway ListConversation).
        """
        if conversation is None:
            return None
        fork = ListConversation()
        for msg in conversation.as_messages():
            fork.append(msg)
        return fork

    @staticmethod
    def _merge_conversation(
        target: ConversationLike | None,
        source: ConversationLike | None,
    ) -> None:
        """Append new messages from *source* back into *target*.

        Only messages added after the fork point (i.e. those beyond the
        original length of *target*) are copied.
        """
        if target is None or source is None:
            return
        base_len = len(target.as_messages())
        for msg in source.as_messages()[base_len:]:
            target.append(msg)

    async def _execute_tools(self, tool_calls: list[ToolCall], messages: MessageAppendable) -> None:
        """Execute tool calls and append results.

        Args:
            tool_calls: Tool calls to execute.
            messages: Object supporting append() - either list[Message] or ConversationLike.
        """
        if not self._config.executor:
            if self._lg:
                self._lg.warning(
                    "tool calls received but no executor configured",
                    extra={"tool_count": len(tool_calls)},
                )
            return
        for tc in tool_calls:
            result = await self._execute_single_tool(tc)
            messages.append(Message(role=Role.TOOL, content=str(result), tool_call_id=tc.id))

    async def _execute_single_tool(self, tc: ToolCall) -> str:
        """Execute a single tool call with logging."""
        self._log_tool_start(tc)
        try:
            result = await self._config.executor(tc.name, tc.arguments)  # type: ignore[misc]
        except Exception as e:
            self._log_tool_error(tc, e)
            return f"Error: {e}"
        self._log_tool_success(tc, result)
        return str(result)

    def _log_tool_start(self, tc: ToolCall) -> None:
        """Log tool execution start."""
        if self._lg:
            self._lg.trace(
                "executing tool...",
                extra={"tool": tc.name, "id": tc.id, "tool_args": tc.arguments},
            )

    def _log_tool_success(self, tc: ToolCall, result: Any) -> None:
        """Log successful tool execution."""
        if self._lg:
            extra = {"tool": tc.name, "id": tc.id, "tool_args": tc.arguments, "result": str(result)}
            self._lg.trace("tool executed", extra=extra)

    def _log_tool_error(self, tc: ToolCall, error: Exception) -> None:
        """Log failed tool execution."""
        if self._lg:
            self._lg.warning(
                "tool execution failed",
                extra={"tool": tc.name, "id": tc.id, "tool_args": tc.arguments, "exception": error},
            )

    async def _finalize(
        self,
        prompt: str,
        content: str,
        schema: type[T] | None,
        trace_id: str = "",
        temperature: float | None = None,
    ) -> tuple[str, T | None]:
        """Finalize result, optionally parsing structured output."""
        if schema:
            # Request structured output with schema
            structured_prompt = f"{prompt}\n\nBased on the following information:\n{content}"
            json_schema = dataclass_to_json_schema(schema)
            response = await self._backend.chat(
                [Message(role=Role.USER, content=structured_prompt)],
                system=self._call.system,
                response_schema=json_schema,
                temperature=temperature,
            )
            response.call_id = self._generate_id()
            self._write_base_trace(response, trace_id=trace_id, phase="finalize")
            try:
                data = json.loads(response.content)
            except json.JSONDecodeError as e:
                if self._lg:
                    preview = self._truncate(response.content, self._PREVIEW_LIMIT)
                    self._lg.warning(
                        "json parse error in finalize",
                        extra={
                            "exception": e,
                            "content_preview": preview,
                            "schema": schema.__name__,
                        },
                    )
                raise self._structured_output_error(e, response.content, schema.__name__) from e
            result = parse_json_to_dataclass(data, schema)
            return content, result
        return content, None

    # --- High-level helpers for verbs ---

    async def _complete(
        self,
        prompt: str,
        run: CallOptions | None = None,
        conversation: ConversationLike | None = None,
    ) -> str:
        """Complete with tools if available, otherwise direct.

        Applies output guards if configured.
        """
        if self._has_tools():
            content, _ = await self._loop(prompt, run=run, conversation=conversation)
            return await self._apply_text_guards(prompt, content, run, conversation=conversation)
        conv = conversation if conversation is not None else ListConversation()
        conv.append(Message(role=Role.USER, content=prompt))
        trace_id = self._generate_id()
        response = await self._backend.chat(
            conv.as_messages(),
            system=self._call.system,
            temperature=self._resolve_temperature(run),
        )
        response.call_id = self._generate_id()
        conv.append(self._to_message(response))
        self._write_base_trace(response, trace_id=trace_id, phase="direct")
        return await self._apply_text_guards(
            prompt, response.content, run, conversation=conversation
        )

    async def _complete_text_attempt(
        self,
        prompt: str,
        run: CallOptions | None = None,
        phase: str = "direct",
        conversation: ConversationLike | None = None,
    ) -> str:
        """Single attempt at text completion without applying guards.

        Used by guard retry logic to avoid recursion.
        """
        if self._has_tools():
            content, _ = await self._loop(prompt, run=run, conversation=conversation)
            return content
        conv = conversation if conversation is not None else ListConversation()
        conv.append(Message(role=Role.USER, content=prompt))
        trace_id = self._generate_id()
        response = await self._backend.chat(
            conv.as_messages(),
            system=self._call.system,
            temperature=self._resolve_temperature(run),
        )
        response.call_id = self._generate_id()
        conv.append(self._to_message(response))
        self._write_base_trace(response, trace_id=trace_id, phase=phase)
        return response.content

    async def _complete_structured(
        self,
        prompt: str,
        schema: type[T],
        run: CallOptions | None = None,
        conversation: ConversationLike | None = None,
    ) -> T:
        """Complete structured with retry on StructuredOutputError.

        Retry count controlled by CallOptions.parse_retries (default: 0).
        Output guards are applied after successful parsing.
        """
        config = self._get_call_options(run)
        max_attempts = 1 + config.parse_retries
        last_error: StructuredOutputError | None = None

        for attempt in range(max_attempts):
            try:
                use_prompt = self._retry_prompt_or_original(
                    prompt,
                    schema,
                    last_error,
                    attempt,
                )
                attempt_conv = self._fork_conversation(conversation)
                result = await self._complete_structured_attempt(
                    use_prompt, schema, run, conversation=attempt_conv
                )
                self._merge_conversation(conversation, attempt_conv)
                return await self._apply_guards(
                    prompt, result, schema, run, conversation=conversation
                )
            except StructuredOutputError as e:
                last_error = e
                if attempt < max_attempts - 1:
                    self._log_parse_retry(schema.__name__, attempt + 1, max_attempts, e)
                    continue
                raise

        # Should not reach here, but satisfy type checker
        raise last_error  # type: ignore[misc]

    async def _complete_structured_attempt(
        self,
        prompt: str,
        schema: type[T],
        run: CallOptions | None = None,
        conversation: ConversationLike | None = None,
    ) -> T:
        """Single attempt at structured completion."""
        if self._has_tools():
            _, result = await self._loop(prompt, run=run, schema=schema, conversation=conversation)
            if result is not None:
                return result
        # Direct structured completion
        conv = conversation if conversation is not None else ListConversation()
        conv.append(Message(role=Role.USER, content=prompt))
        trace_id = self._generate_id()
        json_schema = dataclass_to_json_schema(schema)
        config = self._get_call_options(run)
        max_tokens = config.max_call_tokens if config.max_call_tokens > 0 else None
        response = await self._backend.chat(
            conv.as_messages(),
            system=self._call.system,
            response_schema=json_schema,
            max_tokens=max_tokens,
            temperature=self._resolve_temperature(run),
        )
        response.call_id = self._generate_id()
        conv.append(self._to_message(response))
        self._write_base_trace(response, trace_id=trace_id, phase="direct")
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError as e:
            raise self._structured_output_error(e, response.content, schema.__name__) from e
        return parse_json_to_dataclass(data, schema)

    def _retry_prompt_or_original(
        self,
        prompt: str,
        schema: type[T],
        last_error: StructuredOutputError | None,
        attempt: int,
    ) -> str:
        """Return original prompt on first attempt, retry prompt on subsequent ones."""
        if attempt == 0:
            return prompt
        return self._build_parse_retry_prompt(
            prompt,
            schema,
            last_error,  # type: ignore[arg-type]
        )

    def _build_parse_retry_prompt(
        self, original_prompt: str, schema: type[T], error: StructuredOutputError
    ) -> str:
        """Build a retry prompt with feedback about the parse failure."""
        parts = [original_prompt, "\n\n---\n\nYour previous response could not be parsed."]

        if error.parse_error:
            parts.append(f"\n\nParse error: {error.parse_error}")

        if error.raw_content:
            # Truncate raw content to avoid token bloat
            raw = error.raw_content
            if len(raw) > 500:
                raw = raw[:500] + "... (truncated)"
            parts.append(f"\n\nYour response was:\n```\n{raw}\n```")

        parts.append(
            f"\n\nPlease provide a valid JSON response matching the {schema.__name__} schema."
        )
        return "".join(parts)

    def _log_parse_retry(
        self, schema_name: str, attempt: int, max_attempts: int, error: StructuredOutputError
    ) -> None:
        """Log parse retry attempt."""
        if self._lg:
            self._lg.debug(
                "parse retry",
                extra={
                    "schema": schema_name,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "parse_error": error.parse_error,
                },
            )

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the verb."""
        ...

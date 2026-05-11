"""Tests for IterationGuard feature and built-in iteration guards."""

from __future__ import annotations

from typing import Any

import pytest

from llm_saia import IterationContext, IterationGuard, OutputGuard
from llm_saia.core.backend import ChatResponse
from llm_saia.core.config import CallOptions, Config, TerminalConfig
from llm_saia.core.conversation import Message, ToolCall
from llm_saia.core.logger import NullLogger
from llm_saia.core.types import ToolDef
from llm_saia.guards import (
    _ordinal,
    contradiction,
    narrative,
    terminal_compliance,
    terminal_deadline,
    terminal_schema,
    terminal_status,
)
from llm_saia.verbs import Ask, Instruct
from llm_saia.verbs.complete import Complete
from tests.unit.conftest import MockBackend

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_response(content: str = "", tool_calls: list[ToolCall] | None = None) -> ChatResponse:
    """Create an ChatResponse with optional tool calls."""
    return ChatResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="tool_use" if tool_calls else "end_turn",
    )


def _make_tool_call(name: str = "search", args: dict[str, Any] | None = None) -> ToolCall:
    return ToolCall(id="tc_1", name=name, arguments=args or {})


SEARCH_TOOL = ToolDef(
    name="search",
    description="Search for something",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)

DONE_TOOL = ToolDef(
    name="done",
    description="Signal completion",
    parameters={
        "type": "object",
        "properties": {"output": {"type": "string"}, "status": {"type": "string"}},
    },
)


def _make_config_with_tools(
    backend: MockBackend,
    call: CallOptions | None = None,
    terminal_tool: str | None = None,
) -> Config:
    """Create a Config with search tool."""
    terminal = TerminalConfig(tool=terminal_tool) if terminal_tool else None
    tools = [SEARCH_TOOL]
    if terminal_tool:
        tools.append(DONE_TOOL)

    async def executor(name: str, args: dict[str, Any]) -> str:
        return f"result for {name}"

    return Config(
        lg=NullLogger(),
        backend=backend,
        tools=tools,
        executor=executor,
        call=call,
        terminal=terminal,
    )


# ---------------------------------------------------------------------------
# IterationGuard dataclass
# ---------------------------------------------------------------------------


class TestIterationGuardDataclass:
    """Tests for IterationGuard construction."""

    def test_create_guard(self) -> None:
        guard = IterationGuard(validator=lambda ctx: None, name="test")
        assert guard.name == "test"
        assert guard.validator is not None

    def test_guard_frozen(self) -> None:
        guard = IterationGuard(validator=lambda ctx: None)
        with pytest.raises(AttributeError):
            guard.name = "changed"  # type: ignore[misc]

    def test_guard_default_name_is_none(self) -> None:
        guard = IterationGuard(validator=lambda ctx: None)
        assert guard.name is None


# ---------------------------------------------------------------------------
# with_guard / with_guards dispatch
# ---------------------------------------------------------------------------


class TestGuardDispatch:
    """Tests that with_guard routes IterationGuard vs OutputGuard correctly."""

    def test_with_guard_routes_output_guard(self) -> None:
        backend = MockBackend()
        config = Config(lg=NullLogger(), backend=backend, tools=[], executor=None)
        verb = Ask(config)
        out_guard = OutputGuard(validator=lambda x: None, retry_instruction="Fix.")
        result = verb.with_guard(out_guard)
        assert out_guard in result._call.output_guards
        assert len(result._call.iteration_guards) == 0

    def test_with_guard_routes_iteration_guard(self) -> None:
        backend = MockBackend()
        config = Config(lg=NullLogger(), backend=backend, tools=[], executor=None)
        verb = Ask(config)
        iter_guard = IterationGuard(validator=lambda ctx: None, name="test")
        result = verb.with_guard(iter_guard)
        assert iter_guard in result._call.iteration_guards
        assert len(result._call.output_guards) == 0

    def test_with_guards_mixed(self) -> None:
        backend = MockBackend()
        config = Config(lg=NullLogger(), backend=backend, tools=[], executor=None)
        verb = Ask(config)
        out_guard = OutputGuard(validator=lambda x: None, retry_instruction="Fix.")
        iter_guard = IterationGuard(validator=lambda ctx: None, name="test")
        result = verb.with_guards(out_guard, iter_guard)
        assert out_guard in result._call.output_guards
        assert iter_guard in result._call.iteration_guards


# ---------------------------------------------------------------------------
# Iteration guard in _loop (base verb with tools)
# ---------------------------------------------------------------------------


class TestIterationGuardInLoop:
    """Tests for iteration guards in the base verb tool-calling loop."""

    async def test_guard_passes_no_feedback(self) -> None:
        """When guard returns None, loop proceeds normally."""
        backend = MockBackend()
        # First call: tool call, second call: final response
        backend.queue_response(
            _tool_response("I'll search", [_make_tool_call("search", {"q": "test"})])
        )
        backend.queue_response(_tool_response("Here are the results"))

        guard = IterationGuard(validator=lambda ctx: None, name="always_pass")
        call = CallOptions(iteration_guards=(guard,), max_iterations=5)
        config = _make_config_with_tools(backend, call=call)
        verb = Instruct(config)
        result = await verb("do something")
        assert result.value == "Here are the results"

    async def test_guard_injects_feedback(self) -> None:
        """When guard returns feedback, it's injected as a user message."""
        backend = MockBackend()
        call_count = 0
        captured_messages: list[list[Message]] = []

        original_chat = backend.chat

        async def tracking_chat(messages: list[Message], **kwargs: Any) -> ChatResponse:
            nonlocal call_count
            call_count += 1
            captured_messages.append(list(messages))
            return await original_chat(messages, **kwargs)

        backend.chat = tracking_chat  # type: ignore[assignment]

        # First: tool call without explanation → guard fires
        backend.queue_response(_tool_response("", [_make_tool_call("search", {"q": "test"})]))
        # Second: tool call with explanation → guard passes
        backend.queue_response(
            _tool_response(
                "Searching for test results",
                [_make_tool_call("search", {"q": "test2"})],
            )
        )
        # Third: final response
        backend.queue_response(_tool_response("Done"))

        def require_narrative(ctx: IterationContext) -> str | None:
            if ctx.response.tool_calls and not (ctx.response.content or "").strip():
                return "Explain what you're doing and why."
            return None

        guard = IterationGuard(validator=require_narrative, name="narrative")
        call = CallOptions(iteration_guards=(guard,), max_iterations=10)
        config = _make_config_with_tools(backend, call=call)
        verb = Instruct(config)
        result = await verb("do something")

        assert result.value == "Done"
        assert call_count == 3

        # The second call should have the guard feedback in messages
        second_call_msgs = captured_messages[1]
        feedback_msgs = [m for m in second_call_msgs if "Explain what" in m.content]
        assert len(feedback_msgs) == 1

    async def test_multiple_guards_combined_feedback(self) -> None:
        """Multiple failing guards combine feedback with double newline."""
        backend = MockBackend()
        # First: triggers both guards (empty content, has tool call)
        backend.queue_response(_tool_response("", [_make_tool_call("search")]))
        # Second (after guard feedback): passes both guards, has tool call
        backend.queue_response(
            _tool_response("Explained and detailed", [_make_tool_call("search")])
        )
        # Third: final response (no tools, passes both guards)
        backend.queue_response(_tool_response("All done with results"))

        guard1 = IterationGuard(
            validator=lambda ctx: (
                "Need explanation." if not (ctx.response.content or "").strip() else None
            ),
            name="explain",
        )
        guard2 = IterationGuard(
            validator=lambda ctx: "Be verbose." if len(ctx.response.content or "") < 5 else None,
            name="verbose",
        )

        call = CallOptions(iteration_guards=(guard1, guard2), max_iterations=10)
        config = _make_config_with_tools(backend, call=call)
        verb = Instruct(config)
        result = await verb("do something")
        assert result.value == "All done with results"

    async def test_guard_outcome_in_trace(self) -> None:
        """Guard outcomes are recorded in the trace Step."""
        backend = MockBackend()
        # Trigger guard, then succeed
        backend.queue_response(_tool_response("", [_make_tool_call("search")]))
        backend.queue_response(_tool_response("Done"))

        guard = IterationGuard(
            validator=lambda ctx: "Explain." if not (ctx.response.content or "").strip() else None,
            name="narrative",
        )
        call = CallOptions(iteration_guards=(guard,), max_iterations=10)
        config = _make_config_with_tools(backend, call=call)
        verb = Instruct(config)
        result = await verb("do something")

        # Check the trace has guard outcomes
        trace = result.trace
        assert trace is not None
        # First step should have guard outcome (failed)
        first_step = trace.steps[0]
        assert len(first_step.guards) > 0
        assert first_step.guards[0].name == "narrative"
        assert first_step.guards[0].passed is False
        assert first_step.guards[0].error == "Explain."

    async def test_validator_exception_becomes_feedback(self) -> None:
        """If a guard validator raises, the exception becomes feedback."""
        backend = MockBackend()
        backend.queue_response(_tool_response("content", [_make_tool_call("search")]))
        backend.queue_response(_tool_response("Done"))

        calls = 0

        def bad_once_validator(ctx: IterationContext) -> str | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("validator broke")
            return None

        guard = IterationGuard(validator=bad_once_validator, name="broken_once")
        call = CallOptions(iteration_guards=(guard,), max_iterations=10)
        config = _make_config_with_tools(backend, call=call)
        verb = Instruct(config)
        result = await verb("do something")
        assert result.value == "Done"
        assert calls == 2


# ---------------------------------------------------------------------------
# Iteration guard in Complete verb
# ---------------------------------------------------------------------------


class TestIterationGuardInComplete:
    """Tests for iteration guards in the Complete verb's loop."""

    async def test_guard_fires_in_complete(self) -> None:
        """Iteration guard fires and injects feedback in Complete verb."""
        backend = MockBackend()
        call_count = 0

        original_chat = backend.chat

        async def tracking_chat(messages: list[Message], **kwargs: Any) -> ChatResponse:
            nonlocal call_count
            call_count += 1
            return await original_chat(messages, **kwargs)

        backend.chat = tracking_chat  # type: ignore[assignment]

        # First: tool call without narrative → guard fires
        backend.queue_response(_tool_response("", [_make_tool_call("search", {"q": "test"})]))
        # Second: explains and calls done tool → controller asks for confirmation
        done_call = ToolCall(
            id="tc_done", name="done", arguments={"output": "found it", "status": "ok"}
        )
        backend.queue_response(_tool_response("Found what I needed", [done_call]))
        # Third: confirms done tool (controller needs confirmation)
        done_confirm = ToolCall(
            id="tc_done2", name="done", arguments={"output": "found it", "status": "ok"}
        )
        backend.queue_response(_tool_response("Confirming", [done_confirm]))

        def require_narrative(ctx: IterationContext) -> str | None:
            if ctx.response.tool_calls and not (ctx.response.content or "").strip():
                return "Explain."
            return None

        guard = IterationGuard(validator=require_narrative, name="narrative")
        call = CallOptions(iteration_guards=(guard,), max_iterations=10)
        config = _make_config_with_tools(backend, call=call, terminal_tool="done")
        verb = Complete(config)
        result = await verb("find something")

        assert result.completed
        assert call_count == 3

    async def test_guard_nudge_scored_as_non_productive(self) -> None:
        """Iteration guard nudges count as non-productive in LoopScore."""
        backend = MockBackend()
        # First: triggers guard
        backend.queue_response(_tool_response("", [_make_tool_call("search")]))
        # Second: passes guard, calls done
        done_call = ToolCall(id="tc_done", name="done", arguments={"output": "ok", "status": "ok"})
        backend.queue_response(_tool_response("Explained", [done_call]))

        guard = IterationGuard(
            validator=lambda ctx: "Explain." if not (ctx.response.content or "").strip() else None,
            name="narrative",
        )
        call = CallOptions(iteration_guards=(guard,), max_iterations=10)
        config = _make_config_with_tools(backend, call=call, terminal_tool="done")
        verb = Complete(config)
        result = await verb("do it")

        assert result.score is not None
        assert result.score.nudges >= 1

    async def test_guard_step_recorded_in_trace(self) -> None:
        """Guard nudge step is recorded in Complete's VerbTrace."""
        backend = MockBackend()
        # Trigger guard, then complete
        backend.queue_response(_tool_response("", [_make_tool_call("search")]))
        done_call = ToolCall(id="tc_done", name="done", arguments={"output": "ok", "status": "ok"})
        backend.queue_response(_tool_response("Explained", [done_call]))

        guard = IterationGuard(
            validator=lambda ctx: "Explain." if not (ctx.response.content or "").strip() else None,
            name="narrative",
        )
        call = CallOptions(iteration_guards=(guard,), max_iterations=10)
        config = _make_config_with_tools(backend, call=call, terminal_tool="done")
        verb = Complete(config)
        result = await verb("do it")

        assert result.trace is not None
        guard_steps = [s for s in result.trace.steps if s.reason == "iteration_guard"]
        assert len(guard_steps) >= 1
        assert guard_steps[0].nudge_preview == "Explain."


# ---------------------------------------------------------------------------
# Complete verb conversation support
# ---------------------------------------------------------------------------


class CompactingConversation:
    """Mock conversation that tracks appends and can return compacted view."""

    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._compacted_view: list[Message] | None = None

    def append(self, msg: Message) -> None:
        self._messages.append(msg)

    def as_messages(self) -> list[Message]:
        """Return compacted view if set, otherwise full messages."""
        return self._compacted_view if self._compacted_view is not None else self._messages

    def set_compacted_view(self, messages: list[Message]) -> None:
        """Set what as_messages() returns (simulates compaction)."""
        self._compacted_view = messages

    @property
    def full_history(self) -> list[Message]:
        """Access to full internal history for assertions."""
        return self._messages


class TestCompleteConversationSupport:
    """Tests for Complete verb conversation parameter."""

    async def test_messages_appended_to_conversation(self) -> None:
        """Complete appends messages to provided conversation."""
        backend = MockBackend()
        # Done call + confirmation (terminal requires confirmation by default)
        done_call = ToolCall(id="tc_done", name="done", arguments={"output": "ok", "status": "ok"})
        backend.queue_response(_tool_response("Done", [done_call]))
        done_confirm = ToolCall(
            id="tc_done2", name="done", arguments={"output": "ok", "status": "ok"}
        )
        backend.queue_response(_tool_response("Confirming", [done_confirm]))

        config = _make_config_with_tools(backend, terminal_tool="done")
        verb = Complete(config)

        conv = CompactingConversation()
        result = await verb("test task", conversation=conv)

        assert result.completed
        # Conversation should have initial user message + assistant responses
        assert len(conv.full_history) >= 2
        assert conv.full_history[0].role == "user"
        assert conv.full_history[0].content == "test task"

    async def test_history_contains_full_messages(self) -> None:
        """TaskResult.history contains full history even if conv compacts."""
        backend = MockBackend()
        # First: tool call
        backend.queue_response(_tool_response("Searching", [_make_tool_call("search")]))
        # Second: done + confirmation
        done_call = ToolCall(id="tc_done", name="done", arguments={"output": "ok", "status": "ok"})
        backend.queue_response(_tool_response("Done", [done_call]))
        done_confirm = ToolCall(
            id="tc_done2", name="done", arguments={"output": "ok", "status": "ok"}
        )
        backend.queue_response(_tool_response("Confirming", [done_confirm]))

        config = _make_config_with_tools(backend, terminal_tool="done")
        verb = Complete(config)

        conv = CompactingConversation()
        result = await verb("test task", conversation=conv)

        # Result history should be complete (multiple messages)
        assert len(result.history) >= 3
        # Conversation also has full history (no compaction triggered in this test)
        assert len(conv.full_history) == len(result.history)

    async def test_llm_sees_conversation_view(self) -> None:
        """LLM receives conv.as_messages() which may be compacted."""
        backend = MockBackend()
        messages_sent_to_llm: list[list[Message]] = []

        original_chat = backend.chat

        async def tracking_chat(messages: list[Message], **kwargs: Any) -> ChatResponse:
            messages_sent_to_llm.append(list(messages))
            return await original_chat(messages, **kwargs)

        backend.chat = tracking_chat  # type: ignore[assignment]

        # Queue: search tool, then done + confirmation
        backend.queue_response(_tool_response("Searching", [_make_tool_call("search")]))
        done_call = ToolCall(id="tc_done", name="done", arguments={"output": "ok", "status": "ok"})
        backend.queue_response(_tool_response("Done", [done_call]))
        done_confirm = ToolCall(
            id="tc_done2", name="done", arguments={"output": "ok", "status": "ok"}
        )
        backend.queue_response(_tool_response("Confirming", [done_confirm]))

        config = _make_config_with_tools(backend, terminal_tool="done")
        verb = Complete(config)

        conv = CompactingConversation()
        # After a few messages, simulate compaction
        original_append = conv.append

        def compacting_append(msg: Message) -> None:
            original_append(msg)
            # After 4 messages, set compacted view (only latest 2 messages)
            if len(conv._messages) >= 4:
                conv.set_compacted_view(conv._messages[-2:])

        conv.append = compacting_append  # type: ignore[method-assign]

        await verb("test task", conversation=conv)

        # Should have 3 LLM calls (search, done, confirm)
        assert len(messages_sent_to_llm) == 3
        # First call sees just the initial message
        assert len(messages_sent_to_llm[0]) == 1
        # Third call sees compacted view (exactly 2 messages after threshold)
        assert len(messages_sent_to_llm[2]) == 2

    async def test_no_conversation_works_as_before(self) -> None:
        """Complete works without conversation parameter (backward compatible)."""
        backend = MockBackend()
        done_call = ToolCall(id="tc_done", name="done", arguments={"output": "ok", "status": "ok"})
        backend.queue_response(_tool_response("Done", [done_call]))
        done_confirm = ToolCall(
            id="tc_done2", name="done", arguments={"output": "ok", "status": "ok"}
        )
        backend.queue_response(_tool_response("Confirming", [done_confirm]))

        config = _make_config_with_tools(backend, terminal_tool="done")
        verb = Complete(config)

        # No conversation parameter
        result = await verb("test task")

        assert result.completed
        assert len(result.history) >= 2


class AsyncCompactingConversation:
    """Mock async conversation that tracks which append method is called."""

    def __init__(self) -> None:
        self._messages: list[Message] = []
        self.sync_append_count = 0
        self.async_append_count = 0

    def append(self, msg: Message) -> None:
        self._messages.append(msg)
        self.sync_append_count += 1

    async def append_async(self, msg: Message) -> None:
        self._messages.append(msg)
        self.async_append_count += 1

    def as_messages(self) -> list[Message]:
        return self._messages

    @property
    def full_history(self) -> list[Message]:
        return self._messages


class TestAsyncConversationLikeSupport:
    """Tests for AsyncConversationLike protocol support."""

    async def test_async_append_used_for_async_conversation(self) -> None:
        """Complete uses append_async when conversation supports it."""
        backend = MockBackend()
        done_call = ToolCall(id="tc_done", name="done", arguments={"output": "ok", "status": "ok"})
        backend.queue_response(_tool_response("Done", [done_call]))
        done_confirm = ToolCall(
            id="tc_done2", name="done", arguments={"output": "ok", "status": "ok"}
        )
        backend.queue_response(_tool_response("Confirming", [done_confirm]))

        config = _make_config_with_tools(backend, terminal_tool="done")
        verb = Complete(config)

        conv = AsyncCompactingConversation()
        result = await verb("test task", conversation=conv)

        assert result.completed
        # All appends should use async method, none should use sync
        assert conv.async_append_count > 0
        assert conv.sync_append_count == 0
        assert len(conv.full_history) == conv.async_append_count

    async def test_sync_append_used_for_sync_conversation(self) -> None:
        """Complete uses sync append when conversation doesn't support async."""
        backend = MockBackend()
        done_call = ToolCall(id="tc_done", name="done", arguments={"output": "ok", "status": "ok"})
        backend.queue_response(_tool_response("Done", [done_call]))
        done_confirm = ToolCall(
            id="tc_done2", name="done", arguments={"output": "ok", "status": "ok"}
        )
        backend.queue_response(_tool_response("Confirming", [done_confirm]))

        config = _make_config_with_tools(backend, terminal_tool="done")
        verb = Complete(config)

        conv = CompactingConversation()  # Sync-only conversation
        result = await verb("test task", conversation=conv)

        assert result.completed
        assert len(conv.full_history) >= 2

    async def test_non_complete_verb_uses_async_append(self) -> None:
        """Non-Complete verbs (Ask, etc.) use append_async when conversation supports it."""
        backend = MockBackend()
        backend.queue_response(_tool_response("The answer is 42."))

        config = Config(lg=NullLogger(), backend=backend, tools=[], executor=None)
        verb = Ask(config)

        conv = AsyncCompactingConversation()
        result = await verb("some artifact", "what is the answer?", conversation=conv)

        assert result.value == "The answer is 42."
        # All appends should use async method, none should use sync
        assert conv.async_append_count > 0
        assert conv.sync_append_count == 0
        assert len(conv.full_history) == conv.async_append_count

    async def test_tool_execution_syncs_via_async_append(self) -> None:
        """Tool execution results are synced to async conversation via append_async."""
        backend = MockBackend()
        # First response: non-terminal tool call (search)
        search_call = ToolCall(id="tc_search", name="search", arguments={"q": "test"})
        backend.queue_response(_tool_response("Searching...", [search_call]))
        # Second response: terminal tool call (done)
        done_call = ToolCall(id="tc_done", name="done", arguments={"output": "ok", "status": "ok"})
        backend.queue_response(_tool_response("Done", [done_call]))
        # Third response: confirmation
        done_confirm = ToolCall(
            id="tc_done2", name="done", arguments={"output": "ok", "status": "ok"}
        )
        backend.queue_response(_tool_response("Confirming", [done_confirm]))

        config = _make_config_with_tools(backend, terminal_tool="done")
        verb = Complete(config)

        conv = AsyncCompactingConversation()
        result = await verb("test task", conversation=conv)

        assert result.completed
        # All appends should use async method, none should use sync
        assert conv.async_append_count > 0
        assert conv.sync_append_count == 0
        # Verify tool result message was synced (Role.TOOL in history)
        tool_messages = [m for m in conv.full_history if m.role == "tool"]
        assert len(tool_messages) >= 1, "Tool result should be synced to async conversation"


# ---------------------------------------------------------------------------
# Built-in iteration guard factories
# ---------------------------------------------------------------------------


def _make_response(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
) -> ChatResponse:
    """Create a minimal ChatResponse for testing."""
    return ChatResponse(
        content=content,
        tool_calls=tool_calls,
        input_tokens=10,
        output_tokens=10,
        call_id="test-call",
        finish_reason="stop",
    )


def _make_ctx(
    response: ChatResponse,
    iteration: int = 0,
    max_iterations: int = 10,
) -> IterationContext:
    """Wrap ChatResponse in IterationContext for guard testing."""
    return IterationContext(response=response, iteration=iteration, max_iterations=max_iterations)


class TestTerminalStatusGuard:
    """Tests for terminal_status guard factory."""

    def test_no_tool_calls_passes(self) -> None:
        guard = terminal_status("done", "status", ("stuck", "failed"))
        response = _make_response("Just some text")
        assert guard.validator(_make_ctx(response)) is None

    def test_non_terminal_tool_passes(self) -> None:
        guard = terminal_status("done", "status", ("stuck", "failed"))
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="other_tool", arguments={"status": "stuck"})]
        )
        assert guard.validator(_make_ctx(response)) is None

    def test_success_status_passes(self) -> None:
        guard = terminal_status("done", "status", ("stuck", "failed"))
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="done", arguments={"status": "complete"})]
        )
        assert guard.validator(_make_ctx(response)) is None

    def test_failure_status_returns_feedback(self) -> None:
        guard = terminal_status("done", "status", ("stuck", "failed"))
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="done", arguments={"status": "stuck"})]
        )
        feedback = guard.validator(_make_ctx(response))
        assert feedback is not None
        assert "stuck" in feedback

    def test_escalation_on_repeated_failures(self) -> None:
        guard = terminal_status("done", "status", ("stuck",), max_retries=3, escalate=True)
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="done", arguments={"status": "stuck"})]
        )
        # Use incrementing iterations to simulate consecutive failures within same task
        feedback1 = guard.validator(_make_ctx(response, iteration=0))
        assert feedback1 is not None
        assert "MUST" not in feedback1

        feedback2 = guard.validator(_make_ctx(response, iteration=1))
        assert feedback2 is not None
        assert "MUST" in feedback2

    def test_exhausted_retries_passes(self) -> None:
        guard = terminal_status("done", "status", ("stuck",), max_retries=2)
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="done", arguments={"status": "stuck"})]
        )
        # Use incrementing iterations to simulate consecutive failures within same task
        assert guard.validator(_make_ctx(response, iteration=0)) is not None
        assert guard.validator(_make_ctx(response, iteration=1)) is not None
        assert guard.validator(_make_ctx(response, iteration=2)) is None

    def test_state_resets_on_new_task(self) -> None:
        """Guard state resets when iteration=0 (new task starts)."""
        guard = terminal_status("done", "status", ("stuck",), max_retries=2)
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="done", arguments={"status": "stuck"})]
        )
        # First task: exhaust retries
        assert guard.validator(_make_ctx(response, iteration=0)) is not None
        assert guard.validator(_make_ctx(response, iteration=1)) is not None
        assert guard.validator(_make_ctx(response, iteration=2)) is None  # exhausted

        # Second task: iteration=0 resets state, guard fires again
        assert guard.validator(_make_ctx(response, iteration=0)) is not None
        assert guard.validator(_make_ctx(response, iteration=1)) is not None
        assert guard.validator(_make_ctx(response, iteration=2)) is None  # exhausted again

    def test_non_dict_arguments_passes(self) -> None:
        guard = terminal_status("done", "status", ("stuck",))
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="done", arguments="not a dict")]  # type: ignore
        )
        assert guard.validator(_make_ctx(response)) is None


class TestTerminalSchemaGuard:
    """Tests for terminal_schema guard factory."""

    @pytest.fixture
    def tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                name="report",
                description="Report findings",
                parameters={
                    "type": "object",
                    "required": ["summary", "score"],
                    "properties": {
                        "summary": {"type": "string"},
                        "score": {"type": "integer"},
                    },
                },
            ),
        ]

    def test_no_tool_calls_passes(self, tools: list[ToolDef]) -> None:
        guard = terminal_schema(tools, "report")
        response = _make_response("Just some text")
        assert guard.validator(_make_ctx(response)) is None

    def test_valid_arguments_passes(self, tools: list[ToolDef]) -> None:
        guard = terminal_schema(tools, "report")
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="report", arguments={"summary": "done", "score": 5})]
        )
        assert guard.validator(_make_ctx(response)) is None

    def test_missing_required_field_returns_feedback(self, tools: list[ToolDef]) -> None:
        guard = terminal_schema(tools, "report")
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="report", arguments={"summary": "done"})]
        )
        feedback = guard.validator(_make_ctx(response))
        assert feedback is not None
        assert "schema errors" in feedback
        assert "score" in feedback

    def test_wrong_type_returns_feedback(self, tools: list[ToolDef]) -> None:
        guard = terminal_schema(tools, "report")
        response = _make_response(
            tool_calls=[
                ToolCall(id="1", name="report", arguments={"summary": "done", "score": "five"})
            ]
        )
        feedback = guard.validator(_make_ctx(response))
        assert feedback is not None
        assert "expected integer" in feedback

    def test_escalation_on_repeated_failures(self, tools: list[ToolDef]) -> None:
        guard = terminal_schema(tools, "report", max_retries=3, escalate=True)
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="report", arguments={"summary": "done"})]
        )
        # Use incrementing iterations to simulate consecutive failures within same task
        feedback1 = guard.validator(_make_ctx(response, iteration=0))
        assert feedback1 is not None
        assert "STILL" not in feedback1

        feedback2 = guard.validator(_make_ctx(response, iteration=1))
        assert feedback2 is not None
        assert "STILL" in feedback2

    def test_exhausted_retries_passes(self, tools: list[ToolDef]) -> None:
        guard = terminal_schema(tools, "report", max_retries=2)
        response = _make_response(tool_calls=[ToolCall(id="1", name="report", arguments={})])
        # Use incrementing iterations to simulate consecutive failures within same task
        assert guard.validator(_make_ctx(response, iteration=0)) is not None
        assert guard.validator(_make_ctx(response, iteration=1)) is not None
        assert guard.validator(_make_ctx(response, iteration=2)) is None

    def test_unknown_tool_creates_noop_guard(self) -> None:
        guard = terminal_schema([], "nonexistent")
        response = _make_response(tool_calls=[ToolCall(id="1", name="nonexistent", arguments={})])
        assert guard.validator(_make_ctx(response)) is None


class TestContradictionGuard:
    """Tests for contradiction guard factory."""

    def test_no_tool_calls_passes(self) -> None:
        guard = contradiction("done")
        response = _make_response("I can't do this")
        assert guard.validator(_make_ctx(response)) is None

    def test_non_terminal_tool_passes(self) -> None:
        guard = contradiction("done")
        response = _make_response(
            content="However, I can't do this",
            tool_calls=[ToolCall(id="1", name="other_tool", arguments={})],
        )
        assert guard.validator(_make_ctx(response)) is None

    def test_terminal_without_contradiction_passes(self) -> None:
        guard = contradiction("done")
        response = _make_response(
            content="Task completed successfully!",
            tool_calls=[ToolCall(id="1", name="done", arguments={})],
        )
        assert guard.validator(_make_ctx(response)) is None

    def test_terminal_with_empty_content_passes(self) -> None:
        guard = contradiction("done")
        response = _make_response(
            content="",
            tool_calls=[ToolCall(id="1", name="done", arguments={})],
        )
        assert guard.validator(_make_ctx(response)) is None

    def test_contradiction_detected(self) -> None:
        guard = contradiction("done")
        response = _make_response(
            content="However, I wasn't able to complete everything",
            tool_calls=[ToolCall(id="1", name="done", arguments={})],
        )
        feedback = guard.validator(_make_ctx(response))
        assert feedback is not None
        assert "contradictory" in feedback

    def test_detects_various_signals(self) -> None:
        signals = ["unfortunately", "i can't", "i cannot", "not possible", "unable to"]
        for signal in signals:
            guard = contradiction("done")
            response = _make_response(
                content=f"Task done, {signal} verify everything",
                tool_calls=[ToolCall(id="1", name="done", arguments={})],
            )
            feedback = guard.validator(_make_ctx(response))
            assert feedback is not None, f"Should detect signal: {signal}"

    def test_escalation_on_repeated_contradictions(self) -> None:
        guard = contradiction("done", max_retries=3, escalate=True)
        response = _make_response(
            content="However, something went wrong",
            tool_calls=[ToolCall(id="1", name="done", arguments={})],
        )
        # Use incrementing iterations to simulate consecutive failures within same task
        feedback1 = guard.validator(_make_ctx(response, iteration=0))
        assert feedback1 is not None
        assert "2nd" not in feedback1

        feedback2 = guard.validator(_make_ctx(response, iteration=1))
        assert feedback2 is not None
        assert "2nd" in feedback2

    def test_exhausted_retries_passes(self) -> None:
        guard = contradiction("done", max_retries=2)
        response = _make_response(
            content="Unfortunately this failed",
            tool_calls=[ToolCall(id="1", name="done", arguments={})],
        )
        # Use incrementing iterations to simulate consecutive failures within same task
        assert guard.validator(_make_ctx(response, iteration=0)) is not None
        assert guard.validator(_make_ctx(response, iteration=1)) is not None
        assert guard.validator(_make_ctx(response, iteration=2)) is None


class TestOrdinalHelper:
    """Tests for _ordinal helper."""

    def test_ordinals(self) -> None:
        assert _ordinal(1) == "1st"
        assert _ordinal(2) == "2nd"
        assert _ordinal(3) == "3rd"
        assert _ordinal(4) == "4th"
        assert _ordinal(11) == "11th"
        assert _ordinal(12) == "12th"
        assert _ordinal(13) == "13th"
        assert _ordinal(21) == "21st"
        assert _ordinal(22) == "22nd"
        assert _ordinal(23) == "23rd"


# ---------------------------------------------------------------------------
# Non-blocking (advisory) guards
# ---------------------------------------------------------------------------


class TestNonBlockingGuards:
    """Tests for non-blocking (advisory) iteration guards."""

    async def test_advisory_guard_executes_tools_then_injects_feedback(self) -> None:
        """Non-blocking guard allows tool execution, then injects feedback."""
        backend = MockBackend()
        executed_tools: list[str] = []

        async def tracking_executor(name: str, args: dict[str, Any]) -> str:
            executed_tools.append(name)
            return f"result for {name}"

        # First: tool call without explanation
        backend.queue_response(_tool_response("", [_make_tool_call("search", {"q": "test"})]))
        # Second: final response (after tool result + advisory feedback)
        backend.queue_response(_tool_response("Done with results"))

        guard = IterationGuard(
            validator=lambda ctx: "Explain what you're doing." if ctx.response.tool_calls else None,
            name="narrative",
            blocking=False,  # Advisory mode
        )

        call = CallOptions(iteration_guards=(guard,), max_iterations=10)
        config = Config(
            lg=NullLogger(),
            backend=backend,
            tools=[SEARCH_TOOL],
            executor=tracking_executor,
            call=call,
        )
        verb = Instruct(config)
        result = await verb("do something")

        # Tool was executed despite guard firing
        assert "search" in executed_tools
        assert result.value == "Done with results"

        # Verify message ordering: tool result comes before advisory feedback
        messages = backend.last_messages
        tool_msg_idx = next(i for i, m in enumerate(messages) if m.role == "tool")
        feedback_idx = next(
            i for i, m in enumerate(messages) if m.role == "user" and "Explain" in m.content
        )
        assert tool_msg_idx < feedback_idx, "Tool result must precede advisory feedback"

    async def test_blocking_guard_takes_precedence(self) -> None:
        """When both blocking and advisory guards fire, blocking wins."""
        backend = MockBackend()
        executed_tools: list[str] = []

        async def tracking_executor(name: str, args: dict[str, Any]) -> str:
            executed_tools.append(name)
            return f"result for {name}"

        # First: tool call → blocking guard fires → no execution
        backend.queue_response(_tool_response("", [_make_tool_call("search", {"q": "test"})]))
        # Second: retry with explanation → guards pass
        backend.queue_response(
            _tool_response("Searching now", [_make_tool_call("search", {"q": "test2"})])
        )
        # Third: final response
        backend.queue_response(_tool_response("Done"))

        blocking_guard = IterationGuard(
            validator=lambda ctx: (
                "STOP: Explain first." if not (ctx.response.content or "").strip() else None
            ),
            name="require_content",
            blocking=True,
        )
        advisory_guard = IterationGuard(
            validator=lambda ctx: "Also be verbose." if ctx.response.tool_calls else None,
            name="verbose",
            blocking=False,
        )

        call = CallOptions(iteration_guards=(blocking_guard, advisory_guard), max_iterations=10)
        config = Config(
            lg=NullLogger(),
            backend=backend,
            tools=[SEARCH_TOOL],
            executor=tracking_executor,
            call=call,
        )
        verb = Instruct(config)
        result = await verb("do something")

        # First tool call was blocked (no execution), second one succeeded
        assert executed_tools == ["search"]
        assert result.value == "Done"

    async def test_advisory_guard_outcome_records_blocking_false(self) -> None:
        """Guard outcome records blocking=False for advisory guards."""
        backend = MockBackend()
        backend.queue_response(_tool_response("", [_make_tool_call("search")]))
        backend.queue_response(_tool_response("Done"))

        guard = IterationGuard(
            validator=lambda ctx: "Be verbose." if ctx.response.tool_calls else None,
            name="verbose",
            blocking=False,
        )

        call = CallOptions(iteration_guards=(guard,), max_iterations=10)
        config = _make_config_with_tools(backend, call=call)
        verb = Instruct(config)
        result = await verb("do something")

        trace = result.trace
        assert trace is not None
        first_step = trace.steps[0]
        assert len(first_step.guards) > 0
        assert first_step.guards[0].blocking is False

    async def test_advisory_feedback_skipped_on_task_completion(self) -> None:
        """Advisory feedback is NOT injected when task completes in same iteration."""
        backend = MockBackend()

        # First: initial done call → controller asks for confirmation
        done1 = ToolCall(id="tc_done1", name="done", arguments={"output": "result", "status": "ok"})
        backend.queue_response(_tool_response("Here's my answer", [done1]))
        # Second: confirmation → task completes (advisory guard fires but shouldn't inject)
        done2 = ToolCall(id="tc_done2", name="done", arguments={"output": "result", "status": "ok"})
        backend.queue_response(_tool_response("Confirming", [done2]))

        # Advisory guard that fires when there are tool calls
        guard = IterationGuard(
            validator=lambda ctx: "Explain more." if ctx.response.tool_calls else None,
            name="verbose",
            blocking=False,
        )

        call = CallOptions(iteration_guards=(guard,), max_iterations=10)
        config = _make_config_with_tools(backend, call=call, terminal_tool="done")
        verb = Complete(config)
        result = await verb("do something")

        # Task completed
        assert result.completed
        assert result.output == "result"

        # Verify advisory feedback was NOT injected after completion.
        # The final messages should be: assistant (confirm) -> tool (ack)
        # NOT: assistant (confirm) -> tool (ack) -> user (Explain more.)
        last_msg = result.history[-1]
        assert last_msg.role.value == "tool", (
            "Last message should be tool ack, not advisory feedback"
        )
        second_last = result.history[-2]
        assert second_last.role.value == "assistant", "Second-to-last should be assistant confirm"


# ---------------------------------------------------------------------------
# Behavioral iteration guards
# ---------------------------------------------------------------------------


class TestNarrativeGuard:
    """Tests for narrative guard factory."""

    def test_no_tool_calls_passes(self) -> None:
        guard = narrative("report")
        response = _make_response("Just explaining something")
        assert guard.validator(_make_ctx(response)) is None

    def test_tool_call_with_explanation_passes(self) -> None:
        guard = narrative("report")
        response = _make_response(
            content="Searching for relevant data",
            tool_calls=[ToolCall(id="1", name="search", arguments={"q": "test"})],
        )
        assert guard.validator(_make_ctx(response)) is None

    def test_tool_call_without_explanation_returns_feedback(self) -> None:
        guard = narrative("report")
        response = _make_response(
            content="",
            tool_calls=[ToolCall(id="1", name="search", arguments={"q": "test"})],
        )
        feedback = guard.validator(_make_ctx(response))
        assert feedback is not None
        assert "search" in feedback
        assert "explain" in feedback.lower()

    def test_whitespace_only_counts_as_no_explanation(self) -> None:
        guard = narrative("report")
        response = _make_response(
            content="   \n  ",
            tool_calls=[ToolCall(id="1", name="search", arguments={})],
        )
        feedback = guard.validator(_make_ctx(response))
        assert feedback is not None

    def test_terminal_tool_skipped(self) -> None:
        guard = narrative("report")
        response = _make_response(
            content="",
            tool_calls=[ToolCall(id="1", name="report", arguments={"summary": "done"})],
        )
        assert guard.validator(_make_ctx(response)) is None

    def test_multiple_tool_names_in_feedback(self) -> None:
        guard = narrative("report")
        response = _make_response(
            content="",
            tool_calls=[
                ToolCall(id="1", name="search", arguments={}),
                ToolCall(id="2", name="fetch", arguments={}),
            ],
        )
        feedback = guard.validator(_make_ctx(response))
        assert feedback is not None
        assert "search" in feedback
        assert "fetch" in feedback

    def test_is_non_blocking(self) -> None:
        guard = narrative("report")
        assert guard.blocking is False

    def test_max_retries_stops_feedback(self) -> None:
        guard = narrative("report", max_retries=2)
        response = _make_response(
            content="",
            tool_calls=[ToolCall(id="1", name="search", arguments={})],
        )
        ctx = _make_ctx(response, iteration=1)
        # First two attempts return feedback
        assert guard.validator(ctx) is not None
        assert guard.validator(ctx) is not None
        # Third attempt returns None (max_retries exceeded)
        assert guard.validator(ctx) is None

    def test_escalate_changes_message(self) -> None:
        guard = narrative("report", max_retries=3, escalate=True)
        response = _make_response(
            content="",
            tool_calls=[ToolCall(id="1", name="search", arguments={})],
        )
        ctx = _make_ctx(response, iteration=1)
        first = guard.validator(ctx)
        second = guard.validator(ctx)
        assert first is not None
        assert second is not None
        assert "REQUIRED" in second  # Escalated message
        assert "REQUIRED" not in first  # Base message

    def test_state_resets_on_new_task(self) -> None:
        guard = narrative("report", max_retries=1)
        response = _make_response(
            content="",
            tool_calls=[ToolCall(id="1", name="search", arguments={})],
        )
        # Exhaust retries on iteration 1
        ctx1 = _make_ctx(response, iteration=1)
        assert guard.validator(ctx1) is not None
        assert guard.validator(ctx1) is None  # Exhausted
        # New task (iteration 0) resets state
        ctx0 = _make_ctx(response, iteration=0)
        assert guard.validator(ctx0) is not None  # Feedback again


class TestTerminalDeadlineGuard:
    """Tests for terminal_deadline guard factory."""

    def test_many_iterations_remaining_passes(self) -> None:
        guard = terminal_deadline("done")
        response = _make_response(tool_calls=[ToolCall(id="1", name="search", arguments={})])
        # 5 remaining > 3, so no enforcement
        ctx = _make_ctx(response, iteration=5, max_iterations=10)
        assert guard.validator(ctx) is None

    def test_low_iterations_no_terminal_returns_feedback(self) -> None:
        guard = terminal_deadline("done")
        response = _make_response(tool_calls=[ToolCall(id="1", name="search", arguments={})])
        # 3 remaining triggers enforcement
        ctx = _make_ctx(response, iteration=7, max_iterations=10)
        feedback = guard.validator(ctx)
        assert feedback is not None
        assert "3 iteration(s) remaining" in feedback
        assert "done" in feedback

    def test_low_iterations_with_terminal_passes(self) -> None:
        guard = terminal_deadline("done")
        response = _make_response(
            tool_calls=[ToolCall(id="1", name="done", arguments={"output": "result"})]
        )
        ctx = _make_ctx(response, iteration=7, max_iterations=10)
        assert guard.validator(ctx) is None

    def test_low_iterations_mixed_tools_returns_feedback(self) -> None:
        guard = terminal_deadline("done")
        response = _make_response(
            tool_calls=[
                ToolCall(id="1", name="done", arguments={"output": "result"}),
                ToolCall(id="2", name="search", arguments={}),
            ]
        )
        ctx = _make_ctx(response, iteration=8, max_iterations=10)
        feedback = guard.validator(ctx)
        assert feedback is not None
        assert "mix" in feedback.lower()
        assert "search" in feedback

    def test_no_tool_calls_low_iterations_fires(self) -> None:
        """When iterations are low and no tools called, still demands terminal tool."""
        guard = terminal_deadline("done")
        response = _make_response("Just thinking")
        ctx = _make_ctx(response, iteration=9, max_iterations=10)
        # Low iterations + no terminal tool = guard fires
        feedback = guard.validator(ctx)
        assert feedback is not None
        assert "MUST call done" in feedback

    def test_remaining_one_enforces(self) -> None:
        guard = terminal_deadline("done")
        response = _make_response(tool_calls=[ToolCall(id="1", name="search", arguments={})])
        ctx = _make_ctx(response, iteration=9, max_iterations=10)
        feedback = guard.validator(ctx)
        assert feedback is not None
        assert "1 iteration(s) remaining" in feedback

    def test_max_retries_stops_feedback(self) -> None:
        guard = terminal_deadline("done", max_retries=2)
        response = _make_response(tool_calls=[ToolCall(id="1", name="search", arguments={})])
        ctx = _make_ctx(response, iteration=7, max_iterations=10)  # 3 remaining
        # First two attempts return feedback
        assert guard.validator(ctx) is not None
        assert guard.validator(ctx) is not None
        # Third attempt returns None (max_retries exceeded)
        assert guard.validator(ctx) is None

    def test_escalate_changes_message(self) -> None:
        guard = terminal_deadline("done", max_retries=3, escalate=True)
        response = _make_response(tool_calls=[ToolCall(id="1", name="search", arguments={})])
        ctx = _make_ctx(response, iteration=7, max_iterations=10)  # 3 remaining
        first = guard.validator(ctx)
        second = guard.validator(ctx)
        assert first is not None
        assert second is not None
        assert "CRITICAL" in second  # Escalated message
        assert "CRITICAL" not in first  # Base message

    def test_custom_threshold(self) -> None:
        """Custom threshold changes when guard fires."""
        guard = terminal_deadline("done", threshold=5)
        response = _make_response(tool_calls=[ToolCall(id="1", name="search", arguments={})])
        # 4 remaining - would pass with default threshold=3, but fails with threshold=5
        ctx = _make_ctx(response, iteration=6, max_iterations=10)
        feedback = guard.validator(ctx)
        assert feedback is not None
        # 6 remaining - passes even with threshold=5
        ctx_high = _make_ctx(response, iteration=4, max_iterations=10)
        guard2 = terminal_deadline("done", threshold=5)  # Fresh guard
        assert guard2.validator(ctx_high) is None

    def test_unlimited_iterations_never_fires(self) -> None:
        """Guard never fires when max_iterations=0 (UNLIMITED)."""
        guard = terminal_deadline("done")
        response = _make_response(tool_calls=[ToolCall(id="1", name="search", arguments={})])
        # max_iterations=0 means unlimited - remaining becomes UNLIMITED (2^63-1)
        # Guard should never fire regardless of iteration count
        for iteration in [0, 1, 10, 100, 1000]:
            ctx = _make_ctx(response, iteration=iteration, max_iterations=0)
            assert guard.validator(ctx) is None, f"Should not fire at iteration {iteration}"


class TestTerminalComplianceGuard:
    """Tests for terminal_compliance guard factory."""

    def test_no_mention_passes(self) -> None:
        guard = terminal_compliance("report_findings")
        response = _make_response("I'm going to search for more data")
        ctx = _make_ctx(response, iteration=8, max_iterations=10)
        assert guard.validator(ctx) is None

    def test_mentions_and_calls_passes(self) -> None:
        guard = terminal_compliance("report_findings")
        response = _make_response(
            content="Calling report_findings now",
            tool_calls=[ToolCall(id="1", name="report_findings", arguments={})],
        )
        ctx = _make_ctx(response, iteration=8, max_iterations=10)
        assert guard.validator(ctx) is None

    def test_mentions_but_doesnt_call_low_iterations(self) -> None:
        guard = terminal_compliance("report_findings")
        response = _make_response("I will call report_findings")
        ctx = _make_ctx(response, iteration=8, max_iterations=10)  # 2 remaining
        feedback = guard.validator(ctx)
        assert feedback is not None
        assert "DID NOT" in feedback
        assert "MUST" in feedback

    def test_mentions_report_word_boundary(self) -> None:
        guard = terminal_compliance("report_findings")
        response = _make_response("I'll report my findings now")
        ctx = _make_ctx(response, iteration=9, max_iterations=10)  # 1 remaining
        feedback = guard.validator(ctx)
        assert feedback is not None

    def test_reported_past_tense_no_match(self) -> None:
        """'reported' should not trigger the guard (word boundary check)."""
        guard = terminal_compliance("report_findings")
        response = _make_response("I reported the issue earlier")
        ctx = _make_ctx(response, iteration=9, max_iterations=10)
        # 'reported' doesn't match word boundary for 'report'
        assert guard.validator(ctx) is None

    def test_high_iterations_remaining_passes(self) -> None:
        guard = terminal_compliance("report_findings")
        response = _make_response("I'll call report_findings soon")
        ctx = _make_ctx(response, iteration=5, max_iterations=10)  # 5 remaining > 2
        assert guard.validator(ctx) is None

    def test_has_other_tool_calls_passes(self) -> None:
        """If there are other tool calls, this pattern doesn't apply."""
        guard = terminal_compliance("report_findings")
        response = _make_response(
            content="I'll report_findings after this search",
            tool_calls=[ToolCall(id="1", name="search", arguments={})],
        )
        ctx = _make_ctx(response, iteration=8, max_iterations=10)
        assert guard.validator(ctx) is None

    def test_max_retries_stops_feedback(self) -> None:
        guard = terminal_compliance("report_findings", max_retries=2)
        response = _make_response("I will call report_findings")
        ctx = _make_ctx(response, iteration=8, max_iterations=10)  # 2 remaining
        # First two attempts return feedback
        assert guard.validator(ctx) is not None
        assert guard.validator(ctx) is not None
        # Third attempt returns None (max_retries exceeded)
        assert guard.validator(ctx) is None

    def test_escalate_changes_message(self) -> None:
        guard = terminal_compliance("report_findings", max_retries=3, escalate=True)
        response = _make_response("I will call report_findings")
        ctx = _make_ctx(response, iteration=8, max_iterations=10)  # 2 remaining
        first = guard.validator(ctx)
        second = guard.validator(ctx)
        assert first is not None
        assert second is not None
        assert "FAILURE" in second  # Escalated message
        assert "FAILURE" not in first  # Base message

    def test_custom_threshold(self) -> None:
        """Custom threshold changes when guard fires."""
        guard = terminal_compliance("report_findings", threshold=4)
        response = _make_response("I will call report_findings")
        # 3 remaining - would pass with default threshold=2, but fails with threshold=4
        ctx = _make_ctx(response, iteration=7, max_iterations=10)
        feedback = guard.validator(ctx)
        assert feedback is not None
        # 5 remaining - passes even with threshold=4
        ctx_high = _make_ctx(response, iteration=5, max_iterations=10)
        guard2 = terminal_compliance("report_findings", threshold=4)  # Fresh guard
        assert guard2.validator(ctx_high) is None

    def test_unlimited_iterations_never_fires(self) -> None:
        """Guard never fires when max_iterations=0 (UNLIMITED)."""
        guard = terminal_compliance("report_findings")
        response = _make_response("I will call report_findings")
        # max_iterations=0 means unlimited - remaining becomes UNLIMITED (2^63-1)
        # Guard should never fire regardless of iteration count
        for iteration in [0, 1, 10, 100, 1000]:
            ctx = _make_ctx(response, iteration=iteration, max_iterations=0)
            assert guard.validator(ctx) is None, f"Should not fire at iteration {iteration}"

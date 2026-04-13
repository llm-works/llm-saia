"""Tests for conversation threading through verb __call__ and helper methods."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from llm_saia.core.backend import AgentResponse
from llm_saia.core.config import CallOptions, Config
from llm_saia.core.conversation import ListConversation, Message, Role
from llm_saia.core.types import ToolDef
from llm_saia.verbs import Ask, Constrain, Extract, Instruct
from tests.unit.conftest import MockBackend

pytestmark = pytest.mark.unit


def make_config(backend: MockBackend, call: CallOptions | None = None) -> Config:
    """Create a Config with no tools (direct backend calls)."""
    return Config(backend=backend, tools=[], executor=None, call=call)


# ---------------------------------------------------------------------------
# Basic threading: messages appended and history sent to backend
# ---------------------------------------------------------------------------


class TestConversationThreadingText:
    """Test conversation threading through text-completion verbs."""

    async def test_text_verb_appends_user_and_assistant(self) -> None:
        """Calling a text verb should append user + assistant messages to conversation."""
        backend = MockBackend()
        backend.set_complete_response("the answer")
        conv = ListConversation()

        ask = Ask(make_config(backend))
        result = await ask("artifact", "question?", conversation=conv)

        assert result.value == "the answer"
        msgs = conv.as_messages()
        assert len(msgs) == 2
        assert msgs[0].role == Role.USER
        assert msgs[1].role == Role.ASSISTANT
        assert msgs[1].content == "the answer"

    async def test_text_verb_sends_prior_history_to_backend(self) -> None:
        """Prior conversation messages should be included in the backend call."""
        backend = MockBackend()
        conv = ListConversation()
        conv.append(Message(role=Role.USER, content="earlier question"))
        conv.append(Message(role=Role.ASSISTANT, content="earlier answer"))

        instruct = Instruct(make_config(backend))
        await instruct("do something", conversation=conv)

        # Backend receives reference to live list; after call it has prior + user + assistant
        assert len(backend.last_messages) == 4
        assert backend.last_messages[0].content == "earlier question"
        assert backend.last_messages[1].content == "earlier answer"
        assert "do something" in backend.last_messages[2].content
        assert backend.last_messages[2].role == Role.USER

    async def test_text_verb_without_conversation_works(self) -> None:
        """Passing no conversation should behave identically to before."""
        backend = MockBackend()
        backend.set_complete_response("result")
        instruct = Instruct(make_config(backend))
        result = await instruct("directive")

        assert result.value == "result"
        # Backend gets a reference to the live internal list (user + assistant after append)
        assert len(backend.last_messages) == 2
        assert backend.last_messages[0].role == Role.USER

    async def test_constrain_empty_rules_skips_backend(self) -> None:
        """Constrain with empty rules returns input unchanged, no conversation side-effects."""
        backend = MockBackend()
        conv = ListConversation()
        constrain = Constrain(make_config(backend))
        result = await constrain("text", [], conversation=conv)

        assert result.value == "text"
        assert len(conv.as_messages()) == 0


# ---------------------------------------------------------------------------
# Structured output threading
# ---------------------------------------------------------------------------


@dataclass
class SimpleResult:
    """Simple structured result for tests."""

    value: str


class TestConversationThreadingStructured:
    """Test conversation threading through structured-output verbs."""

    async def test_structured_verb_appends_messages(self) -> None:
        """Structured verbs should append user + assistant to conversation."""
        backend = MockBackend()
        backend.set_structured_response(SimpleResult, SimpleResult(value="extracted"))
        conv = ListConversation()

        extract = Extract(make_config(backend))
        result = await extract("content", SimpleResult, conversation=conv)

        assert result.value.value == "extracted"
        msgs = conv.as_messages()
        assert len(msgs) == 2
        assert msgs[0].role == Role.USER
        assert msgs[1].role == Role.ASSISTANT

    async def test_structured_verb_sends_prior_history(self) -> None:
        """Prior conversation should be sent to backend for structured calls."""
        backend = MockBackend()
        backend.set_structured_response(SimpleResult, SimpleResult(value="ok"))
        conv = ListConversation()
        conv.append(Message(role=Role.USER, content="prior"))
        conv.append(Message(role=Role.ASSISTANT, content="prior response"))

        extract = Extract(make_config(backend))
        await extract("new content", SimpleResult, conversation=conv)

        # Live list reference: 2 prior + user + assistant
        assert len(backend.last_messages) == 4
        assert backend.last_messages[0].content == "prior"
        assert backend.last_messages[2].role == Role.USER


# ---------------------------------------------------------------------------
# Parse retry isolation
# ---------------------------------------------------------------------------


class SequencedMockBackend(MockBackend):
    """Mock backend that returns queued raw responses for structured calls."""

    def __init__(self) -> None:
        super().__init__()
        self._raw_sequence: list[str] = []
        self._sequence_index = 0

    @property
    def call_count(self) -> int:
        return self._sequence_index

    def queue_raw_response(self, content: str) -> None:
        self._raw_sequence.append(content)

    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AgentResponse:
        self.last_messages = messages
        self.last_system = system
        self.last_tools = tools
        self.last_response_schema = response_schema
        self.last_temperature = temperature

        if self._sequence_index < len(self._raw_sequence):
            content = self._raw_sequence[self._sequence_index]
            self._sequence_index += 1
            return self._make_response(content)

        return await super().chat(messages, system, tools, response_schema, max_tokens, temperature)


class TestParseRetryIsolation:
    """Parse retries should not pollute the caller's conversation."""

    async def test_retry_does_not_leak_failed_attempts(self) -> None:
        """After retry, caller's conversation should only have the final exchange."""
        backend = SequencedMockBackend()
        # First attempt: invalid JSON -> triggers retry
        backend.queue_raw_response("not json {")
        # Second attempt: valid JSON -> success
        backend.queue_raw_response(json.dumps({"value": "success"}))

        call = CallOptions(parse_retries=1)
        conv = ListConversation()

        extract = Extract(make_config(backend, call=call))
        result = await extract("content", SimpleResult, conversation=conv)

        assert result.value.value == "success"
        # Only the successful exchange should be in the conversation
        # (not the failed attempt's user+assistant pair)
        msgs = conv.as_messages()
        assert len(msgs) == 2
        assert msgs[0].role == Role.USER
        assert msgs[1].role == Role.ASSISTANT

    async def test_retry_without_conversation_still_works(self) -> None:
        """Parse retry with no conversation should work as before."""
        backend = SequencedMockBackend()
        backend.queue_raw_response("bad json")
        backend.queue_raw_response(json.dumps({"value": "ok"}))

        call = CallOptions(parse_retries=1)
        extract = Extract(make_config(backend, call=call))
        result = await extract("content", SimpleResult)

        assert result.value.value == "ok"

    async def test_retry_preserves_prior_history(self) -> None:
        """Parse retry should still send prior conversation to backend."""
        backend = SequencedMockBackend()
        backend.queue_raw_response("invalid")
        backend.queue_raw_response(json.dumps({"value": "ok"}))

        call = CallOptions(parse_retries=1)
        conv = ListConversation()
        conv.append(Message(role=Role.USER, content="prior"))
        conv.append(Message(role=Role.ASSISTANT, content="prior answer"))

        extract = Extract(make_config(backend, call=call))
        result = await extract("content", SimpleResult, conversation=conv)

        assert result.value.value == "ok"
        # Conversation should have prior + final exchange only
        msgs = conv.as_messages()
        assert len(msgs) == 4  # 2 prior + 2 final
        assert msgs[0].content == "prior"
        assert msgs[1].content == "prior answer"
        assert msgs[2].role == Role.USER
        assert msgs[3].role == Role.ASSISTANT


# ---------------------------------------------------------------------------
# Guard retry threading
# ---------------------------------------------------------------------------


class TestGuardRetryConversation:
    """Guard retries should thread conversation through."""

    async def test_text_guard_retry_appends_to_conversation(self) -> None:
        """Guard retry exchanges should appear in the conversation."""
        call_count = 0

        class CountingBackend(MockBackend):
            async def chat(
                self,
                messages: list[Message],
                system: str | None = None,
                tools: list[ToolDef] | None = None,
                response_schema: dict[str, Any] | None = None,
                max_tokens: int | None = None,
                temperature: float | None = None,
            ) -> AgentResponse:
                nonlocal call_count
                self.last_messages = messages
                call_count += 1
                if call_count == 1:
                    return self._make_response("BAD output")
                return self._make_response("GOOD output")

        from llm_saia import OutputGuard

        def reject_bad(text: str) -> str | None:
            if "BAD" in text:
                return "Contains BAD"
            return None

        guard = OutputGuard(reject_bad, "Fix it", max_retries=1)
        call = CallOptions(output_guards=(guard,))
        backend = CountingBackend()
        conv = ListConversation()

        instruct = Instruct(make_config(backend, call=call))
        result = await instruct("do something", conversation=conv)

        assert result.value == "GOOD output"
        # Conversation should have: initial user+assistant, then guard retry user+assistant
        msgs = conv.as_messages()
        assert len(msgs) == 4
        assert msgs[0].role == Role.USER
        assert msgs[1].role == Role.ASSISTANT
        assert msgs[1].content == "BAD output"
        assert msgs[2].role == Role.USER  # guard retry prompt
        assert msgs[3].role == Role.ASSISTANT
        assert msgs[3].content == "GOOD output"

    async def test_text_guard_retry_sees_prior_history(self) -> None:
        """Guard retry should send prior conversation context to backend."""
        seen_messages: list[list[Message]] = []

        class TrackingBackend(MockBackend):
            async def chat(
                self,
                messages: list[Message],
                system: str | None = None,
                tools: list[ToolDef] | None = None,
                response_schema: dict[str, Any] | None = None,
                max_tokens: int | None = None,
                temperature: float | None = None,
            ) -> AgentResponse:
                seen_messages.append(list(messages))
                self.last_messages = messages
                if len(seen_messages) == 1:
                    return self._make_response("REJECT")
                return self._make_response("ACCEPT")

        from llm_saia import OutputGuard

        def reject_first(text: str) -> str | None:
            if "REJECT" in text:
                return "Rejected"
            return None

        guard = OutputGuard(reject_first, "Fix it", max_retries=1)
        call = CallOptions(output_guards=(guard,))
        backend = TrackingBackend()
        conv = ListConversation()
        conv.append(Message(role=Role.USER, content="prior q"))
        conv.append(Message(role=Role.ASSISTANT, content="prior a"))

        instruct = Instruct(make_config(backend, call=call))
        await instruct("directive", conversation=conv)

        # First call: prior (2) + new user (1) = 3
        assert len(seen_messages[0]) == 3
        # Second call (guard retry): prior (2) + initial user+assistant (2) + retry user (1) = 5
        assert len(seen_messages[1]) == 5

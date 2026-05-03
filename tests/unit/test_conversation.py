"""Tests for ConversationLike protocol and ListConversation."""

import pytest

from llm_saia.core.conversation import (
    AsyncConversationLike,
    ConversationLike,
    ListConversation,
    Message,
    MessageAppendable,
    ToolCall,
)

pytestmark = pytest.mark.unit


class TestListConversation:
    """Tests for the default ListConversation implementation."""

    def test_append_adds_message(self) -> None:
        conv = ListConversation()
        msg = Message(role="user", content="hello")
        conv.append(msg)

        assert len(conv.as_messages()) == 1
        assert conv.as_messages()[0] == msg

    def test_as_messages_returns_view_not_copy(self) -> None:
        conv = ListConversation()
        conv.append(Message(role="user", content="first"))

        messages = conv.as_messages()
        conv.append(Message(role="assistant", content="second"))

        # The view should reflect the new message
        assert len(messages) == 2

    def test_multiple_messages(self) -> None:
        conv = ListConversation()
        conv.append(Message(role="user", content="hello"))
        conv.append(Message(role="assistant", content="hi there"))
        conv.append(Message(role="user", content="how are you?"))

        messages = conv.as_messages()
        assert len(messages) == 3
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[2].role == "user"


class TestConversationLikeProtocol:
    """Tests for the ConversationLike protocol compatibility."""

    def test_list_conversation_satisfies_protocol(self) -> None:
        conv = ListConversation()
        assert isinstance(conv, ConversationLike)

    def test_list_conversation_satisfies_appendable(self) -> None:
        conv = ListConversation()
        assert isinstance(conv, MessageAppendable)

    def test_raw_list_satisfies_appendable(self) -> None:
        messages: list[Message] = []
        # list[Message] satisfies MessageAppendable (has append method)
        assert isinstance(messages, MessageAppendable)


class TestCustomConversation:
    """Test that custom implementations work with the protocol."""

    def test_custom_implementation(self) -> None:
        class TrackingConversation:
            """Custom conversation that tracks append count."""

            def __init__(self) -> None:
                self._messages: list[Message] = []
                self.append_count = 0

            def append(self, msg: Message) -> None:
                self._messages.append(msg)
                self.append_count += 1

            def as_messages(self) -> list[Message]:
                return self._messages

        conv = TrackingConversation()
        assert isinstance(conv, ConversationLike)

        conv.append(Message(role="user", content="test"))
        conv.append(Message(role="assistant", content="response"))

        assert conv.append_count == 2
        assert len(conv.as_messages()) == 2


class TestAsyncConversationLike:
    """Tests for the AsyncConversationLike protocol."""

    def test_sync_conversation_not_async(self) -> None:
        """ListConversation should NOT be detected as AsyncConversationLike."""
        conv = ListConversation()
        assert isinstance(conv, ConversationLike)
        assert not isinstance(conv, AsyncConversationLike)

    def test_async_conversation_detected(self) -> None:
        """Custom async conversation should be detected as AsyncConversationLike."""

        class AsyncConversation:
            def __init__(self) -> None:
                self._messages: list[Message] = []

            def append(self, msg: Message) -> None:
                self._messages.append(msg)

            async def append_async(self, msg: Message) -> None:
                self._messages.append(msg)

            def as_messages(self) -> list[Message]:
                return self._messages

        conv = AsyncConversation()
        assert isinstance(conv, ConversationLike)
        assert isinstance(conv, AsyncConversationLike)

    async def test_append_async_called(self) -> None:
        """Verify append_async is actually called for async conversations."""

        class TrackingAsyncConversation:
            def __init__(self) -> None:
                self._messages: list[Message] = []
                self.sync_count = 0
                self.async_count = 0

            def append(self, msg: Message) -> None:
                self._messages.append(msg)
                self.sync_count += 1

            async def append_async(self, msg: Message) -> None:
                self._messages.append(msg)
                self.async_count += 1

            def as_messages(self) -> list[Message]:
                return self._messages

        conv = TrackingAsyncConversation()
        msg = Message(role="user", content="test")

        # Directly test the async method
        await conv.append_async(msg)
        assert conv.async_count == 1
        assert conv.sync_count == 0
        assert len(conv.as_messages()) == 1


class TestToolCallSerialization:
    """Tests for ToolCall serialization."""

    def test_to_dict(self) -> None:
        tc = ToolCall(id="call_1", name="search", arguments={"query": "python"})
        d = tc.to_dict()

        assert d == {"id": "call_1", "name": "search", "arguments": {"query": "python"}}

    def test_from_dict(self) -> None:
        d = {"id": "call_2", "name": "read", "arguments": {"path": "/tmp/file.txt"}}
        tc = ToolCall.from_dict(d)

        assert tc.id == "call_2"
        assert tc.name == "read"
        assert tc.arguments == {"path": "/tmp/file.txt"}

    def test_roundtrip(self) -> None:
        original = ToolCall(id="call_3", name="write", arguments={"data": "hello"})
        restored = ToolCall.from_dict(original.to_dict())

        assert restored == original


class TestMessageSerialization:
    """Tests for Message serialization."""

    def test_to_dict_basic(self) -> None:
        msg = Message(role="user", content="hello")
        d = msg.to_dict()

        assert d == {"role": "user", "content": "hello"}
        assert "tool_calls" not in d
        assert "tool_call_id" not in d

    def test_to_dict_with_tool_calls(self) -> None:
        msg = Message(
            role="assistant",
            content="Let me search",
            tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "x"})],
        )
        d = msg.to_dict()

        assert d["role"] == "assistant"
        assert d["content"] == "Let me search"
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0] == {"id": "c1", "name": "search", "arguments": {"q": "x"}}

    def test_to_dict_with_tool_call_id(self) -> None:
        msg = Message(role="tool", content="result data", tool_call_id="c1")
        d = msg.to_dict()

        assert d == {"role": "tool", "content": "result data", "tool_call_id": "c1"}

    def test_from_dict_basic(self) -> None:
        d = {"role": "user", "content": "hello"}
        msg = Message.from_dict(d)

        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_from_dict_with_tool_calls(self) -> None:
        d = {
            "role": "assistant",
            "content": "searching",
            "tool_calls": [{"id": "c2", "name": "search", "arguments": {"q": "y"}}],
        }
        msg = Message.from_dict(d)

        assert msg.role == "assistant"
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "c2"
        assert msg.tool_calls[0].name == "search"

    def test_from_dict_with_tool_call_id(self) -> None:
        d = {"role": "tool", "content": "done", "tool_call_id": "c3"}
        msg = Message.from_dict(d)

        assert msg.role == "tool"
        assert msg.tool_call_id == "c3"

    def test_roundtrip_complex(self) -> None:
        original = Message(
            role="assistant",
            content="multiple tools",
            tool_calls=[
                ToolCall(id="a", name="t1", arguments={"x": 1}),
                ToolCall(id="b", name="t2", arguments={"y": 2}),
            ],
        )
        restored = Message.from_dict(original.to_dict())

        assert restored == original

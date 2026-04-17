"""Tests for ConversationLike protocol and ListConversation."""

import pytest

from llm_saia.core.conversation import (
    AsyncConversationLike,
    ConversationLike,
    ListConversation,
    Message,
    MessageAppendable,
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

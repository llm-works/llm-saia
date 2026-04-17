"""Conversation and message types for SAIA.

This module contains the core types for managing conversation history:
- Role: Standard message roles (extensible via StrEnum)
- Message: A single message in conversation history
- ConversationLike: Protocol for pluggable conversation management
- ListConversation: Default implementation using a plain list
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AsyncConversationLike",
    "ConversationLike",
    "ListConversation",
    "Message",
    "MessageAppendable",
    "Role",
    "ToolCall",
]


class Role(StrEnum):
    """Standard message roles.

    Use these constants for common roles. The Message.role field accepts any string,
    allowing extension for custom roles (e.g., "memory", "summary" for RAG/compaction).

    Example extending roles::

        class CustomRole(StrEnum):
            MEMORY = "memory"  # For RAG-injected context
            SUMMARY = "summary"  # For compacted history
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"  # Tool result (message has tool_call_id)


@dataclass
class ToolCall:
    """A tool invocation from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """A message in the conversation history.

    Attributes:
        role: Message role - use Role enum constants or custom strings.
            Standard roles: "user", "assistant", "system", "tool".
        content: Message text content.
        tool_calls: Tool calls made by assistant (role="assistant" only).
        tool_call_id: ID of the tool call this message responds to (role="tool" only).
    """

    role: str  # Role enum or custom string
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


@runtime_checkable
class MessageAppendable(Protocol):
    """Protocol for objects that can append messages.

    Satisfied by both ``list[Message]`` and ``ConversationLike``.
    Used internally where only append is needed (e.g., tool execution).
    """

    def append(self, msg: Message) -> None:
        """Append a message."""
        ...


@runtime_checkable
class ConversationLike(Protocol):
    """Protocol for pluggable conversation/message management.

    Allows external systems (e.g., kelt) to provide conversation objects that
    handle compaction, token tracking, and session persistence while saia's
    tool loop remains stateless.

    Contract:
        - ``append()`` adds a message to the conversation
        - ``as_messages()`` returns a **view** (not copy) of the current messages
        - saia calls ``as_messages()`` fresh before each LLM call
        - Implementors may mutate the underlying list between iterations (e.g., compaction)
    """

    def append(self, msg: Message) -> None:
        """Append a message to the conversation."""
        ...

    def as_messages(self) -> list[Message]:
        """Return current messages as a list (view, not copy)."""
        ...


@runtime_checkable
class AsyncConversationLike(ConversationLike, Protocol):
    """Extended protocol with async append support for non-blocking compaction.

    Use this when conversation operations may trigger I/O (e.g., LLM-based
    compaction). The ``append_async()`` method allows compaction to run without
    blocking the event loop.

    Implementations should support both ``append()`` and ``append_async()``:
        - ``append()``: Synchronous append (may block or raise if async compaction needed)
        - ``append_async()``: Non-blocking append with async compaction support
    """

    async def append_async(self, msg: Message) -> None:
        """Append a message asynchronously.

        Async variant of ``append()`` that supports non-blocking compaction.
        Use this when the conversation may trigger I/O during append (e.g.,
        LLM-based summarization for compaction).
        """
        ...


class ListConversation:
    """Default ConversationLike implementation using a plain list.

    Provides the same behavior as saia's original internal list[Message].
    Used when no external conversation object is provided.
    """

    __slots__ = ("_messages",)

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def append(self, msg: Message) -> None:
        """Append a message to the conversation."""
        self._messages.append(msg)

    def as_messages(self) -> list[Message]:
        """Return current messages (view, not copy)."""
        return self._messages

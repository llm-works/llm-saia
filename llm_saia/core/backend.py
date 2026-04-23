"""Backend interface that any LLM framework must implement.

SAIA defines what it needs from an LLM backend - implementations live elsewhere
(e.g., llm-infer/client). This keeps SAIA as a pure language layer.

Usage:
    from llm_saia.core.backend import Backend, ToolDef, ChatResponse

    class MyBackend(Backend):
        async def chat(self, messages, system=None, tools=None, ...):
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .conversation import Message, ToolCall

__all__ = [
    "Backend",
    "ChatResponse",
    "ToolDef",
]


@dataclass
class ToolDef:
    """Definition of a tool that can be called by the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass
class ChatResponse:
    """Response from Backend.chat().

    `raw` holds the unmodified backend-native response. Reading from it couples
    the caller to a specific backend; SAIA makes no stability guarantees about
    its shape.
    """

    content: str
    tool_calls: list[ToolCall]
    finish_reason: str | None = None  # "end_turn", "tool_use", etc.
    input_tokens: int = 0
    output_tokens: int = 0
    call_id: str = ""  # Set by SAIA per chat() call for tracing
    model: str | None = None  # Resolved model name returned by the backend
    raw: Any = None  # Backend-native response object; consumers reading it accept coupling


class Backend(ABC):
    """Stateless interface for LLM backends.

    A single chat() method handles all LLM interactions. SAIA handles
    structured output parsing, tool loop logic, and prompt construction.

    Resource management (close(), context managers) is the responsibility
    of implementations and their callers, not SAIA.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            messages: Conversation history.
            system: Optional system prompt.
            tools: Optional tools the LLM can call.
            response_schema: Optional JSON schema for structured output.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (None = backend default).

        Returns:
            ChatResponse with content, tool calls, and token usage.
        """
        ...

"""Backend interface that any LLM framework must implement.

SAIA defines what it needs from an LLM backend - implementations live elsewhere
(e.g., llm-infer/client). This keeps SAIA as a pure language layer.

Usage:
    from llm_saia.core.backend import SAIABackend, Message, ToolDef, AgentResponse

    class MyBackend(SAIABackend):
        async def chat(self, messages, system=None, tools=None, ...):
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# --- Types for the backend interface ---


@dataclass
class ToolDef:
    """Definition of a tool that can be called by the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass
class ToolCall:
    """A tool invocation from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """A message in the conversation history."""

    role: str  # "user", "assistant", "tool_result"
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool_result messages


@dataclass
class AgentResponse:
    """Response from LLM that may include tool calls."""

    content: str
    tool_calls: list[ToolCall]
    stop_reason: str | None = None  # "end_turn", "tool_use", etc.
    input_tokens: int = 0
    output_tokens: int = 0


# --- Backend ABC ---


class SAIABackend(ABC):
    """Stateless interface for LLM backends.

    A single chat() method handles all LLM interactions. SAIA handles
    structured output parsing, tool loop logic, and prompt construction.

    Implementations handle HTTP communication and resource management.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> AgentResponse:
        """Send a chat completion request.

        Args:
            messages: Conversation history.
            system: Optional system prompt.
            tools: Optional tools the LLM can call.
            response_schema: Optional JSON schema for structured output.
            max_tokens: Maximum tokens to generate.

        Returns:
            AgentResponse with content, tool calls, and token usage.
        """
        ...

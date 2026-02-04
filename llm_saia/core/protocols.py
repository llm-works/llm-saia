"""Backend protocol that any LLM framework must implement."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from llm_saia.core.types import AgentResponse, Message, ToolDef

T = TypeVar("T")


class SAIABackend(ABC):
    """Interface that any LLM framework must implement.

    Backends should also implement async context manager protocol (__aenter__/__aexit__)
    for proper resource cleanup.
    """

    @abstractmethod
    async def complete(self, prompt: str) -> str:
        """Basic LLM completion.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            The LLM's text response.
        """
        ...

    @abstractmethod
    async def complete_structured(self, prompt: str, schema: type[T]) -> T:
        """LLM completion with structured output.

        Args:
            prompt: The prompt to send to the LLM.
            schema: A dataclass type to parse the response into.

        Returns:
            An instance of the schema type populated from the LLM response.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the backend and release resources.

        Should be idempotent (safe to call multiple times).
        """
        ...

    @abstractmethod
    async def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> AgentResponse:
        """LLM completion with tool calling support.

        Args:
            messages: Conversation history.
            tools: Available tools the LLM can call.
            system: Optional system prompt.
            max_tokens: Maximum tokens for this call.

        Returns:
            AgentResponse with content, tool calls, and token usage.
        """
        ...

    async def __aenter__(self) -> "SAIABackend":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

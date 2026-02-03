"""Backend protocol that any LLM framework must implement."""

from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar("T")


class SAIABackend(ABC):
    """Interface that any LLM framework must implement."""

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

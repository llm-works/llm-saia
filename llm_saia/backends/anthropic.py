"""Anthropic Claude backend implementation."""

from typing import TypeVar

import anthropic
from anthropic.types import TextBlock, ToolParam, ToolUseBlock

from llm_saia.backends._schema import (
    dataclass_to_json_schema,
    parse_json_to_dataclass,
)
from llm_saia.core.protocols import SAIABackend

T = TypeVar("T")


class AnthropicBackend(SAIABackend):
    """Backend using Anthropic Claude API directly."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        """Initialize the Anthropic backend.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: Model identifier to use.
        """
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def complete(self, prompt: str) -> str:
        """Basic LLM completion."""
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content:
            raise ValueError("API returned empty content")
        block = response.content[0]
        if not isinstance(block, TextBlock):
            raise ValueError(f"Expected TextBlock, got {type(block)}")
        return str(block.text)

    async def complete_structured(self, prompt: str, schema: type[T]) -> T:
        """LLM completion with structured output using tool_use."""
        json_schema = dataclass_to_json_schema(schema)
        tool_param: ToolParam = {
            "name": json_schema["name"],
            "description": json_schema["description"],
            # Anthropic uses "input_schema" while our shared util uses "schema"
            "input_schema": json_schema["schema"],
        }

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool_param],
            tool_choice={"type": "tool", "name": json_schema["name"]},
        )

        # Extract the tool use block
        for block in response.content:
            if isinstance(block, ToolUseBlock):
                return parse_json_to_dataclass(block.input, schema)

        raise ValueError(f"No tool_use block in response: {response.content}")

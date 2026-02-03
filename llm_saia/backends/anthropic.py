"""Anthropic Claude backend implementation."""

import dataclasses
from typing import Any, TypeVar, get_type_hints

import anthropic
from anthropic.types import TextBlock, ToolParam, ToolUseBlock

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
        return block.text

    async def complete_structured(self, prompt: str, schema: type[T]) -> T:
        """LLM completion with structured output using tool_use."""
        tool_schema = _dataclass_to_tool_schema(schema)
        tool_param: ToolParam = {
            "name": tool_schema["name"],
            "description": tool_schema["description"],
            "input_schema": tool_schema["input_schema"],
        }

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool_param],
            tool_choice={"type": "tool", "name": tool_schema["name"]},
        )

        # Extract the tool use block
        for block in response.content:
            if isinstance(block, ToolUseBlock):
                return _parse_tool_result(block.input, schema)

        raise ValueError(f"No tool_use block in response: {response.content}")


def _dataclass_to_tool_schema(schema: type) -> dict[str, Any]:
    """Convert a dataclass to an Anthropic tool schema."""
    if not dataclasses.is_dataclass(schema):
        raise TypeError(f"Schema must be a dataclass, got {type(schema)}")

    hints = get_type_hints(schema)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for field in dataclasses.fields(schema):
        field_type = hints[field.name]
        properties[field.name] = _python_type_to_json_schema(field_type)

        # Check if field has a default
        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            required.append(field.name)

    return {
        "name": schema.__name__,
        "description": schema.__doc__ or f"Structured output for {schema.__name__}",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _python_type_to_json_schema(python_type: type) -> dict[str, Any]:
    """Convert Python type hints to JSON schema."""
    origin = getattr(python_type, "__origin__", None)

    if python_type is str:
        return {"type": "string"}
    elif python_type is int:
        return {"type": "integer"}
    elif python_type is float:
        return {"type": "number"}
    elif python_type is bool:
        return {"type": "boolean"}
    elif origin is list:
        args = getattr(python_type, "__args__", (Any,))
        return {"type": "array", "items": _python_type_to_json_schema(args[0])}
    elif origin is dict:
        return {"type": "object"}
    elif python_type is Any:
        return {"type": "string"}
    else:
        raise TypeError(
            f"Unsupported type for JSON schema: {python_type}. "
            "Supported types: str, int, float, bool, list[T], dict, Any."
        )


def _parse_tool_result(data: object, schema: type[T]) -> T:
    """Parse tool result into dataclass instance."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")
    return schema(**data)

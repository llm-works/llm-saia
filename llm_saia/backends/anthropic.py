"""Anthropic Claude backend implementation."""

from typing import Any, TypeVar, cast

import anthropic
from anthropic.types import MessageParam, TextBlock, ToolParam, ToolResultBlockParam, ToolUseBlock

from llm_saia.backends._schema import (
    dataclass_to_json_schema,
    parse_json_to_dataclass,
)
from llm_saia.core.protocols import SAIABackend
from llm_saia.core.types import AgentResponse, Message, ToolCall, ToolDef

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

    async def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> AgentResponse:
        """LLM completion with tool calling support."""
        anthropic_messages = self._convert_messages(messages)
        tool_params = self._convert_tools(tools)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "tools": tool_params,
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        return self._parse_response(response)

    def _convert_tools(self, tools: list[ToolDef]) -> list[ToolParam]:
        """Convert SAIA tools to Anthropic tool format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    def _parse_response(self, response: Any) -> AgentResponse:
        """Parse Anthropic response into AgentResponse."""
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if isinstance(block, TextBlock):
                content_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=cast(dict[str, Any], block.input) if block.input else {},
                    )
                )

        # Extract token usage from response
        input_tokens = getattr(response.usage, "input_tokens", 0) if response.usage else 0
        output_tokens = getattr(response.usage, "output_tokens", 0) if response.usage else 0

        return AgentResponse(
            content="\n".join(content_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _convert_messages(self, messages: list[Message]) -> list[MessageParam]:
        """Convert SAIA messages to Anthropic message format."""
        result: list[MessageParam] = []
        for msg in messages:
            converted = self._convert_single_message(msg)
            if converted:
                result.append(converted)
        return result

    def _convert_single_message(self, msg: Message) -> MessageParam | None:
        """Convert a single SAIA message to Anthropic format."""
        if msg.role == "user":
            return {"role": "user", "content": msg.content}
        elif msg.role == "assistant":
            return self._convert_assistant_message(msg)
        elif msg.role == "tool_result":
            return self._convert_tool_result(msg)
        return None

    def _convert_assistant_message(self, msg: Message) -> MessageParam:
        """Convert assistant message, handling tool calls if present."""
        if not msg.tool_calls:
            return {"role": "assistant", "content": msg.content}

        content: list[Any] = []
        if msg.content:
            content.append({"type": "text", "text": msg.content})
        for tc in msg.tool_calls:
            content.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
            )
        return {"role": "assistant", "content": content}

    def _convert_tool_result(self, msg: Message) -> MessageParam:
        """Convert tool result to Anthropic format."""
        tool_result: ToolResultBlockParam = {
            "type": "tool_result",
            "tool_use_id": msg.tool_call_id or "",
            "content": msg.content,
        }
        return {"role": "user", "content": [tool_result]}

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

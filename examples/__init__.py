"""Simple OpenAI backend for examples.

This is a minimal implementation of Backend for demonstration purposes.
Production implementations should live in llm-infer/client.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from llm_saia.core.backend import (
    AgentResponse,
    Backend,
    Message,
    ToolCall,
    ToolDef,
)


class OpenAIBackend(Backend):
    """Simple OpenAI-compatible backend for examples.

    Works with OpenAI, local LLMs (ollama, llama.cpp, vLLM), and other
    OpenAI-compatible APIs.

    Environment variables:
        LLM_BASE_URL: Base URL for API (default: http://localhost:8000/v1)
        LLM_MODEL: Model name (default: gpt-4o-mini)
        OPENAI_API_KEY: API key (optional for local LLMs)
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self._model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
        self._client = httpx.AsyncClient(timeout=60.0)

    def _build_api_messages(
        self, messages: list[Message], system: str | None
    ) -> list[dict[str, Any]]:
        """Convert SAIA messages to OpenAI API format."""
        api_messages: list[dict[str, Any]] = []
        if system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            api_messages.append(self._convert_message(msg))

        return api_messages

    def _convert_message(self, msg: Message) -> dict[str, Any]:
        """Convert a single SAIA message to OpenAI format."""
        if msg.role == "tool_result":
            if not msg.tool_call_id:
                raise ValueError("tool_call_id is required for tool_result messages")
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            }
        if msg.role == "assistant" and msg.tool_calls:
            return {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in msg.tool_calls
                ],
            }
        return {"role": msg.role, "content": msg.content}

    def _build_request(
        self,
        api_messages: list[dict[str, Any]],
        tools: list[ToolDef] | None,
        response_schema: dict[str, Any] | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Build the OpenAI API request body."""
        request: dict[str, Any] = {"model": self._model, "messages": api_messages}

        if max_tokens:
            request["max_tokens"] = max_tokens

        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        if response_schema:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.get("name", "response"),
                    "strict": True,
                    "schema": response_schema.get("schema", {}),
                },
            }

        return request

    def _parse_tool_arguments(self, args_str: str) -> dict[str, Any]:
        """Parse tool arguments JSON, returning error dict on malformed input."""
        try:
            result: dict[str, Any] = json.loads(args_str)
            return result
        except json.JSONDecodeError:
            return {"_error": "malformed_json", "_raw": args_str[:200]}

    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        """Parse OpenAI API response into AgentResponse."""
        choices = data.get("choices")
        if not choices:
            raise ValueError(f"API response missing 'choices': {data}")
        choice = choices[0]
        message = choice.get("message")
        if not message:
            raise ValueError(f"API response missing 'message': {choice}")
        usage = data.get("usage", {})

        tool_calls: list[ToolCall] = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=self._parse_tool_arguments(tc["function"]["arguments"]),
                    )
                )

        return AgentResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason"),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> AgentResponse:
        """Send a chat completion request to OpenAI."""
        api_messages = self._build_api_messages(messages, system)
        request = self._build_request(api_messages, tools, response_schema, max_tokens)

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        response = await self._client.post(
            f"{self._base_url}/chat/completions",
            headers=headers,
            json=request,
        )
        response.raise_for_status()

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise ValueError(f"API returned invalid JSON: {e}") from e

        return self._parse_response(data)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> OpenAIBackend:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

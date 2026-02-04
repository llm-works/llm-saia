"""OpenAI-compatible backend.

Connects to any server implementing the OpenAI API format,
such as vLLM, llama.cpp, Ollama, or OpenAI itself.

Usage:
    from llm_saia import SAIA
    from llm_saia.backends.openai import OpenAIBackend

    # Local server (no auth needed)
    backend = OpenAIBackend()  # defaults to http://localhost:8000

    # OpenAI API (requires api_key or OPENAI_API_KEY env var)
    backend = OpenAIBackend(
        base_url="https://api.openai.com",
        model="gpt-4",
        api_key="sk-...",  # or set OPENAI_API_KEY env var
    )

    saia = SAIA(backend=backend)
"""

from __future__ import annotations

import json
import os
from typing import Any, TypeVar, cast

import httpx

from llm_saia.backends._schema import dataclass_to_json_schema, parse_json_to_dataclass
from llm_saia.core.protocols import SAIABackend
from llm_saia.core.types import AgentResponse, Message, RunConfig, ToolCall, ToolDef

T = TypeVar("T")

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "default"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_TOKENS = 4096


class OpenAIBackend(SAIABackend):
    """Backend for OpenAI-compatible servers."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        api_key: str | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers)
        self._run: RunConfig | None = None

    @property
    def _max_tokens(self) -> int:
        """Get max tokens from run config or default."""
        if self._run and self._run.max_call_tokens > 0:
            return self._run.max_call_tokens
        return DEFAULT_MAX_TOKENS

    def set_run_config(self, run: RunConfig) -> None:
        """Set the run configuration for token limits."""
        self._run = run

    async def complete(self, prompt: str) -> str:
        """Basic completion."""
        response = await self._client.post(
            f"{self._base_url}/v1/chat/completions",
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self._max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        content: str = data["choices"][0]["message"]["content"]
        return content

    async def complete_structured(self, prompt: str, schema: type[T]) -> T:
        """Completion with structured output via JSON schema."""
        json_schema = dataclass_to_json_schema(schema)

        # Use only the raw schema in prompts, not the OpenAI wrapper format
        # (name/description/schema). The wrapper confuses models into echoing it back.
        raw_schema = json_schema["schema"]

        structured_prompt = (
            f"{prompt}\n\n"
            f"Respond with valid JSON matching this schema:\n"
            f"{json.dumps(raw_schema, indent=2)}"
        )

        response = await self._client.post(
            f"{self._base_url}/v1/chat/completions",
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": structured_prompt}],
                "response_format": {"type": "json_object"},
                "max_tokens": self._max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]

        parsed = json.loads(content)
        return parse_json_to_dataclass(parsed, schema)

    async def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system: str | None = None,
    ) -> AgentResponse:
        """Completion with tool calling support."""
        response = await self._client.post(
            f"{self._base_url}/v1/chat/completions",
            json={
                "model": self._model,
                "messages": self._convert_messages(messages, system),
                "tools": self._convert_tools(tools) or None,
                "max_tokens": self._max_tokens,
            },
        )
        response.raise_for_status()
        return self._parse_tool_response(response.json())

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def _parse_tool_response(self, data: dict[str, Any]) -> AgentResponse:
        """Parse OpenAI response into AgentResponse."""
        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage", {})

        tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=self._parse_tool_arguments(tc["function"]["arguments"]),
            )
            for tc in message.get("tool_calls", [])
        ]

        return AgentResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason"),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    def _parse_tool_arguments(self, args_str: str) -> dict[str, Any]:
        """Parse tool arguments JSON, returning error dict on malformed input."""
        try:
            return cast(dict[str, Any], json.loads(args_str))
        except json.JSONDecodeError:
            return {"_error": "malformed_json", "_raw": args_str[:200]}

    def _convert_messages(
        self, messages: list[Message], system: str | None
    ) -> list[dict[str, Any]]:
        """Convert SAIA messages to OpenAI format."""
        result: list[dict[str, Any]] = []
        if system:
            result.append({"role": "system", "content": system})
        for msg in messages:
            result.append(self._convert_message(msg))
        return result

    def _convert_message(self, msg: Message) -> dict[str, Any]:
        """Convert a single SAIA message to OpenAI format."""
        if msg.role == "user":
            return {"role": "user", "content": msg.content}
        if msg.role == "tool_result":
            if not msg.tool_call_id:
                raise ValueError("tool_call_id is required for tool_result messages")
            return {"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content}
        # assistant
        entry: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in msg.tool_calls
            ]
        return entry

    def _convert_tools(self, tools: list[ToolDef]) -> list[dict[str, Any]]:
        """Convert SAIA tools to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

"""OpenClaw backend implementation.

OpenClaw is an open-source AI agent platform that supports multiple LLM providers
including Claude, OpenRouter, and local models via Ollama.

This backend routes SAIA verb calls through the OpenClaw gateway's HTTP API,
allowing you to leverage OpenClaw's LLM routing and configuration.

Usage:
    from llm_saia import SAIA
    from llm_saia.backends.openclaw import OpenClawBackend

    # Connect to local OpenClaw gateway
    backend = OpenClawBackend()

    # Or specify custom gateway URL and token
    backend = OpenClawBackend(
        gateway_url="http://192.168.1.100:18789",
        token="your-gateway-token"
    )

    saia = SAIA(backend=backend)
    result = await saia.verify("claim", "factually accurate")
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, TypeVar

import httpx

from llm_saia.backends._schema import (
    dataclass_to_json_schema,
    parse_json_to_dataclass,
)
from llm_saia.core.protocols import SAIABackend
from llm_saia.core.types import AgentResponse, Message, RunConfig, ToolCall, ToolDef

T = TypeVar("T")

DEFAULT_GATEWAY_URL = "http://127.0.0.1:18789"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_TOKENS = 4096


class OpenClawBackend(SAIABackend):
    """Backend using OpenClaw gateway for LLM routing.

    OpenClaw supports multiple LLM providers:
    - Anthropic Claude (via API key)
    - OpenRouter (auto-routing to best model)
    - Local models via Ollama (Llama, Mixtral, etc.)

    The backend connects to the OpenClaw gateway's HTTP API and uses the
    llm-task tool for completions.
    """

    def __init__(
        self,
        gateway_url: str | None = None,
        token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """Initialize the OpenClaw backend.

        Args:
            gateway_url: OpenClaw gateway URL. Defaults to http://127.0.0.1:18789.
                Can also be set via OPENCLAW_GATEWAY_URL env var.
            token: Gateway authentication token. Defaults to OPENCLAW_GATEWAY_TOKEN env var.
            timeout: Request timeout in seconds. Defaults to 60.
        """
        self._gateway_url = (
            gateway_url or os.environ.get("OPENCLAW_GATEWAY_URL") or DEFAULT_GATEWAY_URL
        )
        self._token = token or os.environ.get("OPENCLAW_GATEWAY_TOKEN")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._run: RunConfig | None = None

    @property
    def gateway_url(self) -> str:
        """Return the gateway URL (for health checks)."""
        return self._gateway_url

    @property
    def _max_tokens(self) -> int:
        """Get max tokens from run config or default."""
        if self._run and self._run.max_call_tokens > 0:
            return self._run.max_call_tokens
        return DEFAULT_MAX_TOKENS

    def set_run_config(self, run: RunConfig) -> None:
        """Set the run configuration for token limits."""
        self._run = run

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client (thread-safe)."""
        if self._client is None:
            async with self._client_lock:
                # Double-check inside lock to avoid race condition
                if self._client is None:
                    headers: dict[str, str] = {"Content-Type": "application/json"}
                    if self._token:
                        headers["Authorization"] = f"Bearer {self._token}"
                    self._client = httpx.AsyncClient(
                        base_url=self._gateway_url,
                        headers=headers,
                        timeout=self._timeout,
                    )
        return self._client

    async def _invoke_tool(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """Invoke a tool on the OpenClaw gateway.

        Args:
            tool: The tool name to invoke.
            args: Tool arguments.

        Returns:
            The tool result.

        Raises:
            httpx.HTTPStatusError: If the request fails.
            ValueError: If the response indicates an error.
        """
        client = await self._get_client()
        response = await client.post(
            "/tools/invoke",
            json={"tool": tool, "args": args},
        )
        response.raise_for_status()

        data: dict[str, Any] = response.json()
        if not data.get("ok"):
            error = data.get("error", "Unknown error")
            raise ValueError(f"OpenClaw tool invocation failed: {error}")

        result: dict[str, Any] = data.get("result", {})
        return result

    async def complete(self, prompt: str) -> str:
        """Basic LLM completion via OpenClaw llm-task tool.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            The LLM's text response.
        """
        result = await self._invoke_tool(
            "llm-task",
            {
                "action": "text",
                "prompt": prompt,
                "max_tokens": self._max_tokens,
            },
        )

        # Extract text from result
        text = result.get("text") or result.get("output") or result.get("content")
        if text is None:
            # Try to get from details.json if present
            details = result.get("details", {})
            text = details.get("text") or details.get("output")

        if text is None:
            raise ValueError(f"No text in OpenClaw response: {result}")

        return str(text)

    async def complete_structured(self, prompt: str, schema: type[T]) -> T:
        """LLM completion with structured output via OpenClaw llm-task tool.

        Args:
            prompt: The prompt to send to the LLM.
            schema: A dataclass type to parse the response into.

        Returns:
            An instance of the schema type populated from the LLM response.
        """
        json_schema = dataclass_to_json_schema(schema)

        result = await self._invoke_tool(
            "llm-task",
            {
                "action": "json",
                "prompt": prompt,
                "schema": json_schema["schema"],
                "max_tokens": self._max_tokens,
            },
        )

        # Extract JSON from result
        json_data = result.get("json") or result.get("output")
        if json_data is None:
            # Try details.json
            details = result.get("details", {})
            json_data = details.get("json")

        if json_data is None:
            raise ValueError(f"No JSON in OpenClaw response: {result}")

        return parse_json_to_dataclass(json_data, schema)

    async def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system: str | None = None,
    ) -> AgentResponse:
        """LLM completion with tool calling support via OpenClaw gateway."""
        openclaw_messages = self._convert_messages(messages)
        openclaw_tools = self._convert_tools(tools)

        args: dict[str, Any] = {
            "action": "tools",
            "messages": openclaw_messages,
            "tools": openclaw_tools,
            "max_tokens": self._max_tokens,
        }
        if system:
            args["system"] = system

        result = await self._invoke_tool("llm-task", args)
        return self._parse_tool_response(result)

    def _convert_tools(self, tools: list[ToolDef]) -> list[dict[str, Any]]:
        """Convert SAIA tools to OpenClaw format."""
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in tools
        ]

    def _parse_tool_response(self, result: dict[str, Any]) -> AgentResponse:
        """Parse OpenClaw response into AgentResponse."""
        content = result.get("content") or result.get("text") or ""
        raw_tool_calls = result.get("tool_calls") or []

        tool_calls = [
            ToolCall(
                id=tc.get("id", ""),
                name=tc.get("name", ""),
                arguments=tc.get("arguments", {}),
            )
            for tc in raw_tool_calls
        ]

        # Extract token usage if available
        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        return AgentResponse(
            content=str(content),
            tool_calls=tool_calls,
            stop_reason=result.get("stop_reason"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert SAIA messages to OpenClaw message format."""
        result: list[dict[str, Any]] = []

        for msg in messages:
            converted: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.tool_calls:
                converted["tool_calls"] = [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                converted["tool_call_id"] = msg.tool_call_id
            result.append(converted)

        return result

    async def close(self) -> None:
        """Close the HTTP client."""
        async with self._client_lock:
            if self._client is not None:
                await self._client.aclose()
                self._client = None

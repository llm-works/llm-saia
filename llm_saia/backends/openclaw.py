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

import asyncio
import os
from typing import Any, TypeVar

import httpx

from llm_saia.backends._schema import (
    dataclass_to_json_schema,
    parse_json_to_dataclass,
)
from llm_saia.core.protocols import SAIABackend

T = TypeVar("T")

DEFAULT_GATEWAY_URL = "http://127.0.0.1:18789"
DEFAULT_TIMEOUT = 60.0


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

    @property
    def gateway_url(self) -> str:
        """Return the gateway URL (for health checks)."""
        return self._gateway_url

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

    async def close(self) -> None:
        """Close the HTTP client."""
        async with self._client_lock:
            if self._client is not None:
                await self._client.aclose()
                self._client = None

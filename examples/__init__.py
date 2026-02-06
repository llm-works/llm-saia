"""Shared utilities for SAIA examples.

Provides an OpenAI-compatible backend, a simple stderr logger, and trace helpers.
Production backends should live in llm-infer/client.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Awaitable, Callable
from io import StringIO
from typing import Any

import httpx

from llm_saia.core.backend import (
    AgentResponse,
    Backend,
    Message,
    ToolCall,
    ToolDef,
)
from llm_saia.core.logger import Logger as Logger

# ---------------------------------------------------------------------------
# Common tool definitions for examples
# ---------------------------------------------------------------------------

COMMON_TOOLS = [
    ToolDef(
        name="read_file",
        description="Read the contents of a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path to read"}},
            "required": ["path"],
        },
    ),
    ToolDef(
        name="list_files",
        description="List files in a directory",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path"}},
            "required": ["path"],
        },
    ),
    # Terminal tool - signals task completion but is never executed.
    # Must be configured via .terminal("report", output_field="analysis") to mark it
    # as special. When the LLM calls this tool, the controller intercepts it and
    # requests confirmation rather than executing it. On confirmation, the controller
    # extracts the "analysis" field and returns it as the final task output.
    ToolDef(
        name="report",
        description="Submit the final analysis report",
        parameters={
            "type": "object",
            "properties": {"analysis": {"type": "string", "description": "The analysis"}},
            "required": ["analysis"],
        },
    ),
]


async def common_executor(name: str, args: dict[str, Any]) -> str:
    """Execute common tools (read_file, list_files).

    Does not handle terminal tools like 'report' - those are intercepted by
    SAIA's controller and never reach the executor. Terminal tools must be
    configured via .terminal() to mark them as special.

    Args:
        name: Tool name.
        args: Tool arguments.

    Returns:
        Tool execution result as string, or error message.
    """
    from pathlib import Path

    match name:
        case "read_file":
            path = Path(args["path"])
            if not path.exists():
                return f"Error: {path} not found"
            try:
                return path.read_text()
            except (PermissionError, UnicodeDecodeError, OSError) as e:
                return f"Error reading {path}: {e}"
        case "list_files":
            path = Path(args["path"])
            if not path.is_dir():
                return f"Error: {path} is not a directory"
            try:
                return "\n".join(
                    p.name for p in sorted(path.iterdir()) if not p.name.startswith(".")
                )
            except (PermissionError, OSError) as e:
                return f"Error listing {path}: {e}"
        case _:
            return f"Unknown tool: {name}"


def make_executor(
    *handlers: Callable[[str, dict[str, Any]], Awaitable[str]],
) -> Callable[[str, dict[str, Any]], Awaitable[str]]:
    """Create an executor that chains multiple handlers.

    Handlers are tried in order. The first handler that doesn't return
    "Unknown tool: ..." wins. If all handlers return unknown, the last
    error is returned.

    Args:
        *handlers: Async callables with signature (name: str, args: dict) -> str.
                  If not provided, uses common_executor as default.

    Returns:
        Async callable that tries handlers in order.

    Example:
        # Use default common tools only
        executor = make_executor()

        # Extend with custom tools
        async def custom_handler(name: str, args: dict[str, Any]) -> str:
            if name == "custom_tool":
                return "custom result"
            return f"Unknown tool: {name}"

        executor = make_executor(custom_handler, common_executor)
    """
    if not handlers:
        handlers = (common_executor,)

    async def executor(name: str, args: dict[str, Any]) -> str:
        last_error = f"Unknown tool: {name}"
        for handler in handlers:
            result = await handler(name, args)
            if not result.startswith("Unknown tool:"):
                return result
            last_error = result
        return last_error

    return executor


class StderrLogger:
    """Simple logger that prints to stderr. Satisfies the SAIA Logger protocol."""

    def __init__(self, level: str = "debug") -> None:
        self._levels = ("trace", "debug", "info", "warning", "error")
        self._min = self._levels.index(level)

    def _log(self, level: str, msg: str, extra: dict[str, Any] | None) -> None:
        if self._levels.index(level) < self._min:
            return
        parts = [f"[{level.upper():7s}] {msg}"]
        if extra:
            for k, v in extra.items():
                parts.append(f"  {k}={v}")
        print("\n".join(parts), file=sys.stderr)

    def trace(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self._log("trace", msg, extra)

    def debug(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self._log("debug", msg, extra)

    def info(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self._log("info", msg, extra)

    def warning(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self._log("warning", msg, extra)

    def error(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self._log("error", msg, extra)


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
        self._model = model or os.environ.get("LLM_MODEL", "qwen3-4b-instruct-2507-awq")
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
            finish_reason=choice.get("finish_reason"),
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


def print_trace_json(trace_buf: StringIO) -> None:
    """Print the captured JSONL trace as pretty-printed JSON."""
    content = trace_buf.getvalue().strip()
    if not content:
        print("No trace records captured.")
        return

    try:
        records = [json.loads(line) for line in content.split("\n") if line.strip()]
        print(json.dumps(records, indent=2))
    except json.JSONDecodeError as e:
        print(f"Error parsing trace: {e}", file=sys.stderr)


def print_trace_full(record: dict[str, Any]) -> None:
    """Print each trace record as pretty JSON."""
    print(json.dumps(record, indent=2))


def _format_trace_line(record: dict[str, Any]) -> str:
    """Format a single trace record as a compact one-liner."""
    it = record.get("iteration", "?")
    action = record.get("action", "?")
    reason = record.get("reason", "")
    tools = ",".join(record.get("tool_names_used", [])) or "-"
    tokens = record.get("input_tokens", 0) + record.get("output_tokens", 0)
    finish = record.get("finish_reason", "?")
    content_len = len(record.get("content_preview", ""))
    has_nudge = bool(record.get("nudge_preview"))

    parts = [
        f"[{it:>2}]",
        f"{action:<14s}",
        f"reason={reason:<28s}",
        f"tools={tools:<12s}",
        f"tok={tokens:<5}",
        f"fin={finish}",
    ]
    line = " ".join(parts)
    if has_nudge:
        line += "  nudge"
    if content_len:
        line += f"  content={content_len}ch"
    return line


def print_trace_compact(record: dict[str, Any]) -> None:
    """Print a single informative line per trace record."""
    if "_meta" in record:
        meta = record["_meta"]
        print(f"--- trace={meta.get('trace_id', '?')} req={meta.get('request_id', '?')} ---")
        return
    print(_format_trace_line(record))

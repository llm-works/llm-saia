# Implementing a Backend

SAIA is backend-agnostic. It defines what it needs from an LLM client through a single abstract
method - you provide the implementation that talks to your LLM of choice.

## The Backend Protocol

```python
from typing import Any

from llm_saia import Backend, Message, ToolDef, AgentResponse

class MyBackend(Backend):
    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AgentResponse:
        # Your implementation here
        ...
```

That's it. One method. SAIA handles everything else - prompt construction, tool loops, structured
output parsing, state detection.

## Data Types

### Message

```python
@dataclass
class Message:
    role: str           # "user", "assistant", "tool_result"
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool_result messages
```

### ToolDef

```python
@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
```

### ToolCall

```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
```

### AgentResponse

```python
@dataclass
class AgentResponse:
    content: str
    tool_calls: list[ToolCall]
    finish_reason: str | None = None  # "end_turn", "tool_use", etc.
    input_tokens: int = 0
    output_tokens: int = 0
    call_id: str = ""  # Set by SAIA per chat() call for tracing
```

## Example: OpenAI-Compatible Backend

Here's a minimal backend for OpenAI-compatible APIs (works with OpenAI, Ollama, vLLM, llama.cpp):

```python
import json
import httpx
from llm_saia import Backend, Message, ToolDef, ToolCall, AgentResponse

class OpenAIBackend(Backend):
    def __init__(self, model: str, api_key: str | None = None, base_url: str = "https://api.openai.com/v1"):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client = httpx.AsyncClient(timeout=60.0)

    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AgentResponse:
        # Build API messages
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            api_messages.append(self._convert_message(msg))

        # Build request
        request = {"model": self._model, "messages": api_messages}
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if temperature is not None:
            request["temperature"] = temperature
        if tools:
            request["tools"] = self._convert_tools(tools)
        if response_schema:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": response_schema}
            }

        # Make request
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        response = await self._client.post(
            f"{self._base_url}/chat/completions",
            headers=headers,
            json=request,
        )
        response.raise_for_status()
        data = response.json()

        # Parse response
        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage", {})

        tool_calls = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"]),
                ))

        return AgentResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    def _convert_message(self, msg: Message) -> dict:
        if msg.role == "tool_result":
            return {"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content}
        if msg.role == "assistant" and msg.tool_calls:
            return {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in msg.tool_calls
                ],
            }
        return {"role": msg.role, "content": msg.content}

    def _convert_tools(self, tools: list[ToolDef]) -> list[dict]:
        return [
            {"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters
            }}
            for t in tools
        ]

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OpenAIBackend":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
```

## Resource Management

SAIA does not manage backend resources. Your backend owns its connections, and your code owns the
backend lifecycle:

```python
# Option 1: Context manager (recommended)
async with OpenAIBackend("gpt-4o") as backend:
    saia = SAIA.builder().backend(backend).build()
    result = await saia.verify(code, "no SQL injection")

# Option 2: Manual cleanup
backend = OpenAIBackend("gpt-4o")
try:
    saia = SAIA.builder().backend(backend).build()
    result = await saia.verify(code, "no SQL injection")
finally:
    await backend.close()
```

This keeps SAIA as a pure language layer - no hidden connection pools, no surprise network calls on
import.

## Structured Output

When `response_schema` is passed, the backend should request structured JSON output from the LLM.
The schema is a JSON Schema dict. How you request this depends on your LLM:

- **OpenAI**: Use `response_format.json_schema`
- **Anthropic**: Use `tool_use` with a fake tool, or the native structured output feature
- **Ollama/vLLM**: Use `response_format.json_schema` (OpenAI-compatible)
- **Other**: Append schema to the prompt and parse JSON from response

SAIA parses the JSON response and validates it against the schema.

## Tool Calling

When `tools` is passed, the backend should register them with the LLM for function calling. The
response should populate `tool_calls` when the LLM wants to call tools.

If the LLM doesn't support native tool calling, you can:
1. Include tool descriptions in the system prompt
2. Parse tool calls from the response text
3. Return them in `AgentResponse.tool_calls`

SAIA will execute the tools and continue the conversation.

## What Not to Implement

Don't add these to your backend - SAIA handles them:

- **Prompt construction**: SAIA builds the prompts, passes them via `messages`
- **Tool execution**: SAIA calls your executor, passes results back as `tool_result` messages
- **State detection**: SAIA detects loops, repetition, degenerate states
- **Structured output parsing**: SAIA parses JSON and validates against schemas
- **Retry logic**: Handle at the transport layer or let errors propagate

Keep your backend simple - just translate between SAIA's types and your LLM's API.

## See Also

- [Custom Verbs](./custom-verbs.md) - Create your own verb operations
- [Production Guide](./production.md) - Error handling, tracing, best practices
- [examples/](../examples/__init__.py) - Working OpenAI backend implementation

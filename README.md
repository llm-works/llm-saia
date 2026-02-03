# llm-saia

Framework-agnostic verb vocabulary for LLM agents.

## What is SAIA?

SAIA (Structured AI Agent Interface) provides a consistent set of semantic verbs for interacting
with LLMs, independent of the underlying framework (Anthropic, OpenAI, LangChain, etc.).

Instead of writing raw prompts, you express intent through verbs like ASK, VERIFY, CRITIQUE, REFINE.

## Installation

```bash
pip install llm-saia
```

## Quick Start

```python
import asyncio
from llm_saia import SAIA
from llm_saia.backends.anthropic import AnthropicBackend

async def main():
    saia = SAIA(backend=AnthropicBackend())

    # Investigate a claim
    claim = "Python is slower than C for all computational tasks"

    evidence = await saia.ask(claim, "What evidence supports or refutes this?")
    result = await saia.verify(evidence, "factually accurate")
    counter = await saia.critique(claim)

    print(f"Verified: {result.passed}")
    print(f"Reason: {result.reason}")
    print(f"Counter-argument: {counter.counter_argument}")

asyncio.run(main())
```

## Verbs

| Verb | Purpose | Returns |
|------|---------|---------|
| `ask` | Query an artifact with a question | `str` |
| `constrain` | Parse response into structured schema | `T` (dataclass) |
| `verify` | Check if artifact satisfies predicate | `VerifyResult` |
| `critique` | Generate strongest counter-argument | `Critique` |
| `refine` | Improve artifact based on feedback | `str` |
| `synthesize` | Combine multiple artifacts | `T` (dataclass) |
| `ground` | Anchor artifact against sources | `list[Evidence]` |
| `decompose` | Break task into subtasks | `list[str]` |
| `store` | Save value to memory | `None` |
| `recall` | Retrieve from memory | `list[Any]` |

## Backend Configuration

### Anthropic

```python
from llm_saia.backends.anthropic import AnthropicBackend

# Uses ANTHROPIC_API_KEY environment variable
backend = AnthropicBackend()

# Or pass API key directly
backend = AnthropicBackend(api_key="sk-...")

# Specify model
backend = AnthropicBackend(model="claude-sonnet-4-20250514")
```

### OpenClaw

Use any LLM supported by [OpenClaw](https://openclaw.ai/) - Claude, OpenRouter, Ollama, and more.

```python
from llm_saia.backends.openclaw import OpenClawBackend

# Connect to local OpenClaw gateway (default: http://127.0.0.1:18789)
backend = OpenClawBackend()

# Or specify custom gateway URL and token
backend = OpenClawBackend(
    gateway_url="http://192.168.1.100:18789",
    token="your-gateway-token"
)

# Environment variables also supported:
# OPENCLAW_GATEWAY_URL - Gateway URL
# OPENCLAW_GATEWAY_TOKEN - Authentication token
```

Benefits of using OpenClaw:
- **Multi-model**: Switch between Claude, Llama, Mixtral without code changes
- **Local models**: Run entirely offline with Ollama
- **OpenRouter**: Auto-route to most cost-effective model
- **Unified config**: Use your existing OpenClaw setup

## Why SAIA?

- **Framework-agnostic**: Switch backends without changing application code
- **Structured outputs**: Type-safe dataclass responses via tool_use
- **Composable**: Chain verbs together for complex workflows
- **Explicit semantics**: Clear intent makes code readable and debuggable

## License

Apache 2.0

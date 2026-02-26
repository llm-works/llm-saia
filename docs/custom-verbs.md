# Creating Custom Verbs

SAIA ships with 12 built-in verbs, but you can create your own. A verb is a reusable operation with
a defined input/output contract.

## The Verb Base Class

```python
from llm_saia import Verb, Config
from dataclasses import dataclass

class MyVerb(Verb):
    def __init__(self, config: Config):
        super().__init__(config)

    async def __call__(self, input: str) -> str:
        # Your implementation
        ...
```

Every verb:
1. Takes a `Config` in `__init__` (holds backend, tools, system prompt, etc.)
2. Implements `__call__` with your operation's signature
3. Uses inherited helpers for LLM calls

## Inherited Helpers

The `Verb` base class provides:

### `_complete(prompt: str) -> str`

Execute a prompt with tools if available, or direct call otherwise.

```python
async def __call__(self, task: str) -> str:
    prompt = f"Do this task: {task}"
    return await self._complete(prompt)
```

### `_complete_structured(prompt: str, schema: type[T]) -> T`

Execute and parse into a dataclass.

```python
@dataclass
class Analysis:
    summary: str
    score: int
    issues: list[str]

async def __call__(self, code: str) -> Analysis:
    prompt = f"Analyze this code:\n{code}"
    return await self._complete_structured(prompt, Analysis)
```

### `_loop(prompt, run, schema, trace_id) -> tuple[str, T | None]`

Full control over the tool-calling loop with optional structured finalization.

```python
async def __call__(self, task: str) -> str:
    content, _ = await self._loop(
        prompt=f"Complete: {task}",
        run=RunConfig(max_iterations=5, timeout_secs=60),
    )
    return content
```

### `_chat(messages, max_tokens) -> AgentResponse`

Single LLM call. Use when you need fine-grained control.

```python
async def __call__(self, question: str) -> str:
    response = await self._chat(
        messages=[Message(role="user", content=question)],
        max_tokens=500,
    )
    return response.content
```

## Example: Summarize Verb

```python
from dataclasses import dataclass
from llm_saia import Verb, Config

@dataclass
class Summary:
    title: str
    key_points: list[str]
    word_count: int

class Summarize(Verb):
    """Summarize text into structured output."""

    def __init__(self, config: Config):
        super().__init__(config)

    async def __call__(self, text: str, max_points: int = 5) -> Summary:
        prompt = f"""Summarize the following text.
Extract up to {max_points} key points.

Text:
{text}"""
        return await self._complete_structured(prompt, Summary)
```

Usage:

```python
saia = SAIA.builder().backend(backend).build()
summarize = Summarize(saia._config)

summary = await summarize(long_text, max_points=3)
print(summary.title)
print(summary.key_points)
```

## Example: Multi-Step Verb with Tools

```python
from dataclasses import dataclass
from llm_saia import Verb, Config, RunConfig

@dataclass
class Investigation:
    findings: list[str]
    conclusion: str
    confidence: float

class Investigate(Verb):
    """Investigate using tools and return structured findings."""

    def __init__(self, config: Config):
        super().__init__(config)

    async def __call__(
        self,
        question: str,
        max_iterations: int = 10,
    ) -> Investigation:
        prompt = f"""Investigate this question using available tools:

{question}

Use tools to gather information, then provide your findings."""

        run = RunConfig(max_iterations=max_iterations)
        content, result = await self._loop(prompt, run=run, schema=Investigation)

        if result:
            return result

        # Fallback: parse from content if finalization failed
        return await self._complete_structured(
            f"Based on: {content}\n\nProvide investigation results.",
            Investigation,
        )
```

## Accessing Config

The config is available via `self._config`:

```python
class MyVerb(Verb):
    async def __call__(self, input: str) -> str:
        # Access config properties
        if self._has_tools():
            # Tools and executor are configured
            ...

        if self._lg:
            # Logger is configured
            self._lg.info("Starting operation", extra={"input_len": len(input)})

        # Use config backend directly
        response = await self._backend.chat(...)
```

Key properties:
- `self._backend` - The configured backend
- `self._lg` - Logger (or None)
- `self._config.tools` - Tool definitions
- `self._config.executor` - Tool executor function
- `self._config.system` - System prompt
- `self._config.tracer` - Tracer for observability

## Registering with SAIA

To make your verb accessible via `saia.my_verb()`:

```python
from llm_saia import SAIA

# Option 1: Instantiate directly
saia = SAIA.builder().backend(backend).build()
my_verb = MyVerb(saia._config)
result = await my_verb(input)

# Option 2: Extend SAIA (for a framework)
class MySAIA(SAIA):
    @property
    def summarize(self) -> Summarize:
        return Summarize(self._config)

    @property
    def investigate(self) -> Investigate:
        return Investigate(self._config)

# Usage
saia = MySAIA.builder().backend(backend).build()
summary = await saia.summarize(text)
findings = await saia.investigate(question)
```

## Logging

Use the inherited logger for consistent logging:

```python
class MyVerb(Verb):
    async def __call__(self, input: str) -> str:
        if self._lg:
            self._lg.debug("verb started", extra={"input_len": len(input)})

        result = await self._complete(f"Process: {input}")

        if self._lg:
            self._lg.debug("verb completed", extra={"output_len": len(result)})

        return result
```

Log levels: `trace`, `debug`, `info`, `warning`, `error`

## Tracing

Traces are automatic when a tracer is configured. For custom trace data:

```python
class MyVerb(Verb):
    async def __call__(self, input: str) -> str:
        trace_id = self._generate_id()

        response = await self._chat(
            messages=[Message(role="user", content=input)],
            max_tokens=1000,
        )

        # Write trace record
        self._write_base_trace(
            response,
            trace_id=trace_id,
            iteration=0,
            phase="direct",
        )

        return response.content
```

## Best Practices

1. **Single responsibility** - One verb, one operation
2. **Clear contracts** - Use dataclasses for structured output
3. **Sensible defaults** - Make common cases easy
4. **Use helpers** - `_complete` and `_complete_structured` handle the common paths
5. **Log at appropriate levels** - `debug` for progress, `warning` for problems
6. **Handle errors gracefully** - Let SAIA errors propagate, wrap domain errors

## See Also

- [Backend Implementation](./backend.md) - How backends work
- [Production Guide](./production.md) - Error handling, tracing, best practices

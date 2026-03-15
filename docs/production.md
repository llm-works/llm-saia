# Production Guide

Best practices for running SAIA in production: error handling, tracing, resource management, and
operational patterns.

## Recommended: llm-infer

For production use, we recommend [llm-infer](https://github.com/llm-works/llm-infer) as your LLM
client layer. It provides:

- **SAIAAdapter** - Drop-in Backend implementation for SAIA
- **Connection pooling** - Efficient HTTP client management
- **Retries with backoff** - Automatic retry on transient failures
- **Rate limiting** - Respect API rate limits
- **Multiple providers** - Anthropic, OpenAI, and OpenAI-compatible APIs
- **Streaming support** - For real-time output

```python
from llm_infer.client import Factory, SAIAAdapter
from llm_saia import SAIA

factory = Factory(logger)
async with factory.anthropic(model="claude-sonnet-4-20250514") as client:
    backend = SAIAAdapter(client)
    saia = SAIA.builder().backend(backend).build()
    result = await saia.verify(code, "no SQL injection")
```

If you're building your own backend, see the [Backend Guide](./backend.md).

## Error Handling

SAIA provides a hierarchy of exceptions for structured error handling:

```
Error (base)
├── StructuredOutputError    # LLM returned invalid structured output
│   └── TruncatedResponseError  # Response cut off (token limit)
├── ToolExecutionError       # Tool execution failed
├── ConfigurationError       # Invalid SAIA configuration
└── BackendError             # Backend communication failed
```

### Catching Errors

```python
from llm_saia import (
    Error,
    StructuredOutputError,
    TruncatedResponseError,
    ToolExecutionError,
    BackendError,
)

try:
    result = await saia.verify(code, "no SQL injection")
except TruncatedResponseError as e:
    # Response was cut off - increase token limit
    logger.warning("Response truncated", extra={
        "schema": e.schema_name,
        "raw_content": e.raw_content[:200],
    })
    # Retry with higher limit
    result = await saia.with_max_call_tokens(8192).verify(code, "no SQL injection")

except StructuredOutputError as e:
    # LLM returned malformed output
    logger.error("Invalid output", extra={
        "schema": e.schema_name,
        "parse_error": e.parse_error,
        "raw_content": e.raw_content[:500],
    })
    raise

except BackendError as e:
    # Network/API error
    logger.error("Backend failed", extra={
        "status_code": e.status_code,
        "response_body": e.response_body,
    })
    raise

except Error as e:
    # Catch-all for SAIA errors
    logger.error("SAIA error", extra={"error": str(e)})
    raise
```

### Error Attributes

Each error type carries context:

```python
# StructuredOutputError / TruncatedResponseError
e.raw_content    # The raw response that failed to parse
e.schema_name    # Name of the expected schema
e.parse_error    # The parse error message

# ToolExecutionError
e.tool_name      # Name of the failed tool
e.arguments      # Arguments passed to the tool
e.cause          # The underlying exception

# ConfigurationError
e.field          # The invalid config field
e.value          # The invalid value
e.reason         # Why it's invalid

# BackendError
e.status_code    # HTTP status code (if applicable)
e.response_body  # Raw response body
e.cause          # The underlying exception
```

## Tracing

SAIA writes JSONL traces for every LLM call. Use traces for debugging, monitoring, and analysis.

### Enable Tracing

```python
# File-based tracing
saia = (
    SAIA.builder()
    .backend(backend)
    .tracing.file("/var/log/saia/traces.jsonl")
    .build()
)

# Console tracing (stdout)
saia = SAIA.builder().backend(backend).tracing.console().build()

# Callback tracing (custom handler)
def handle_trace(record: dict) -> None:
    # Send to your observability stack
    metrics.increment("saia.calls", tags={"verb": record.get("verb")})
    if record.get("action") == "error":
        alerts.send(record)

saia = SAIA.builder().backend(backend).tracing.callback(handle_trace).build()
```

### Trace Record Fields

Each trace record contains:

```python
{
    "trace_id": "a1b2c3d4",      # Constant across one verb invocation
    "call_id": "e5f6g7h8",       # Unique per LLM call
    "iteration": 0,              # Loop iteration (0-indexed)
    "ts": 1708444800.123,        # Epoch seconds
    "verb": "Verify",            # Verb class name
    "phase": "loop",             # "loop", "direct", or "finalize"
    "request_id": "req-123",     # User-provided correlation ID

    # Observation (what the LLM returned)
    "has_content": true,
    "has_tool_calls": false,
    "tool_call_count": 0,
    "tool_names_used": [],
    "input_tokens": 150,
    "output_tokens": 50,
    "finish_reason": "end_turn",
    "content_preview": "The code is safe...",

    # Decision (what SAIA did)
    "action": "complete",
    "reason": "terminal_content",
    "nudge_preview": null,
}
```

### Correlation IDs

Tag requests for tracing across systems:

```python
# Set request ID for correlation
result = await saia.with_request_id("order-12345").verify(code, predicate)

# The request_id appears in all trace records for this call
```

### Analyzing Traces

```python
import pandas as pd

# Load traces
df = pd.read_json("/var/log/saia/traces.jsonl", lines=True)

# Token usage by verb
df.groupby("verb")[["input_tokens", "output_tokens"]].sum()

# Average iterations per verb
df.groupby("verb")["iteration"].max().mean()

# Find slow calls
df[df["output_tokens"] > 1000]
```

## Resource Management

SAIA doesn't manage backend resources. Your code owns the lifecycle:

```python
# Recommended: context manager
async with MyBackend() as backend:
    saia = SAIA.builder().backend(backend).build()
    # Use saia...
# Backend closed automatically

# Alternative: explicit cleanup
backend = MyBackend()
try:
    saia = SAIA.builder().backend(backend).build()
    # Use saia...
finally:
    await backend.close()
```

### Graceful Shutdown

Handle shutdown signals properly:

```python
import signal
import asyncio

shutdown_event = asyncio.Event()

def handle_signal(signum, frame):
    shutdown_event.set()

async def main():
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    async with MyBackend() as backend:
        saia = SAIA.builder().backend(backend).build()

        while not shutdown_event.is_set():
            # Process work...
            await process_task(saia)

        # Cleanup happens via context manager
```

## Timeouts and Limits

Protect against runaway loops:

```python
saia = (
    SAIA.builder()
    .backend(backend)
    .max_iterations(20)          # Stop after N tool loops
    .timeout_secs(120)           # Stop after N seconds
    .max_total_tokens(50000)     # Stop after N total tokens
    .max_call_tokens(4096)       # Limit per-call output
    .temperature(0.7)            # Sampling temperature (default: backend decides)
    .build()
)

# Override per-call
result = await saia.with_max_iterations(5).with_timeout(30).complete(task)
```

### Single-Call Mode

For simple operations without tool loops:

```python
# No iteration, no timeout - just one LLM call
result = await saia.with_single_call().verify(code, predicate)
```

## Logging

SAIA accepts any logger implementing the `Logger` protocol:

```python
class Logger(Protocol):
    def trace(self, msg: str, *, extra: dict | None = None) -> None: ...
    def debug(self, msg: str, *, extra: dict | None = None) -> None: ...
    def info(self, msg: str, *, extra: dict | None = None) -> None: ...
    def warning(self, msg: str, *, extra: dict | None = None) -> None: ...
    def error(self, msg: str, *, extra: dict | None = None) -> None: ...
```

### Integration Example

```python
import structlog

class StructlogAdapter:
    def __init__(self):
        self._log = structlog.get_logger()

    def trace(self, msg: str, *, extra: dict | None = None) -> None:
        self._log.debug(msg, **(extra or {}))

    def debug(self, msg: str, *, extra: dict | None = None) -> None:
        self._log.debug(msg, **(extra or {}))

    def info(self, msg: str, *, extra: dict | None = None) -> None:
        self._log.info(msg, **(extra or {}))

    def warning(self, msg: str, *, extra: dict | None = None) -> None:
        self._log.warning(msg, **(extra or {}))

    def error(self, msg: str, *, extra: dict | None = None) -> None:
        self._log.error(msg, **(extra or {}))

saia = SAIA.builder().backend(backend).logger(StructlogAdapter()).build()
```

## Monitoring Checklist

For production deployments:

- [ ] **Tracing enabled** - File or callback tracer configured
- [ ] **Correlation IDs** - Request IDs passed through for tracing
- [ ] **Error handling** - All SAIA exceptions caught and logged
- [ ] **Timeouts configured** - max_iterations, timeout_secs, max_total_tokens set
- [ ] **Token limits** - max_call_tokens set to prevent truncation
- [ ] **Resource cleanup** - Backend closed on shutdown
- [ ] **Metrics** - Token usage, latency, error rates tracked
- [ ] **Alerts** - Unusual patterns (high iteration count, frequent truncation)

## See Also

- [llm-infer](https://github.com/llm-works/llm-infer) - Production LLM client with SAIAAdapter
- [Backend Implementation](./backend.md) - How to implement custom backends
- [Custom Verbs](./custom-verbs.md) - Creating your own verbs
- [SECURITY.md](../SECURITY.md) - Security considerations

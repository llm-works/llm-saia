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
# File-based tracing (at build time)
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

# Per-call tracer override (fluent API)
from llm_saia.core.trace import FileTracer

tracer = FileTracer("/tmp/debug.jsonl")
try:
    result = await saia.with_tracer(tracer).complete(task)
finally:
    tracer.close()
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

## Guards

### Output Guards

Validate the final result and retry if validation fails. See the
[README](../README.md#output-guards) for basic usage.

```python
from llm_saia import OutputGuard

guard = OutputGuard(
    validator=lambda text: None if len(text) < 500 else "Too long",
    retry_instruction="Keep your response under 500 characters.",
)
result = await saia.with_guard(guard).ask(doc, question)
```

Guards run after completion. If validation fails, the request is retried with the guard's
`retry_instruction` appended to the prompt. Multiple guards are applied in order; if a retry
produces a different result, all guards are re-validated from the beginning (capped at 10 rounds).

### Iteration Guards

Enforce behavioral constraints *during* tool-calling loops. Unlike output guards, iteration guards
run after each LLM response — not at the end.

```python
from llm_saia import IterationContext, IterationGuard

def require_narrative(ctx: IterationContext) -> str | None:
    """Require the LLM to explain its actions, not just call tools silently."""
    if ctx.response.tool_calls and not (ctx.response.content or "").strip():
        return "Explain what you're doing and why."
    return None

result = await (
    saia
    .with_guard(IterationGuard(require_narrative, name="narrative"))
    .complete(task)
)
```

`IterationContext` provides:
- `response` — the current `ChatResponse`
- `iteration` — current iteration number (0-indexed)
- `max_iterations` — maximum iterations configured
- `remaining` — iterations remaining (including current)

Use iteration info for adaptive guards:

```python
from llm_saia import UNLIMITED

def force_terminal(ctx: IterationContext) -> str | None:
    """Force terminal tool call when iterations are running low."""
    # Skip if unlimited iterations
    if ctx.remaining == UNLIMITED:
        return None
    # Check if response already calls the terminal tool
    has_terminal = any(
        tc.name == "report_findings" for tc in (ctx.response.tool_calls or [])
    )
    if ctx.remaining <= 3 and not has_terminal:
        return "You must call report_findings now to complete the task."
    return None
```

When a guard fires:
1. Pending tool calls are acknowledged (keeping the conversation valid)
2. The feedback string is injected as a user message
3. The loop continues — no retry, just a course correction

#### Blocking vs Advisory Guards

By default, guards **block** tool execution when they fire — tools are acknowledged but not run.
This is correct for guards that reject invalid tool calls:

```python
# Blocking (default): reject terminal tool with bad status
guard = IterationGuard(validator=check_status, blocking=True)
```

For guards that want to shape behavior without blocking progress, use `blocking=False`. Tools
execute first, then feedback is injected:

```python
# Advisory: execute tools, then ask for explanation
def require_narrative(ctx: IterationContext) -> str | None:
    if ctx.response.tool_calls and not (ctx.response.content or "").strip():
        return "Explain what you're doing and why."
    return None

guard = IterationGuard(require_narrative, name="narrative", blocking=False)
```

With `blocking=False`, the LLM gets tool results AND the feedback in the same iteration,
preventing deadlocks where it can't explain without seeing results.

Guard outcomes are recorded in trace steps (`Step.guards`) for observability. If a validator
raises an exception, the error message becomes the feedback string.

`with_guard()` and `with_guards()` accept both guard types and route them automatically.

## Terminal Tools

A terminal tool signals task completion. When the model calls the terminal tool, SAIA extracts
the result and stops the loop.

```python
saia = (
    SAIA.builder()
    .backend(backend)
    .tools(tools, executor)
    .terminal_tool("report_findings")
    .build()
)

result = await saia.complete(task)
print(result.terminal_data)  # Arguments from the terminal tool call
```

### Confirmation Behavior

By default, SAIA asks the model to call the terminal tool twice to confirm completion. This
prevents accidental termination but can cause issues: many models respond to "call again to
confirm" with text like "Confirming..." instead of a second tool call.

When confirmation fails, `terminal_data` is `None` even though the model called the terminal
tool with valid data. The data is still accessible via `result.history`, but this is
inconvenient.

**Recommendation:** Disable confirmation unless you specifically need it:

```python
# Recommended: complete immediately on first terminal call
saia = (
    SAIA.builder()
    .backend(backend)
    .tools(tools, executor)
    .terminal_tool("report_findings", require_confirmation=False)
    .build()
)

# Legacy: require confirmation (default)
saia = (
    SAIA.builder()
    .backend(backend)
    .tools(tools, executor)
    .terminal_tool("report_findings")  # require_confirmation=True by default
    .build()
)
```

### Terminal Tool Configuration

For advanced control, use `.terminal()` instead of `.terminal_tool()`:

```python
saia = (
    SAIA.builder()
    .backend(backend)
    .tools(tools, executor)
    .terminal(
        tool="complete_task",
        output_field="summary",        # Field to use for result.output
        status_field="status",         # Field containing completion status
        failure_values=("stuck", "failed"),  # Status values that indicate failure
        require_confirmation=False,
    )
    .build()
)
```

### Terminal Iteration Guards

SAIA provides built-in iteration guards for common terminal tool behaviors. These are opt-in
and run during each iteration of the tool loop.

**Reject Failure Status:**

```python
from llm_saia.guards import terminal_status

# Reject terminal calls with "stuck" or "failed" status, ask to retry
guard = terminal_status(
    tool="complete_task",
    status_field="status",
    failure_values=("stuck", "failed"),
    max_retries=3,
    escalate=True,  # Increasingly forceful retry messages
)

result = await saia.with_guard(guard).complete(task)
```

**Validate Schema:**

```python
from llm_saia.guards import terminal_schema

# Validate terminal tool arguments against JSON schema
guard = terminal_schema(tools, "report_findings", max_retries=2)

result = await saia.with_guard(guard).complete(task)
```

**Detect Contradictions:**

```python
from llm_saia.guards import contradiction

# Detect hedging language ("however", "unfortunately", "I can't")
# when the terminal tool is called
guard = contradiction("complete_task", max_retries=2)

result = await saia.with_guard(guard).complete(task)
```

### Extracting Data from History

If `terminal_data` is `None` (due to confirmation failure), you can extract the terminal tool
call from `result.history`:

```python
result = await saia.complete(task)

if result.terminal_data is None:
    # Fallback: scan history for terminal tool call
    for msg in reversed(result.history):
        if msg.tool_calls:
            for call in msg.tool_calls:
                if call.name == "report_findings":
                    findings = call.arguments
                    break
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

### Trace-Level Observability

Enable trace-level logging to see detailed execution flow. This is invaluable for debugging stuck
loops, understanding why the LLM repeated an action, or verifying that guards fired correctly.

> **Privacy Warning:** Trace logs may contain sensitive information including user messages,
> tool results, and LLM responses. Apply appropriate redaction, retention policies, and access
> controls before enabling trace-level logging in production environments.

**What trace logs show:**

| Log Message | What It Contains |
|-------------|------------------|
| `"verb started"` | Verb name, trace_id |
| `"verb completed"` | Duration, step count |
| `"sending messages to llm"` | Message count by role, last user message preview, recent tool results |
| `"tool result returned to llm"` | Tool name, result length, result (up to 50k chars) |
| `"running iteration guards"` | List of guard names being checked |
| `"iteration guards triggered feedback"` | Which guards fired, feedback (up to 50k chars) |
| `"guard feedback injected into conversation"` | Feedback content (up to 50k chars), acknowledged tool names |
| `"controller decision details"` | Action, reason, terminal_tool, terminal_data |
| `"checking guard"` / `"guard passed"` | Output guard validation progress |

**Debugging stuck loops:**

When a loop gets stuck (LLM repeats the same action), trace logs answer:

1. **Did the LLM receive the tool result?** Check `"tool result returned to llm"` — shows the
   exact string (e.g., "BLOCKED" or "WRAP UP") that was sent back.

2. **Did guards fire and inject feedback?** Check `"guard feedback injected into conversation"` —
   shows the feedback message and which tool calls were acknowledged.

3. **What context did it see before repeating?** Check `"sending messages to llm"` — shows
   message counts and the last user message (often a guard nudge or tool result).

**Example trace session:**

```
TRACE sending messages to llm {"call_id": "abc123", "msg_count": 8, "by_role": {"user": 3, "assistant": 3, "tool": 2}, "last_user_msg": "You reported status='stuck' but the task is not complete..."}
TRACE tool result returned to llm {"tool": "read_file", "result_len": 2500, "result": "def process_data(x):..."}
TRACE iteration guards triggered feedback {"guards_fired": ["terminal_status"], "feedback": "You reported status='stuck' but the task is not complete. Try a different approach."}
TRACE guard feedback injected into conversation {"feedback_len": 85, "feedback": "You reported status='stuck' but the task is not complete. Try a different approach.", "acked_tools": ["complete_task"]}
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

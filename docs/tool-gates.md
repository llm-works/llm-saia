# Tool Gates

Tool gates dynamically control which tools are visible to the model on each iteration of a
tool-calling loop. When a gate blocks a tool, SAIA strips it from the outbound schema — the model
never sees it and cannot spend output tokens generating a call to it.

## Motivation

In tool-calling loops, models often generate expensive calls that will be rejected. A common
pattern: a `min_iterations` guard forces 3 research iterations before the model can self-terminate,
but the model still generates the terminal payload every iteration — then an `IterationGuard`
rejects it and injects feedback. The terminal payload is often large (structured output with
reasoning), so those rejected generations waste significant tokens.

Tool gates solve this by hiding tools the model cannot use yet. If the terminal tool is invisible
on iterations 0–2, the model cannot generate a call to it and the wasted tokens disappear entirely.

Measured on `gemini-3.1-pro-preview` with a 3-iteration minimum: **42.7% of output tokens** were
wasted on rejected terminal payloads (avg 2,168 tokens each). Schema hiding avoids that entirely.

## API

### Gate callback

A gate is a callable with signature:

```python
ToolGate = Callable[[ToolGateContext], bool | str | None]
```

Return values:
- `True` or `None` — allow (tool remains in the schema)
- `False` — block silently
- `str` — block with a reason recorded in trace/log

Reasons are diagnostic for the operator. They are never surfaced to the model — the point of
schema-hiding is to prevent the model from thinking about a tool we've decided to withhold.

### ToolGateContext

```python
@dataclass(frozen=True)
class ToolGateContext:
    iteration: int  # Current 0-indexed loop iteration
    tool_call_counts: Mapping[str, int]  # Cumulative tool calls by name
    last_response: ChatResponse | None  # Previous response, None on first iteration
    messages: Sequence[Message]  # Outbound transcript (read-only)
    extra: Any = None  # Factory-computed shared state (see below)
```

## Registration

### Builder pattern

```python
saia = (
    SAIA.builder()
    .backend(backend)
    .tools(tools, executor)
    .tool_gate("search", lambda ctx: ctx.iteration < 10)  # single gate
    .tool_gates({"fetch": gate_fn, "summarize": other_gate})  # multiple gates
    .tool_gate_context_factory(compute_shared_state)  # optional
    .build()
)
```

### Runtime modifiers

```python
# Per-call gates
result = await saia.with_tool_gate(
    "expensive_tool", lambda ctx: ctx.tool_call_counts.get("cheap_tool", 0) >= 2
).complete(task)
```

### TerminalConfig shortcut

The `min_iterations` parameter on `TerminalConfig` is a shortcut for the common "force N research
iterations before self-termination" pattern:

```python
saia = (
    SAIA.builder()
    .backend(backend)
    .tools(tools, executor)
    .terminal("complete_task", min_iterations=3)
    .build()
)
```

This is equivalent to registering a gate that blocks the terminal tool until iteration >= 3.
An explicit gate for the same tool wins over the shortcut.

## Context Factory

When multiple gates need the same derived signal (e.g. a transcript walk extracting a specific
tool's arguments), computing it in each gate is wasteful. The `ToolGateContextFactory` runs once
per LLM call; its return value is attached as `ctx.extra` for every gate on the same iteration.

```python
ToolGateContextFactory = Callable[[ToolGateContext], Any]
```

Example — count how many searches returned no results:

```python
def compute_empty_searches(ctx: ToolGateContext) -> int:
    """Walk transcript once; result shared by all gates."""
    count = 0
    for msg in ctx.messages:
        if msg.role == "tool" and msg.tool_call_id:
            # Check if this was a search that returned empty
            if "no results" in (msg.content or "").lower():
                count += 1
    return count


def gate_expensive_tool(ctx: ToolGateContext) -> bool | str:
    """Only allow expensive tool after confirming searches found something."""
    if ctx.extra >= 2:  # ctx.extra is the count from compute_empty_searches
        return "too many empty searches"
    return True


saia = (
    SAIA.builder()
    .backend(backend)
    .tools(tools, executor)
    .tool_gate_context_factory(compute_empty_searches)
    .tool_gate("expensive_tool", gate_expensive_tool)
    .build()
)
```

## Common Patterns

### Staged access

Reveal tools progressively as the task advances:

```python
def staged_gate(after_iteration: int) -> ToolGate:
    def gate(ctx: ToolGateContext) -> bool | str:
        if ctx.iteration < after_iteration:
            return f"requires iteration >= {after_iteration}"
        return True

    return gate


# Register same gate for multiple tools
gates = {tool: staged_gate(2) for tool in ["summarize", "conclude"]}
saia = SAIA.builder().tool_gates(gates).build()
```

### Rate limiting

Prevent a tool from being called more than N times:

```python
def max_calls(tool: str, limit: int) -> ToolGate:
    def gate(ctx: ToolGateContext) -> bool | str:
        count = ctx.tool_call_counts.get(tool, 0)
        if count >= limit:
            return f"{tool} limit reached ({limit})"
        return True

    return gate


saia = SAIA.builder().tool_gate("search", max_calls("search", 5)).build()
```

### Conditional on previous tool

Only allow tool B after tool A has been called:

```python
def requires(prerequisite: str) -> ToolGate:
    def gate(ctx: ToolGateContext) -> bool | str:
        if ctx.tool_call_counts.get(prerequisite, 0) == 0:
            return f"requires {prerequisite} first"
        return True

    return gate


saia = SAIA.builder().tool_gate("submit", requires("validate")).build()
```

## Comparison: Gates vs Iteration Guards

| Aspect | Tool Gates | Iteration Guards |
|--------|------------|------------------|
| **When** | Before LLM call (schema filtering) | After LLM response (validation) |
| **Effect** | Tool invisible to model | Feedback injected, loop continues |
| **Token cost** | Zero (tool never seen) | Model generates rejected call |
| **Use case** | Prevent generation | Enforce behavior with feedback |

Use gates when you know ahead of time a tool shouldn't be used. Use guards when you need to
inspect the actual call the model made.

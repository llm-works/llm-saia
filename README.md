# SAIA

**Framework-agnostic verb vocabulary for LLM agents**

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen.svg)
[![Coverage](https://codecov.io/gh/llm-works/llm-saia/branch/main/graph/badge.svg)](https://codecov.io/gh/llm-works/llm-saia)
[![Typed](https://img.shields.io/badge/typed-PEP%20561-brightgreen.svg)](https://peps.python.org/pep-0561/)
[![Linting: Ruff](https://img.shields.io/badge/linting-ruff-brightgreen)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/llm-works/llm-saia/actions/workflows/ci.yml/badge.svg)](https://github.com/llm-works/llm-saia/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

SAIA provides a fixed vocabulary of semantic verbs for LLM interactions. Instead of writing raw
prompts, you express intent through verbs like `ask`, `verify`, `critique`, and `refine`.

**Why a fixed vocabulary?** The same insight that made SCUMM work for adventure games in 1987:
constraints enable tooling. A finite set of operations means every interaction is debuggable,
testable, and composable.

## Why SAIA?

**"Can't I just type this into Claude?"** For a one-off question, yes. SAIA is for when you're
building software, not chatting:

- **Structured outputs** - `verify()` returns `VerifyResult(passed: bool, reason: str)`, not text
  you parse
- **Composable** - chain `verify → critique → refine` in 3 lines of code
- **Testable** - mock the backend, unit test your verbs
- **Traceable** - every verb call logged with inputs/outputs for debugging production issues
- **Backend-agnostic** - same code works with Anthropic, OpenAI, or local models
- **Zero dependencies** - pure Python core; bring your own LLM client

**"Why not raw tool calling?"** You could write the iteration loop yourself (~50-100 lines). SAIA
gives you `complete()` with terminal detection, tracing, timeouts, and max iterations built in.
It's `requests` vs `urllib` - both work, one is cleaner.

SAIA's value compounds when combining verbs, switching backends, or building a team around
consistent patterns.

**"Is this novel?"** Perhaps not. These are the patterns that emerge when you build LLM agents that
need to actually work. SAIA extracts them into ~2500 lines anyone can use, inspect, and build on.

## Installation

```bash
pip install llm-saia
```

## Quick Start

```python
from llm_saia import SAIA

async def main():
    saia = (
        SAIA.builder()
        .backend(your_backend)
        .build()
    )

    # Verify a claim
    result = await saia.verify(
        "This code handles null input safely",
        "no null pointer exceptions possible"
    )
    print(f"Passed: {result.passed}, Reason: {result.reason}")

    # Generate counter-arguments
    critique = await saia.critique("Python is slow for all tasks")
    print(f"Counter: {critique.counter_argument}")

    # Break down a complex task
    subtasks = await saia.decompose("Build a REST API with authentication")
    for task in subtasks:
        print(f"- {task}")
```

## Verb Reference

| Verb | Purpose | Returns |
|------|---------|---------|
| `ask` | Query an artifact with a question | `str` |
| `verify` | Check if artifact satisfies predicate | `VerifyResult(passed, reason)` |
| `critique` | Generate strongest counter-argument | `Critique(counter_argument, weaknesses, strength)` |
| `refine` | Improve artifact based on feedback | `str` |
| `synthesize` | Combine multiple artifacts into one | `T` (structured) |
| `decompose` | Break complex task into subtasks | `list[str]` |
| `extract` | Pull structured data from text | `T` (structured) |
| `classify` | Categorize into predefined classes | `ClassifyResult(label, confidence)` |
| `choose` | Select best option from choices | `ChooseResult(choice, reasoning)` |
| `constrain` | Parse into structured schema | `T` (structured) |
| `ground` | Anchor claims to source evidence | `list[Evidence]` |
| `instruct` | Execute open-ended instructions | `str` |

### Memory Verbs

| Verb | Purpose |
|------|---------|
| `store` | Save value to memory |
| `recall` | Retrieve from memory by query |

## Configuration

### Builder Pattern

```python
saia = (
    SAIA.builder()
    .backend(backend)                    # Required: LLM backend
    .tools(tool_defs, executor)          # Optional: tool calling
    .logger(logger)                      # Optional: logging
    .system("You are a helpful assistant")  # Optional: system prompt
    .max_iterations(10)                  # Optional: tool loop limit
    .max_call_tokens(4096)               # Optional: per-call token limit
    .build()
)
```

### Runtime Modifiers

```python
# Single LLM call (no tool loop)
result = await saia.with_single_call().verify(code, "compiles")

# Custom iteration limit
result = await saia.with_max_iterations(5).instruct(task)

# Timeout
result = await saia.with_timeout(30).decompose(problem)

# Correlation ID for tracing
result = await saia.with_request_id("req-123").ask(doc, question)
```

## Examples

See the [examples/](examples/) directory:

- `investigate.py` - Investigate a claim (verify → critique → refine)
- `build.py` - Build an app (decompose → instruct → synthesize)
- `build_multi.py` - Two LLMs collaborate (local generates, smart verifies)

## Design Philosophy

1. **Verbs express intent** - not implementation details
2. **Structured over strings** - type-safe dataclass responses
3. **Composable primitives** - build complex flows from simple verbs
4. **Backend-agnostic** - same code works with any LLM
5. **Zero dependencies** - pure Python core, you control your LLM client
6. **Debuggable** - every operation is traceable

## Research Directions

SAIA's structured verb outputs create opportunities beyond inference:

- **Consistency tuning** - traces capture (prompt, decision) pairs that can fine-tune models for
  stable verb behavior. Same semantic question → same semantic answer.
- **Structured generation** - backends can use grammar-constrained decoding (xgrammar, outlines) to
  guarantee valid outputs from verbs like `extract`, `verify`, and `classify` without retry loops.

## License

Apache 2.0 - see [LICENSE](LICENSE)

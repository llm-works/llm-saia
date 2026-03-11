# SAIA: Typed Contracts for LLM Outputs

Prompts are suggestions. Verbs are contracts.

Instead of parsing free text from "verify this claim and explain why", `verify()` returns
`{passed: bool, reason: str}`. The structure follows from the operation.

SAIA gives you 12 typed verbs for controlling LLM outputs.

![SAIA demo](./demo/demo.gif)

Or in practice:

```python
saia = SAIA.builder().backend(backend).build()

result = await saia.verify(generated_code, "no SQL injection vulnerabilities")
# result.passed = False, result.reason = "User input concatenated directly into query"

critique = await saia.critique(llm_response)
# critique.weaknesses = ["Claims unsupported by sources", "Contradicts earlier statement"]

fixed = await saia.refine(llm_response, critique.weaknesses)
```

## The Analogy

SCUMM (1987) reduced adventure games to a fixed set of verbs. Monkey Island had 9: Open, Close,
Push, Pull, Give, Pick up, Look at, Talk to, Use. Every puzzle was built from these primitives.
Designers knew exactly what players could do.

SAIA does the same for LLMs. 12 verbs, typed outputs, predictable behavior.

## The Verbs

- `verify` - check a predicate -> `{passed: bool, reason: str}`
- `critique` - find weaknesses -> `{weaknesses: list[str]}`
- `decompose` - break into subtasks -> `list[str]`
- `extract` - pull structured data -> your schema
- `synthesize` - combine inputs -> structured output or text

Plus `ask`, `refine`, `classify`, `choose`, `constrain`, `ground`, `instruct` - see
[docs](../README.md).

## Example

```python
from llm_saia import SAIA

saia = SAIA.builder().backend(anthropic_backend).build()

subtasks = await saia.decompose("Build a web scraper")
results = [await saia.instruct(t) for t in subtasks]
output = await saia.synthesize(results, goal="single working Python script")
```

## Loop Controller

Verbs run until the LLM stops calling tools. `complete()` adds a controller that detects stuck
states and manages clean termination.

```python
result = await saia.complete("Analyze the files in /src and summarize")
# [1] read_file("/src")
# [2] "Should I continue?" -> nudged
# [3] read_file("/src/main.py")
# [4] done(summary="...")
# result.iterations = 4, result.score.nudges = 1
```

It detects when the LLM repeats itself, contradicts itself, or asks permission - and nudges it
back on track. Termination uses confirmation: "is this your final answer?"

## Design

- Pure Python, no runtime dependencies - bring your own LLM client
- Backend-agnostic: Anthropic, OpenAI, Ollama, vLLM
- ~2500 lines total

## vs. Other Tools

Some related tools in this space:

Instructor validates structured output from a single call. SAIA runs multi-step loops with state
detection.

LangChain provides an end-to-end framework. SAIA is just the semantic layer.

## Background

Part of a larger project, currently being developed:
- llm-gent  - agent orchestration
- llm-saia  - semantic actions (this library)
- llm-kelt  - knowledge, embedding, learning, training
- llm-infer - inference backends
- appinfra  - logging, configs, databases, CLIs

## Links

- [llm-saia](https://github.com/llm-works/llm-saia) - this repo
- [serendip-ml](https://github.com/serendip-ml) - full agent stack

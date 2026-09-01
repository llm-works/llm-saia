# SAIA Examples

Runnable examples demonstrating the SAIA verb vocabulary. All ship in the
installed wheel — invoke via:

```bash
python -m llm_saia.examples.<name>
```

Or run the source file directly (every `.py` here is executable):

```bash
./llm_saia/examples/<name>.py
```

The root-level `examples/` in the repo is a symlink to this directory, so
`examples/foo.py` and `llm_saia/examples/foo.py` refer to the same file.

## No backend required

These use an in-process demo backend that synthesizes JSON matching the
requested schema. No API keys, no network. Exercised by CI.

| Example | What it shows |
|---------|---------------|
| `quickstart.py` | The README quick-start: `verify`, `critique`, `decompose` end-to-end. |
| `pydantic_scorer.py` | `complete_structured` with a `pydantic.BaseModel` — `Field(ge=, le=, max_length=, Literal[...])`. Requires `pip install llm-saia[pydantic]`. |
| `compose_example.py` | The `compose()` prompt-building utility. No LLM at all — pure string mechanics. |

## Real LLM required

These call an actual model via the OpenAI-compatible or Anthropic backend
from `__init__.py`. Configure via env vars (see next section).

| Example | Verbs used | What it shows |
|---------|------------|---------------|
| `investigate.py` | `verify` → `critique` → `refine` | Fact-check-and-improve a claim. `./investigate.py "your claim"` or default. |
| `build.py` | `decompose` → `instruct` → `ask` | Build an artifact from a natural-language task. |
| `build_multi.py` | `decompose` → `instruct` → `verify` → `critique` → `refine` → `ask` | Two-model orchestration: cheap local generates, smart model verifies. |
| `scraper.py` | `decompose` → `instruct` → `synthesize` | Build a web scraper by decomposition then synthesis. |
| `agent.py` | `complete()` with tools | Tool-calling agent loop. Needs a model with robust tool support (claude-haiku+, gpt-4o-mini+, or 14B+ local). |
| `analyze.py` | `complete()` with tools | Agent that reads and analyzes its own source code. Tracing on every LLM call. |

## Configuring the backend

The real-LLM examples share `get_backend()` in `__init__.py`, driven by
these environment variables:

| Var | Default | Meaning |
|-----|---------|---------|
| `LLM_BACKEND` | `openai` | `openai` for any OpenAI-compatible API (vLLM, llama.cpp, ollama, real OpenAI). `anthropic` for Claude via the Messages API. |
| `LLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible base URL. Ignored when `LLM_BACKEND=anthropic`. |
| `LLM_MODEL` | `gpt-4o-mini` | Model id passed to the backend. |
| `OPENAI_API_KEY` | — | Required for real OpenAI; a dummy string is fine for most local backends. |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_BACKEND=anthropic`. |

For a local vLLM at `localhost:18300` serving `qwen3.5-27b-gptq-int4`:

```bash
export LLM_BASE_URL=http://localhost:18300/v1
export LLM_MODEL=qwen3.5-27b-gptq-int4
export OPENAI_API_KEY=local-dummy
python -m llm_saia.examples.investigate "Water boils at 100 degrees Celsius"
```

For Anthropic:

```bash
export LLM_BACKEND=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python -m llm_saia.examples.build
```

## Note on the demo backend

`quickstart.py` and `pydantic_scorer.py` each embed their own `DemoBackend`
that inspects the JSON schema and returns a canned payload satisfying its
constraints. These prove the SAIA pipeline is wired end-to-end — imports,
schema conversion, parsing — without spending tokens on a real model.
They are what the `smoke-wheel` CI job runs against the installed wheel
on every push.

# Pydantic Schemas

SAIA's structured-output primitives — every built-in verb and the public
`SAIA.complete_structured(prompt, schema)` — accept either a stdlib
`@dataclass` or a `pydantic.BaseModel` subclass. Pydantic is an optional
extra. Zero-dep users pay nothing.

`@dataclass` is fine for simple typed shapes. `BaseModel` unlocks the full
JSON-Schema vocabulary — `Field(ge=..., le=..., pattern=...,
max_length=..., discriminator=..., ...)` — plus Python-side validators
that trigger SAIA's schema-retry loop when the model produces something
the decoder didn't catch.

## Installation

```bash
pip install llm-saia[pydantic]
```

The extra pulls in `pydantic>=2.0`. Base `pip install llm-saia` remains
zero-dep; no top-level import of `pydantic` happens unless a caller
actually hands a `BaseModel` to `complete_structured`.

## Basic usage

```python
from pydantic import BaseModel, Field
from typing import Literal
from llm_saia import SAIA


class EssayScore(BaseModel):
    """LLM-judge score for a short piece of writing."""

    score: float = Field(ge=0.0, le=10.0, description="0-10 overall grade")
    confidence: float = Field(ge=0.0, le=1.0)
    verdict: Literal["pass", "fail", "borderline"]
    feedback: str = Field(max_length=400, description="1-2 sentences")


saia = SAIA.builder().backend(backend).build()

result = await saia.complete_structured(
    f"Grade this essay for clarity and evidence:\n\n{essay}",
    EssayScore,
)

print(result.value.score, result.value.verdict)
```

`result.value` is a fully-validated `EssayScore` instance. `result.trace`
holds the per-step execution trace, same as every other verb.

## What's structurally enforced vs. advisory

SAIA forwards the entire JSON schema produced by `model_json_schema()` to
the backend verbatim. Which keywords the model *literally cannot* violate
depends on the backend's constrained decoder:

| `Field(...)` keyword | JSON Schema | Enforced by vLLM / OpenAI strict | Notes |
|----------------------|-------------|----------------------------------|-------|
| `ge=`, `le=`, `gt=`, `lt=` | `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` | Yes | Numeric range. Rejects out-of-range tokens at generation. |
| `min_length=`, `max_length=` | `minLength`, `maxLength` | Yes (most backends) | Length bounds on strings. |
| `pattern=` | `pattern` | Yes on vLLM (regex-constrained decoding); OpenAI strict mode enforces it too | Regex constraint on string content. |
| `Literal[...]`, `Enum` | `enum` | Yes | The classic tool for closed sets. |
| Discriminated `Union` with `Field(discriminator="type")` | `oneOf` + `discriminator` | Partial — depends on backend version | See below. |
| `multiple_of=` | `multipleOf` | Rarely | Emitted in the schema; most decoders skip it. |
| `format=` (email, uri, date, uuid, …) | `format` | Advisory | Emitted; not decoder-enforced. |
| `description=` | `description` | Advisory | Model reads it in context; compliance varies. |
| `examples=` | `examples` | Advisory | Emitted; not decoder-enforced. |

Everything the decoder doesn't catch falls to Pydantic's Python-side
validation on parse — see the next section.

## Validators and the schema-retry loop

Pydantic's `@field_validator` and `@model_validator` run after
`model_validate` receives the JSON. If they raise, Pydantic wraps the
error in `ValidationError`, which inherits from `ValueError`. SAIA's
existing structured-output handler catches `(TypeError, ValueError)`,
wraps it in `StructuredOutputError`, and hands it to the iteration
guard's `parse_max_retries` budget — same path as any dataclass parse
failure.

```python
from pydantic import BaseModel, field_validator


class ClaimReview(BaseModel):
    verdict: str
    citations: list[str]

    @field_validator("citations")
    @classmethod
    def at_least_two_citations(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("need at least two citations")
        return v


# Configure a retry budget on the verb call:
result = await saia.with_guard(schema_retry(max_retries=2)).complete_structured(prompt, ClaimReview)
```

If the first response has one citation, the validator raises, the retry
guard consumes budget, and SAIA reprompts with the validation error in
the user turn. Pydantic's default error messages are LLM-friendly
("List should have at least 2 items after validation, not 1"), which
helps the model self-correct.

## Discriminated unions

For payloads that can be one of several shapes, Pydantic's discriminated
unions produce a `oneOf` with a `discriminator` block. Useful for tool
routing, event streams, and typed agent commands.

```python
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field


class SearchAction(BaseModel):
    kind: Literal["search"]
    query: str


class FetchAction(BaseModel):
    kind: Literal["fetch"]
    url: str


class DoneAction(BaseModel):
    kind: Literal["done"]
    reason: str


class Turn(BaseModel):
    action: Annotated[
        Union[SearchAction, FetchAction, DoneAction],
        Field(discriminator="kind"),
    ]


result = await saia.complete_structured(prompt, Turn)
match result.value.action:
    case SearchAction(query=q):
        ...
    case FetchAction(url=u):
        ...
    case DoneAction(reason=r):
        ...
```

Backend support for discriminated unions varies. Modern vLLM with
`xgrammar` handles them; older backends fall back to a plain `oneOf` and
rely on the model matching the `kind` literal. Either way, `Turn(...)`
construction on the parse side is fully validated.

## Nested models

Pydantic emits nested models as `$ref` + `$defs`. SAIA forwards the
envelope untouched; backends resolve refs internally. Cycles are handled
by Pydantic (unlike `dataclass_to_json_schema`, which rejects
self-referential dataclasses).

```python
class Address(BaseModel):
    street: str
    city: str


class Person(BaseModel):
    name: str
    address: Address
    manager: "Person | None" = None  # cycles OK with pydantic
```

## Migrating from `@dataclass`

Most `@dataclass`-based schemas port to `BaseModel` mechanically:

```python
# Before
from dataclasses import dataclass


@dataclass
class Score:
    value: float
    label: str


# After
from pydantic import BaseModel, Field


class Score(BaseModel):
    value: float = Field(ge=0.0, le=1.0)  # now enforced structurally
    label: str
```

The dispatch in `complete_structured` is by duck-type detection on the
schema type — no caller changes needed. Nothing about SAIA's public
surface (verb signatures, `VerbResult`, tracing) changes.

Built-in verb result types (`Critique`, `Evidence`, `ClassifyResult`,
`TaskResult`, …) remain stdlib `@dataclass`. Users who want structural
constraint enforcement on those specific verbs are free to define their
own Pydantic mirrors and route via `complete_structured` — the built-in
verbs are recipes over the same primitive.

## Backend enforcement matrix

The structural-enforcement column above collapses several backend
capabilities into "yes / partial / advisory." For anything beyond
`ge`/`le`/`enum`/basic length, verify against your actual backend
before relying on the constraint:

- **vLLM (`outlines` / `xgrammar`)** — modern versions honor most of
  JSON-Schema Draft-2020-12. `xgrammar` is the more complete of the two;
  check the vLLM version's grammar backend.
- **OpenAI `response_format: {type: json_schema, strict: true}`** —
  enforces a subset of JSON Schema; `pattern`, `minLength`, `maxLength`
  work; some advanced constructs (`$dynamicRef`, complex `oneOf`) are
  rejected as unsupported at request time (400).
- **Anthropic (via tool-use)** — no server-side constrained decoding;
  the schema is advisory only. Pydantic's parse-side validation is what
  catches violations; the schema-retry loop handles them.

If a backend rejects your schema, the failure surfaces as a
`BackendError` on the first call — visible immediately.

## See also

- `examples/pydantic_scorer.py` — runnable smoke, exercised by the
  `smoke-wheel` CI job with the `[pydantic]` extra installed.
- `README.md` § Custom typed output (Pydantic) — one-page overview.
- `SAIA.complete_structured` docstring — API reference.

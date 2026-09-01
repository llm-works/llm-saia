#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Pydantic scorer smoke — proves llm-saia[pydantic] wires end-to-end.

Runs against an installed wheel via
``python -m llm_saia.examples.pydantic_scorer``. Exercised by the smoke-wheel
CI job with the ``[pydantic]`` extra installed to catch regressions in the
BaseModel-schema and model_validate dispatch paths.

Uses an in-process demo backend so the run needs no API keys and no network.
The point is to prove Field(ge=..., le=..., description=...) reaches the
JSON schema envelope and round-trips through complete_structured — not to
validate LLM output. For a real-LLM run, swap DemoBackend for the
OpenAI-compatible backend from ``examples/__init__.py``.

Requires ``pip install llm-saia[pydantic]``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llm_saia import SAIA, Backend, ChatResponse, Message  # noqa: E402

try:
    from pydantic import BaseModel, Field
except ImportError:
    print(
        "This example needs pydantic. Install it with:\n  pip install llm-saia[pydantic]",
        file=sys.stderr,
    )
    raise SystemExit(2) from None


class EssayScore(BaseModel):
    """LLM-judge score for a short piece of writing."""

    score: float = Field(ge=0.0, le=10.0, description="0.0 to 10.0 — overall grade")
    confidence: float = Field(ge=0.0, le=1.0, description="0.0 to 1.0 — judge confidence")
    verdict: Literal["pass", "fail", "borderline"]
    feedback: str = Field(max_length=400, description="1-2 sentences, no lists")


class DemoBackend(Backend):
    """Backend that synthesizes a JSON payload matching the requested schema."""

    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        context: dict[str, Any] | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> ChatResponse:
        payload = _payload_for(response_schema)
        return ChatResponse(content=json.dumps(payload), tool_calls=[])


def _payload_for(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Build a minimal object satisfying the pydantic-produced JSON schema."""
    inner = (schema or {}).get("schema") or (schema or {})
    props = inner.get("properties", {})
    return {name: _value_for(spec) for name, spec in props.items()}


def _value_for(spec: dict[str, Any]) -> Any:
    """Pick a value that satisfies a single property spec (honors ge/le/enum)."""
    if "enum" in spec:
        return spec["enum"][0]
    t = spec.get("type")
    if t == "boolean":
        return True
    if t in ("number", "integer"):
        low = spec.get("minimum", 0)
        high = spec.get("maximum", low)
        return low if low <= high else 0
    if t == "array":
        return [_value_for(spec.get("items") or {})]
    return "smoke"


async def main() -> int:
    saia = SAIA.builder().backend(DemoBackend()).build()

    prompt = "Grade this essay for clarity and evidence:\n\nDemo essay body."
    result = await saia.complete_structured(prompt, EssayScore)

    print(f"score:      {result.value.score}")
    print(f"confidence: {result.value.confidence}")
    print(f"verdict:    {result.value.verdict!r}")
    print(f"feedback:   {result.value.feedback!r}")

    assert isinstance(result.value, EssayScore)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

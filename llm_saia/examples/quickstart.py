#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Minimal quick-start smoke that mirrors the README example.

Runs against an installed wheel via `python -m llm_saia.examples.quickstart`.
Exercised by the smoke-wheel CI job from a working directory outside the
source tree to catch README-vs-installed API drift, broken top-level
exports, and missing wheel resources.

Uses an in-process demo backend so the run needs no API keys and no network —
the goal is to prove the surface is wired, not to validate LLM output.
Production backends live in llm-infer/client (see examples/agent.py).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from llm_saia import SAIA, Backend, ChatResponse, Message


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
    """Build a minimal object satisfying the top-level JSON schema properties.

    SAIA passes the schema in the wrapped form dataclass_to_json_schema produces:
    ``{"name", "description", "schema": {"properties": {...}}}``. Fall back to
    the bare schema dict for defensive handling.
    """
    inner = (schema or {}).get("schema") or (schema or {})
    props = inner.get("properties", {})
    out: dict[str, Any] = {}
    for name, spec in props.items():
        t = spec.get("type")
        if t == "boolean":
            out[name] = True
        elif t in ("number", "integer"):
            out[name] = 0
        elif t == "array":
            item_type = (spec.get("items") or {}).get("type", "string")
            out[name] = [_scalar_default(item_type) for _ in range(2)]
        else:
            out[name] = "smoke"
    return out


def _scalar_default(t: str) -> Any:
    if t == "boolean":
        return True
    if t in ("number", "integer"):
        return 0
    return "smoke"


async def main() -> int:
    saia = SAIA.builder().backend(DemoBackend()).build()

    verified = await saia.verify(
        "def add(a, b): return a + b",
        "the function returns the sum of its two arguments",
    )
    print(f"verify: passed={verified.value.passed} reason={verified.value.reason!r}")

    critique = await saia.critique(
        "Adding more programmers to a late project always makes it finish faster."
    )
    print(f"critique: counter={critique.value.counter_argument!r}")

    subtasks = await saia.decompose("Build a REST API with authentication")
    for i, task in enumerate(subtasks.value):
        print(f"decompose[{i}]: {task}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

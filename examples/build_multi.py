#!/usr/bin/env python3
"""Two LLMs collaborate: local generates, smart verifies.

This example demonstrates SAIA's power for multi-model orchestration:

- Cost optimization: cheap local model does bulk generation, expensive model only for quality
- Full verb vocabulary: 6 verbs in one workflow (decompose → instruct → verify → critique →
  refine → synthesize)
- Feedback loop: verify fails → critique finds issues → refine improves
- Real collaboration: models with different strengths working together

The whole orchestration is ~50 lines of readable code.

Local LLM (cheap): decompose, instruct, refine
Smart LLM (quality): verify, critique, synthesize

Usage:
    # Local OpenAI + Smart Anthropic
    LOCAL_URL=http://localhost:18000/v1 SMART_BACKEND=anthropic ./build_multi.py

    # Both local
    LOCAL_URL=http://localhost:18000/v1 SMART_URL=http://localhost:18000/v1 ./build_multi.py

Example output:
    Local: http://localhost:18000/v1 | Smart: anthropic
    Task: Build a Python unit converter: km/miles, celsius/fahrenheit, kg/pounds

    [decompose] breaking down task...
    [decompose] 4 subtasks
      - Create function to convert kilometers to miles and vice versa
      - Create function to convert celsius to fahrenheit and vice versa
      - Create function to convert kilograms to pounds and vice versa
      - Test the functions with various test cases

    [instruct] generating code...
      [1/4] Create function to convert kilometers to miles and... done
      [2/4] Create function to convert celsius to fahrenheit a... done
      [3/4] Create function to convert kilograms to pounds and... done
      [4/4] Test the functions with various test cases... done

    [verify] checking code quality...
      [1/4] verifying... ✓ The artifact satisfies the predicate
      [2/4] verifying... ✗ The artifact contains valid Python syntax but...
      [critique]... 8 issues
      [refine]... improved
      [3/4] verifying... ✗ The artifact contains syntactically valid...
      [critique]... 5 issues
      [refine]... improved
      [4/4] verifying... ✓ The artifact is valid Python code

    [synthesize] combining into final script...
    [synthesize]
    ```python
    def convert_distance(value, from_unit, to_unit):
        ...
    ```
"""

import asyncio
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples import DEFAULT_ANTHROPIC_MODEL, OpenAIBackend
from examples import Colors as C
from llm_saia import SAIA
from llm_saia.core.backend import Backend

TASK = "Build a Python unit converter: km/miles, celsius/fahrenheit, kg/pounds"


@asynccontextmanager
async def get_smart_backend() -> AsyncGenerator[Backend, None]:
    """Get smart backend based on SMART_BACKEND env var."""
    backend_type = os.environ.get("SMART_BACKEND", "openai").lower()
    model = os.environ.get("SMART_MODEL")

    if backend_type == "anthropic":
        from appinfra.log import Logger
        from llm_infer.client import Factory, SAIAAdapter

        lg = Logger("build-multi")
        factory = Factory(lg)
        async with factory.anthropic(model=model or DEFAULT_ANTHROPIC_MODEL) as client:
            yield SAIAAdapter(client)
    else:
        url = os.environ.get("SMART_URL", "http://localhost:8000/v1")
        async with OpenAIBackend(model=model, base_url=url) as backend:
            yield backend


async def main() -> None:  # cq: exempt
    local_model = os.environ.get("LOCAL_MODEL")
    local_url = os.environ.get("LOCAL_URL", "http://localhost:8000/v1")
    smart_type = os.environ.get("SMART_BACKEND", "openai")

    print(f"Local: {local_url} | Smart: {smart_type}")
    print(f"Task: {TASK}\n")

    async with OpenAIBackend(model=local_model, base_url=local_url) as local_backend:
        async with get_smart_backend() as smart_backend:
            local = SAIA.builder().backend(local_backend).build()
            smart = SAIA.builder().backend(smart_backend).build()

            # 1. Decompose (local)
            print("[decompose] breaking down task...")
            subtasks = await local.decompose(TASK)
            print(f"[decompose] {len(subtasks)} subtasks")
            for t in subtasks:
                print(f"  - {t[:60]}")

            # 2. Instruct each (local)
            print("\n[instruct] generating code...")
            parts = []
            for i, t in enumerate(subtasks):
                print(f"  [{i + 1}/{len(subtasks)}] {t[:50]}...", end=" ", flush=True)
                parts.append(await local.instruct(f"Write Python for: {t}"))
                print("done")

            # 3. Verify each (smart)
            print("\n[verify] checking code quality...")
            for i, code in enumerate(parts):
                print(f"  [{i + 1}/{len(parts)}] verifying...", end=" ", flush=True)
                result = await smart.verify(code, "valid Python, handles errors")
                status = "✓" if result.passed else "✗"
                print(f"{status} {result.reason[:40]}")

                # 4. Critique failures (smart)
                if not result.passed:
                    print("  [critique]...", end=" ", flush=True)
                    critique = await smart.critique(code)
                    print(f"{len(critique.weaknesses)} issues")

                    # 5. Refine (local)
                    print("  [refine]...", end=" ", flush=True)
                    feedback = "\n".join(critique.weaknesses)
                    parts[i] = await local.refine(code, feedback)
                    print("improved")

            # 6. Synthesize (smart)
            print("\n[synthesize] combining into final script...")
            final = await smart.ask(
                "Combine into one script:\n" + "\n---\n".join(parts),
                "Output only working Python code",
            )
            print(f"\n[synthesize]\n{final}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n{C.RED}[error]{C.RESET} {type(e).__name__}: {e}")

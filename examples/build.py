#!/usr/bin/env python3
"""Build an app: decompose → instruct → ask (combine)

Usage:
    LLM_BACKEND=anthropic ./build.py

Example output:
    [decompose] 4 subtasks
      - Create conversion functions for km/miles
      - Create conversion functions for celsius/fahrenheit
      - Create conversion functions for kg/pounds
      - Create a main function with user interface

    [instruct]
      ✓ Create conversion functions for km/miles
      ✓ Create conversion functions for celsius/fahrenheit
      ✓ Create conversion functions for kg/pounds
      ✓ Create a main function with user interface

    [ask]
    ```python
    def km_to_miles(km): return km * 0.621371
    def miles_to_km(miles): return miles / 0.621371
    ...
    ```
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples import Colors as C
from examples import get_backend
from llm_saia import SAIA

TASK = "Build a Python unit converter: km/miles, celsius/fahrenheit, kg/pounds"


async def main() -> None:
    async with get_backend() as backend:
        saia = SAIA.builder().backend(backend).build()

        # Break into subtasks
        subtasks = await saia.decompose(TASK)
        print(f"{C.CYAN}[decompose]{C.RESET} {len(subtasks)} subtasks")
        for t in subtasks:
            print(f"  - {t}")

        # Execute each subtask
        print(f"\n{C.GREEN}[instruct]{C.RESET}")
        parts = []
        for t in subtasks:
            code = await saia.instruct(f"Write Python for: {t}")
            parts.append(code)
            print(f"  ✓ {t[:50]}")

        # Combine into final result
        final = await saia.ask(
            "Combine into one script:\n\n" + "\n---\n".join(parts),
            "Output only working Python code",
        )
        print(f"\n{C.MAGENTA}[ask]{C.RESET}\n{final}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n{C.RED}[error]{C.RESET} {type(e).__name__}: {e}")

#!/usr/bin/env python3
"""Build a web scraper: decompose → instruct → synthesize

Usage:
    LLM_BACKEND=anthropic ./scraper.py

Example output:
    [decompose] breaking down: Build a web scraper
       - Set up HTTP client with proper headers
       - Parse HTML content with BeautifulSoup
       - Extract links and text from pages
       - Handle errors and retries

    [instruct] generating code...
       ✓ Set up HTTP client with proper headers
       ✓ Parse HTML content with BeautifulSoup
       ✓ Extract links and text from pages
       ✓ Handle errors and retries

    [synthesize] combining into working scraper...

    [output]
    import httpx
    from bs4 import BeautifulSoup
    ...
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples import Colors as C
from examples import get_backend
from llm_saia import SAIA

TASK = "Build a web scraper"


async def main() -> None:
    async with get_backend() as backend:
        saia = SAIA.builder().backend(backend).build()

        # Break into subtasks
        print(f"{C.CYAN}[decompose]{C.RESET} breaking down: {TASK}")
        subtasks = await saia.decompose(TASK)
        for t in subtasks:
            print(f"   - {t}")

        # Execute each subtask
        print(f"\n{C.GREEN}[instruct]{C.RESET} generating code...")
        results = []
        for t in subtasks:
            code = await saia.instruct(t)
            results.append(code)
            print(f"   ✓ {t[:50]}")

        # Combine into final result
        print(f"\n{C.MAGENTA}[synthesize]{C.RESET} combining into working scraper...")
        output = await saia.with_max_call_tokens(4096).synthesize(
            results, goal="single working Python script, raw code without markdown fences"
        )

        # Print the output (writing to file would create untracked repo files)
        print(f"\n{C.GREEN}[output]{C.RESET}\n{output}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n{C.RED}[error]{C.RESET} {type(e).__name__}: {e}")

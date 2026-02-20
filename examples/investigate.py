#!/usr/bin/env python3
"""Investigate a claim: verify → critique → refine

Usage:
    ./investigate.py "Water boils at 100 degrees Celsius"
    LLM_BACKEND=anthropic ./investigate.py "The sky is green"

Example output:
    Claim: Water boils at 100 degrees Celsius

    [verify] ✓ PASS
       The statement is factually accurate under standard atmospheric pressure.

    [critique] questioning claim, weaknesses:
       - Doesn't consider environmental factors like altitude
       - Limited scope to only one condition

    [refine] improved claim:
       Water boils at 100 degrees Celsius at sea level under standard atmospheric pressure.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples import Colors as C
from examples import get_backend
from llm_saia import SAIA

DEFAULT_CLAIM = "2 + 2 equals 5"


async def main(claim: str) -> None:
    async with get_backend() as backend:
        saia = (
            SAIA.builder()
            .backend(backend)
            .system("One sentence max. No markdown. No lists.")
            .build()
        )

        print(f"Claim: {C.YELLOW}{claim}{C.RESET}\n")

        # Verify - is this claim accurate?
        result = await saia.verify(claim, "factually accurate and not misleading")
        status = f"{C.GREEN}✓ PASS{C.RESET}" if result.passed else f"{C.RED}✗ FAIL{C.RESET}"
        print(f"{C.BLUE}[verify]{C.RESET} {status}")
        print(f"   {result.reason[:150]}\n")

        # Critique the verification
        print(f"{C.YELLOW}[critique]{C.RESET} questioning claim, weaknesses:")
        critique = await saia.critique(claim)
        for w in critique.weaknesses[:3]:
            print(f"   - {w}")
        print()

        # Refine - fix the claim
        refined = await saia.refine(claim, "\n".join(critique.weaknesses))
        print(f"{C.MAGENTA}[refine]{C.RESET} improved claim:")
        print(f"   {C.GREEN}{refined}{C.RESET}")


if __name__ == "__main__":
    claim = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CLAIM
    try:
        asyncio.run(main(claim))
    except Exception as e:
        print(f"\n{C.RED}[error]{C.RESET} {type(e).__name__}: {e}")

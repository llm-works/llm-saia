#!/usr/bin/env python3
"""Example: Investigate a claim using SAIA verbs.

This example demonstrates a simple investigation workflow:
1. ASK for evidence about a claim
2. VERIFY the evidence is factually accurate
3. CRITIQUE the original claim
4. REFINE the claim based on critique

Usage:
    ./examples/investigate.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_saia import SAIA
from llm_saia.backends.openai import OpenAIBackend
from llm_saia.core.types import Critique, RunConfig, VerifyResult


async def gather_evidence(saia: SAIA, claim: str) -> str:
    """ASK for evidence about a claim."""
    print("\n[ASK] Gathering evidence...")
    evidence = await saia.ask(claim, "What evidence supports or refutes this claim?")
    print(f"Evidence:\n{evidence[:500]}...")
    return evidence


async def verify_evidence(saia: SAIA, evidence: str) -> VerifyResult:
    """VERIFY the evidence is factually accurate."""
    print("\n[VERIFY] Checking factual accuracy...")
    verification = await saia.verify(evidence, "factually accurate and well-sourced")
    print(f"Passed: {verification.passed}")
    print(f"Reason: {verification.reason}")
    return verification


async def critique_claim(saia: SAIA, claim: str) -> Critique:
    """CRITIQUE the original claim."""
    print("\n[CRITIQUE] Finding counter-arguments...")
    critique = await saia.critique(claim)
    print(f"Counter-argument: {critique.counter_argument}")
    print(f"Weaknesses: {critique.weaknesses}")
    print(f"Strength score: {critique.strength:.2f}")
    return critique


async def refine_claim(saia: SAIA, claim: str, critique: Critique) -> str:
    """REFINE the claim based on critique."""
    print("\n[REFINE] Improving the claim...")
    feedback = f"Address these weaknesses: {critique.weaknesses}"
    refined = await saia.refine(claim, feedback)
    print(f"Refined claim:\n{refined}")
    return refined


def store_result(saia: SAIA, claim: str, refined: str, verification: Any, critique: Any) -> None:
    """Store investigation result in memory."""
    saia.store(
        "investigation_result",
        {
            "original_claim": claim,
            "refined_claim": refined,
            "verification": verification,
            "critique": critique,
        },
    )


async def investigate_claim(saia: SAIA, claim: str) -> None:
    """Run investigation workflow on a claim."""
    print(f"Investigating claim: {claim}\n")
    print("=" * 60)

    evidence = await gather_evidence(saia, claim)
    verification = await verify_evidence(saia, evidence)
    critique = await critique_claim(saia, claim)
    refined = await refine_claim(saia, claim, critique)
    store_result(saia, claim, refined, verification, critique)

    print("\n" + "=" * 60)
    print("Investigation complete. Result stored in memory.")


async def main() -> None:
    """Run the example."""
    async with OpenAIBackend() as backend:
        # Create SAIA with custom run config
        saia = SAIA(
            backend=backend,
            run=RunConfig(max_iterations=3, max_call_tokens=4096),
        )

        print(f"Run config: {saia.run_config}")

        claim = "Python is slower than C for all computational tasks"
        await investigate_claim(saia, claim)

        results = saia.recall("investigation")
        print(f"\nRecalled {len(results)} result(s) from memory")


if __name__ == "__main__":
    asyncio.run(main())

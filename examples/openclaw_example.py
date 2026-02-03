#!/usr/bin/env python3
"""Example: Using SAIA with OpenClaw backend.

This example demonstrates using SAIA verbs through the OpenClaw gateway,
which supports multiple LLM providers (Claude, OpenRouter, Ollama, etc.).

Prerequisites:
    1. Install OpenClaw: npm install -g openclaw@latest
    2. Run onboarding: openclaw onboard
    3. Start gateway: openclaw gateway

Usage:
    python examples/openclaw_example.py
"""

import asyncio
import sys

import httpx

from llm_saia import SAIA
from llm_saia.backends.openclaw import OpenClawBackend


async def check_gateway(gateway_url: str) -> bool:
    """Check if OpenClaw gateway is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{gateway_url}/health")
            return response.status_code == 200
    except Exception:
        return False


def print_gateway_setup_instructions() -> None:
    """Print instructions for setting up OpenClaw gateway."""
    print("OpenClaw gateway not available at http://127.0.0.1:18789")
    print("\nTo set up OpenClaw:")
    print("  1. Install: npm install -g openclaw@latest")
    print("  2. Onboard: openclaw onboard")
    print("  3. Start:   openclaw gateway")
    print("\nOr set OPENCLAW_GATEWAY_URL to point to a running gateway.")


async def demo_ask(saia: SAIA, claim: str) -> str:
    """Demo the ASK verb."""
    print("\n[ASK] Querying about Python performance...")
    response = await saia.ask(claim, "Is this claim accurate? Why or why not?")
    print(f"Response: {response[:300]}...")
    return response


async def demo_verify(saia: SAIA, text: str) -> None:
    """Demo the VERIFY verb."""
    print("\n[VERIFY] Checking factual accuracy...")
    result = await saia.verify(text, "factually accurate and nuanced")
    print(f"Passed: {result.passed}")
    print(f"Reason: {result.reason}")


async def demo_critique(saia: SAIA, claim: str) -> None:
    """Demo the CRITIQUE verb."""
    print("\n[CRITIQUE] Finding counter-arguments...")
    critique = await saia.critique(claim)
    print(f"Counter-argument: {critique.counter_argument}")
    print(f"Weaknesses: {critique.weaknesses}")
    print(f"Strength score: {critique.strength:.2f}")


async def demo_decompose(saia: SAIA) -> None:
    """Demo the DECOMPOSE verb."""
    print("\n[DECOMPOSE] Breaking down a task...")
    task = "Build a REST API with authentication and database integration"
    subtasks = await saia.decompose(task)
    print("Subtasks:")
    for i, subtask in enumerate(subtasks, 1):
        print(f"  {i}. {subtask}")


async def main() -> None:
    """Run the OpenClaw example."""
    backend = OpenClawBackend()

    if not await check_gateway(backend.gateway_url):
        print_gateway_setup_instructions()
        sys.exit(1)

    async with backend:
        saia = SAIA(backend=backend)
        print("Connected to OpenClaw gateway")
        print("=" * 50)

        claim = "Python is slower than C for all computational tasks"
        response = await demo_ask(saia, claim)
        await demo_verify(saia, response)
        await demo_critique(saia, claim)
        await demo_decompose(saia)

    print("\n" + "=" * 50)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())

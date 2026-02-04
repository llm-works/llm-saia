#!/usr/bin/env python3
"""Example: Using SAIA with OpenClaw backend.

This example demonstrates using SAIA verbs through the OpenClaw gateway,
which supports multiple LLM providers (Claude, OpenRouter, Ollama, etc.).

Also demonstrates the fluent run config API:
- with_single_call() - single LLM call, no looping
- with_max_iterations(n) - limit tool-calling rounds
- with_timeout_secs(s) - set timeout
- with_max_tokens(n) - set token budget

Prerequisites:
    1. Install OpenClaw: npm install -g openclaw@latest
    2. Run onboarding: openclaw onboard
    3. Start gateway: openclaw gateway

Usage:
    ./examples/openclaw_example.py
"""

import asyncio
import sys
from pathlib import Path

import httpx

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_saia import SAIA
from llm_saia.backends.openclaw import OpenClawBackend
from llm_saia.core.types import RunConfig


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
    """Demo the VERIFY verb with single-call mode."""
    print("\n[VERIFY] Checking factual accuracy (single-call mode)...")
    # Use with_single_call() for quick verification without tool looping
    result = await saia.with_single_call().verify(text, "factually accurate and nuanced")
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
    """Demo the DECOMPOSE verb with timeout."""
    print("\n[DECOMPOSE] Breaking down a task (with 30s timeout)...")
    task = "Build a REST API with authentication and database integration"
    # Use with_timeout_secs() to limit response time
    subtasks = await saia.with_timeout_secs(30).decompose(task)
    print("Subtasks:")
    for i, subtask in enumerate(subtasks, 1):
        print(f"  {i}. {subtask}")


async def demo_fluent_api(saia: SAIA) -> None:
    """Demo the fluent run config API."""
    print("\n[FLUENT API] Demonstrating run config modifiers...")

    print(f"\nDefault config: {saia.run_config}")
    print(f"Single-call config: {saia.with_single_call().run_config}")
    print(f"Max 10 iterations: {saia.with_max_iterations(10).run_config}")
    print(f"With 60s timeout: {saia.with_timeout_secs(60).run_config}")
    print(f"With 50k token budget: {saia.with_max_tokens(50000).run_config}")


async def main() -> None:
    """Run the OpenClaw example."""
    backend = OpenClawBackend()

    if not await check_gateway(backend.gateway_url):
        print_gateway_setup_instructions()
        sys.exit(1)

    async with backend:
        # Create SAIA with custom default run config
        saia = SAIA(
            backend=backend,
            run=RunConfig(max_iterations=5, max_total_tokens=100000),
        )

        print("Connected to OpenClaw gateway")
        print(f"Default run config: {saia.run_config}")
        print("=" * 50)

        claim = "Python is slower than C for all computational tasks"
        response = await demo_ask(saia, claim)
        await demo_verify(saia, response)
        await demo_critique(saia, claim)
        await demo_decompose(saia)
        await demo_fluent_api(saia)

    print("\n" + "=" * 50)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())

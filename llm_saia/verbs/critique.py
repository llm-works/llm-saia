"""CRITIQUE verb: Generate strongest counter-argument."""

from typing import Any

from llm_saia.core.protocols import SAIABackend
from llm_saia.core.types import Critique


async def critique(backend: SAIABackend, artifact: Any) -> Critique:
    """CRITIQUE verb: Generate strongest counter-argument.

    Args:
        backend: The LLM backend to use.
        artifact: The artifact to critique (claim, argument, etc.).

    Returns:
        Critique with counter_argument, weaknesses, and strength score.
    """
    prompt = f"""You are a rigorous critic. Your job is to find the strongest counter-argument
and identify real weaknesses. Do not be agreeable or diplomatic.

Artifact to critique:
{artifact}

Provide:
1. The strongest counter-argument against this
2. A list of specific weaknesses
3. A strength score (0.0-1.0) indicating how strong the original artifact is
   (lower = more weaknesses found)

Be thorough and find real problems."""
    return await backend.complete_structured(prompt, Critique)

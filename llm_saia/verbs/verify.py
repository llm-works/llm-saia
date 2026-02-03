"""VERIFY verb: Check if artifact satisfies predicate."""

from typing import Any

from llm_saia.core.protocols import SAIABackend
from llm_saia.core.types import VerifyResult


async def verify(backend: SAIABackend, artifact: Any, predicate: str) -> VerifyResult:
    """VERIFY verb: Check if artifact satisfies predicate.

    Args:
        backend: The LLM backend to use.
        artifact: The artifact to verify.
        predicate: The condition to check (e.g., "factually accurate").

    Returns:
        VerifyResult with passed (bool) and reason (str).
    """
    prompt = f"""Evaluate whether this artifact satisfies the given predicate.
Be rigorous and objective in your assessment.

Artifact:
{artifact}

Predicate: {predicate}

Evaluate whether the artifact satisfies the predicate."""
    return await backend.complete_structured(prompt, VerifyResult)

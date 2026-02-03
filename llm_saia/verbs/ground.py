"""GROUND verb: Anchor artifact against sources for evidence."""

from typing import Any

from llm_saia.core.protocols import SAIABackend
from llm_saia.core.types import Evidence


async def ground(backend: SAIABackend, artifact: Any, sources: list[Any]) -> list[Evidence]:
    """GROUND verb: Anchor artifact against sources for evidence.

    Args:
        backend: The LLM backend to use.
        artifact: The artifact (claim, hypothesis) to ground.
        sources: List of sources to check against.

    Returns:
        List of Evidence objects extracted from sources.
    """
    # Process each source individually to extract evidence
    results: list[Evidence] = []

    for i, source in enumerate(sources):
        prompt = f"""Analyze whether this source provides evidence for or against the artifact.

Artifact (claim/hypothesis):
{artifact}

Source {i + 1}:
{source}

Extract any relevant evidence. Determine:
1. The specific content that relates to the artifact
2. The source identifier
3. Whether it supports, refutes, or is neutral to the artifact
4. How strong the evidence is (0.0-1.0)"""
        evidence = await backend.complete_structured(prompt, Evidence)
        results.append(evidence)

    return results

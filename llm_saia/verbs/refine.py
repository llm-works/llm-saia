"""REFINE verb: Improve artifact based on feedback."""

from typing import Any

from llm_saia.core.protocols import SAIABackend


async def refine(backend: SAIABackend, artifact: Any, feedback: str) -> str:
    """REFINE verb: Improve artifact based on feedback.

    Args:
        backend: The LLM backend to use.
        artifact: The artifact to refine.
        feedback: Feedback describing how to improve the artifact.

    Returns:
        The refined artifact as a string.
    """
    prompt = f"""Improve the following artifact based on the feedback provided.
Make targeted improvements that address the feedback while preserving
the artifact's core purpose and strengths.

Original artifact:
{artifact}

Feedback:
{feedback}

Refined artifact:"""
    return await backend.complete(prompt)

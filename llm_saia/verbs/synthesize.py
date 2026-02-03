"""SYNTHESIZE verb: Combine multiple artifacts into structured output."""

from typing import Any, TypeVar

from llm_saia.core.protocols import SAIABackend

T = TypeVar("T")


async def synthesize(backend: SAIABackend, artifacts: list[Any], schema: type[T]) -> T:
    """SYNTHESIZE verb: Combine multiple artifacts into structured output.

    Args:
        backend: The LLM backend to use.
        artifacts: List of artifacts to synthesize.
        schema: A dataclass type for the structured output.

    Returns:
        An instance of the schema type combining information from all artifacts.
    """
    artifacts_text = "\n\n---\n\n".join(
        f"Artifact {i + 1}:\n{artifact}" for i, artifact in enumerate(artifacts)
    )

    prompt = f"""Synthesize the following artifacts into a coherent, structured output.
Combine and reconcile information across all sources.

{artifacts_text}

Synthesize these into a unified output."""
    return await backend.complete_structured(prompt, schema)

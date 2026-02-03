"""ASK verb: Query an artifact with a question."""

from typing import Any

from llm_saia.core.protocols import SAIABackend


async def ask(backend: SAIABackend, artifact: Any, question: str) -> str:
    """ASK verb: Query an artifact with a question.

    Args:
        backend: The LLM backend to use.
        artifact: The artifact to query (will be converted to string).
        question: The question to ask about the artifact.

    Returns:
        The LLM's response to the question.
    """
    prompt = f"""Given the following artifact, answer the question.

Artifact:
{artifact}

Question: {question}

Answer:"""
    return await backend.complete(prompt)

"""DECOMPOSE verb: Break down task into subtasks."""

from dataclasses import dataclass

from llm_saia.core.protocols import SAIABackend


@dataclass
class DecomposeResult:
    """Internal schema for decompose structured output."""

    subtasks: list[str]


async def decompose(backend: SAIABackend, task: str) -> list[str]:
    """DECOMPOSE verb: Break down task into subtasks.

    Args:
        backend: The LLM backend to use.
        task: The task description to decompose.

    Returns:
        List of subtask descriptions.
    """
    prompt = f"""Break down the following task into concrete, actionable subtasks.
Each subtask should be specific and independently executable.
Order subtasks logically (dependencies first).

Task:
{task}

Provide a list of subtasks."""
    result = await backend.complete_structured(prompt, DecomposeResult)
    return result.subtasks

"""RECALL and STORE verbs: In-memory storage operations."""

from typing import Any


def store(memory: dict[str, Any], key: str, value: Any) -> None:
    """STORE verb: Save a value to memory.

    Args:
        memory: The memory dictionary.
        key: The key to store under.
        value: The value to store.
    """
    memory[key] = value


def recall(memory: dict[str, Any], query: str) -> list[Any]:
    """RECALL verb: Retrieve values from memory matching query.

    Simple implementation: returns values whose keys contain the query string.
    V1 will add semantic search.

    Args:
        memory: The memory dictionary.
        query: The search query (substring match on keys).

    Returns:
        List of values whose keys contain the query.
    """
    results = []
    query_lower = query.lower()
    for key, value in memory.items():
        if query_lower in key.lower():
            results.append(value)
    return results

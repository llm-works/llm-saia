"""Shared helpers for guard implementations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class GuardState:
    """Tracks retry count for iteration guards.

    Automatically resets when a new task starts (iteration 0).
    """

    __slots__ = ("max_retries", "count")

    def __init__(self, max_retries: int) -> None:
        self.max_retries = max_retries
        self.count = 0

    def reset_if_new(self, iteration: int) -> None:
        """Reset counter if this is the start of a new task."""
        if iteration == 0:
            self.count = 0

    def feedback(
        self, fn: Callable[..., tuple[str, str]], escalate: bool, **kwargs: Any
    ) -> str | None:
        """Increment counter and return feedback if under limit."""
        self.count += 1
        if self.count > self.max_retries:
            return None
        base, forceful = fn(self.count, **kwargs)
        return forceful if escalate and self.count > 1 else base


def ordinal(n: int) -> str:
    """Return ordinal string for a number (1st, 2nd, 3rd, etc.)."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# Type mapping for JSON schema validation
JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def validate_schema(data: Any, schema: dict[str, Any]) -> list[str]:
    """Basic JSON schema validation for terminal tool arguments.

    Returns list of error messages. Only validates:
    - Data is a dict
    - Required top-level fields are present
    - Top-level field types match (via _type_matches)

    Does NOT validate: nested objects, array item types, enums, patterns,
    additionalProperties, or other JSON Schema features. For full validation,
    use a dedicated JSON Schema library like jsonschema.
    """
    errors: list[str] = []
    if not isinstance(data, dict):
        return [f"expected object, got {type(data).__name__}"]

    for field in schema.get("required", []):
        if field not in data:
            errors.append(f"missing required field: {field}")

    properties = schema.get("properties", {})
    for field, value in data.items():
        if field in properties:
            expected = properties[field].get("type")
            if expected and not _type_matches(value, expected):
                errors.append(f"field '{field}': expected {expected}, got {type(value).__name__}")
    return errors


def _type_matches(value: Any, json_type: str) -> bool:
    """Check if a Python value matches a JSON schema type."""
    type_class = JSON_TYPE_MAP.get(json_type)
    return type_class is None or isinstance(value, type_class)

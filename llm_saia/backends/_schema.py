"""Shared schema conversion utilities for backends."""

import dataclasses
from typing import Any, TypeVar, get_type_hints

T = TypeVar("T")


def dataclass_to_json_schema(schema: type) -> dict[str, Any]:
    """Convert a dataclass to a JSON schema.

    Args:
        schema: A dataclass type to convert.

    Returns:
        JSON schema dict with name, description, and schema properties.
    """
    if not dataclasses.is_dataclass(schema):
        raise TypeError(f"Schema must be a dataclass, got {type(schema)}")

    hints = get_type_hints(schema)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for field in dataclasses.fields(schema):
        field_type = hints[field.name]
        properties[field.name] = python_type_to_json_schema(field_type)

        # Check if field has a default
        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            required.append(field.name)

    return {
        "name": schema.__name__,
        "description": schema.__doc__ or f"Structured output for {schema.__name__}",
        "schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def python_type_to_json_schema(python_type: type) -> dict[str, Any]:
    """Convert Python type hints to JSON schema."""
    origin = getattr(python_type, "__origin__", None)

    if python_type is str:
        return {"type": "string"}
    elif python_type is int:
        return {"type": "integer"}
    elif python_type is float:
        return {"type": "number"}
    elif python_type is bool:
        return {"type": "boolean"}
    elif origin is list:
        args = getattr(python_type, "__args__", (Any,))
        return {"type": "array", "items": python_type_to_json_schema(args[0])}
    elif origin is dict:
        return {"type": "object"}
    elif python_type is Any:
        return {"type": "string"}
    else:
        raise TypeError(
            f"Unsupported type for JSON schema: {python_type}. "
            "Supported types: str, int, float, bool, list[T], dict, Any."
        )


def parse_json_to_dataclass(data: object, schema: type[T]) -> T:
    """Parse JSON data into a dataclass instance.

    Args:
        data: The JSON data (should be a dict).
        schema: The dataclass type to instantiate.

    Returns:
        An instance of the schema type.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")
    return schema(**data)

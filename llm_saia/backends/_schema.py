"""Shared schema conversion utilities for backends."""

import dataclasses
import types
from typing import Any, TypeVar, Union, cast, get_args, get_origin, get_type_hints

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


def _unwrap_optional(python_type: type) -> type | None:
    """Unwrap Optional[T] or T | None to T. Returns None if not an Optional."""
    origin = get_origin(python_type)
    if origin is not Union and origin is not types.UnionType:
        return None

    args = [a for a in get_args(python_type) if a is not type(None)]
    if len(args) == 1:
        inner_type: type = args[0]
        return inner_type

    raise TypeError(
        f"Union types with multiple non-None types not supported: {python_type}. "
        "Use Optional[T] for nullable fields."
    )


# Mapping of Python primitive types to JSON schema types
_PRIMITIVE_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def python_type_to_json_schema(python_type: type) -> dict[str, Any]:
    """Convert Python type hints to JSON schema.

    Supported types:
        - Primitives: str, int, float, bool
        - Collections: list[T], dict
        - Special: Any, Optional[T], T | None

    Note:
        Nested dataclasses are not supported. If you need nested structures,
        flatten them or use dict for the nested portion.
    """
    # Handle Optional[T] / T | None
    unwrapped = _unwrap_optional(python_type)
    if unwrapped is not None:
        return python_type_to_json_schema(unwrapped)

    # Handle primitives
    if python_type in _PRIMITIVE_TYPE_MAP:
        return {"type": _PRIMITIVE_TYPE_MAP[python_type]}

    # Handle generic types
    origin = get_origin(python_type)
    if origin is list:
        args = get_args(python_type) or (Any,)
        return {"type": "array", "items": python_type_to_json_schema(args[0])}
    if origin is dict:
        return {"type": "object"}
    if python_type is Any:
        return {"type": "string"}

    raise TypeError(
        f"Unsupported type for JSON schema: {python_type}. "
        "Supported types: str, int, float, bool, list[T], dict, Any, Optional[T]."
    )


def parse_json_to_dataclass(data: object, schema: type[T]) -> T:
    """Parse JSON data into a dataclass instance.

    Extra fields in the data that are not defined in the schema are ignored.
    This allows flexibility when LLMs return additional fields beyond the schema.

    Args:
        data: The JSON data (should be a dict).
        schema: The dataclass type to instantiate.

    Returns:
        An instance of the schema type.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")
    # Filter to only fields defined in the dataclass (ignore extra LLM output)
    # Cast needed because type[T] doesn't guarantee dataclass to mypy
    field_names = {f.name for f in dataclasses.fields(cast(type, schema))}
    filtered_data = {k: v for k, v in data.items() if k in field_names}
    return schema(**filtered_data)

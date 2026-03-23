"""Schema conversion utilities for structured output.

SAIA handles structured output by building prompts with JSON schema
instructions and parsing responses into dataclasses.
"""

import dataclasses
import enum
import types
from typing import Any, Literal, TypeVar, Union, cast, get_args, get_origin, get_type_hints

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
        - Constrained: Literal[...], Enum subclasses
        - Nested: dataclasses (recursive)
        - Special: Any, Optional[T], T | None
    """
    # Handle Optional[T] / T | None
    unwrapped = _unwrap_optional(python_type)
    if unwrapped is not None:
        return python_type_to_json_schema(unwrapped)

    # Handle primitives
    if python_type in _PRIMITIVE_TYPE_MAP:
        return {"type": _PRIMITIVE_TYPE_MAP[python_type]}

    if python_type is Any:
        return {"type": "string"}

    # Try complex types (generic, enum, dataclass)
    result = _try_complex_type_to_json_schema(python_type)
    if result is not None:
        return result

    raise TypeError(
        f"Unsupported type for JSON schema: {python_type}. "
        "Supported: str, int, float, bool, list[T], dict, Literal[...], Enum, "
        "dataclass, Any, Optional[T]."
    )


def _try_complex_type_to_json_schema(python_type: type) -> dict[str, Any] | None:
    """Try to convert complex types (generic, enum, dataclass) to JSON schema."""
    origin = get_origin(python_type)

    if origin is Literal:
        return _literal_to_json_schema(get_args(python_type))
    if origin is list:
        args = get_args(python_type) or (Any,)
        return {"type": "array", "items": python_type_to_json_schema(args[0])}
    if origin is dict:
        return {"type": "object"}
    if isinstance(python_type, type) and issubclass(python_type, enum.Enum):
        return _enum_to_json_schema(python_type)
    if dataclasses.is_dataclass(python_type):
        return _nested_dataclass_to_json_schema(python_type)

    return None


def _literal_to_json_schema(args: tuple[Any, ...]) -> dict[str, Any]:
    """Convert Literal type arguments to JSON schema with enum."""
    if not args:
        raise TypeError("Literal type must have at least one value")

    first = args[0]
    if isinstance(first, int) and not isinstance(first, bool):
        return {"type": "integer", "enum": list(args)}
    elif isinstance(first, str):
        return {"type": "string", "enum": list(args)}
    elif isinstance(first, bool):
        return {"type": "boolean", "enum": list(args)}
    elif isinstance(first, float):
        return {"type": "number", "enum": list(args)}
    else:
        # Mixed or complex types - just use enum without type constraint
        return {"enum": list(args)}


def _enum_to_json_schema(enum_type: type[enum.Enum]) -> dict[str, Any]:
    """Convert Enum type to JSON schema with enum values."""
    values = [member.value for member in enum_type]
    if not values:
        raise TypeError(f"Enum {enum_type.__name__} has no members")

    first = values[0]
    if isinstance(first, int) and not isinstance(first, bool):
        return {"type": "integer", "enum": values}
    elif isinstance(first, str):
        return {"type": "string", "enum": values}
    else:
        return {"enum": values}


def _nested_dataclass_to_json_schema(schema: type) -> dict[str, Any]:
    """Convert a nested dataclass to inline JSON schema (object type)."""
    hints = get_type_hints(schema)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for field in dataclasses.fields(schema):
        field_type = hints[field.name]
        properties[field.name] = python_type_to_json_schema(field_type)

        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            required.append(field.name)

    result: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        result["required"] = required
    return result


def parse_json_to_dataclass(data: object, schema: type[T]) -> T:
    """Parse JSON data into a dataclass instance.

    Extra fields in the data that are not defined in the schema are ignored.
    This allows flexibility when LLMs return additional fields beyond the schema.

    Handles nested dataclasses, enums, and lists of dataclasses recursively.

    Args:
        data: The JSON data (should be a dict).
        schema: The dataclass type to instantiate.

    Returns:
        An instance of the schema type.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")

    hints = get_type_hints(schema)
    parsed_data: dict[str, Any] = {}

    for field in dataclasses.fields(cast(type, schema)):
        if field.name not in data:
            continue

        field_type = hints[field.name]
        value = data[field.name]
        parsed_data[field.name] = _parse_field_value(value, field_type)

    return schema(**parsed_data)


def _parse_field_value(value: Any, field_type: type) -> Any:
    """Parse a field value according to its type hint."""
    if value is None:
        return None

    # Unwrap Optional
    unwrapped = _unwrap_optional(field_type)
    if unwrapped is not None:
        field_type = unwrapped

    origin = get_origin(field_type)

    # Literal - value is already the right type from JSON
    if origin is Literal:
        return value
    # list[T] - may contain nested dataclasses
    if origin is list:
        return _parse_list_value(value, field_type)
    # Enum - convert value back to enum member
    if isinstance(field_type, type) and issubclass(field_type, enum.Enum):
        return field_type(value)
    # Nested dataclass
    if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
        return parse_json_to_dataclass(value, field_type)
    # Primitives and other types - return as-is
    return value


def _parse_list_value(value: Any, field_type: type) -> Any:
    """Parse a list value, recursively parsing items if needed."""
    if not isinstance(value, list):
        return value
    args = get_args(field_type)
    if not args:
        return value
    item_type = args[0]
    return [_parse_field_value(item, item_type) for item in value]

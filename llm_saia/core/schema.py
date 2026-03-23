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

    return {
        "name": schema.__name__,
        "description": schema.__doc__ or f"Structured output for {schema.__name__}",
        "schema": _build_object_schema(schema, seen=set()),
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


def python_type_to_json_schema(
    python_type: type, *, _seen: set[type] | None = None
) -> dict[str, Any]:
    """Convert Python type hints to JSON schema.

    Supported types:
        - Primitives: str, int, float, bool
        - Collections: list[T], dict
        - Constrained: Literal[...], Enum subclasses
        - Nested: dataclasses (recursive, but not self-referential)
        - Special: Any, Optional[T], T | None

    Args:
        python_type: The Python type to convert.
        _seen: Internal parameter for cycle detection. Do not pass directly.

    Raises:
        TypeError: If the type is unsupported or contains cycles.
    """
    if _seen is None:
        _seen = set()

    # Handle Optional[T] / T | None
    unwrapped = _unwrap_optional(python_type)
    if unwrapped is not None:
        return python_type_to_json_schema(unwrapped, _seen=_seen)

    # Handle primitives
    if python_type in _PRIMITIVE_TYPE_MAP:
        return {"type": _PRIMITIVE_TYPE_MAP[python_type]}

    if python_type is Any:
        return {"type": "string"}

    # Try complex types (generic, enum, dataclass)
    result = _try_complex_type_to_json_schema(python_type, _seen)
    if result is not None:
        return result

    raise TypeError(
        f"Unsupported type for JSON schema: {python_type}. "
        "Supported: str, int, float, bool, list[T], dict, Literal[...], Enum, "
        "dataclass, Any, Optional[T]."
    )


def _try_complex_type_to_json_schema(python_type: type, seen: set[type]) -> dict[str, Any] | None:
    """Try to convert complex types (generic, enum, dataclass) to JSON schema."""
    origin = get_origin(python_type)

    if origin is Literal:
        return _literal_to_json_schema(get_args(python_type))
    if origin is list:
        args = get_args(python_type) or (Any,)
        return {"type": "array", "items": python_type_to_json_schema(args[0], _seen=seen)}
    if origin is dict:
        return {"type": "object"}
    if isinstance(python_type, type) and issubclass(python_type, enum.Enum):
        return _enum_to_json_schema(python_type)
    if dataclasses.is_dataclass(python_type):
        return _build_object_schema(python_type, seen)

    return None


def _get_literal_type_info(value: Any) -> tuple[str, type]:
    """Get JSON type string and Python type for a Literal value."""
    if isinstance(value, bool):
        return "boolean", bool
    if isinstance(value, int):
        return "integer", int
    if isinstance(value, str):
        return "string", str
    if isinstance(value, float):
        return "number", float
    raise TypeError(
        f"Unsupported Literal value type: {type(value).__name__}. "
        "Literal values must be str, int, float, or bool."
    )


def _validate_literal_type_consistency(args: tuple[Any, ...], expected_type: type) -> None:
    """Validate all Literal args match the expected type."""
    for i, arg in enumerate(args[1:], start=1):
        # bool is subclass of int, so check bool first
        if isinstance(arg, bool) and expected_type is not bool:
            raise TypeError(
                f"Mixed types in Literal: index 0 is {expected_type.__name__}, "
                f"index {i} is bool. All values must be the same type."
            )
        if not isinstance(arg, expected_type) or (expected_type is int and isinstance(arg, bool)):
            raise TypeError(
                f"Mixed types in Literal: index 0 is {expected_type.__name__}, "
                f"index {i} is {type(arg).__name__}. All values must be the same type."
            )


def _literal_to_json_schema(args: tuple[Any, ...]) -> dict[str, Any]:
    """Convert Literal type arguments to JSON schema with enum.

    All values must be of the same type. Mixed-type Literals (e.g., Literal["a", 1])
    are not supported as they produce invalid JSON schemas.
    """
    if not args:
        raise TypeError("Literal type must have at least one value")

    json_type, expected_type = _get_literal_type_info(args[0])
    _validate_literal_type_consistency(args, expected_type)

    return {"type": json_type, "enum": list(args)}


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


def _build_object_schema(schema: type, seen: set[type]) -> dict[str, Any]:
    """Build JSON schema object type from a dataclass with cycle detection.

    Args:
        schema: The dataclass type to convert.
        seen: Set of types already being processed (for cycle detection).

    Raises:
        TypeError: If a cycle is detected (self-referential dataclass).
    """
    if schema in seen:
        raise TypeError(
            f"Recursive type detected: {schema.__name__}. "
            "Self-referential dataclasses are not supported in JSON schema generation."
        )

    seen = seen | {schema}  # Create new set to avoid mutating caller's set

    hints = get_type_hints(schema)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for field in dataclasses.fields(schema):
        field_type = hints[field.name]
        properties[field.name] = python_type_to_json_schema(field_type, _seen=seen)

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
    """Parse a field value according to its type hint.

    Raises:
        TypeError: If value type doesn't match expected field type for structured types.
    """
    if value is None:
        return None

    # Unwrap Optional
    unwrapped = _unwrap_optional(field_type)
    if unwrapped is not None:
        field_type = unwrapped

    origin = get_origin(field_type)

    if origin is Literal:
        return value
    if origin is list:
        return _parse_list_field(value, field_type)
    if isinstance(field_type, type) and issubclass(field_type, enum.Enum):
        return field_type(value)
    if dataclasses.is_dataclass(field_type):
        return _parse_dataclass_field(value, field_type)

    return value


def _parse_list_field(value: Any, field_type: type) -> list[Any]:
    """Parse a list field, validating type and recursively parsing items."""
    if not isinstance(value, list):
        raise TypeError(f"Expected list for field type {field_type}, got {type(value).__name__}")
    args = get_args(field_type)
    if not args:
        return value
    item_type = args[0]
    return [_parse_field_value(item, item_type) for item in value]


def _parse_dataclass_field(value: Any, field_type: type) -> Any:
    """Parse a nested dataclass field, validating type."""
    if not isinstance(value, dict):
        raise TypeError(
            f"Expected dict for dataclass field {field_type.__name__}, got {type(value).__name__}"
        )
    return parse_json_to_dataclass(value, field_type)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Schema conversion utilities for structured output.

SAIA handles structured output by building prompts with JSON schema
instructions and parsing responses into dataclasses.
"""

import dataclasses
import enum
import types
from typing import Any, Literal, TypeVar, Union, cast, get_args, get_origin, get_type_hints

T = TypeVar("T")


def _is_pydantic_model(schema: type) -> bool:
    """Duck-type check for pydantic.BaseModel subclasses.

    Uses attribute presence rather than isinstance so pydantic remains a
    genuinely optional dependency — the import is deferred to callers
    that reach the pydantic branch. Zero-dep users pay nothing.
    """
    return (
        isinstance(schema, type)
        and hasattr(schema, "model_json_schema")
        and hasattr(schema, "model_validate")
    )


def to_json_schema(schema: type) -> dict[str, Any]:
    """Convert a schema type to the JSON schema envelope SAIA sends to backends.

    Dispatches on ``schema``: pydantic ``BaseModel`` subclasses go through
    ``model_json_schema()`` (full JSON-Schema vocabulary — ``ge``/``le``,
    ``pattern``, ``format``, discriminated unions, ``$ref``/``$defs``,
    validators, everything pydantic v2 supports). Stdlib dataclasses
    continue through :func:`dataclass_to_json_schema`.

    Requires ``pip install llm-saia[pydantic]`` for BaseModel schemas;
    dataclass callers pay no dependency cost.
    """
    if _is_pydantic_model(schema):
        return _pydantic_to_json_schema(schema)
    return dataclass_to_json_schema(schema)


def parse(data: Any, schema: type[T]) -> T:
    """Parse JSON data into an instance of ``schema``.

    Dispatches on ``schema``: pydantic models validate via
    ``model_validate`` (full pydantic pipeline — coercion, custom
    ``@field_validator``/``@model_validator``, discriminated union
    dispatch); dataclasses use :func:`parse_json_to_dataclass`.

    Pydantic's :class:`ValidationError` inherits from :class:`ValueError`,
    so it flows into SAIA's existing structured-output retry path just
    like a dataclass parse error.
    """
    if _is_pydantic_model(schema):
        return cast(T, schema.model_validate(data))  # type: ignore[attr-defined]
    return parse_json_to_dataclass(data, schema)


def _pydantic_to_json_schema(schema: type) -> dict[str, Any]:
    """Build SAIA's schema envelope from a pydantic BaseModel."""
    raw = schema.model_json_schema()  # type: ignore[attr-defined]
    _inject_additional_properties_false(raw)
    return {
        "name": schema.__name__,
        "description": schema.__doc__ or f"Structured output for {schema.__name__}",
        "schema": raw,
    }


def _inject_additional_properties_false(schema: dict[str, Any]) -> None:
    """Recursively add additionalProperties: false to all object types.

    Required for OpenAI strict mode. Pydantic's model_json_schema() does not
    emit this by default (unless extra='forbid' is set on every model).
    """
    if schema.get("type") == "object" and "additionalProperties" not in schema:
        schema["additionalProperties"] = False

    # Handle $defs (nested model definitions)
    for defn in schema.get("$defs", {}).values():
        _inject_additional_properties_false(defn)

    # Handle nested properties
    for prop in schema.get("properties", {}).values():
        _inject_additional_properties_false(prop)

    # Handle array items
    if "items" in schema:
        _inject_additional_properties_false(schema["items"])

    # Handle allOf/anyOf/oneOf
    for key in ("allOf", "anyOf", "oneOf"):
        for sub in schema.get(key, []):
            _inject_additional_properties_false(sub)


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
    """Convert Enum type to JSON schema with enum values.

    All member values must be of the same type. Mixed-type Enums will raise TypeError.
    """
    values = [member.value for member in enum_type]
    if not values:
        raise TypeError(f"Enum {enum_type.__name__} has no members")

    first = values[0]
    json_type, expected_type = _get_enum_type_info(first, enum_type)
    _validate_enum_type_consistency(values, expected_type, enum_type)

    if json_type is None:
        return {"enum": values}
    return {"type": json_type, "enum": values}


def _get_enum_type_info(first: Any, enum_type: type) -> tuple[str | None, type | None]:
    """Get JSON type string and Python type for an enum's first value."""
    if isinstance(first, bool):
        return "boolean", bool
    if isinstance(first, int):
        return "integer", int
    if isinstance(first, str):
        return "string", str
    if isinstance(first, float):
        return "number", float
    # Unsupported type - will emit enum without type constraint
    return None, None


def _validate_enum_type_consistency(
    values: list[Any], expected_type: type | None, enum_type: type
) -> None:
    """Validate all enum values are the same type."""
    if expected_type is None:
        return  # No type constraint for unsupported types

    for i, val in enumerate(values[1:], start=1):
        if isinstance(val, bool) and expected_type is not bool:
            raise TypeError(
                f"Mixed types in Enum {enum_type.__name__}: "
                f"member 0 is {expected_type.__name__}, member {i} is bool."
            )
        if not isinstance(val, expected_type) or (expected_type is int and isinstance(val, bool)):
            raise TypeError(
                f"Mixed types in Enum {enum_type.__name__}: "
                f"member 0 is {expected_type.__name__}, member {i} is {type(val).__name__}."
            )


def _build_object_schema(schema: type, seen: set[type]) -> dict[str, Any]:
    """Build JSON schema object type from a dataclass with cycle detection.

    All fields are marked required regardless of Python defaults, and
    additionalProperties is set to false. This is required for OpenAI strict
    mode; the parser (parse_json_to_dataclass) handles missing fields via
    dataclass defaults.

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
        required.append(field.name)

    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
        "required": required,
    }
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
        return _parse_literal_field(value, field_type)
    if origin is list:
        return _parse_list_field(value, field_type)
    if isinstance(field_type, type) and issubclass(field_type, enum.Enum):
        return _parse_enum_field(value, field_type)
    if dataclasses.is_dataclass(field_type):
        return _parse_dataclass_field(value, field_type)

    return value


def _parse_literal_field(value: Any, field_type: type) -> Any:
    """Parse a Literal field, validating value is in allowed set."""
    allowed = get_args(field_type)
    if value not in allowed:
        raise TypeError(f"Value {value!r} not in Literal{list(allowed)}")
    return value


def _parse_enum_field(value: Any, enum_type: type[enum.Enum]) -> enum.Enum:
    """Parse an enum field, converting value to enum member."""
    try:
        return enum_type(value)
    except ValueError as e:
        raise TypeError(f"Invalid value {value!r} for enum {enum_type.__name__}: {e}") from e


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

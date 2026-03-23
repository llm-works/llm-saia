"""Tests for shared schema conversion utilities."""

import enum
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import pytest

from llm_saia.core.schema import (
    dataclass_to_json_schema,
    parse_json_to_dataclass,
    python_type_to_json_schema,
)

pytestmark = pytest.mark.unit


class TestPythonTypeToJsonSchema:
    def test_primitives(self) -> None:
        assert python_type_to_json_schema(str) == {"type": "string"}
        assert python_type_to_json_schema(int) == {"type": "integer"}
        assert python_type_to_json_schema(float) == {"type": "number"}
        assert python_type_to_json_schema(bool) == {"type": "boolean"}

    def test_list(self) -> None:
        schema = python_type_to_json_schema(list[str])
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "string"

    def test_nested_list(self) -> None:
        schema = python_type_to_json_schema(list[list[int]])
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "array"
        assert schema["items"]["items"]["type"] == "integer"

    def test_dict(self) -> None:
        schema = python_type_to_json_schema(dict[str, int])
        assert schema["type"] == "object"

    def test_any(self) -> None:
        schema = python_type_to_json_schema(Any)
        assert schema["type"] == "string"

    def test_optional_type(self) -> None:
        # Optional[str] should unwrap to string
        schema = python_type_to_json_schema(Optional[str])  # noqa: UP045 - testing Optional syntax
        assert schema["type"] == "string"

    def test_optional_list(self) -> None:
        # Optional[list[int]] should unwrap to array of integers
        schema = python_type_to_json_schema(Optional[list[int]])  # noqa: UP045 - testing Optional
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "integer"

    def test_union_with_none_syntax(self) -> None:
        # str | None should behave like Optional[str]
        schema = python_type_to_json_schema(str | None)
        assert schema["type"] == "string"

    def test_union_multiple_types_raises(self) -> None:
        # Union[str, int] (without None) should raise
        with pytest.raises(TypeError, match="Union types with multiple non-None"):
            python_type_to_json_schema(Union[str, int])  # noqa: UP007 - testing Union syntax

    def test_unsupported_type(self) -> None:
        with pytest.raises(TypeError, match="Unsupported type"):
            python_type_to_json_schema(set)  # type: ignore[arg-type]

    def test_literal_int(self) -> None:
        schema = python_type_to_json_schema(Literal[1, 2, 3, 4, 5])
        assert schema["type"] == "integer"
        assert schema["enum"] == [1, 2, 3, 4, 5]

    def test_literal_str(self) -> None:
        schema = python_type_to_json_schema(Literal["low", "medium", "high"])
        assert schema["type"] == "string"
        assert schema["enum"] == ["low", "medium", "high"]

    def test_literal_bool(self) -> None:
        schema = python_type_to_json_schema(Literal[True, False])
        assert schema["type"] == "boolean"
        assert schema["enum"] == [True, False]

    def test_enum_str_values(self) -> None:
        class Priority(enum.Enum):
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"

        schema = python_type_to_json_schema(Priority)
        assert schema["type"] == "string"
        assert schema["enum"] == ["low", "medium", "high"]

    def test_enum_int_values(self) -> None:
        class Rating(enum.Enum):
            ONE = 1
            TWO = 2
            THREE = 3

        schema = python_type_to_json_schema(Rating)
        assert schema["type"] == "integer"
        assert schema["enum"] == [1, 2, 3]

    def test_nested_dataclass(self) -> None:
        @dataclass
        class Inner:
            value: int

        schema = python_type_to_json_schema(Inner)
        assert schema["type"] == "object"
        assert schema["properties"]["value"]["type"] == "integer"
        assert schema["required"] == ["value"]

    def test_list_of_dataclass(self) -> None:
        @dataclass
        class Item:
            name: str

        schema = python_type_to_json_schema(list[Item])
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "object"
        assert schema["items"]["properties"]["name"]["type"] == "string"

    def test_literal_mixed_types_raises(self) -> None:
        with pytest.raises(TypeError, match="Mixed types in Literal"):
            python_type_to_json_schema(Literal["a", 1])  # type: ignore[arg-type]

    def test_literal_mixed_int_bool_raises(self) -> None:
        # bool is subclass of int, but we should still reject mixing them
        with pytest.raises(TypeError, match="Mixed types in Literal"):
            python_type_to_json_schema(Literal[1, True])  # type: ignore[arg-type]

    def test_recursive_dataclass_raises(self) -> None:
        # Test cycle detection via the internal function directly.
        # Forward references in local scopes don't work with get_type_hints,
        # but cycle detection works when called with already-seen types.
        from llm_saia.core.schema import _build_object_schema

        @dataclass
        class Node:
            value: int

        # Simulate cycle detection by pre-populating the seen set
        with pytest.raises(TypeError, match="Recursive type detected"):
            _build_object_schema(Node, seen={Node})


class TestDataclassToJsonSchema:
    def test_simple_dataclass(self) -> None:
        @dataclass
        class Simple:
            """A simple test dataclass."""

            name: str
            value: int

        schema = dataclass_to_json_schema(Simple)

        assert schema["name"] == "Simple"
        assert schema["description"] == "A simple test dataclass."
        assert schema["schema"]["type"] == "object"
        assert schema["schema"]["properties"]["name"]["type"] == "string"
        assert schema["schema"]["properties"]["value"]["type"] == "integer"
        assert "name" in schema["schema"]["required"]
        assert "value" in schema["schema"]["required"]

    def test_dataclass_with_defaults(self) -> None:
        @dataclass
        class WithDefaults:
            required_field: str
            optional_field: int = 42

        schema = dataclass_to_json_schema(WithDefaults)

        assert "required_field" in schema["schema"]["required"]
        assert "optional_field" not in schema["schema"]["required"]

    def test_dataclass_with_default_factory(self) -> None:
        @dataclass
        class WithFactory:
            required: str
            items: list[str] = field(default_factory=list)

        schema = dataclass_to_json_schema(WithFactory)

        assert "required" in schema["schema"]["required"]
        assert "items" not in schema["schema"]["required"]

    def test_dataclass_with_complex_types(self) -> None:
        @dataclass
        class Complex:
            strings: list[str]
            mapping: dict[str, int]

        schema = dataclass_to_json_schema(Complex)

        assert schema["schema"]["properties"]["strings"]["type"] == "array"
        assert schema["schema"]["properties"]["mapping"]["type"] == "object"

    def test_non_dataclass_raises(self) -> None:
        with pytest.raises(TypeError, match="Schema must be a dataclass"):
            dataclass_to_json_schema(str)  # type: ignore[arg-type]

    def test_dataclass_without_docstring(self) -> None:
        @dataclass
        class NoDoc:
            field: str

        schema = dataclass_to_json_schema(NoDoc)
        # Dataclass without explicit docstring gets auto-generated repr-style doc
        assert "NoDoc" in schema["description"]


class TestParseJsonToDataclass:
    def test_parse_simple(self) -> None:
        @dataclass
        class Simple:
            name: str
            value: int

        result = parse_json_to_dataclass({"name": "test", "value": 42}, Simple)

        assert isinstance(result, Simple)
        assert result.name == "test"
        assert result.value == 42

    def test_parse_with_list(self) -> None:
        @dataclass
        class WithList:
            items: list[str]

        result = parse_json_to_dataclass({"items": ["a", "b", "c"]}, WithList)

        assert result.items == ["a", "b", "c"]

    def test_parse_non_dict_raises(self) -> None:
        @dataclass
        class Schema:
            field: str

        with pytest.raises(TypeError, match="Expected dict"):
            parse_json_to_dataclass(["not", "a", "dict"], Schema)

    def test_parse_missing_field_raises(self) -> None:
        @dataclass
        class Required:
            required_field: str

        with pytest.raises(TypeError):
            parse_json_to_dataclass({}, Required)

    def test_parse_ignores_extra_fields(self) -> None:
        """Extra fields from LLM response should be ignored, not cause errors."""

        @dataclass
        class Simple:
            name: str
            value: int

        # Simulate LLM returning extra fields beyond the schema
        data = {
            "name": "test",
            "value": 42,
            "_note": "This is an extra field from the LLM",
            "confidence": 0.95,
        }
        result = parse_json_to_dataclass(data, Simple)

        assert isinstance(result, Simple)
        assert result.name == "test"
        assert result.value == 42
        # Extra fields should not be present
        assert not hasattr(result, "_note")
        assert not hasattr(result, "confidence")

    def test_parse_nested_dataclass(self) -> None:
        @dataclass
        class Inner:
            value: int

        @dataclass
        class Outer:
            name: str
            inner: Inner

        data = {"name": "test", "inner": {"value": 42}}
        result = parse_json_to_dataclass(data, Outer)

        assert isinstance(result, Outer)
        assert result.name == "test"
        assert isinstance(result.inner, Inner)
        assert result.inner.value == 42

    def test_parse_list_of_dataclass(self) -> None:
        @dataclass
        class Item:
            name: str
            score: int

        @dataclass
        class Container:
            items: list[Item]

        data = {"items": [{"name": "a", "score": 1}, {"name": "b", "score": 2}]}
        result = parse_json_to_dataclass(data, Container)

        assert isinstance(result, Container)
        assert len(result.items) == 2
        assert isinstance(result.items[0], Item)
        assert result.items[0].name == "a"
        assert result.items[1].score == 2

    def test_parse_enum(self) -> None:
        class Priority(enum.Enum):
            LOW = "low"
            HIGH = "high"

        @dataclass
        class Task:
            name: str
            priority: Priority

        data = {"name": "test", "priority": "high"}
        result = parse_json_to_dataclass(data, Task)

        assert isinstance(result, Task)
        assert result.priority == Priority.HIGH

    def test_parse_literal(self) -> None:
        @dataclass
        class Rating:
            score: Literal[1, 2, 3, 4, 5]

        data = {"score": 4}
        result = parse_json_to_dataclass(data, Rating)

        assert isinstance(result, Rating)
        assert result.score == 4

    def test_parse_optional_nested(self) -> None:
        @dataclass
        class Inner:
            value: int

        @dataclass
        class Outer:
            inner: Inner | None

        # With value
        result = parse_json_to_dataclass({"inner": {"value": 42}}, Outer)
        assert result.inner is not None
        assert result.inner.value == 42

        # With None
        result = parse_json_to_dataclass({"inner": None}, Outer)
        assert result.inner is None

    def test_parse_nested_wrong_type_raises(self) -> None:
        @dataclass
        class Inner:
            value: int

        @dataclass
        class Outer:
            inner: Inner

        with pytest.raises(TypeError, match="Expected dict for dataclass field"):
            parse_json_to_dataclass({"inner": "not a dict"}, Outer)

    def test_parse_list_wrong_type_raises(self) -> None:
        @dataclass
        class Container:
            items: list[str]

        with pytest.raises(TypeError, match="Expected list for field type"):
            parse_json_to_dataclass({"items": "not a list"}, Container)

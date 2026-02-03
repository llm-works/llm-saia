"""Tests for backend implementations using SAIA core types."""

import pytest

from llm_saia.backends._schema import (
    dataclass_to_json_schema,
    parse_json_to_dataclass,
    python_type_to_json_schema,
)
from llm_saia.core.types import Critique, VerifyResult

pytestmark = pytest.mark.unit


class TestDataclassToToolSchema:
    """Test schema conversion with actual SAIA types."""

    def test_simple_dataclass(self) -> None:
        schema = dataclass_to_json_schema(VerifyResult)

        assert schema["name"] == "VerifyResult"
        assert "schema" in schema
        assert schema["schema"]["type"] == "object"
        assert "passed" in schema["schema"]["properties"]
        assert "reason" in schema["schema"]["properties"]

    def test_required_fields(self) -> None:
        schema = dataclass_to_json_schema(VerifyResult)

        # Both fields are required (no defaults)
        assert "passed" in schema["schema"]["required"]
        assert "reason" in schema["schema"]["required"]

    def test_complex_dataclass(self) -> None:
        schema = dataclass_to_json_schema(Critique)

        assert schema["name"] == "Critique"
        props = schema["schema"]["properties"]
        assert props["counter_argument"]["type"] == "string"
        assert props["weaknesses"]["type"] == "array"
        assert props["strength"]["type"] == "number"

    def test_non_dataclass_raises(self) -> None:
        with pytest.raises(TypeError):
            dataclass_to_json_schema(str)  # type: ignore[arg-type]


class TestPythonTypeToJsonSchema:
    """Test type conversion with standard types."""

    def test_primitives(self) -> None:
        assert python_type_to_json_schema(str) == {"type": "string"}
        assert python_type_to_json_schema(int) == {"type": "integer"}
        assert python_type_to_json_schema(float) == {"type": "number"}
        assert python_type_to_json_schema(bool) == {"type": "boolean"}

    def test_list(self) -> None:
        schema = python_type_to_json_schema(list[str])
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "string"

    def test_dict(self) -> None:
        schema = python_type_to_json_schema(dict[str, int])
        assert schema["type"] == "object"


class TestParseToolResult:
    """Test JSON to dataclass parsing with SAIA types."""

    def test_parse_verify_result(self) -> None:
        data = {"passed": True, "reason": "looks good"}
        result = parse_json_to_dataclass(data, VerifyResult)

        assert isinstance(result, VerifyResult)
        assert result.passed is True
        assert result.reason == "looks good"

    def test_parse_critique(self) -> None:
        data = {
            "counter_argument": "counter",
            "weaknesses": ["w1", "w2"],
            "strength": 0.7,
        }
        result = parse_json_to_dataclass(data, Critique)

        assert isinstance(result, Critique)
        assert result.counter_argument == "counter"
        assert result.weaknesses == ["w1", "w2"]
        assert result.strength == 0.7

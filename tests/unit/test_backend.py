"""Tests for backend implementations."""

import pytest

from llm_saia.backends.anthropic import (
    _dataclass_to_tool_schema,
    _parse_tool_result,
    _python_type_to_json_schema,
)
from llm_saia.core.types import Critique, VerifyResult

pytestmark = pytest.mark.unit


class TestDataclassToToolSchema:
    def test_simple_dataclass(self) -> None:
        schema = _dataclass_to_tool_schema(VerifyResult)

        assert schema["name"] == "VerifyResult"
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"
        assert "passed" in schema["input_schema"]["properties"]
        assert "reason" in schema["input_schema"]["properties"]

    def test_required_fields(self) -> None:
        schema = _dataclass_to_tool_schema(VerifyResult)

        # Both fields are required (no defaults)
        assert "passed" in schema["input_schema"]["required"]
        assert "reason" in schema["input_schema"]["required"]

    def test_complex_dataclass(self) -> None:
        schema = _dataclass_to_tool_schema(Critique)

        assert schema["name"] == "Critique"
        props = schema["input_schema"]["properties"]
        assert props["counter_argument"]["type"] == "string"
        assert props["weaknesses"]["type"] == "array"
        assert props["strength"]["type"] == "number"

    def test_non_dataclass_raises(self) -> None:
        with pytest.raises(TypeError):
            _dataclass_to_tool_schema(str)  # type: ignore[arg-type]


class TestPythonTypeToJsonSchema:
    def test_primitives(self) -> None:
        assert _python_type_to_json_schema(str) == {"type": "string"}
        assert _python_type_to_json_schema(int) == {"type": "integer"}
        assert _python_type_to_json_schema(float) == {"type": "number"}
        assert _python_type_to_json_schema(bool) == {"type": "boolean"}

    def test_list(self) -> None:
        schema = _python_type_to_json_schema(list[str])
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "string"

    def test_dict(self) -> None:
        schema = _python_type_to_json_schema(dict[str, int])
        assert schema["type"] == "object"


class TestParseToolResult:
    def test_parse_verify_result(self) -> None:
        data = {"passed": True, "reason": "looks good"}
        result = _parse_tool_result(data, VerifyResult)

        assert isinstance(result, VerifyResult)
        assert result.passed is True
        assert result.reason == "looks good"

    def test_parse_critique(self) -> None:
        data = {
            "counter_argument": "counter",
            "weaknesses": ["w1", "w2"],
            "strength": 0.7,
        }
        result = _parse_tool_result(data, Critique)

        assert isinstance(result, Critique)
        assert result.counter_argument == "counter"
        assert result.weaknesses == ["w1", "w2"]
        assert result.strength == 0.7

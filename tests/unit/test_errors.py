"""Tests for SAIA exception classes."""

import pytest

from llm_saia import (
    BackendError,
    ConfigurationError,
    Error,
    StructuredOutputError,
    ToolExecutionError,
    TruncatedResponseError,
)

pytestmark = pytest.mark.unit


class TestError:
    """Tests for base Error class."""

    def test_base_error(self) -> None:
        err = Error("base error")
        assert str(err) == "base error"
        assert isinstance(err, Exception)


class TestStructuredOutputError:
    """Tests for StructuredOutputError."""

    def test_message_only(self) -> None:
        err = StructuredOutputError("parse failed")
        assert str(err) == "parse failed"
        assert err.raw_content is None
        assert err.schema_name is None
        assert err.parse_error is None

    def test_with_all_kwargs(self) -> None:
        err = StructuredOutputError(
            "parse failed",
            raw_content='{"incomplete": ',
            schema_name="VerifyResult",
            parse_error="Expecting value: line 1 column 15",
        )
        assert str(err) == "parse failed"
        assert err.raw_content == '{"incomplete": '
        assert err.schema_name == "VerifyResult"
        assert err.parse_error == "Expecting value: line 1 column 15"

    def test_inheritance(self) -> None:
        err = StructuredOutputError("test")
        assert isinstance(err, Error)
        assert isinstance(err, Exception)


class TestTruncatedResponseError:
    """Tests for TruncatedResponseError."""

    def test_default_message(self) -> None:
        err = TruncatedResponseError()
        assert "truncated" in str(err).lower()
        assert "max_call_tokens" in str(err)

    def test_custom_message(self) -> None:
        err = TruncatedResponseError("custom truncation message")
        assert str(err) == "custom truncation message"

    def test_with_all_kwargs(self) -> None:
        err = TruncatedResponseError(
            "response cut off",
            raw_content='{"partial": "da',
            schema_name="Critique",
            parse_error="Unterminated string",
        )
        assert str(err) == "response cut off"
        assert err.raw_content == '{"partial": "da'
        assert err.schema_name == "Critique"
        assert err.parse_error == "Unterminated string"

    def test_inheritance(self) -> None:
        err = TruncatedResponseError()
        assert isinstance(err, StructuredOutputError)
        assert isinstance(err, Error)


class TestToolExecutionError:
    """Tests for ToolExecutionError."""

    def test_message_only(self) -> None:
        err = ToolExecutionError("tool failed")
        assert str(err) == "tool failed"
        assert err.tool_name is None
        assert err.arguments is None
        assert err.cause is None

    def test_with_all_kwargs(self) -> None:
        cause = ValueError("invalid input")
        err = ToolExecutionError(
            "tool failed",
            tool_name="read_file",
            arguments={"path": "/etc/passwd"},
            cause=cause,
        )
        assert str(err) == "tool failed"
        assert err.tool_name == "read_file"
        assert err.arguments == {"path": "/etc/passwd"}
        assert err.cause is cause

    def test_inheritance(self) -> None:
        err = ToolExecutionError("test")
        assert isinstance(err, Error)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_message_only(self) -> None:
        err = ConfigurationError("invalid config")
        assert str(err) == "invalid config"
        assert err.field is None
        assert err.value is None
        assert err.reason is None

    def test_with_all_kwargs(self) -> None:
        err = ConfigurationError(
            "invalid config",
            field="max_iterations",
            value=-1,
            reason="must be positive",
        )
        assert str(err) == "invalid config"
        assert err.field == "max_iterations"
        assert err.value == -1
        assert err.reason == "must be positive"

    def test_inheritance(self) -> None:
        err = ConfigurationError("test")
        assert isinstance(err, Error)


class TestBackendError:
    """Tests for BackendError."""

    def test_message_only(self) -> None:
        err = BackendError("backend failed")
        assert str(err) == "backend failed"
        assert err.status_code is None
        assert err.response_body is None
        assert err.cause is None

    def test_with_all_kwargs(self) -> None:
        cause = ConnectionError("connection refused")
        err = BackendError(
            "backend failed",
            status_code=503,
            response_body='{"error": "service unavailable"}',
            cause=cause,
        )
        assert str(err) == "backend failed"
        assert err.status_code == 503
        assert err.response_body == '{"error": "service unavailable"}'
        assert err.cause is cause

    def test_inheritance(self) -> None:
        err = BackendError("test")
        assert isinstance(err, Error)


class TestExceptionRaising:
    """Tests that exceptions can be raised and caught properly."""

    def test_raise_and_catch_structured_output_error(self) -> None:
        with pytest.raises(StructuredOutputError) as exc_info:
            raise StructuredOutputError("bad output", schema_name="Test")
        assert exc_info.value.schema_name == "Test"

    def test_raise_and_catch_truncated_as_structured(self) -> None:
        """TruncatedResponseError should be catchable as StructuredOutputError."""
        with pytest.raises(StructuredOutputError):
            raise TruncatedResponseError()

    def test_raise_and_catch_as_base_error(self) -> None:
        """All errors should be catchable as Error."""
        errors = [
            StructuredOutputError("test"),
            TruncatedResponseError(),
            ToolExecutionError("test"),
            ConfigurationError("test"),
            BackendError("test"),
        ]
        for err in errors:
            with pytest.raises(Error):
                raise err

"""SAIA exception classes."""

from __future__ import annotations


class Error(Exception):
    """Base exception for all SAIA errors."""


# --- Structured Output Errors ---


class StructuredOutputError(Error):
    """LLM returned invalid structured output.

    Attributes:
        raw_content: The raw response content that failed to parse.
        schema_name: Name of the expected schema/dataclass.
        parse_error: The underlying parse error message.
    """

    def __init__(
        self,
        message: str,
        *,
        raw_content: str | None = None,
        schema_name: str | None = None,
        parse_error: str | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_content = raw_content
        self.schema_name = schema_name
        self.parse_error = parse_error


class TruncatedResponseError(StructuredOutputError):
    """LLM response was truncated, likely due to token limits.

    This typically happens when max_tokens is too low for the requested output.
    Consider increasing max_call_tokens in your config.
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        raw_content: str | None = None,
        schema_name: str | None = None,
        parse_error: str | None = None,
    ) -> None:
        if message is None:
            message = (
                "LLM response appears truncated. "
                "Consider increasing max_call_tokens in your config."
            )
        super().__init__(
            message,
            raw_content=raw_content,
            schema_name=schema_name,
            parse_error=parse_error,
        )


# --- Execution Errors ---


class ToolExecutionError(Error):
    """Tool execution failed.

    Attributes:
        tool_name: Name of the tool that failed.
        arguments: Arguments passed to the tool.
        cause: The underlying exception that caused the failure.
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        arguments: dict[str, object] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.arguments = arguments
        self.cause = cause


class ConfigurationError(Error):
    """Invalid SAIA configuration.

    Attributes:
        field: The configuration field that is invalid.
        value: The invalid value.
        reason: Why the value is invalid.
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        value: object = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.value = value
        self.reason = reason


# --- Backend Errors ---


class BackendError(Error):
    """Backend communication failed.

    Attributes:
        status_code: HTTP status code if applicable.
        response_body: Raw response body if available.
        cause: The underlying exception that caused the failure.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.cause = cause

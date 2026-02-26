"""Logger interface for SAIA instrumentation."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Logger(Protocol):
    """Logger interface for SAIA instrumentation.

    Implementations (like appinfra's Logger) satisfy this via structural typing.
    No explicit inheritance required - just implement the methods.
    """

    def trace(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        """Log at TRACE level (more verbose than DEBUG)."""
        ...

    def debug(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        """Log at DEBUG level."""
        ...

    def info(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        """Log at INFO level."""
        ...

    def warning(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        """Log at WARNING level."""
        ...

    def error(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        """Log at ERROR level."""
        ...


class NullLogger:
    """No-op logger for when logging is disabled.

    Satisfies Logger protocol without doing anything.
    """

    def trace(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        """No-op."""
        pass

    def debug(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        """No-op."""
        pass

    def info(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        """No-op."""
        pass

    def warning(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        """No-op."""
        pass

    def error(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        """No-op."""
        pass

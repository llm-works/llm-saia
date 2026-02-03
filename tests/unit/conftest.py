"""Pytest configuration and fixtures."""

from typing import Any, TypeVar

import pytest

from llm_saia.core.protocols import SAIABackend
from llm_saia.core.types import Critique, Evidence, VerifyResult

T = TypeVar("T")


class MockBackend(SAIABackend):
    """Mock backend for testing that returns predetermined responses."""

    def __init__(self) -> None:
        self.last_prompt: str = ""
        self._complete_response: str = "mock response"
        self._structured_responses: dict[type, Any] = {
            VerifyResult: VerifyResult(passed=True, reason="test reason"),
            Critique: Critique(
                counter_argument="test counter",
                weaknesses=["weakness 1"],
                strength=0.5,
            ),
            Evidence: Evidence(
                content="test content",
                source="test source",
                direction="supports",
                strength=0.8,
            ),
        }

    def set_complete_response(self, response: str) -> None:
        """Set the response for complete() calls."""
        self._complete_response = response

    def set_structured_response(self, schema: type[T], response: T) -> None:
        """Set the response for complete_structured() calls."""
        self._structured_responses[schema] = response

    async def complete(self, prompt: str) -> str:
        """Return predetermined response."""
        self.last_prompt = prompt
        return self._complete_response

    async def complete_structured(self, prompt: str, schema: type[T]) -> T:
        """Return predetermined structured response."""
        self.last_prompt = prompt
        if schema in self._structured_responses:
            return self._structured_responses[schema]
        raise ValueError(f"No mock response configured for {schema}")


@pytest.fixture
def mock_backend() -> MockBackend:
    """Provide a mock backend for tests."""
    return MockBackend()

"""Pytest configuration and fixtures."""

from __future__ import annotations

from typing import Any, TypeVar

import pytest

from llm_saia.core.protocols import SAIABackend
from llm_saia.core.types import (
    AgentResponse,
    ChooseResult,
    ClassifyResult,
    ConfirmResult,
    Critique,
    Evidence,
    Message,
    RunConfig,
    ToolDef,
    VerifyResult,
)

T = TypeVar("T")


def _default_structured_responses() -> dict[type, Any]:
    """Default structured responses for mock backend."""
    return {
        VerifyResult: VerifyResult(passed=True, reason="test reason"),
        Critique: Critique(
            counter_argument="test counter", weaknesses=["weakness 1"], strength=0.5
        ),
        Evidence: Evidence(
            content="test content", source="test source", direction="supports", strength=0.8
        ),
        ClassifyResult: ClassifyResult(
            category="test_category", confidence=0.9, reason="test classification reason"
        ),
        ConfirmResult: ConfirmResult(confirmed=True, reason="test confirmation reason"),
        ChooseResult: ChooseResult(choice="option_a", reason="test choice reason"),
    }


class MockBackend(SAIABackend):
    """Mock backend for testing that returns predetermined responses."""

    def __init__(self) -> None:
        self.last_prompt: str = ""
        self.last_messages: list[Message] = []
        self.last_tools: list[ToolDef] = []
        self._complete_response: str = "mock response"
        self._tool_responses: list[AgentResponse] = []
        self._structured_responses: dict[type, Any] = _default_structured_responses()
        self._run: RunConfig | None = None

    def set_run_config(self, run: RunConfig) -> None:
        """Set the run configuration."""
        self._run = run

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

    async def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system: str | None = None,
    ) -> AgentResponse:
        """Return predetermined response with optional tool calls."""
        self.last_prompt = messages[-1].content if messages else ""
        self.last_messages = messages
        self.last_tools = tools

        # Check if we have a queued response
        if self._tool_responses:
            return self._tool_responses.pop(0)

        # Default: return text response (no tool calls)
        return AgentResponse(
            content=self._complete_response,
            tool_calls=[],
            stop_reason="end_turn",
        )

    def queue_tool_response(self, response: AgentResponse) -> None:
        """Queue a response for complete_with_tools."""
        if not hasattr(self, "_tool_responses"):
            self._tool_responses: list[AgentResponse] = []
        self._tool_responses.append(response)

    async def close(self) -> None:
        """No-op for mock backend."""
        pass


@pytest.fixture
def mock_backend() -> MockBackend:
    """Provide a mock backend for tests."""
    return MockBackend()

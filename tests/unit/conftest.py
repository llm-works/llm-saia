"""Pytest configuration and fixtures."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import pytest

from llm_saia import SAIA
from llm_saia.core.backend import Backend
from llm_saia.core.config import CallOptions, Config, TerminalConfig
from llm_saia.core.logger import Logger
from llm_saia.core.types import (
    AgentResponse,
    Message,
    ToolDef,
)

T = TypeVar("T")


def _default_structured_responses() -> dict[str, dict[str, Any]]:
    """Default structured responses for mock backend (as JSON-serializable dicts)."""
    return {
        "VerifyResult": {"passed": True, "reason": "test reason"},
        "Critique": {
            "counter_argument": "test counter",
            "weaknesses": ["weakness 1"],
            "strength": 0.5,
        },
        "Evidence": {
            "content": "test content",
            "source": "test source",
            "direction": "supports",
            "strength": 0.8,
        },
        "ClassifyResult": {
            "category": "test_category",
            "confidence": 0.9,
            "reason": "test classification reason",
        },
        "ChooseResult": {"choice": "option_a", "reason": "test choice reason"},
        # Generic fallbacks
        "DecomposeResult": {"subtasks": ["task 1", "task 2"]},
    }


class MockBackend(Backend):
    """Mock backend for testing that returns predetermined responses."""

    def __init__(self) -> None:
        self.last_messages: list[Message] = []
        self.last_system: str | None = None
        self.last_tools: list[ToolDef] | None = None
        self.last_response_schema: dict[str, Any] | None = None
        self.last_temperature: float | None = None
        self._response_content: str = "mock response"
        self._queued_responses: list[AgentResponse] = []
        self._structured_responses: dict[str, dict[str, Any]] = _default_structured_responses()
        self._queued_structured: dict[str, list[dict[str, Any]]] = {}

    @property
    def last_prompt(self) -> str:
        """Get the content of the last user message (for backwards compatibility)."""
        for msg in reversed(self.last_messages):
            if msg.role == "user":
                return msg.content
        return ""

    def set_complete_response(self, response: str) -> None:
        """Set the response content for chat() calls."""
        self._response_content = response

    def set_structured_response(self, schema: type | str, response: Any) -> None:
        """Set the response for structured output requests.

        Args:
            schema: Either a dataclass type or schema name string.
            response: Either a dataclass instance or a dict.
        """
        import dataclasses

        if isinstance(schema, str):
            self._structured_responses[schema] = response
        else:
            # schema is a type, response is an instance
            schema_name = schema.__name__
            if dataclasses.is_dataclass(response):
                self._structured_responses[schema_name] = dataclasses.asdict(response)
            else:
                self._structured_responses[schema_name] = response

    def queue_structured_response(self, schema: type | str, response: Any) -> None:
        """Queue a structured response to be consumed in order.

        Use this when you need different responses for the same schema type
        across multiple calls. Queued responses are consumed before falling
        back to set_structured_response defaults.

        Args:
            schema: Either a dataclass type or schema name string.
            response: Either a dataclass instance or a dict.
        """
        import dataclasses

        if isinstance(schema, str):
            schema_name = schema
        else:
            schema_name = schema.__name__

        if dataclasses.is_dataclass(response):
            response_dict = dataclasses.asdict(response)
        else:
            response_dict = response

        if schema_name not in self._queued_structured:
            self._queued_structured[schema_name] = []
        self._queued_structured[schema_name].append(response_dict)

    def queue_response(self, response: AgentResponse) -> None:
        """Queue a response for the next chat() call."""
        self._queued_responses.append(response)

    # Backwards compatibility alias
    def queue_tool_response(self, response: AgentResponse) -> None:
        """Queue a response for the next chat() call (alias for queue_response)."""
        self._queued_responses.append(response)

    def _make_response(self, content: str) -> AgentResponse:
        """Create a simple AgentResponse with given content."""
        return AgentResponse(content=content, tool_calls=[], finish_reason="end_turn")

    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AgentResponse:
        """Return predetermined response."""
        self.last_messages = messages
        self.last_system = system
        self.last_tools = tools
        self.last_response_schema = response_schema
        self.last_temperature = temperature

        # Check structured output first (these calls have response_schema set)
        if response_schema:
            schema_name = response_schema.get("name", "")
            # Check queued structured responses first
            if schema_name in self._queued_structured and self._queued_structured[schema_name]:
                response_dict = self._queued_structured[schema_name].pop(0)
                return self._make_response(json.dumps(response_dict))
            # Fall back to default responses
            if schema_name in self._structured_responses:
                return self._make_response(json.dumps(self._structured_responses[schema_name]))
            return self._make_response("{}")

        # Regular chat calls use queued responses
        if self._queued_responses:
            return self._queued_responses.pop(0)

        return self._make_response(self._response_content)


@pytest.fixture
def mock_backend() -> MockBackend:
    """Provide a mock backend for tests."""
    return MockBackend()


def make_saia(
    backend: Backend,
    tools: list[ToolDef] | None = None,
    executor: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
    system: str | None = None,
    terminal_tool: str | None = None,
    lg: Logger | None = None,
) -> SAIA:
    """Helper to create SAIA instances for tests."""
    terminal = TerminalConfig(tool=terminal_tool) if terminal_tool else None
    call = CallOptions(system=system) if system else None
    config = Config(
        backend=backend,
        tools=tools or [],
        executor=executor,
        call=call,
        terminal=terminal,
        lg=lg,
    )
    return SAIA(config)

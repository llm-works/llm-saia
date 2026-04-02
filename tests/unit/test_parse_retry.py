"""Tests for parse retry with feedback feature."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from llm_saia.core.backend import AgentResponse
from llm_saia.core.config import CallOptions, Config
from llm_saia.core.conversation import Message
from llm_saia.core.errors import StructuredOutputError
from llm_saia.core.types import ToolDef
from llm_saia.verbs import Extract
from tests.unit.conftest import MockBackend

pytestmark = pytest.mark.unit


class SequencedMockBackend(MockBackend):
    """Mock backend that can return a sequence of raw responses for structured calls."""

    def __init__(self) -> None:
        super().__init__()
        self._raw_sequence: list[str] = []
        self._sequence_index = 0

    @property
    def call_count(self) -> int:
        """Number of chat calls made to this backend."""
        return self._sequence_index

    def queue_raw_response(self, content: str) -> None:
        """Queue a raw response (can be invalid JSON)."""
        self._raw_sequence.append(content)

    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AgentResponse:
        """Return queued raw responses before falling back to normal behavior."""
        self.last_messages = messages
        self.last_system = system
        self.last_tools = tools
        self.last_response_schema = response_schema
        self.last_temperature = temperature

        # Use raw sequence first if available
        if self._sequence_index < len(self._raw_sequence):
            content = self._raw_sequence[self._sequence_index]
            self._sequence_index += 1
            return self._make_response(content)

        # Fall back to normal behavior
        return await super().chat(messages, system, tools, response_schema, max_tokens, temperature)


def make_config(backend: MockBackend, call: CallOptions | None = None) -> Config:
    """Create a Config with no tools (direct backend calls)."""
    return Config(backend=backend, tools=[], executor=None, call=call)


@dataclass
class SimpleSchema:
    """Test schema for parse retry tests."""

    value: str
    score: int


class TestParseRetry:
    """Tests for parse retry with feedback feature."""

    async def test_retry_succeeds_on_second_attempt(self) -> None:
        """Parse retry should succeed when second attempt returns valid JSON."""
        backend = SequencedMockBackend()
        # First call returns invalid JSON
        backend.queue_raw_response("not valid json {")
        # Second call returns valid JSON
        backend.queue_raw_response(json.dumps({"value": "success", "score": 42}))

        call = CallOptions(parse_retries=1)
        extract = Extract(make_config(backend, call=call))
        result = await extract("test content", SimpleSchema)

        assert result.value == "success"
        assert result.score == 42

    async def test_retry_exhausted_raises_error(self) -> None:
        """Should raise StructuredOutputError when all retries are exhausted."""
        backend = SequencedMockBackend()
        # Both attempts return invalid JSON
        backend.queue_raw_response("invalid 1")
        backend.queue_raw_response("invalid 2")

        call = CallOptions(parse_retries=1)
        extract = Extract(make_config(backend, call=call))
        with pytest.raises(StructuredOutputError):
            await extract("test content", SimpleSchema)

    async def test_parse_retries_zero_disables_retry(self) -> None:
        """With parse_retries=0, should fail immediately without retry."""
        backend = SequencedMockBackend()
        # First call returns invalid JSON
        backend.queue_raw_response("invalid json")
        # Second call would succeed (but should never be reached)
        backend.queue_raw_response(json.dumps({"value": "success", "score": 1}))

        call = CallOptions(parse_retries=0)
        extract = Extract(make_config(backend, call=call))

        with pytest.raises(StructuredOutputError):
            await extract("test content", SimpleSchema)

        # Verify only one call was made (sequence index should be 1)
        assert backend.call_count == 1

    async def test_retry_prompt_includes_error_feedback(self) -> None:
        """Retry prompt should include the parse error from the first attempt."""
        backend = SequencedMockBackend()
        # First call returns invalid JSON
        backend.queue_raw_response("{incomplete")
        # Second call returns valid JSON
        backend.queue_raw_response(json.dumps({"value": "ok", "score": 1}))

        call = CallOptions(parse_retries=1)
        extract = Extract(make_config(backend, call=call))
        await extract("test content", SimpleSchema)

        # Check the retry prompt (second message sent to backend)
        assert len(backend.last_messages) > 0
        retry_prompt = backend.last_prompt

        # Should include feedback about the error
        assert "could not be parsed" in retry_prompt
        assert "{incomplete" in retry_prompt  # Original invalid response
        assert "SimpleSchema" in retry_prompt  # Schema name

    async def test_multiple_retries_allowed(self) -> None:
        """Should support multiple retry attempts via parse_retries setting."""
        backend = SequencedMockBackend()
        # First two calls return invalid JSON
        backend.queue_raw_response("invalid 1")
        backend.queue_raw_response("invalid 2")
        # Third call succeeds
        backend.queue_raw_response(json.dumps({"value": "third", "score": 3}))

        call = CallOptions(parse_retries=2)  # 2 retries = 3 total attempts
        extract = Extract(make_config(backend, call=call))
        result = await extract("test content", SimpleSchema)

        assert result.value == "third"
        assert result.score == 3
        assert backend.call_count == 3

    async def test_first_success_no_retry(self) -> None:
        """When first attempt succeeds, no retry should happen."""
        backend = SequencedMockBackend()
        # First call returns valid JSON
        backend.queue_raw_response(json.dumps({"value": "immediate", "score": 100}))

        extract = Extract(make_config(backend))
        result = await extract("test content", SimpleSchema)

        assert result.value == "immediate"
        assert result.score == 100
        assert backend.call_count == 1  # Only one call made

    async def test_with_parse_retries_fluent_api(self) -> None:
        """Test using with_parse_retries() fluent API."""
        from llm_saia import SAIA

        backend = SequencedMockBackend()
        # First call returns invalid JSON
        backend.queue_raw_response("bad json")
        # Second call succeeds
        backend.queue_raw_response(json.dumps({"value": "fluent", "score": 5}))

        config = Config(backend=backend, tools=[], executor=None)
        saia = SAIA(config)

        # Use fluent API to set parse retries
        result = await saia.with_parse_retries(1).extract("content", SimpleSchema)

        assert result.value == "fluent"
        assert result.score == 5

    def test_negative_parse_retries_rejected(self) -> None:
        """parse_retries must be non-negative."""
        with pytest.raises(ValueError, match="parse_retries must be non-negative"):
            CallOptions(parse_retries=-1)

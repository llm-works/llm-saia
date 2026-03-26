"""Tests for output guards feature."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from llm_saia import OutputGuard, OutputGuardError
from llm_saia.core.backend import AgentResponse, Message
from llm_saia.core.config import Config
from llm_saia.core.types import ToolDef
from llm_saia.guards import (
    ascii_only,
    english_only,
    max_length,
    no_emoji,
    no_markdown,
    no_preamble,
)
from llm_saia.verbs import Extract
from tests.unit.conftest import MockBackend

pytestmark = pytest.mark.unit


class SequencedMockBackend(MockBackend):
    """Mock backend that can return a sequence of responses."""

    def __init__(self) -> None:
        super().__init__()
        self._response_sequence: list[str] = []
        self._sequence_index = 0

    @property
    def call_count(self) -> int:
        """Number of chat calls made to this backend."""
        return self._sequence_index

    def queue_json_response(self, content: str) -> None:
        """Queue a raw JSON string response."""
        self._response_sequence.append(content)

    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AgentResponse:
        """Return queued responses before falling back to normal behavior."""
        self.last_messages = messages
        self.last_system = system
        self.last_tools = tools
        self.last_response_schema = response_schema
        self.last_temperature = temperature

        if self._sequence_index < len(self._response_sequence):
            content = self._response_sequence[self._sequence_index]
            self._sequence_index += 1
            return self._make_response(content)

        return await super().chat(messages, system, tools, response_schema, max_tokens, temperature)


def make_config(backend: MockBackend) -> Config:
    """Create a Config with no tools (direct backend calls)."""
    return Config(backend=backend, tools=[], executor=None)


@dataclass
class SimpleResult:
    """Simple test result schema."""

    text: str
    score: int


class TestOutputGuard:
    """Tests for OutputGuard dataclass."""

    def test_create_guard(self) -> None:
        """Can create an OutputGuard with validator and instruction."""
        guard = OutputGuard(
            validator=lambda x: None,
            retry_instruction="Fix it.",
            name="test_guard",
        )
        assert guard.name == "test_guard"
        assert guard.max_retries == 1  # default
        assert guard.retry_instruction == "Fix it."

    def test_guard_frozen(self) -> None:
        """OutputGuard is immutable (frozen dataclass)."""
        guard = OutputGuard(validator=lambda x: None, retry_instruction="Fix it.")
        with pytest.raises(AttributeError):  # FrozenInstanceError raises AttributeError
            guard.name = "changed"  # type: ignore[misc]

    def test_guard_rejects_negative_max_retries(self) -> None:
        """OutputGuard raises ValueError for negative max_retries."""
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            OutputGuard(
                validator=lambda x: None,
                retry_instruction="Fix it.",
                max_retries=-1,
            )


class TestGuardExecution:
    """Tests for guard execution during verb calls."""

    async def test_guard_passes_on_valid_output(self) -> None:
        """Guard that returns None allows output through."""
        backend = SequencedMockBackend()
        backend.queue_json_response(json.dumps({"text": "Hello", "score": 10}))

        guard = OutputGuard(
            validator=lambda x: None,  # Always valid
            retry_instruction="Fix it.",
        )

        config = make_config(backend)
        extract = Extract(config).with_guard(guard)
        result = await extract("test", SimpleResult)

        assert result.text == "Hello"
        assert result.score == 10
        assert backend.call_count == 1  # No retry needed

    async def test_guard_retries_on_invalid_output(self) -> None:
        """Guard triggers retry when validator returns error string."""
        backend = SequencedMockBackend()
        # First response fails validation
        backend.queue_json_response(json.dumps({"text": "Bad 日本語", "score": 1}))
        # Second response passes
        backend.queue_json_response(json.dumps({"text": "Good English", "score": 2}))

        def reject_non_ascii(result: Any) -> str | None:
            text = str(result)
            for c in text:
                if ord(c) > 127:
                    return f"Non-ASCII character: {c}"
            return None

        guard = OutputGuard(
            validator=reject_non_ascii,
            retry_instruction="Use ASCII only.",
            max_retries=1,
        )

        config = make_config(backend)
        extract = Extract(config).with_guard(guard)
        result = await extract("test", SimpleResult)

        assert result.text == "Good English"
        assert backend.call_count == 2

    async def test_guard_exhausts_retries(self) -> None:
        """OutputGuardError raised when all retries exhausted."""
        backend = SequencedMockBackend()
        # All responses fail validation
        backend.queue_json_response(json.dumps({"text": "Bad 1", "score": 1}))
        backend.queue_json_response(json.dumps({"text": "Bad 2", "score": 2}))

        guard = OutputGuard(
            validator=lambda x: "Always fails",
            retry_instruction="Try again.",
            max_retries=1,
            name="always_fail",
        )

        config = make_config(backend)
        extract = Extract(config).with_guard(guard)

        with pytest.raises(OutputGuardError) as exc_info:
            await extract("test", SimpleResult)

        assert exc_info.value.guard_name == "always_fail"
        assert exc_info.value.error == "Always fails"
        assert exc_info.value.attempts == 2
        assert backend.call_count == 2

    async def test_multiple_guards_applied_in_order(self) -> None:
        """Multiple guards are applied sequentially."""
        backend = SequencedMockBackend()
        backend.queue_json_response(json.dumps({"text": "Short", "score": 1}))

        call_order: list[str] = []

        def guard1_validator(x: Any) -> str | None:
            call_order.append("guard1")
            return None

        def guard2_validator(x: Any) -> str | None:
            call_order.append("guard2")
            return None

        guard1 = OutputGuard(validator=guard1_validator, retry_instruction="G1")
        guard2 = OutputGuard(validator=guard2_validator, retry_instruction="G2")

        config = make_config(backend)
        extract = Extract(config).with_guard(guard1).with_guard(guard2)
        await extract("test", SimpleResult)

        assert call_order == ["guard1", "guard2"]

    async def test_with_guards_adds_multiple_at_once(self) -> None:
        """with_guards() adds multiple guards in a single call."""
        backend = SequencedMockBackend()
        backend.queue_json_response(json.dumps({"text": "Short", "score": 1}))

        call_order: list[str] = []

        def guard1_validator(x: Any) -> str | None:
            call_order.append("guard1")
            return None

        def guard2_validator(x: Any) -> str | None:
            call_order.append("guard2")
            return None

        def guard3_validator(x: Any) -> str | None:
            call_order.append("guard3")
            return None

        guard1 = OutputGuard(validator=guard1_validator, retry_instruction="G1")
        guard2 = OutputGuard(validator=guard2_validator, retry_instruction="G2")
        guard3 = OutputGuard(validator=guard3_validator, retry_instruction="G3")

        config = make_config(backend)
        extract = Extract(config).with_guards(guard1, guard2, guard3)
        await extract("test", SimpleResult)

        assert call_order == ["guard1", "guard2", "guard3"]

    async def test_retry_prompt_includes_feedback(self) -> None:
        """Retry prompt includes error and instruction."""
        backend = SequencedMockBackend()
        backend.queue_json_response(json.dumps({"text": "Too long text here", "score": 1}))
        backend.queue_json_response(json.dumps({"text": "Short", "score": 2}))

        # Validator checks if "Too long" appears in the string representation
        guard = OutputGuard(
            validator=lambda x: "Contains forbidden text!" if "Too long" in str(x) else None,
            retry_instruction="Be concise.",
            max_retries=1,
        )

        config = make_config(backend)
        extract = Extract(config).with_guard(guard)
        await extract("test", SimpleResult)

        # Check the retry prompt
        retry_prompt = backend.last_prompt
        assert "Contains forbidden text!" in retry_prompt
        assert "Be concise." in retry_prompt
        assert "did not meet requirements" in retry_prompt


class TestPrebuiltGuards:
    """Tests for pre-built guards in llm_saia.guards."""

    def test_english_only_passes_english(self) -> None:
        """english_only passes ASCII/Latin text."""
        guard = english_only()
        assert guard.validator("Hello world!") is None
        assert guard.validator("Café résumé") is None  # Latin extended OK

    def test_english_only_rejects_non_latin(self) -> None:
        """english_only rejects CJK, Arabic, etc."""
        guard = english_only()
        error = guard.validator("Hello 日本語")
        assert error is not None
        assert "U+65E5" in error  # Japanese character code

    def test_max_length_passes_short(self) -> None:
        """max_length passes text under limit."""
        guard = max_length(10)
        assert guard.validator("Short") is None

    def test_max_length_rejects_long(self) -> None:
        """max_length rejects text over limit."""
        guard = max_length(10)
        error = guard.validator("This is too long")
        assert error is not None
        assert "16 chars" in error
        assert "max 10" in error

    def test_no_markdown_passes_plain(self) -> None:
        """no_markdown passes plain text."""
        guard = no_markdown()
        assert guard.validator("Just plain text.") is None

    def test_no_markdown_rejects_formatting(self) -> None:
        """no_markdown rejects markdown formatting."""
        guard = no_markdown()

        assert guard.validator("# Header") is not None
        assert guard.validator("- bullet") is not None
        assert guard.validator("**bold**") is not None
        assert guard.validator("```code```") is not None
        assert guard.validator("`inline`") is not None

    def test_no_preamble_passes_direct(self) -> None:
        """no_preamble passes direct responses."""
        guard = no_preamble()
        assert guard.validator("The answer is 42.") is None

    def test_no_preamble_rejects_preambles(self) -> None:
        """no_preamble rejects conversational starters."""
        guard = no_preamble()

        assert guard.validator("Sure! Here's the answer.") is not None
        assert guard.validator("Certainly, I can help.") is not None
        assert guard.validator("I'd be happy to help!") is not None
        assert guard.validator("Here is the information:") is not None

    def test_ascii_only_passes_ascii(self) -> None:
        """ascii_only passes ASCII text."""
        guard = ascii_only()
        assert guard.validator("Hello, World! 123") is None

    def test_ascii_only_rejects_unicode(self) -> None:
        """ascii_only rejects non-ASCII characters."""
        guard = ascii_only()

        assert guard.validator("Café") is not None  # é
        assert guard.validator("Hello 👋") is not None  # emoji
        assert guard.validator("日本語") is not None  # Japanese


class TestNoEmojiGuard:
    """Tests for no_emoji guard."""

    def test_no_emoji_passes_plain_text(self) -> None:
        """no_emoji passes plain ASCII text."""
        guard = no_emoji()
        assert guard.validator("Hello, World!") is None

    def test_no_emoji_passes_unicode_letters(self) -> None:
        """no_emoji allows non-ASCII letters (accents, CJK, etc.)."""
        guard = no_emoji()
        assert guard.validator("Café résumé") is None
        assert guard.validator("日本語テキスト") is None
        assert guard.validator("Привет мир") is None
        assert guard.validator("مرحبا بالعالم") is None

    def test_no_emoji_rejects_common_emoji(self) -> None:
        """no_emoji rejects common emoji."""
        guard = no_emoji()

        # Emoticons
        assert guard.validator("Hello 😀") is not None
        assert guard.validator("Goodbye 👋") is not None

        # Symbols
        assert guard.validator("Check ✅") is not None
        assert guard.validator("Star ⭐") is not None
        assert guard.validator("Heart ❤") is not None

    def test_no_emoji_rejects_zwj_sequences(self) -> None:
        """no_emoji rejects ZWJ emoji sequences."""
        guard = no_emoji()

        # Family emoji (multiple code points joined with ZWJ)
        family = "👨‍👩‍👧‍👦"
        error = guard.validator(f"Family: {family}")
        assert error is not None

    def test_no_emoji_error_shows_character(self) -> None:
        """no_emoji error message includes the emoji."""
        guard = no_emoji()
        error = guard.validator("Hello 🎉 party")
        assert error is not None
        assert "🎉" in error
        assert "U+1F389" in error


class TestUnicodeHandling:
    """Tests for Unicode handling across all guards."""

    def test_english_only_rejects_emoji(self) -> None:
        """english_only rejects emoji (not Latin script)."""
        guard = english_only()
        error = guard.validator("Hello 👋 World")
        assert error is not None
        assert "U+1F44B" in error

    def test_english_only_allows_smart_quotes(self) -> None:
        """english_only allows common typographic punctuation."""
        guard = english_only()
        # Smart quotes, em-dash, ellipsis
        assert guard.validator('He said "hello" — well…') is None

    def test_ascii_only_rejects_emoji(self) -> None:
        """ascii_only rejects emoji."""
        guard = ascii_only()
        error = guard.validator("Hello 👋")
        assert error is not None

    def test_ascii_only_rejects_accents(self) -> None:
        """ascii_only rejects accented characters."""
        guard = ascii_only()
        error = guard.validator("Café")
        assert error is not None
        assert "é" in error

    def test_max_length_counts_code_points(self) -> None:
        """max_length counts Unicode code points, not bytes."""
        guard = max_length(5)

        # 5 single-codepoint emoji = 5 chars
        assert guard.validator("👋👋👋👋👋") is None

        # 6 emoji = 6 chars, exceeds limit
        error = guard.validator("👋👋👋👋👋👋")
        assert error is not None
        assert "6 chars" in error

    def test_max_length_zwj_counts_all_codepoints(self) -> None:
        """max_length counts all code points in ZWJ sequences."""
        guard = max_length(10)

        # Family emoji = 7 code points (4 people + 3 ZWJ)
        family = "👨‍👩‍👧‍👦"
        assert len(family) == 7
        assert guard.validator(family) is None

        # With limit of 5, family emoji exceeds
        guard_strict = max_length(5)
        error = guard_strict.validator(family)
        assert error is not None

    def test_no_markdown_works_with_unicode(self) -> None:
        """no_markdown works correctly with Unicode text."""
        guard = no_markdown()

        # Plain Unicode text passes
        assert guard.validator("日本語で書かれたテキスト") is None
        assert guard.validator("Ümläuts and äccénts") is None

        # Markdown in Unicode text still detected
        assert guard.validator("# 日本語ヘッダー") is not None
        assert guard.validator("**太字テキスト**") is not None

    def test_no_preamble_works_with_unicode(self) -> None:
        """no_preamble works correctly with Unicode text."""
        guard = no_preamble()

        # Unicode text without preamble passes
        assert guard.validator("日本語で始まる文章です") is None
        assert guard.validator("Réponse directe") is None

        # English preamble still detected
        assert guard.validator("Sure! 日本語で答えます") is not None


class TestGuardWithParseRetries:
    """Tests for interaction between guards and parse_retries."""

    async def test_guards_applied_after_parse_retries(self) -> None:
        """Guards are applied after successful parsing (including parse retries)."""
        backend = SequencedMockBackend()
        # First: invalid JSON (triggers parse retry)
        backend.queue_json_response("not json")
        # Second: valid JSON but fails guard
        backend.queue_json_response(json.dumps({"text": "Bad 日本語", "score": 1}))
        # Third: valid JSON and passes guard
        backend.queue_json_response(json.dumps({"text": "Good", "score": 2}))

        guard = OutputGuard(
            validator=lambda x: "Non-ASCII" if any(ord(c) > 127 for c in str(x)) else None,
            retry_instruction="ASCII only.",
            max_retries=1,
        )

        from llm_saia.core.config import CallOptions

        call = CallOptions(parse_retries=1)
        config = Config(backend=backend, tools=[], executor=None, call=call)
        extract = Extract(config).with_guard(guard)
        result = await extract("test", SimpleResult)

        assert result.text == "Good"
        assert backend.call_count == 3  # 1 parse fail + 1 guard fail + 1 success

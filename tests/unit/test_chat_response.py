"""Tests for ChatResponse: new fields and backward-compatible construction."""

from __future__ import annotations

import pytest

from llm_saia.core.backend import ChatResponse

pytestmark = pytest.mark.unit


class TestDefaults:
    """Construction without the new fields must still work (backward compat)."""

    def test_minimal_construction(self) -> None:
        r = ChatResponse(content="hi", tool_calls=[])
        assert r.content == "hi"
        assert r.tool_calls == []
        assert r.finish_reason is None
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.call_id == ""
        assert r.model is None
        assert r.raw is None

    def test_legacy_kwargs_still_work(self) -> None:
        r = ChatResponse(
            content="hi",
            tool_calls=[],
            finish_reason="end_turn",
            input_tokens=100,
            output_tokens=20,
            call_id="abc123",
        )
        assert r.finish_reason == "end_turn"
        assert r.input_tokens == 100
        assert r.output_tokens == 20
        assert r.call_id == "abc123"


class TestNewFields:
    """model and raw round-trip unchanged."""

    def test_model_roundtrip(self) -> None:
        r = ChatResponse(content="", tool_calls=[], model="claude-haiku-4-5-20251001")
        assert r.model == "claude-haiku-4-5-20251001"

    def test_raw_roundtrip_dict(self) -> None:
        payload = {"vendor_field": 42, "nested": {"a": 1}}
        r = ChatResponse(content="", tool_calls=[], raw=payload)
        assert r.raw is payload

    def test_raw_accepts_arbitrary_object(self) -> None:
        class VendorResponse:
            pass

        obj = VendorResponse()
        r = ChatResponse(content="", tool_calls=[], raw=obj)
        assert r.raw is obj

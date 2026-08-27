# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Tests for per-iteration tool visibility gates."""

from __future__ import annotations

from typing import Any

import pytest

from llm_saia import (
    ToolGateContext,
    ToolGateContextFactory,
)
from llm_saia.core.backend import ChatResponse
from llm_saia.core.config import CallOptions, Config, TerminalConfig
from llm_saia.core.conversation import Message, ToolCall
from llm_saia.core.logger import NullLogger
from llm_saia.core.tool_gate import apply_tool_gates
from llm_saia.core.types import ToolDef
from llm_saia.verbs import Instruct
from llm_saia.verbs.complete import Complete
from tests.unit.conftest import MockBackend

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


SEARCH_TOOL = ToolDef(
    name="search",
    description="Search",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)
FETCH_TOOL = ToolDef(
    name="fetch",
    description="Fetch",
    parameters={"type": "object", "properties": {"url": {"type": "string"}}},
)
DONE_TOOL = ToolDef(
    name="done",
    description="Signal completion",
    parameters={
        "type": "object",
        "properties": {"output": {"type": "string"}, "status": {"type": "string"}},
    },
)


def _tool_response(content: str = "", tool_calls: list[ToolCall] | None = None) -> ChatResponse:
    return ChatResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="tool_use" if tool_calls else "end_turn",
    )


def _mk_tc(name: str, args: dict[str, Any] | None = None, tc_id: str = "tc") -> ToolCall:
    return ToolCall(id=tc_id, name=name, arguments=args or {})


async def _executor(name: str, args: dict[str, Any]) -> str:
    return f"result:{name}"


def _config(
    backend: MockBackend,
    *,
    tools: list[ToolDef],
    terminal: TerminalConfig | None = None,
    tool_gates: dict | None = None,
    tool_gate_context_factory: ToolGateContextFactory | None = None,
    max_iterations: int = 10,
) -> Config:
    return Config(
        lg=NullLogger(),
        backend=backend,
        tools=tools,
        executor=_executor,
        call=CallOptions(max_iterations=max_iterations),
        terminal=terminal,
        tool_gates=tool_gates or {},
        tool_gate_context_factory=tool_gate_context_factory,
    )


# ---------------------------------------------------------------------------
# TerminalConfig validation
# ---------------------------------------------------------------------------


class TestTerminalConfigValidation:
    def test_default_min_iterations_zero(self) -> None:
        tc = TerminalConfig(tool="done")
        assert tc.min_iterations == 0

    def test_positive_min_iterations_ok(self) -> None:
        tc = TerminalConfig(tool="done", min_iterations=3)
        assert tc.min_iterations == 3

    def test_negative_min_iterations_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_iterations"):
            TerminalConfig(tool="done", min_iterations=-1)


# ---------------------------------------------------------------------------
# apply_tool_gates unit tests
# ---------------------------------------------------------------------------


class TestApplyToolGates:
    def test_no_gates_returns_tools_unchanged(self) -> None:
        cfg = _config(MockBackend(), tools=[SEARCH_TOOL, DONE_TOOL])
        filtered, blocked = apply_tool_gates(cfg, 0, [], None, [SEARCH_TOOL, DONE_TOOL])
        assert filtered == [SEARCH_TOOL, DONE_TOOL]
        assert blocked == {}

    def test_tools_none_returns_none(self) -> None:
        cfg = _config(MockBackend(), tools=[SEARCH_TOOL])
        filtered, blocked = apply_tool_gates(cfg, 0, [], None, None)
        assert filtered is None
        assert blocked == {}

    def test_gate_allows_via_true(self) -> None:
        cfg = _config(
            MockBackend(),
            tools=[SEARCH_TOOL],
            tool_gates={"search": lambda ctx: True},
        )
        filtered, blocked = apply_tool_gates(cfg, 0, [], None, [SEARCH_TOOL])
        assert filtered == [SEARCH_TOOL]
        assert blocked == {}

    def test_gate_allows_via_none(self) -> None:
        cfg = _config(
            MockBackend(),
            tools=[SEARCH_TOOL],
            tool_gates={"search": lambda ctx: None},
        )
        filtered, blocked = apply_tool_gates(cfg, 0, [], None, [SEARCH_TOOL])
        assert filtered == [SEARCH_TOOL]
        assert blocked == {}

    def test_gate_blocks_via_false(self) -> None:
        cfg = _config(
            MockBackend(),
            tools=[SEARCH_TOOL, FETCH_TOOL],
            tool_gates={"search": lambda ctx: False},
        )
        filtered, blocked = apply_tool_gates(cfg, 0, [], None, [SEARCH_TOOL, FETCH_TOOL])
        assert filtered == [FETCH_TOOL]
        assert "search" in blocked
        assert "False" in blocked["search"]

    def test_gate_blocks_with_reason_string(self) -> None:
        cfg = _config(
            MockBackend(),
            tools=[SEARCH_TOOL],
            tool_gates={"search": lambda ctx: "need more evidence"},
        )
        filtered, blocked = apply_tool_gates(cfg, 0, [], None, [SEARCH_TOOL])
        assert filtered == []
        assert blocked == {"search": "need more evidence"}

    def test_gate_receives_tool_call_counts(self) -> None:
        captured: list[dict[str, int]] = []

        def gate(ctx: ToolGateContext) -> bool:
            captured.append(dict(ctx.tool_call_counts))
            return True

        cfg = _config(MockBackend(), tools=[SEARCH_TOOL], tool_gates={"search": gate})
        messages = [
            Message(role="assistant", content="", tool_calls=[_mk_tc("search")]),
            Message(role="tool", content="ok", tool_call_id="tc"),
            Message(role="assistant", content="", tool_calls=[_mk_tc("search"), _mk_tc("fetch")]),
        ]
        apply_tool_gates(cfg, 2, messages, None, [SEARCH_TOOL])
        assert captured == [{"search": 2, "fetch": 1}]

    def test_gate_receives_iteration_and_last_response(self) -> None:
        seen: dict[str, Any] = {}

        def gate(ctx: ToolGateContext) -> bool:
            seen["iteration"] = ctx.iteration
            seen["last_response"] = ctx.last_response
            return True

        cfg = _config(MockBackend(), tools=[SEARCH_TOOL], tool_gates={"search": gate})
        last = _tool_response("prev")
        apply_tool_gates(cfg, 5, [], last, [SEARCH_TOOL])
        assert seen["iteration"] == 5
        assert seen["last_response"] is last

    def test_factory_populates_extra(self) -> None:
        captured: list[Any] = []

        def factory(ctx: ToolGateContext) -> dict[str, int]:
            return {"walked_at_iter": ctx.iteration}

        def gate(ctx: ToolGateContext) -> bool:
            captured.append(ctx.extra)
            return True

        cfg = _config(
            MockBackend(),
            tools=[SEARCH_TOOL],
            tool_gates={"search": gate},
            tool_gate_context_factory=factory,
        )
        apply_tool_gates(cfg, 7, [], None, [SEARCH_TOOL])
        assert captured == [{"walked_at_iter": 7}]

    def test_factory_runs_once_per_call_shared_across_gates(self) -> None:
        run_count = 0

        def factory(ctx: ToolGateContext) -> str:
            nonlocal run_count
            run_count += 1
            return "shared"

        cfg = _config(
            MockBackend(),
            tools=[SEARCH_TOOL, FETCH_TOOL],
            tool_gates={
                "search": lambda ctx: True if ctx.extra == "shared" else "wrong",
                "fetch": lambda ctx: True if ctx.extra == "shared" else "wrong",
            },
            tool_gate_context_factory=factory,
        )
        filtered, blocked = apply_tool_gates(cfg, 0, [], None, [SEARCH_TOOL, FETCH_TOOL])
        assert blocked == {}
        assert filtered == [SEARCH_TOOL, FETCH_TOOL]
        assert run_count == 1

    def test_min_iterations_shortcut_blocks_below_threshold(self) -> None:
        cfg = _config(
            MockBackend(),
            tools=[SEARCH_TOOL, DONE_TOOL],
            terminal=TerminalConfig(tool="done", min_iterations=3),
        )
        filtered, blocked = apply_tool_gates(cfg, 2, [], None, [SEARCH_TOOL, DONE_TOOL])
        assert filtered == [SEARCH_TOOL]
        assert "done" in blocked
        assert "min_iterations" in blocked["done"]

    def test_min_iterations_shortcut_allows_at_threshold(self) -> None:
        cfg = _config(
            MockBackend(),
            tools=[SEARCH_TOOL, DONE_TOOL],
            terminal=TerminalConfig(tool="done", min_iterations=3),
        )
        filtered, blocked = apply_tool_gates(cfg, 3, [], None, [SEARCH_TOOL, DONE_TOOL])
        assert filtered == [SEARCH_TOOL, DONE_TOOL]
        assert blocked == {}

    def test_explicit_gate_wins_over_shortcut(self) -> None:
        """An explicit tool_gates entry for the terminal tool overrides the shortcut."""
        cfg = _config(
            MockBackend(),
            tools=[DONE_TOOL],
            terminal=TerminalConfig(tool="done", min_iterations=10),
            tool_gates={"done": lambda ctx: True},  # explicit: always allow
        )
        filtered, blocked = apply_tool_gates(cfg, 0, [], None, [DONE_TOOL])
        assert filtered == [DONE_TOOL]
        assert blocked == {}


# ---------------------------------------------------------------------------
# Integration: gate inside the tool-calling loop
# ---------------------------------------------------------------------------


class TestGateInLoop:
    async def test_terminal_hidden_until_min_iterations(self) -> None:
        """min_iterations=2 hides 'done' on iters 0-1, exposes it on iter 2."""
        backend = MockBackend()
        seen_tool_names: list[list[str]] = []

        original_chat = backend.chat

        async def tracking_chat(messages: list[Message], **kwargs: Any) -> ChatResponse:
            tools = kwargs.get("tools") or []
            seen_tool_names.append([t.name for t in tools])
            return await original_chat(messages, **kwargs)

        backend.chat = tracking_chat  # type: ignore[assignment]

        # iter 0: search
        backend.queue_response(_tool_response("looking", [_mk_tc("search", tc_id="s0")]))
        # iter 1: search again
        backend.queue_response(_tool_response("more", [_mk_tc("search", tc_id="s1")]))
        # iter 2: done (now allowed)
        backend.queue_response(
            _tool_response("done!", [_mk_tc("done", {"output": "x", "status": "ok"}, "d1")])
        )
        # iter 3: confirmation
        backend.queue_response(
            _tool_response("confirm", [_mk_tc("done", {"output": "x", "status": "ok"}, "d2")])
        )

        cfg = _config(
            backend,
            tools=[SEARCH_TOOL, DONE_TOOL],
            terminal=TerminalConfig(tool="done", min_iterations=2),
        )
        result = await Complete(cfg)("go")

        assert result.completed
        assert seen_tool_names[0] == ["search"], "iter 0 must not expose 'done'"
        assert seen_tool_names[1] == ["search"], "iter 1 must not expose 'done'"
        assert "done" in seen_tool_names[2], "iter 2 must expose 'done'"

    async def test_arbitrary_tool_gate_hides_non_terminal_tool(self) -> None:
        """A gate keyed on a non-terminal tool works the same as terminal gating."""
        backend = MockBackend()
        seen_tool_names: list[list[str]] = []

        original_chat = backend.chat

        async def tracking_chat(messages: list[Message], **kwargs: Any) -> ChatResponse:
            tools = kwargs.get("tools") or []
            seen_tool_names.append([t.name for t in tools])
            return await original_chat(messages, **kwargs)

        backend.chat = tracking_chat  # type: ignore[assignment]

        # iter 0: search only (fetch hidden)
        backend.queue_response(_tool_response("s", [_mk_tc("search", tc_id="a")]))
        # iter 1: search only (fetch still hidden)
        backend.queue_response(_tool_response("s", [_mk_tc("search", tc_id="b")]))
        # iter 2: fetch now visible; model calls it
        backend.queue_response(_tool_response("f", [_mk_tc("fetch", tc_id="c")]))
        # iter 3: final
        backend.queue_response(_tool_response("all done"))

        # Hide fetch until 2 search calls have happened.
        def fetch_gate(ctx: ToolGateContext) -> bool | str:
            n = ctx.tool_call_counts.get("search", 0)
            return True if n >= 2 else f"need 2 searches, have {n}"

        cfg = _config(
            backend,
            tools=[SEARCH_TOOL, FETCH_TOOL],
            tool_gates={"fetch": fetch_gate},
        )
        result = await Instruct(cfg)("go")

        assert result.value == "all done"
        assert seen_tool_names[0] == ["search"]
        assert seen_tool_names[1] == ["search"]
        assert "fetch" in seen_tool_names[2]

    async def test_no_gates_no_regression(self) -> None:
        """With no gates configured, every call sees the full tool list."""
        backend = MockBackend()
        seen_tool_names: list[list[str]] = []

        original_chat = backend.chat

        async def tracking_chat(messages: list[Message], **kwargs: Any) -> ChatResponse:
            tools = kwargs.get("tools") or []
            seen_tool_names.append([t.name for t in tools])
            return await original_chat(messages, **kwargs)

        backend.chat = tracking_chat  # type: ignore[assignment]

        backend.queue_response(_tool_response("hi", [_mk_tc("search", tc_id="x")]))
        backend.queue_response(_tool_response("done"))

        cfg = _config(backend, tools=[SEARCH_TOOL, FETCH_TOOL])
        await Instruct(cfg)("go")

        assert all(set(names) == {"search", "fetch"} for names in seen_tool_names)


# ---------------------------------------------------------------------------
# Fluent API
# ---------------------------------------------------------------------------


class TestFluentAPI:
    def test_with_tool_gate_returns_new_instance_with_gate(self) -> None:
        cfg = _config(MockBackend(), tools=[SEARCH_TOOL])
        verb = Instruct(cfg)
        gate = lambda ctx: True  # noqa: E731
        result = verb.with_tool_gate("search", gate)
        assert result is not verb
        assert result._config.tool_gates == {"search": gate}
        assert verb._config.tool_gates == {}

    def test_with_tool_gates_merges(self) -> None:
        gate_a = lambda ctx: True  # noqa: E731
        gate_b = lambda ctx: False  # noqa: E731
        cfg = _config(MockBackend(), tools=[SEARCH_TOOL], tool_gates={"search": gate_a})
        verb = Instruct(cfg)
        result = verb.with_tool_gates({"fetch": gate_b})
        assert result._config.tool_gates == {"search": gate_a, "fetch": gate_b}

    def test_with_tool_gate_context_factory(self) -> None:
        cfg = _config(MockBackend(), tools=[SEARCH_TOOL])
        verb = Instruct(cfg)
        factory = lambda ctx: {"x": 1}  # noqa: E731
        result = verb.with_tool_gate_context_factory(factory)
        assert result._config.tool_gate_context_factory is factory

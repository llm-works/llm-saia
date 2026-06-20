"""Tests for context-merge semantics — ``merge_context`` primitive and
``Configurable.with_context()`` integration.

Covers the acceptance criteria for the REPLACE → MERGE change:
disjoint-key merge, last-write-wins, arbitrary-depth recursion, list-replace,
``None``-as-value, parent immutability, derived independence, leaf-reference
semantics, verb inheritance, dict-subclass result-type coercion,
mixed-container tuples.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, defaultdict
from typing import Any

import pytest

from llm_saia.core.context import merge_context
from tests.unit.conftest import MockBackend, make_saia

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# merge_context primitive
# ---------------------------------------------------------------------------


class TestMergeContextPrimitive:
    """Pure-function tests for the merge_context primitive."""

    def test_disjoint_keys_both_present(self) -> None:
        result = merge_context({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_same_key_last_wins(self) -> None:
        result = merge_context({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_deep_merge_three_levels(self) -> None:
        base = {"a": {"b": {"c": 1, "d": 2}}}
        overlay = {"a": {"b": {"c": 99}}}
        result = merge_context(base, overlay)
        assert result == {"a": {"b": {"c": 99, "d": 2}}}

    def test_deep_merge_four_levels(self) -> None:
        # Guards against any depth-2 cutoff regression.
        base = {"a": {"b": {"c": {"d": 1, "e": 2}}}}
        overlay = {"a": {"b": {"c": {"d": 99}}}}
        result = merge_context(base, overlay)
        assert result == {"a": {"b": {"c": {"d": 99, "e": 2}}}}

    def test_nested_disjoint_keys_preserved(self) -> None:
        base = {"ns": {"a": 1}}
        overlay = {"ns": {"b": 2}}
        result = merge_context(base, overlay)
        assert result == {"ns": {"a": 1, "b": 2}}

    def test_list_leaves_replace(self) -> None:
        result = merge_context({"tags": ["a", "b"]}, {"tags": ["c"]})
        assert result == {"tags": ["c"]}

    def test_set_leaves_replace(self) -> None:
        result = merge_context({"s": {1, 2}}, {"s": {3}})
        assert result == {"s": {3}}

    def test_tuple_leaves_replace(self) -> None:
        result = merge_context({"t": (1, 2)}, {"t": (3,)})
        assert result == {"t": (3,)}

    def test_none_as_value_preserved(self) -> None:
        # None is a value, not a delete signal.
        result = merge_context({"x": 5}, {"x": None})
        assert result == {"x": None}

    def test_overlay_dict_replaces_scalar(self) -> None:
        # base scalar + overlay dict → overlay wins; no merge attempted.
        result = merge_context({"k": 1}, {"k": {"nested": "v"}})
        assert result == {"k": {"nested": "v"}}

    def test_overlay_scalar_replaces_dict(self) -> None:
        result = merge_context({"k": {"nested": "v"}}, {"k": 1})
        assert result == {"k": 1}

    def test_inputs_not_mutated(self) -> None:
        base = {"a": {"b": 1}}
        overlay = {"a": {"c": 2}}
        merge_context(base, overlay)
        assert base == {"a": {"b": 1}}
        assert overlay == {"a": {"c": 2}}

    def test_leaf_list_shared_by_reference(self) -> None:
        # Leaf values are NOT deep-copied: callers expecting to thread shared
        # objects (cost trackers, loggers) through context need identity.
        # As a side effect, mutable builtin containers are also aliased —
        # documented contract, not a bug.
        tags = ["a", "b"]
        result = merge_context({}, {"tags": tags})
        assert result["tags"] is tags

    def test_custom_object_shared_by_reference(self) -> None:
        # Critical use case: cost trackers and similar stateful objects must
        # retain identity across with_context() chains so backend callbacks
        # see the caller's instance, not a copy.
        class Tracker:
            def __init__(self) -> None:
                self.calls = 0

        tracker = Tracker()
        result = merge_context({}, {"tracker": tracker})
        assert result["tracker"] is tracker
        tracker.calls = 5
        assert result["tracker"].calls == 5

    def test_non_copyable_object_passes_through(self) -> None:
        # Locks, file handles, db connections, http clients — none of these
        # are deep-copyable, but all are legitimate things to thread through
        # context for backend callbacks. Reference semantics make this work.
        lock = threading.Lock()
        result = merge_context({}, {"ns": {"lock": lock}})
        assert result["ns"]["lock"] is lock

    def test_dict_subclass_coerced_to_plain_dict(self) -> None:
        # OrderedDict and defaultdict pass isinstance(x, dict) so they recurse,
        # but the result must be a plain dict at every level (the merge
        # primitive doesn't try to preserve subclass type, which would require
        # arbitrary constructor knowledge — e.g. defaultdict's factory).
        base = OrderedDict([("a", OrderedDict([("b", 1)]))])
        dd: defaultdict[str, int] = defaultdict(int)
        dd["c"] = 2
        overlay = {"a": dd}
        result = merge_context(base, overlay)
        assert type(result) is dict
        assert type(result["a"]) is dict
        assert result == {"a": {"b": 1, "c": 2}}

    def test_mixed_container_tuple_does_not_recurse(self) -> None:
        # Merge only recurses into dicts; a tuple leaf passes through by
        # reference, including any dict it contains. The overlay's tuple
        # wins wholesale; its inner dict is not merged with the base's.
        inner_dict_base = {"x": 1}
        inner_dict_overlay = {"y": 2}
        base = {"t": (inner_dict_base, "a")}
        overlay = {"t": (inner_dict_overlay, "b")}
        result = merge_context(base, overlay)
        assert result["t"] is overlay["t"]
        assert result["t"][0] is inner_dict_overlay

    def test_empty_overlay_returns_copy_of_base(self) -> None:
        base = {"a": {"b": 1}}
        result = merge_context(base, {})
        assert result == base
        assert result is not base
        assert result["a"] is not base["a"]

    def test_empty_base_returns_copy_of_overlay(self) -> None:
        overlay = {"a": {"b": 1}}
        result = merge_context({}, overlay)
        assert result == overlay
        assert result is not overlay
        assert result["a"] is not overlay["a"]


# ---------------------------------------------------------------------------
# with_context() integration with the merge semantics
# ---------------------------------------------------------------------------


class TestWithContextMerge:
    """Behavioural tests for ``SAIA.with_context()`` under merge semantics."""

    async def test_disjoint_keys_layer(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)
        merged = saia.with_context({"foo": 1}).with_context({"bar": 2})
        assert merged.config.call.context == {"foo": 1, "bar": 2}

    async def test_same_key_last_wins(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)
        merged = saia.with_context({"k": 1}).with_context({"k": 2})
        assert merged.config.call.context == {"k": 2}

    async def test_deep_merge_namespaced(self, mock_backend: MockBackend) -> None:
        # Motivating use case: two callers layer their own subdicts under a
        # shared namespace without trampling each other.
        saia = (
            make_saia(mock_backend)
            .with_context({"ns": {"phase": "exploration"}})
            .with_context({"ns": {"cost": 0.42}})
        )
        assert saia.config.call.context == {"ns": {"phase": "exploration", "cost": 0.42}}

    async def test_parent_context_unchanged_after_derived(self, mock_backend: MockBackend) -> None:
        parent = make_saia(mock_backend).with_context({"a": {"b": 1}})
        parent_ctx_before: Any = parent.config.call.context
        snapshot = {"a": {"b": 1}}
        # Derive and mutate the derived's context to prove independence.
        derived = parent.with_context({"a": {"c": 2}})
        assert derived.config.call.context == {"a": {"b": 1, "c": 2}}
        assert parent.config.call.context == snapshot
        # The dict object on the parent is not the same one as on the derived.
        assert parent.config.call.context is parent_ctx_before

    async def test_two_derivations_independent(self, mock_backend: MockBackend) -> None:
        parent = make_saia(mock_backend).with_context({"shared": {"k": 1}})
        a = parent.with_context({"shared": {"a": "A"}})
        b = parent.with_context({"shared": {"b": "B"}})
        assert a.config.call.context == {"shared": {"k": 1, "a": "A"}}
        assert b.config.call.context == {"shared": {"k": 1, "b": "B"}}
        # Mutating one must not affect the other.
        assert a.config.call.context is not None
        a.config.call.context["shared"]["a"] = "MUTATED"
        assert b.config.call.context == {"shared": {"k": 1, "b": "B"}}

    async def test_empty_dict_is_noop(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend).with_context({"k": 1})
        result = saia.with_context({})
        assert result is saia
        assert result.config.call.context == {"k": 1}

    async def test_none_clears(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend).with_context({"k": 1})
        cleared = saia.with_context(None)
        assert cleared.config.call.context is None

    async def test_merge_on_no_existing_context(self, mock_backend: MockBackend) -> None:
        # base context is None; first with_context should just set it.
        saia = make_saia(mock_backend)
        assert saia.config.call.context is None
        result = saia.with_context({"k": 1})
        assert result.config.call.context == {"k": 1}

    async def test_none_value_preserved_across_calls(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend).with_context({"x": 5})
        result = saia.with_context({"x": None})
        assert result.config.call.context == {"x": None}

    async def test_list_value_replaces(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend).with_context({"tags": ["a", "b"]})
        result = saia.with_context({"tags": ["c"]})
        assert result.config.call.context == {"tags": ["c"]}

    async def test_verb_inherits_merged_context(self, mock_backend: MockBackend) -> None:
        # Regression guard: a verb derived from a context-laden SAIA must see
        # the fully merged context at chat() time.
        from llm_saia.core.backend import ToolDef

        saia = make_saia(
            mock_backend,
            tools=[ToolDef(name="noop", description="", parameters={})],
            executor=lambda n, a: "ok",
        )
        saia = saia.with_context({"ns": {"phase": "exploration"}}).with_context(
            {"ns": {"cost": 0.42}, "trace_id": "abc"}
        )
        mock_backend.set_complete_response("done")
        await saia.complete("hi")
        assert mock_backend.last_context == {
            "ns": {"phase": "exploration", "cost": 0.42},
            "trace_id": "abc",
        }

    async def test_stateful_object_identity_preserved_through_merge(
        self, mock_backend: MockBackend
    ) -> None:
        # The motivating use case for reference-leaf semantics: a cost
        # tracker (or logger, lock, client) threaded through context retains
        # identity across with_context() chains. Backend callbacks see the
        # caller's instance, so mutations propagate.
        from llm_saia.core.backend import ToolDef

        class CostTracker:
            def __init__(self) -> None:
                self.total = 0.0

        tracker = CostTracker()
        saia = make_saia(
            mock_backend,
            tools=[ToolDef(name="noop", description="", parameters={})],
            executor=lambda n, a: "ok",
        )
        # Layer the tracker in, then add another key — second call forces a
        # merge that must not deep-copy the tracker.
        saia = saia.with_context({"tracker": tracker}).with_context({"trace_id": "abc"})
        mock_backend.set_complete_response("done")
        await saia.complete("hi")
        assert mock_backend.last_context is not None
        assert mock_backend.last_context["tracker"] is tracker

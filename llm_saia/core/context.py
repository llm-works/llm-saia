"""Context dict merge semantics for :meth:`Configurable.with_context`.

The primitive is exposed so consumers that pre-compose context outside the
``with_context()`` flow follow the same rules (no parallel implementations).
"""

from __future__ import annotations

from typing import Any

__all__ = ["merge_context"]


def merge_context(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge *overlay* onto *base* and return a new dict.

    Merge rules:

    - Two dict-valued nodes at the same key recurse to arbitrary depth.
      Everything else replaces wholesale at its leaf — never concat, union,
      or merge-as-delete.
    - Conflict resolution: last write wins, per leaf.
    - Dict structure is independent of both inputs (every dict node in the
      result is a fresh dict), so the caller may freely add or remove keys
      on either argument afterwards without affecting the merged result.
    - **Leaf values are shared by reference** — not copied. Mutating a list,
      set, or custom object in either input (or in the merged result) is
      visible on the other side. This is intentional: callers thread
      objects like cost trackers, loggers, tracing handles, locks, and
      clients through context for backend callbacks, and those use cases
      require shared identity. To isolate a mutable leaf, copy it before
      passing.
    - Result container type is plain ``dict`` at every level. Subclass
      inputs (``OrderedDict``, ``defaultdict``, etc.) recurse correctly but
      lose their type in the output.

    Args:
        base: Starting context. Not mutated.
        overlay: Context layered on top. Not mutated.

    Returns:
        New ``dict`` containing the merged context.
    """
    return _merge(base, overlay)


def _merge(base: dict[Any, Any], overlay: dict[Any, Any]) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for k, v in base.items():
        result[k] = _copy_dicts(v)
    for k, v in overlay.items():
        existing = result.get(k)
        if isinstance(v, dict) and isinstance(existing, dict):
            result[k] = _merge(existing, v)
        else:
            result[k] = _copy_dicts(v)
    return result


def _copy_dicts(v: Any) -> Any:
    """Return *v* with every nested dict replaced by a fresh dict; non-dict
    leaves pass through by reference."""
    if isinstance(v, dict):
        return {k: _copy_dicts(vv) for k, vv in v.items()}
    return v

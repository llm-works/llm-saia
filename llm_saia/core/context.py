"""Context dict merge semantics for :meth:`Configurable.with_context`.

The primitive is exposed so consumers that pre-compose context outside the
``with_context()`` flow follow the same rules (no parallel implementations).
"""

from __future__ import annotations

import copy
from typing import Any

__all__ = ["merge_context"]


def merge_context(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge *overlay* onto *base* and return a new dict.

    Merge rules:

    - Two dict-valued nodes at the same key recurse to arbitrary depth.
      Scalars, lists, sets, tuples, and ``None`` replace wholesale at their
      leaf — never concat, union, or merge-as-delete.
    - Conflict resolution: last write wins, per leaf.
    - The result is fully independent of both inputs (deep-copied), so the
      caller may freely mutate either argument afterwards.
    - Result container type is plain ``dict`` at every level. Subclass inputs
      (``OrderedDict``, ``defaultdict``, etc.) recurse correctly but lose
      their type in the output.
    - Every value must be deep-copyable. Non-copyable values (locks, open
      file handles, db connections) raise :class:`TypeError` with the failing
      key path in the message.

    Args:
        base: Starting context. Not mutated.
        overlay: Context layered on top. Not mutated.

    Returns:
        New ``dict`` containing the merged context.

    Raises:
        TypeError: If any value is not deep-copyable. Message names the key
            path, e.g. ``context value at key path 'ns.leaf' is not
            deep-copyable``.
    """
    return _merge(base, overlay, ())


def _merge(base: dict[Any, Any], overlay: dict[Any, Any], path: tuple[Any, ...]) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for k, v in base.items():
        result[k] = _copy_value(v, path + (k,))
    for k, v in overlay.items():
        sub_path = path + (k,)
        existing = result.get(k)
        if isinstance(v, dict) and isinstance(existing, dict):
            result[k] = _merge(existing, v, sub_path)
        else:
            result[k] = _copy_value(v, sub_path)
    return result


def _copy_value(v: Any, path: tuple[Any, ...]) -> Any:
    if isinstance(v, dict):
        return {k: _copy_value(vv, path + (k,)) for k, vv in v.items()}
    try:
        return copy.deepcopy(v)
    except TypeError as e:
        key_path = ".".join(str(p) for p in path)
        raise TypeError(f"context value at key path '{key_path}' is not deep-copyable: {e}") from e

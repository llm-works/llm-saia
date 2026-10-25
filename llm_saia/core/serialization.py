# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Helpers shared by the ``to_dict`` / ``from_dict`` methods of SAIA dataclasses."""

from __future__ import annotations

from dataclasses import fields
from typing import Any


def known_fields(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Return the entries of *data* whose keys name a field of dataclass *cls*.

    Unknown keys are dropped so dicts written by a newer SAIA still load;
    keys absent from *data* fall through to the field defaults on construction.
    """
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}

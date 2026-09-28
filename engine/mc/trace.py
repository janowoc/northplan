# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Flattening a single-path trace into rows, one per traced month.

Owns exactly one thing: turning a sequence of
:class:`~engine.core.context.MonthRecord` (typically
``engine.mc.simulate.SimulationResult.trace``) into a list of flat ``dict``s, one row per
record, one column per leaf value reached by walking each record's dataclass fields and
tuple elements. Writing that flattening to CSV, or anywhere else, is not this module's job
(#23's).

Every column name is the dotted path of attribute names from the record to the leaf, with
tuple positions written as integers (``persons.0.rrsp.room``); the record itself takes no
prefix. A property, such as ``PersonInflows.to_cash``, is not walked at all -- only
:func:`dataclasses.fields` entries are.

The invariant this module holds: every returned row has exactly the same keys, in the same
order -- the union, over every row, of every column any row produced, in first-seen order.
A row that produced a narrower set of columns than another (``year_record`` is ``None``
outside December, a dataclass with its own sub-columns within it) gets ``None`` for the
columns it did not produce itself.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from engine.core.context import MonthRecord

__all__ = ["trace_rows"]


def trace_rows(trace: Sequence[MonthRecord]) -> list[dict[str, object]]:
    """Flatten a single-path trace into one row per month.

    Every ``(1,)`` ``np.ndarray`` of bool, integer, or float dtype whose ``.item()`` is a
    plain ``bool``, ``int``, or ``float`` becomes that scalar; ``None``, and a leaf whose type
    is exactly ``bool``, ``int``, ``float``, or ``str``, is kept unchanged; a dataclass
    instance or a tuple is recursed into. See the module docstring for the column-naming and
    missing-column rules.

    Args:
        trace: A single traced path's monthly records, in order.

    Returns:
        One dict per record, sharing the same keys in the same order. ``[]`` for an
        empty ``trace``.

    Raises:
        ValueError: A leaf ``np.ndarray``'s shape is not exactly ``(1,)`` -- this function
            flattens a single traced path; slice a multi-path record with
            :meth:`~engine.core.context.MonthRecord.at_path` first. Names the column
            and the shape found.
        TypeError: A leaf is of any other type -- a subclass included, such as
            ``np.float64``, an ``IntEnum`` member, or a masked array -- or a leaf array's
            dtype, or the type ``.item()`` returns, is not bool, integer, or float (a
            ``np.longdouble`` array). Names the column and the dtype or type found.
    """
    rows: list[dict[str, object]] = []
    for record in trace:
        row: dict[str, object] = {}
        _flatten(record, "", row)
        rows.append(row)

    columns: dict[str, None] = {}
    for row in rows:
        for key in row:
            columns.setdefault(key, None)

    return [{column: row.get(column) for column in columns} for row in rows]


def _flatten(value: object, path: str, row: dict[str, object]) -> None:
    """Write every leaf reachable from ``value`` into ``row``, keyed by its dotted path."""
    if type(value) is np.ndarray:
        if value.shape != (1,):
            raise ValueError(
                f"column {path!r}: array shape {value.shape} is not (1,); trace_rows "
                "flattens a single traced path -- slice a multi-path record with "
                "MonthRecord.at_path first."
            )
        item = value.item()
        if value.dtype.kind not in "biuf" or type(item) not in (bool, int, float):
            raise TypeError(
                f"column {path!r}: unsupported array dtype {value.dtype}; "
                "only bool, integer, and float arrays whose .item() is a Python scalar "
                "are flattened."
            )
        row[path] = item
    elif value is None:
        row[path] = None
    elif type(value) in (bool, int, float, str):
        row[path] = value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            child = field.name if not path else f"{path}.{field.name}"
            _flatten(getattr(value, field.name), child, row)
    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            child = str(index) if not path else f"{path}.{index}"
            _flatten(item, child, row)
    else:
        kind = type(value)
        raise TypeError(
            f"column {path!r}: unsupported leaf type {kind.__module__}.{kind.__qualname__}."
        )

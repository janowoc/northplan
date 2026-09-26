# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared machinery for ``engine.core`` tests: the state-tree walker.

Lives here, once, because both ``test_state.py`` (which builds small
synthetic fixtures) and ``test_build.py`` (which builds from
``scenarios/example.yaml``) need to check exactly the same invariants over a
built state, and neither should re-implement the walk. This file defines no
fixtures of its own — ``walk`` is an ordinary function, imported, not
injected — and lives in ``conftest.py`` rather than a third test module only
so that ``from .conftest import walk`` in both of the others is a relative
import within this package rather than a peer import between two test
modules, which pytest's own rootless collection can otherwise resolve to a
second, separate module object for the same file. See this package's
``__init__.py`` for the mechanics.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Final

import numpy as np
import pytest
from numpy.typing import DTypeLike

#: Field names whose array is not money, keyed to the dtype they must
#: carry. Anything not listed here is money and must be float64 — an int
#: array where a float belongs truncates dollars with no error at all.
_EXPECTED_DTYPE: Final[dict[str, DTypeLike]] = {
    "death_month_index": np.int64,
    "alive": np.bool_,
    "wound_up": np.bool_,
    "depleted": np.bool_,
    "gis_band": np.bool_,
    "finished": np.bool_,
}


def walk(value: object, n_paths: int, _field_name: str | None = None) -> int:
    """Recursively assert the shape, dtype, and collection invariants of a built state.

    For every array reached: it is non-writeable, exactly ``(n_paths,)``, and
    carries the dtype :data:`_EXPECTED_DTYPE` says its field name should
    (float64 for every field not listed there). For every collection
    reached: it is a ``tuple``, never a ``list``, ``set``, or ``dict``.
    Recurses into dataclass fields and tuple elements; stops at any other
    value (a scalar, or ``None``).

    Args:
        value: The (sub)state to check.
        n_paths: Expected length of every array.
        _field_name: The dataclass field ``value`` was reached through, used
            only to look up the expected dtype at an array leaf. Callers
            never pass this; it is threaded through the recursion itself.

    Returns:
        The number of arrays visited in the whole subtree. A walk that finds
        zero arrays passes every assertion in this function vacuously — an
        empty ``persons`` tuple or a field that quietly became ``None``
        would say nothing went wrong — so callers must assert this count
        against a known minimum rather than only calling :func:`walk` for
        its side effect.
    """
    if isinstance(value, np.ndarray):
        assert not value.flags.writeable, f"array is writeable: {value!r}"
        assert value.shape == (n_paths,), f"array shape {value.shape} != ({n_paths},)"
        expected = _EXPECTED_DTYPE.get(_field_name, np.float64)
        assert value.dtype == expected, (
            f"field {_field_name!r}: dtype {value.dtype} != expected {np.dtype(expected)}"
        )
        return 1
    if isinstance(value, (list, set, dict)):
        pytest.fail(f"found a {type(value).__name__}, every collection here must be a tuple")
    if isinstance(value, tuple):
        return sum(walk(item, n_paths, _field_name) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return sum(walk(getattr(value, f.name), n_paths, f.name) for f in fields(value))
    return 0  # an ordinary scalar (int, float, str, bool, None) — nothing to check.

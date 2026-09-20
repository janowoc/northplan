# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cash: the household's one hub account.

Every inflow and every outflow passes through cash. A policy moves money
between cash and every other account, and every account module's contribution
or withdrawal is one leg of a transfer whose other leg is here. Cash pays zero
real return (``docs/limitations.md`` L38), which is why there is no ``grow``
in this module and why ``CashState.balance`` is classified ``REAL`` in
``tests/core/test_state_nominal_or_real.py`` — it holds its real value by
assumption, with nothing to erode.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from engine.core.state import CashState, updated


def deposit(state: CashState, amount: ArrayLike) -> CashState:
    """Add ``amount`` to cash.

    Args:
        state: Opening cash state.
        amount: Amount to add, real dollars, ``(n_paths,)``. Negative on any
            path raises.

    Returns:
        Updated state.

    Raises:
        ValueError: If ``amount`` is negative on any path.
    """
    amount_arr = _non_negative(amount)
    return updated(state, balance=state.balance + amount_arr)


def pay(state: CashState, amount: ArrayLike) -> CashState:
    """Subtract ``amount`` from cash.

    May take the balance negative on any path. The no-negative-cash rule is a
    *month-end* rule the step enforces by force-withdrawing from other
    accounts in the policy's order (``docs/limitations.md`` L38); it is not an
    invariant of this function. An implementation that floors the result here
    at zero would break that ordering, by hiding a shortfall the step is the
    only place equipped to resolve.

    Args:
        state: Opening cash state.
        amount: Amount to pay out, real dollars, ``(n_paths,)``. Negative on
            any path raises.

    Returns:
        Updated state, balance possibly negative.

    Raises:
        ValueError: If ``amount`` is negative on any path.
    """
    amount_arr = _non_negative(amount)
    return updated(state, balance=state.balance - amount_arr)


def _non_negative(amount: ArrayLike) -> np.ndarray:
    """``amount`` as a float64 array, or raise if any path is negative."""
    amount_arr = np.asarray(amount, dtype=np.float64)
    if np.any(amount_arr < 0):
        raise ValueError(
            f"amount must be non-negative on every path, got {amount_arr!r}. "
            "A negative request is a caller bug, not a reverse transaction."
        )
    return amount_arr

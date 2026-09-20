# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tax-Free Savings Account.

Contributions are not deductible; growth and withdrawals are entirely
tax-free. Withdrawn amounts are added back to contribution room, but only at
the start of the *following* calendar year — recontributing in the same year is
an over-contribution. That lag is the thing to get right, and a monthly
timestep makes it easier to get wrong: room restored in the month after a
withdrawal rather than in the January after it is a plausible-looking bug that
hands the household eleven months of room it does not have.

Parameters come from ``params/{year}/tfsa.yaml``. It is the smallest parameter
file in the repository and will stay that way: a TFSA has no tax calculation at
all, so the only rules are how much room exists and when it appears.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.accounts import base
from engine.accounts.base import WithdrawalResult
from engine.core.indexation import RealParamSet, nominal_carry_factor
from engine.core.state import TfsaState, updated


def room_accrued(
    age_at_end_of_year: int,
    params: RealParamSet,
    january_month_index: int,
) -> float:
    """New contribution room for one year, granted in January.

    A flat annual amount from ``params``, granted in full on 1 January of each
    year from the year the person reaches the eligibility age. It is not
    accrued monthly: a person may contribute the whole year's room in January.

    Args:
        age_at_end_of_year: Age in whole years on 31 December. Not per-path: a
            person's age does not vary by path.
        params: The ``tfsa`` parameter set, supplying
            ``room.eligibility_age_years`` and ``room.amount_annual``.
        january_month_index: Month index of January of the year the room is
            granted for.

    Returns:
        Room accrued for the year, zero below the eligibility age.
    """
    eligibility_age = params.number("room.eligibility_age_years")
    if age_at_end_of_year < eligibility_age:
        return 0.0
    return params.annual_amount("room.amount_annual", january_month_index)


def contribute(state: TfsaState, requested: ArrayLike) -> tuple[TfsaState, NDArray[np.float64]]:
    """Contribute up to the room remaining right now.

    ``state.room`` is the room left at this point in the year, already
    reduced by every contribution made in earlier months. It is not the
    January grant.

    Args:
        state: Opening TFSA state.
        requested: Desired contribution this month, ``(n_paths,)``. Negative
            on any path raises.

    Returns:
        ``(new_state, contributed)``, ``contributed`` capped at ``state.room``.

    Raises:
        ValueError: If ``requested`` is negative on any path.
    """
    requested_arr = _non_negative(requested)
    contributed = np.minimum(requested_arr, state.room)
    new_state = updated(
        state,
        balance=state.balance + contributed,
        room=state.room - contributed,
    )
    return new_state, np.asarray(contributed, dtype=np.float64)


def withdraw(state: TfsaState, requested: ArrayLike) -> tuple[TfsaState, WithdrawalResult]:
    """Withdraw from a TFSA. Entirely tax-free.

    The withdrawal creates no room this year. It is accumulated into
    ``state.withdrawn_this_year``, which next January's :func:`restore_room`
    acts on.

    Args:
        state: Opening TFSA state.
        requested: Amount wanted this month, ``(n_paths,)``. Negative on any
            path raises.

    Returns:
        ``(new_state, result)``. ``result`` has everything in ``tax_free``.

    Raises:
        ValueError: If ``requested`` is negative on any path.
    """
    requested_arr = _non_negative(requested)
    new_balance, withdrawn, shortfall = base.withdraw(state.balance, requested_arr)
    new_state = updated(
        state,
        balance=new_balance,
        withdrawn_this_year=state.withdrawn_this_year + withdrawn,
    )
    zeros = np.zeros_like(withdrawn)
    result = WithdrawalResult(
        gross=withdrawn,
        fully_taxable=zeros,
        capital_gain=zeros,
        tax_free=withdrawn,
        shortfall=shortfall,
    )
    return new_state, result


def restore_room(state: TfsaState, params: RealParamSet) -> TfsaState:
    """Room added back in January for the whole of last year's withdrawals.

    Adds ``state.withdrawn_this_year`` to ``state.room`` and zeroes the
    counter. Called only from the January phase of the step, which is what
    enforces the lag: ``state.withdrawn_this_year`` is always *last* calendar
    year's total by the time January calls this, because the step resets it to
    zero right after.

    Args:
        state: Opening TFSA state for January.
        params: The ``tfsa`` parameter set, supplying
            ``recontribution.restored_after_n_year_ends``.

    Returns:
        Updated state, room restored and the counter zeroed.

    Raises:
        ValueError: If ``recontribution.restored_after_n_year_ends`` is not
            exactly 1 — a single counter cannot express any other lag.
    """
    lag = params.number("recontribution.restored_after_n_year_ends")
    if lag != 1:
        raise ValueError(
            f"{params.name}.yaml recontribution.restored_after_n_year_ends is "
            f"{lag!r}, but restore_room can only express a lag of exactly one "
            "year — a single counter cannot represent any other lag."
        )
    return updated(
        state,
        room=state.room + state.withdrawn_this_year,
        withdrawn_this_year=np.zeros_like(state.withdrawn_this_year),
    )


def erode_nominal(state: TfsaState, inflation_rate: float) -> TfsaState:
    """One January's decay of ``room`` and ``withdrawn_this_year``.

    Both are nominal dollar figures per
    ``tests/core/test_state_nominal_or_real.py``: unused room is carried
    forward indefinitely, and next January restores to room the *nominal*
    amount withdrawn, not its real value at the time. This applies one year's
    decay at a time — the erosion is annual, not monthly, which is
    ``docs/limitations.md`` L57.

    Args:
        state: Opening TFSA state, before this January's erosion.
        inflation_rate: Assumed annual inflation as a bare fraction.

    Returns:
        Updated state with ``room`` and ``withdrawn_this_year`` eroded; every
        other field unchanged.
    """
    factor = nominal_carry_factor(inflation_rate)
    return updated(
        state,
        room=state.room * factor,
        withdrawn_this_year=state.withdrawn_this_year * factor,
    )


def _non_negative(amount: ArrayLike) -> np.ndarray:
    """``amount`` as a float64 array, or raise if any path is negative."""
    amount_arr = np.asarray(amount, dtype=np.float64)
    if np.any(amount_arr < 0):
        raise ValueError(
            f"amount must be non-negative on every path, got {amount_arr!r}. "
            "A negative request is a caller bug, not a reverse transaction."
        )
    return amount_arr

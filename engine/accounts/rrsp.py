# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registered Retirement Savings Plan.

Contributions are deductible against income in the year made; withdrawals are
fully taxable. Contribution room accrues as a fraction of the *prior* year's
earned income up to an annual dollar limit, plus unused room carried forward
(``docs/limitations.md`` L25). Both come from ``params/``.

Room is granted in January for the year, on the previous year's earnings, and
is then consumed by contributions across the months. It does not accrue monthly
from the current month's income; a person may contribute the whole year's room
in January.

Parameters come from ``params/{year}/rrif.yaml``, which holds both halves of
one program: the RRSP under ``rrsp:`` and the RRIF it converts into under
``rrif:``. The conversion age sits at the top level, joining them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.accounts import base
from engine.accounts.base import WithdrawalResult
from engine.core.indexation import RealParamSet, nominal_carry_factor
from engine.core.state import RrspState, updated


def room_accrued(
    prior_year_earned_income: ArrayLike,
    params: RealParamSet,
    january_month_index: int,
) -> NDArray[np.float64]:
    """New contribution room granted in January, on last year's earned income (L25).

    Args:
        prior_year_earned_income: Earned income over the whole prior calendar
            year, real dollars, ``(n_paths,)``. Last year's, not this year's
            and not this month's.
        params: The ``rrif`` parameter set — RRSP and RRIF share one file —
            supplying ``rrsp.room.accrual_rate`` and
            ``rrsp.room.dollar_limit_annual``.
        january_month_index: Month index of January of the year the room is
            granted for.

    Returns:
        Room accrued for the year, ``(n_paths,)``, floored at zero.
    """
    rate = params.number("rrsp.room.accrual_rate")
    dollar_limit = params.annual_amount("rrsp.room.dollar_limit_annual", january_month_index)
    income = np.asarray(prior_year_earned_income, dtype=np.float64)
    accrued = np.minimum(rate * income, dollar_limit)
    return np.asarray(np.clip(accrued, 0, None), dtype=np.float64)


def contribute(state: RrspState, requested: ArrayLike) -> tuple[RrspState, NDArray[np.float64]]:
    """Contribute up to the room remaining right now.

    ``state.room`` is what is left at this point in the year, already reduced
    by contributions made in earlier months.

    Args:
        state: Opening RRSP state.
        requested: Desired contribution this month, ``(n_paths,)``. Negative
            on any path raises.

    Returns:
        ``(new_state, contributed)``. ``contributed`` is capped at
        ``state.room``; the excess is not contributed rather than penalised.

    Raises:
        ValueError: If ``requested`` is negative on any path.
    """
    requested_arr = _non_negative(requested)
    contributed = np.minimum(requested_arr, state.room)
    new_state = updated(
        state,
        balance=state.balance + contributed,
        room=state.room - contributed,
        contributed_ytd=state.contributed_ytd + contributed,
    )
    return new_state, np.asarray(contributed, dtype=np.float64)


def withdraw(state: RrspState, requested: ArrayLike) -> tuple[RrspState, WithdrawalResult]:
    """Withdraw from an RRSP. The full amount is taxable income.

    Withholding tax at source is remitted in the month of withdrawal and
    reduces the balance owing settled in the following year's filing month;
    this function returns the gross withdrawal only — the step applies the
    withholding and records it in ``remitted_ytd``.

    Args:
        state: Opening RRSP state.
        requested: Amount wanted this month, ``(n_paths,)``. Negative on any
            path raises.

    Returns:
        ``(new_state, result)``. ``result`` has everything in
        ``fully_taxable``.

    Raises:
        ValueError: If ``requested`` is negative on any path.
    """
    requested_arr = _non_negative(requested)
    new_balance, withdrawn, shortfall = base.withdraw(state.balance, requested_arr)
    new_state = updated(state, balance=new_balance)
    zeros = np.zeros_like(withdrawn)
    result = WithdrawalResult(
        gross=withdrawn,
        fully_taxable=withdrawn,
        capital_gain=zeros,
        tax_free=zeros,
        shortfall=shortfall,
    )
    return new_state, result


def convert(state: RrspState, fraction: float) -> tuple[RrspState, NDArray[np.float64]]:
    """Move a fraction of the balance out of the RRSP, for the RRIF to receive.

    ``fraction`` is a policy choice, the same for every path, not a per-path
    quantity.

    Args:
        state: Opening RRSP state.
        fraction: Fraction of the balance converted, ``[0, 1]``. Outside that
            range raises: below zero would move money *into* the RRSP and
            hand a negative amount to
            ``engine.accounts.rrif.receive_conversion``, which does not guard
            against one; above one would drive the balance negative.

    Returns:
        ``(new_state, moved)``. ``new_state.converted_fraction_applied`` is
        ``True``. ``moved`` is the amount the caller adds to the RRIF via
        ``engine.accounts.rrif.receive_conversion``.

    Raises:
        ValueError: If ``fraction`` is outside ``[0, 1]``.
    """
    if not 0 <= fraction <= 1:
        raise ValueError(f"fraction must be in [0, 1], got {fraction!r}.")
    moved = state.balance * fraction
    new_state = updated(
        state,
        balance=state.balance - moved,
        converted_fraction_applied=True,
    )
    return new_state, np.asarray(moved, dtype=np.float64)


def must_convert(age_at_end_of_year: int, params: RealParamSet) -> bool:
    """Whether an RRSP must be converted to a RRIF by the end of this year.

    The deadline is 31 December of the year the holder reaches the conversion
    age, whatever month their birthday falls in, so this is tested against age
    at year end and the conversion is applied by the year-end close. The
    converted plan's first minimum is fixed the following January.

    Args:
        age_at_end_of_year: Age in whole years on 31 December. See
            ``engine.core.timeline.age_at_end_of_year``. Not per-path: a
            person's age does not vary by path.
        params: The ``rrif`` parameter set, supplying ``conversion_age_years``.

    Returns:
        Whether conversion is due this year.
    """
    conversion_age = params.number("conversion_age_years")
    return bool(age_at_end_of_year >= conversion_age)


def erode_nominal(state: RrspState, inflation_rate: float) -> RrspState:
    """One January's decay of ``state.room``, fixed in nominal terms.

    Unused contribution room is a dollar figure fixed in nominal terms; held
    constant it overstates the room
    (``tests/core/test_state_nominal_or_real.py``). This applies one year's
    decay at a time — the erosion is annual, not monthly, which is
    ``docs/limitations.md`` L57.

    Args:
        state: Opening RRSP state, before this January's erosion.
        inflation_rate: Assumed annual inflation as a bare fraction.

    Returns:
        Updated state with ``room`` eroded; every other field unchanged.
    """
    factor = nominal_carry_factor(inflation_rate)
    return updated(state, room=state.room * factor)


def _non_negative(amount: ArrayLike) -> np.ndarray:
    """``amount`` as a float64 array, or raise if any path is negative."""
    amount_arr = np.asarray(amount, dtype=np.float64)
    if np.any(amount_arr < 0):
        raise ValueError(
            f"amount must be non-negative on every path, got {amount_arr!r}. "
            "A negative request is a caller bug, not a reverse transaction."
        )
    return amount_arr

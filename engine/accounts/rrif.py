# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registered Retirement Income Fund.

A RRIF has a mandatory minimum withdrawal each year, set by a factor that
depends on age. Two things about it are easy to get wrong:

- The factor is selected by **age at the start of the year**, not age at
  year end or on the withdrawal date.
- The minimum is computed on the **opening balance**, on 1 January, before
  that year's growth — never on a mid-year balance.

It is an annual obligation with a 31 December deadline: fixed in January and
satisfied over the months that follow, in whatever pattern the policy
chooses, with whatever is left forced out by the year-end close
(``docs/limitations.md`` L26).

Factors come from ``params/{year}/rrif.yaml`` under ``rrif.minimum_factors``.
That file holds one program at two stages of life — the RRSP that accumulates
and the RRIF it becomes — with the conversion age at the top level joining
them. There is no formula in this file: even the pre-table basis is a constant
in the YAML, because ``1 / (C - age)`` puts a ``C`` in a ``.py`` file otherwise.

:func:`minimum_withdrawal` takes plain values rather than a state object so
that ``engine.accounts.lif`` can call it directly: a LIF's minimum *is* the
RRIF minimum, computed on the LIF's own opening balance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.accounts import base
from engine.accounts.base import WithdrawalResult
from engine.core.indexation import RealParamSet
from engine.core.state import RrifState, updated


def minimum_factor(age_at_start_of_year: int, params: RealParamSet) -> float:
    """Mandatory withdrawal factor for a given age.

    Below the table's first age, the factor is ``1 / (C - age)`` where ``C``
    is ``rrif.minimum_factors.pre_table.formula_constant``. At and above
    ``rrif.minimum_factors.terminal_age_years``, the terminal row applies.
    Otherwise, the row for that exact age.

    Args:
        age_at_start_of_year: Age in whole years on 1 January. Not age at
            year end, and not age in the current month. See
            ``engine.core.timeline.age_at_start_of_year``. Not per-path: a
            person's age does not vary by path.
        params: The ``rrif`` parameter set, supplying the factor table and the
            pre-table basis.

    Returns:
        Factor as a bare fraction.
    """
    table = params.get("rrif.minimum_factors.by_age")
    terminal_age = params.number("rrif.minimum_factors.terminal_age_years")
    first_age = min(int(age) for age in table)
    if age_at_start_of_year >= terminal_age:
        return float(table[str(int(terminal_age))])
    if age_at_start_of_year < first_age:
        formula_constant = params.number("rrif.minimum_factors.pre_table.formula_constant")
        return 1 / (formula_constant - age_at_start_of_year)
    return float(table[str(age_at_start_of_year)])


def minimum_withdrawal(
    opening_balance: ArrayLike,
    age_at_start_of_year: int,
    opened_year: int | None,
    year: int,
    params: RealParamSet,
) -> NDArray[np.float64]:
    """Mandatory minimum withdrawal for the **whole year**.

    Called once, by the January phase of the step. The result is an annual
    amount that is then drawn down across the year; it is never recomputed
    mid-year, because the balance it is based on no longer exists after
    January.

    Returns zero when ``opened_year is None`` (nothing has been opened) or
    ``opened_year == year`` (opened this calendar year): the first minimum
    applies only from the January after the plan opens (``docs/limitations.md``
    L26).

    Args:
        opening_balance: Balance on 1 January, before the year's growth and
            before any of the year's withdrawals, ``(n_paths,)``.
        age_at_start_of_year: Age in whole years on 1 January. Not per-path.
        opened_year: The calendar year the plan was opened, or ``None`` if it
            holds no balance yet.
        year: The calendar year the minimum is being fixed for.
        params: The ``rrif`` parameter set.

    Returns:
        Annual minimum withdrawal in real dollars, ``(n_paths,)``.
    """
    opening_balance_arr = np.asarray(opening_balance, dtype=np.float64)
    if opened_year is None or opened_year == year:
        return np.zeros_like(opening_balance_arr)
    factor = minimum_factor(age_at_start_of_year, params)
    return np.asarray(opening_balance_arr * factor, dtype=np.float64)


def minimum_still_required(
    annual_minimum: ArrayLike,
    withdrawn_ytd: ArrayLike,
    months_remaining_in_year: int,
) -> NDArray[np.float64]:
    """How much must come out this month for the annual minimum to be met.

    Zero for most of the year: a policy is free to take nothing in January and
    the whole minimum in December. This returns a non-zero floor only when the
    remaining months can no longer accommodate the shortfall, which in practice
    means December (``months_remaining_in_year == 1``).

    Args:
        annual_minimum: The year's minimum, fixed in January, ``(n_paths,)``.
        withdrawn_ytd: Amount already withdrawn this calendar year,
            ``(n_paths,)``.
        months_remaining_in_year: Months left including this one. One in
            December.

    Returns:
        Amount that must be withdrawn this month, ``(n_paths,)``, floored at
        zero.
    """
    shortfall = np.clip(
        np.asarray(annual_minimum, dtype=np.float64) - np.asarray(withdrawn_ytd, dtype=np.float64),
        0,
        None,
    )
    return np.asarray(np.where(months_remaining_in_year <= 1, shortfall, 0.0), dtype=np.float64)


def withdraw(
    state: RrifState,
    requested: ArrayLike,
    floor: ArrayLike,
) -> tuple[RrifState, WithdrawalResult, NDArray[np.float64]]:
    """Withdraw at least this month's required floor. Fully taxable.

    A policy may request less than the floor; the floor still comes out. That
    interaction is why ``floor`` is an argument rather than being recomputed
    here — and why it is *this month's* floor from :func:`minimum_still_required`,
    not the annual minimum, which would be taken twelve times over.

    Args:
        state: Opening RRIF state.
        requested: Policy-requested withdrawal for this month, ``(n_paths,)``.
            Negative on any path raises.
        floor: Output of :func:`minimum_still_required`. ``gross`` is at least
            this on every path, even when ``requested`` is less.

    Returns:
        ``(new_state, result, above_minimum)``. ``result`` has everything in
        ``fully_taxable``. ``above_minimum`` is the base the step's
        withholding applies to:

        ``above_minimum = max(0, gross - max(0, state.annual_minimum -
        state.withdrawn_ytd))``

        **Not** ``gross - floor``: the floor is zero for eleven months of the
        year, so that reading would withhold on the minimum itself whenever it
        is taken early. This is a deliberate convention
        (``docs/limitations.md`` L56).

    Raises:
        ValueError: If ``requested`` is negative on any path.
    """
    requested_arr = _non_negative(requested)
    floor_arr = np.asarray(floor, dtype=np.float64)
    target = np.maximum(requested_arr, floor_arr)
    new_balance, withdrawn, shortfall = base.withdraw(state.balance, target)
    minimum_remaining = np.clip(state.annual_minimum - state.withdrawn_ytd, 0, None)
    above_minimum = np.clip(withdrawn - minimum_remaining, 0, None)
    new_state = updated(
        state,
        balance=new_balance,
        withdrawn_ytd=state.withdrawn_ytd + withdrawn,
    )
    zeros = np.zeros_like(withdrawn)
    result = WithdrawalResult(
        gross=withdrawn,
        fully_taxable=withdrawn,
        capital_gain=zeros,
        tax_free=zeros,
        shortfall=shortfall,
    )
    return new_state, result, above_minimum


def receive_conversion(state: RrifState, amount: ArrayLike, year: int) -> RrifState:
    """Receive an amount converted in from an RRSP.

    Sets ``opened_year`` to ``year`` when the RRIF held nothing before
    (``opened_year is None``); otherwise leaves it, since the RRIF was already
    open and its minimum schedule already running. Per L26, the first minimum
    then applies from the next January, which falls out of
    :func:`minimum_withdrawal` returning zero when ``opened_year == year``.

    Args:
        state: Opening RRIF state, before receiving the conversion.
        amount: Amount moved in from the RRSP, ``(n_paths,)``.
        year: The calendar year the conversion happens in.

    Returns:
        Updated state.
    """
    amount_arr = np.asarray(amount, dtype=np.float64)
    opened_year = state.opened_year if state.opened_year is not None else year
    return updated(state, balance=state.balance + amount_arr, opened_year=opened_year)


def spousal_rollover(
    balance: ArrayLike,
    surviving_spouse_balance: ArrayLike,
) -> tuple[NDArray[np.float64], ...]:
    """Roll a deceased person's RRIF to their surviving spouse, tax-deferred.

    Without a surviving spouse the balance is instead brought fully into income
    in the year of death; that case is handled by the step, not here. Death is
    resolved monthly, so the rollover happens in the month of death and the
    survivor's own minimum for the year is unaffected — theirs was fixed in
    January from their own opening balance.

    Args:
        balance: The deceased person's RRIF balance, ``(n_paths,)``.
        surviving_spouse_balance: The survivor's RRIF balance, ``(n_paths,)``.

    Returns:
        ``(deceased_balance, survivor_balance)`` after the rollover.
    """
    raise NotImplementedError


def _non_negative(amount: ArrayLike) -> np.ndarray:
    """``amount`` as a float64 array, or raise if any path is negative."""
    amount_arr = np.asarray(amount, dtype=np.float64)
    if np.any(amount_arr < 0):
        raise ValueError(
            f"amount must be non-negative on every path, got {amount_arr!r}. "
            "A negative request is a caller bug, not a reverse transaction."
        )
    return amount_arr

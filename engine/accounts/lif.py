# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Life Income Fund: the decumulation phase of a locked-in account.

A LIF has both a mandatory minimum and, in most jurisdictions, a maximum —
the part that distinguishes it from ``engine.accounts.rrif``. **There is no
``minimum_withdrawal`` in this module.** A LIF's minimum *is* the RRIF
minimum: ``open_year`` fixes it by calling
``engine.accounts.rrif.minimum_withdrawal`` with the LIF's own opening
balance and ``opened_year``, exactly as it would for a RRIF. Do not add a
second copy of that arithmetic here.

**The jurisdiction is not the province of residence.** A LIF is governed by
the pension legislation of the jurisdiction its originating pension was
registered under: a household resident in Alberta may hold an
Ontario-registered LIF, drawn under Ontario's table while filing Alberta
income tax. Every function here that takes a parameter set takes the
*registration* jurisdiction's — ``ParamYear.jurisdiction`` keyed by
``LifState.jurisdiction`` — never ``ParamYear.province(household.province)``.

Both the minimum and the maximum are **annual** figures fixed in January from
the 1 January balance and enforced against the year-to-date total, not a
single month's amount.

Not every jurisdiction imposes a maximum; some prescribe a RRIF-like account
with a minimum and no ceiling. :func:`has_maximum` is what separates that
rule from a table nobody has entered yet.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.accounts import base
from engine.accounts.base import WithdrawalResult
from engine.core.indexation import RealParamSet
from engine.core.state import LifState, updated


def has_maximum(params: RealParamSet) -> bool:
    """Whether this jurisdiction caps annual LIF withdrawals.

    A scalar rule about a jurisdiction, not a per-path quantity, so this
    returns a plain ``bool`` rather than an array.

    Without the flag, a jurisdiction with no ceiling is indistinguishable
    from one whose maximum table has not been entered yet — both would raise
    on lookup. The flag makes "there is no maximum" a fact a human verified,
    not an inference.

    Args:
        params: The parameter set of the jurisdiction the account is
            **registered** in.

    Returns:
        True if :func:`maximum_withdrawal` should be consulted.

    Raises:
        MissingParameterError: If the flag itself is absent. An unverified
            jurisdiction is not assumed to be uncapped.
    """
    return bool(params.get("lif.has_maximum"))


def maximum_withdrawal(
    opening_balance: ArrayLike,
    age_at_start_of_year: int,
    params: RealParamSet,
) -> NDArray[np.float64]:
    """Jurisdiction-specific maximum LIF withdrawal for the **whole year**.

    Called once, by the January phase of the step, and then drawn down. Call
    :func:`has_maximum` first: this raises for a jurisdiction that has no
    ceiling rather than returning infinity, so that an uncapped account is
    handled by the caller deciding not to cap it.

    The factor comes from ``lif.maximum_factors.by_age``, the terminal row
    applying at and above ``lif.maximum_factors.terminal_age_years``. Unlike
    ``engine.accounts.rrif.minimum_factor``, there is no formula below the
    table: an age below the table's first row raises rather than
    extrapolating.

    In reality Alberta's maximum is the greater of the factor result and the
    prior year's investment return; this function applies the factor result
    only and does not read the jurisdiction file's flag for that rule
    (``docs/limitations.md`` L27).

    Args:
        opening_balance: Balance on 1 January, before growth, ``(n_paths,)``.
        age_at_start_of_year: Age in whole years on 1 January. Not per-path.
        params: The parameter set of the jurisdiction the account is
            **registered** in.

    Returns:
        Maximum permitted withdrawal for the year, ``(n_paths,)``.

    Raises:
        MissingParameterError: If the jurisdiction has no maximum table. Guard
            with :func:`has_maximum`.
        ValueError: If ``age_at_start_of_year`` is below the table's first
            row. The maximum is not extrapolated below it.
    """
    table = params.get("lif.maximum_factors.by_age")
    terminal_age = params.number("lif.maximum_factors.terminal_age_years")
    first_age = min(int(age) for age in table)
    if age_at_start_of_year < first_age:
        raise ValueError(
            f"age {age_at_start_of_year} is below the LIF maximum table's "
            f"first row (age {first_age}) for jurisdiction {params.name!r}. "
            "The maximum is not extrapolated below the table."
        )
    if age_at_start_of_year >= terminal_age:
        factor = float(table[str(int(terminal_age))])
    else:
        factor = float(table[str(age_at_start_of_year)])
    opening_balance_arr = np.asarray(opening_balance, dtype=np.float64)
    return np.asarray(opening_balance_arr * factor, dtype=np.float64)


def withdraw(
    state: LifState,
    requested: ArrayLike,
    floor: ArrayLike,
    maximum_remaining: ArrayLike,
) -> tuple[LifState, WithdrawalResult, NDArray[np.float64]]:
    """Withdraw this month within the annual bounds. Fully taxable.

    Mirrors ``engine.accounts.rrif.withdraw``, including the same
    ``above_minimum`` formula from ``state.annual_minimum`` and
    ``state.withdrawn_ytd`` (``docs/limitations.md`` L56):

    ``above_minimum = max(0, gross - max(0, state.annual_minimum -
    state.withdrawn_ytd))``

    Args:
        state: Opening LIF state.
        requested: Policy-requested withdrawal for this month, ``(n_paths,)``,
            clamped into ``[floor, maximum_remaining]``. Negative on any path
            raises.
        floor: RRIF-equivalent floor for this month, from
            ``engine.accounts.rrif.minimum_still_required``.
        maximum_remaining: What is left of the year's maximum, from
            ``engine.accounts.base.remaining_annual_allowance`` applied to
            :func:`maximum_withdrawal`. Not the annual maximum itself. For a
            jurisdiction where :func:`has_maximum` is false, pass the balance:
            the account is still capped by what is in it.

    Returns:
        ``(new_state, result, above_minimum)``. ``result`` has everything in
        ``fully_taxable``. A floor above the *balance* is ordinary: the
        account takes the balance and reports the shortfall.

    Raises:
        ValueError: If ``requested`` is negative on any path, or if ``floor``
            exceeds ``maximum_remaining`` on any path — that can only mean the
            RRIF minimum factor exceeds the LIF maximum factor at this age, a
            parameter error rather than a household's problem.
    """
    requested_arr = _non_negative(requested)
    floor_arr = np.asarray(floor, dtype=np.float64)
    maximum_remaining_arr = np.asarray(maximum_remaining, dtype=np.float64)
    if np.any(floor_arr > maximum_remaining_arr):
        raise ValueError(
            f"floor ({floor_arr!r}) exceeds maximum_remaining "
            f"({maximum_remaining_arr!r}) on at least one path. The RRIF "
            "minimum factor cannot exceed the LIF maximum factor at the same "
            "age; this is a parameter error, not a household's problem."
        )
    target = np.clip(requested_arr, floor_arr, maximum_remaining_arr)
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


def _non_negative(amount: ArrayLike) -> np.ndarray:
    """``amount`` as a float64 array, or raise if any path is negative."""
    amount_arr = np.asarray(amount, dtype=np.float64)
    if np.any(amount_arr < 0):
        raise ValueError(
            f"amount must be non-negative on every path, got {amount_arr!r}. "
            "A negative request is a caller bug, not a reverse transaction."
        )
    return amount_arr

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Non-registered (taxable) investment account.

The only account whose tax depends on how the money got there. It carries an
adjusted cost base: a withdrawal realizes a capital gain proportional to the
embedded gain, and only the included fraction of that gain enters income.
Distributions are taxed in the year received whether or not they are withdrawn,
which is what makes this account's drag different from a registered one's
(``docs/limitations.md`` L36).

On a monthly timestep, distributions arrive monthly and accumulate into the
year-to-date income ledger; the tax on them is assessed once, at the December
close, and paid in the following year's filing month. Real funds distribute
monthly, quarterly, or annually depending on the holding, and that schedule is
not modelled: the assumed yield is spread evenly across the twelve months.
This affects the timing of the ACB increase within a year, not the year's
total.

Nothing here imports ``engine.mc``: the weighted yields
(``MarketInputs.weighted_yields(kind)``) arrive as a plain
``(interest, dividend, gains)`` triple of annual rates, typed structurally.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.accounts import base
from engine.accounts.base import WithdrawalResult
from engine.core.indexation import nominal_carry_factor
from engine.core.state import TaxableState, updated


def distributions_monthly(
    balance: ArrayLike,
    weighted_yields: tuple[float, float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Interest, dividends, and distributed gains for **one month**.

    Taxed in the year received regardless of whether they are withdrawn.
    Reinvested distributions increase the adjusted cost base (see
    :func:`reinvest`).

    ``weighted_yields`` is annual, because that is how a yield assumption is
    stated; the division to a monthly figure happens here, once.

    Args:
        balance: Balance at the start of the month, ``(n_paths,)``.
        weighted_yields: ``(interest, dividend, gains)`` annual yield as a
            bare fraction of balance, each not per-path — one figure for the
            whole holding, from ``MarketInputs.weighted_yields(kind)``.

    Returns:
        ``(interest, eligible_dividends, capital_gains_distributed)`` for this
        month, each ``(n_paths,)``.
    """
    balance_arr = np.asarray(balance, dtype=np.float64)
    interest_yield, dividend_yield, gains_yield = weighted_yields
    interest = balance_arr * interest_yield / 12
    dividends = balance_arr * dividend_yield / 12
    gains = balance_arr * gains_yield / 12
    return (
        np.asarray(interest, dtype=np.float64),
        np.asarray(dividends, dtype=np.float64),
        np.asarray(gains, dtype=np.float64),
    )


def reinvest(
    state: TaxableState,
    distributions: tuple[ArrayLike, ArrayLike, ArrayLike],
) -> TaxableState:
    """Reinvest a month's distributions, raising both balance and ACB.

    Raising ``acb`` by the same amount as ``balance`` is what keeps
    distributions from being taxed twice on the way out (L36): they are
    already taxed as income when received, so no further gain is realized on
    the reinvested dollars themselves.

    Args:
        state: Opening taxable state.
        distributions: ``(interest, eligible_dividends, capital_gains)`` for
            the month, as returned by :func:`distributions_monthly`, each
            ``(n_paths,)``.

    Returns:
        Updated state, ``balance`` and ``acb`` both raised by the total.
    """
    interest, dividends, gains = distributions
    total = (
        np.asarray(interest, dtype=np.float64)
        + np.asarray(dividends, dtype=np.float64)
        + np.asarray(gains, dtype=np.float64)
    )
    return updated(state, balance=state.balance + total, acb=state.acb + total)


def price_growth(
    state: TaxableState,
    monthly_total_return: ArrayLike,
    weighted_yields: tuple[float, float, float],
) -> TaxableState:
    """Apply one month's price change to the balance. Leaves ``acb`` untouched.

    ``monthly_total_return`` is already a monthly figure — the step computes
    it from ``MarketInputs.weights(kind)`` and the month's draws.
    ``weighted_yields`` is annual, exactly as :func:`distributions_monthly`
    takes it. Price change is total return **less** the yields, since the
    yields are paid out as distributions and are not also price appreciation:
    ``balance * (monthly_total_return - sum(weighted_yields) / 12)``. Mixing
    up which of the two arguments is monthly and which is annual is the bug
    this docstring exists to prevent.

    Args:
        state: Opening taxable state.
        monthly_total_return: This month's total return, real, as a bare
            fraction, ``(n_paths,)``.
        weighted_yields: ``(interest, dividend, gains)`` annual yield, as in
            :func:`distributions_monthly`.

    Returns:
        Updated state, ``balance`` grown by the price-only return; ``acb``
        unchanged.
    """
    monthly_total_return_arr = np.asarray(monthly_total_return, dtype=np.float64)
    annual_yield_total = sum(weighted_yields)
    price_return = monthly_total_return_arr - annual_yield_total / 12
    return updated(state, balance=state.balance * (1 + price_return))


def withdraw(state: TaxableState, requested: ArrayLike) -> tuple[TaxableState, WithdrawalResult]:
    """Withdraw, realizing a proportional share of the embedded gain.

    The realized gain is ``gross * (balance - acb) / balance``, where
    ``gross = min(balance, requested)`` — never the raw ``requested``. Pricing
    the gain against ``requested`` instead would let a withdrawal larger than
    the balance realize a gain on money that was never actually taken out of
    the account, and would break ``capital_gain + tax_free == gross``, which
    must hold on every path including a shortfall one. The realized gain may
    itself be negative: ``acb > balance`` is an ordinary loss, not an error
    (``TaxableState``'s own docstring). Where ``balance`` is zero the division
    is guarded rather than left to raise — this function withdraws nothing and
    reports the full shortfall instead.

    Args:
        state: Opening taxable state.
        requested: Amount wanted this month, ``(n_paths,)``. Negative on any
            path raises.

    Returns:
        ``(new_state, result)``. ``result.gross`` is ``min(balance,
        requested)``; ``result.capital_gain`` is the realized gain before the
        inclusion rate; ``result.tax_free`` is the return of capital (the
        matching reduction in ``acb``); ``result.fully_taxable`` is always
        zero.

    Raises:
        ValueError: If ``requested`` is negative on any path.
    """
    requested_arr = _non_negative(requested)
    new_balance, withdrawn, shortfall = base.withdraw(state.balance, requested_arr)
    zero_balance = state.balance == 0
    balance_safe = np.where(zero_balance, 1.0, state.balance)
    acb_reduction = np.where(zero_balance, 0.0, withdrawn * state.acb / balance_safe)
    capital_gain = withdrawn - acb_reduction
    new_state = updated(state, balance=new_balance, acb=state.acb - acb_reduction)
    zeros = np.zeros_like(withdrawn)
    result = WithdrawalResult(
        gross=withdrawn,
        fully_taxable=zeros,
        capital_gain=capital_gain,
        tax_free=acb_reduction,
        shortfall=shortfall,
    )
    return new_state, result


def deemed_disposition(state: TaxableState) -> NDArray[np.float64]:
    """Gain on the whole balance, for the terminal return.

    Returns the gain only; it does not change ``state``. The caller applies
    this at the second death, where the whole holding is deemed disposed of.

    Args:
        state: Taxable state at the moment of deemed disposition.

    Returns:
        ``balance - acb``, ``(n_paths,)``, possibly negative.
    """
    return np.asarray(state.balance - state.acb, dtype=np.float64)


def erode_nominal(state: TaxableState, inflation_rate: float) -> TaxableState:
    """One January's decay of ``acb`` only — never ``balance``.

    A capital gain is taxed on the *nominal* gain, so holding ``acb`` constant
    in real terms would exempt the inflation component of every gain from tax.
    This applies one year's decay at a time — the erosion is annual, not
    monthly, which is ``docs/limitations.md`` L57.

    Args:
        state: Opening taxable state, before this January's erosion.
        inflation_rate: Assumed annual inflation as a bare fraction.

    Returns:
        Updated state with ``acb`` eroded; ``balance`` unchanged.
    """
    factor = nominal_carry_factor(inflation_rate)
    return updated(state, acb=state.acb * factor)


def _non_negative(amount: ArrayLike) -> np.ndarray:
    """``amount`` as a float64 array, or raise if any path is negative."""
    amount_arr = np.asarray(amount, dtype=np.float64)
    if np.any(amount_arr < 0):
        raise ValueError(
            f"amount must be non-negative on every path, got {amount_arr!r}. "
            "A negative request is a caller bug, not a reverse transaction."
        )
    return amount_arr

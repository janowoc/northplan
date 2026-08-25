"""Non-registered (taxable) investment account.

The only account whose tax depends on how the money got there. It carries an
adjusted cost base: a withdrawal realizes a capital gain proportional to the
embedded gain, and only the included fraction of that gain enters income.
Distributions are taxed in the year received whether or not they are withdrawn,
which is what makes this account's drag different from a registered one's.

On a monthly timestep, distributions arrive monthly and accumulate into the
year-to-date income ledger; the tax on them is assessed once, at the December
close, and paid in the following year's filing month. Real funds distribute
monthly, quarterly, or annually depending on the holding, and that schedule is
not modelled: the assumed yield is spread evenly across the twelve months. That
is an approximation and it is stated rather than hidden. It affects the timing
of the ACB increase within a year, not the year's total.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.accounts.base import WithdrawalResult
from engine.params.loader import ParamSet


def monthly_distributions(
    balance: ArrayLike,
    annual_yield_rate: ArrayLike,
) -> tuple[NDArray[np.float64], ...]:
    """Interest, dividends, and distributed gains for **one month**.

    Taxed in the year received regardless of whether they are withdrawn.
    Reinvested distributions increase the adjusted cost base, and failing to
    add them is the classic double-taxation bug in this account type — the
    monthly timestep gives twelve chances a year to forget it.

    The rate argument is annual because that is how a yield assumption is
    stated; the division to a monthly figure happens here, once, and the name
    says which is which.

    Args:
        balance: Balance at the start of the month, ``(n_paths,)``.
        annual_yield_rate: Distribution yield for a full year as a bare
            fraction, ``(n_paths,)``.

    Returns:
        ``(interest, eligible_dividends, capital_gains_distributed)`` for this
        month, each ``(n_paths,)``.
    """
    raise NotImplementedError


def taxable_income_from_distributions(
    interest: ArrayLike,
    eligible_dividends: ArrayLike,
    capital_gains: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Amount entering taxable income from distributions.

    Each component enters at its own rate: interest in full, eligible dividends
    grossed up, capital gains at the inclusion rate. The gross-up and inclusion
    rate are parameters.

    Works on whatever period it is handed. The step feeds it a month's
    distributions and adds the result to the year-to-date ledger; a golden test
    may feed it a year's. The rates do not depend on the period, which is why
    this one function serves both.

    Args:
        interest: Interest income, ``(n_paths,)``.
        eligible_dividends: Actual dividends received, before gross-up.
        capital_gains: Distributed capital gains, before the inclusion rate.
        params: The ``federal`` parameter set.

    Returns:
        Amount added to taxable income, ``(n_paths,)``.
    """
    raise NotImplementedError


def withdraw(
    balance: ArrayLike,
    adjusted_cost_base: ArrayLike,
    requested: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], WithdrawalResult]:
    """Withdraw, realizing a proportional share of the embedded gain.

    Args:
        balance: Market value at the time of the withdrawal, ``(n_paths,)``.
        adjusted_cost_base: ACB of the holdings, including every distribution
            reinvested so far this year, ``(n_paths,)``.
        requested: Amount wanted this month, ``(n_paths,)``.

    Returns:
        ``(new_balance, new_acb, result)``. The result splits into
        ``capital_gain`` and ``tax_free`` return of capital; the caller applies
        the inclusion rate.
    """
    raise NotImplementedError

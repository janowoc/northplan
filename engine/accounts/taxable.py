"""Non-registered (taxable) investment account.

The only account whose tax depends on how the money got there. It carries an
adjusted cost base: a withdrawal realizes a capital gain proportional to the
embedded gain, and only the included fraction of that gain enters income.
Distributions are taxed annually whether or not they are withdrawn, which is
what makes this account's drag different from a registered one's.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.accounts.base import WithdrawalResult
from engine.params.loader import ParamSet


def annual_distributions(
    balance: ArrayLike,
    yield_rate: ArrayLike,
) -> tuple[NDArray[np.float64], ...]:
    """Interest, dividends, and distributed gains for the year.

    Taxed in the year received regardless of whether they are withdrawn.
    Reinvested distributions increase the adjusted cost base, and failing to
    add them is the classic double-taxation bug in this account type.

    Args:
        balance: Opening balance, ``(n_paths,)``.
        yield_rate: Distribution yield as a bare fraction, ``(n_paths,)``.

    Returns:
        ``(interest, eligible_dividends, capital_gains_distributed)``.
    """
    raise NotImplementedError


def taxable_income_from_distributions(
    interest: ArrayLike,
    eligible_dividends: ArrayLike,
    capital_gains: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Amount entering taxable income from a year's distributions.

    Each component enters at its own rate: interest in full, eligible dividends
    grossed up, capital gains at the inclusion rate. The gross-up and inclusion
    rate are parameters.

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
        balance: Market value, ``(n_paths,)``.
        adjusted_cost_base: ACB of the holdings, ``(n_paths,)``.
        requested: Amount wanted, ``(n_paths,)``.

    Returns:
        ``(new_balance, new_acb, result)``. The result splits into
        ``capital_gain`` and ``tax_free`` return of capital; the caller applies
        the inclusion rate.
    """
    raise NotImplementedError

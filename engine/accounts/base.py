"""Shared account primitives.

Growth and the split of a withdrawal into its tax components are the same
arithmetic for every account type; only the tax character differs. That
arithmetic lives here so there is one copy of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class WithdrawalResult:
    """What a withdrawal produced, split by tax treatment.

    Splitting at the point of withdrawal keeps the tax layer from having to
    know which account a dollar came from.

    Attributes:
        gross: Amount removed from the account, real dollars, ``(n_paths,)``.
        fully_taxable: Portion included in income at 100 percent — RRSP and
            RRIF withdrawals, and the taxable portion of an RESP EAP.
        capital_gain: Realized capital gain, before the inclusion rate.
        tax_free: Portion not included in income at all — TFSA withdrawals,
            return of capital, RESP contribution withdrawals.
        shortfall: Amount requested but unavailable, ``(n_paths,)``. Non-zero
            means the account was drained; the caller decides what to do.
    """

    gross: NDArray[np.float64]
    fully_taxable: NDArray[np.float64]
    capital_gain: NDArray[np.float64]
    tax_free: NDArray[np.float64]
    shortfall: NDArray[np.float64]


def grow(balance: ArrayLike, real_return: ArrayLike) -> NDArray[np.float64]:
    """Apply one year of real return to a balance.

    Args:
        balance: Opening balance, real dollars, ``(n_paths,)``.
        real_return: Real return for the year as a bare fraction, ``(n_paths,)``.
            Real, not nominal — see ``engine/__init__.py``.

    Returns:
        Closing balance before any contribution or withdrawal.
    """
    raise NotImplementedError


def withdraw(balance: ArrayLike, requested: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    """Take up to ``requested`` from ``balance``.

    Args:
        balance: Available balance, ``(n_paths,)``.
        requested: Amount wanted, ``(n_paths,)``. Negative requests raise.

    Returns:
        A tuple of ``(new_balance, withdrawn, shortfall)``, each ``(n_paths,)``.
        ``withdrawn + shortfall == requested`` on every path.
    """
    raise NotImplementedError

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

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


def grow(balance: ArrayLike, monthly_real_return: ArrayLike) -> NDArray[np.float64]:
    """Apply one month of real return to a balance.

    Args:
        balance: Opening balance for the month, real dollars, ``(n_paths,)``.
        monthly_real_return: Real, monthly return as a bare fraction,
            ``(n_paths,)``, converted from an annual assumption once in
            ``engine.mc.returns.generate``. An annual return passed here is a
            silent error: the balance comes out wrong by an order of
            magnitude by year end.

    Returns:
        Closing balance for the month, before any contribution or withdrawal.
    """
    balance_arr = np.asarray(balance, dtype=np.float64)
    monthly_real_return_arr = np.asarray(monthly_real_return, dtype=np.float64)
    return np.asarray(balance_arr * (1 + monthly_real_return_arr), dtype=np.float64)


def withdraw(balance: ArrayLike, requested: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    """Take up to ``requested`` from ``balance``.

    Args:
        balance: Available balance, ``(n_paths,)``.
        requested: Amount wanted this month, ``(n_paths,)``. Negative requests
            raise.

    Returns:
        A tuple of ``(new_balance, withdrawn, shortfall)``, each ``(n_paths,)``.
        ``withdrawn + shortfall == requested`` on every path.

    Raises:
        ValueError: If ``requested`` is negative on any path.
    """
    requested_arr = np.asarray(requested, dtype=np.float64)
    if np.any(requested_arr < 0):
        raise ValueError(
            f"requested must be non-negative on every path, got {requested_arr!r}. "
            "A negative request is a caller bug, not a reverse transaction."
        )
    balance_arr = np.asarray(balance, dtype=np.float64)
    withdrawn = np.clip(np.minimum(balance_arr, requested_arr), 0, None)
    shortfall = requested_arr - withdrawn
    new_balance = balance_arr - withdrawn
    return (
        np.asarray(new_balance, dtype=np.float64),
        np.asarray(withdrawn, dtype=np.float64),
        np.asarray(shortfall, dtype=np.float64),
    )


def remaining_annual_allowance(
    annual_limit: ArrayLike,
    taken_ytd: ArrayLike,
) -> NDArray[np.float64]:
    """How much of an annual limit is left for the rest of the year.

    Every annual bound in this package is enforced against a year-to-date
    total, but most account functions take that total directly as an
    argument (``state.grant_received_ytd``, ``state.withdrawn_ytd``, and so
    on) and compare it inline, rather than calling this first. This is the
    one place that comparison is factored out, because
    ``engine.accounts.lif.withdraw`` does not take a year-to-date total at
    all — it takes ``maximum_remaining`` directly, since the LIF maximum has
    no fixed floor the way a minimum does. The step (#19) is expected to call
    this once, in January, to turn the year's LIF maximum and the
    year-to-date withdrawn total into that argument.

    Args:
        annual_limit: The year's limit, fixed in January, ``(n_paths,)``.
        taken_ytd: Amount already used this calendar year, ``(n_paths,)``.

    Returns:
        Remaining allowance, floored at zero, ``(n_paths,)``.
    """
    return np.asarray(
        np.clip(
            np.asarray(annual_limit, dtype=np.float64) - np.asarray(taken_ytd, dtype=np.float64),
            0,
            None,
        ),
        dtype=np.float64,
    )

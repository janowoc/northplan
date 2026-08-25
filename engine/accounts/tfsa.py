"""Tax-Free Savings Account.

Contributions are not deductible; growth and withdrawals are entirely
tax-free. Withdrawn amounts are added back to contribution room, but only at
the start of the *following* calendar year — recontributing in the same year is
an over-contribution. That one-year lag is the thing to get right.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.accounts.base import WithdrawalResult
from engine.params.loader import ParamSet


def room_accrued(age: ArrayLike, params: ParamSet) -> NDArray[np.float64]:
    """New contribution room for one year.

    A flat annual amount from ``params``, granted from the year the person
    reaches the eligibility age.

    Args:
        age: Age in whole years at the end of the year, ``(n_paths,)``.
        params: The ``federal`` parameter set.

    Returns:
        Room accrued, ``(n_paths,)``.
    """
    raise NotImplementedError


def contribute(
    balance: ArrayLike,
    room: ArrayLike,
    requested: ArrayLike,
) -> tuple[NDArray[np.float64], ...]:
    """Contribute up to available room.

    Returns:
        ``(new_balance, new_room, contributed)``, each ``(n_paths,)``.
    """
    raise NotImplementedError


def withdraw(balance: ArrayLike, requested: ArrayLike) -> WithdrawalResult:
    """Withdraw from a TFSA. Entirely tax-free.

    Returns:
        A :class:`~engine.accounts.base.WithdrawalResult` with everything in
        ``tax_free``.
    """
    raise NotImplementedError


def room_restored(withdrawn_last_year: ArrayLike) -> NDArray[np.float64]:
    """Room added back at the start of this year for last year's withdrawals.

    Takes *last* year's withdrawals, not this year's, which is what enforces
    the one-year lag.

    Args:
        withdrawn_last_year: Total withdrawn in the prior calendar year,
            ``(n_paths,)``.

    Returns:
        Room restored, ``(n_paths,)``.
    """
    raise NotImplementedError

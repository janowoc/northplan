"""Tax-Free Savings Account.

Contributions are not deductible; growth and withdrawals are entirely
tax-free. Withdrawn amounts are added back to contribution room, but only at
the start of the *following* calendar year — recontributing in the same year is
an over-contribution. That lag is the thing to get right, and a monthly
timestep makes it easier to get wrong: room restored in the month after a
withdrawal rather than in the January after it is a plausible-looking bug that
hands the household eleven months of room it does not have.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.accounts.base import WithdrawalResult
from engine.params.loader import ParamSet


def room_accrued(age: ArrayLike, params: ParamSet) -> NDArray[np.float64]:
    """New contribution room for one year, granted in January.

    A flat annual amount from ``params``, granted in full on 1 January of each
    year from the year the person reaches the eligibility age. It is not
    accrued monthly: a person may contribute the whole year's room in January.

    Args:
        age: Age in whole years at the end of the year, ``(n_paths,)``.
        params: The ``federal`` parameter set.

    Returns:
        Room accrued for the year, ``(n_paths,)``.
    """
    raise NotImplementedError


def contribute(
    balance: ArrayLike,
    room: ArrayLike,
    requested: ArrayLike,
) -> tuple[NDArray[np.float64], ...]:
    """Contribute up to the room remaining right now.

    ``room`` is the room left at this point in the year, already reduced by
    every contribution made in earlier months. It is not the January grant.

    Returns:
        ``(new_balance, new_room, contributed)``, each ``(n_paths,)``.
    """
    raise NotImplementedError


def withdraw(balance: ArrayLike, requested: ArrayLike) -> WithdrawalResult:
    """Withdraw from a TFSA. Entirely tax-free.

    The withdrawal creates no room this year. It is accumulated into the
    running total that next January's :func:`room_restored` acts on.

    Returns:
        A :class:`~engine.accounts.base.WithdrawalResult` with everything in
        ``tax_free``.
    """
    raise NotImplementedError


def room_restored(withdrawn_last_year: ArrayLike) -> NDArray[np.float64]:
    """Room added back in January for the whole of last year's withdrawals.

    Takes *last calendar year's* total withdrawals, not this year's and not the
    last twelve months'. Called only from the January phase of the step, which
    is what enforces the lag.

    Args:
        withdrawn_last_year: Total withdrawn over the prior calendar year,
            summed across its twelve months, ``(n_paths,)``.

    Returns:
        Room restored, ``(n_paths,)``.
    """
    raise NotImplementedError

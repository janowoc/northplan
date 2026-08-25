"""Registered Retirement Savings Plan.

Contributions are deductible against income in the year made; withdrawals are
fully taxable. Contribution room accrues as a fraction of the *prior* year's
earned income up to an annual dollar limit, plus unused room carried forward.
Both come from ``params/``.

Room is granted in January for the year, on the previous year's earnings, and
is then consumed by contributions across the months. It does not accrue monthly
from the current month's income; a person may contribute the whole year's room
in January.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.accounts.base import WithdrawalResult
from engine.params.loader import ParamSet


def room_accrued(prior_year_earned_income: ArrayLike, params: ParamSet) -> NDArray[np.float64]:
    """New contribution room granted in January, on last year's earned income.

    Args:
        prior_year_earned_income: Earned income over the whole prior calendar
            year, real dollars, ``(n_paths,)``. Last year's, not this year's
            and not this month's.
        params: The ``federal`` parameter set, supplying the accrual rate and
            the annual dollar limit.

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

    ``room`` is what is left at this point in the year, already reduced by
    contributions made in earlier months.

    Args:
        balance: Balance at the start of this month, ``(n_paths,)``.
        room: Contribution room still available, ``(n_paths,)``.
        requested: Desired contribution this month, ``(n_paths,)``.

    Returns:
        ``(new_balance, new_room, contributed)``. ``contributed`` is capped at
        ``room``; the excess is not contributed rather than being penalised.
    """
    raise NotImplementedError


def withdraw(balance: ArrayLike, requested: ArrayLike) -> WithdrawalResult:
    """Withdraw from an RRSP. The full amount is taxable income.

    Withholding tax at source *is* relevant on a monthly timestep, because it
    changes when cash leaves rather than only how much: it is remitted in the
    month of the withdrawal and reduces the balance owing settled in the
    following year's filing month. This function returns the gross withdrawal;
    the step applies the withholding and records it in ``remitted_ytd``.

    Args:
        balance: Balance at the start of this month, ``(n_paths,)``.
        requested: Amount wanted this month, ``(n_paths,)``.

    Returns:
        A :class:`~engine.accounts.base.WithdrawalResult` with everything in
        ``fully_taxable``.
    """
    raise NotImplementedError


def must_convert_to_rrif(age_at_end_of_year: ArrayLike, params: ParamSet) -> NDArray[np.bool_]:
    """Whether an RRSP must be converted to a RRIF by the end of this year.

    The deadline is 31 December of the year the holder reaches the conversion
    age, whatever month their birthday falls in, so this is tested against age
    at year end and the conversion is applied by the year-end close. The
    converted plan's first minimum is fixed the following January.

    The conversion age is a parameter, not a literal.

    Args:
        age_at_end_of_year: Age in whole years on 31 December, ``(n_paths,)``.
            See ``engine.core.timeline.age_at_end_of_year``.
        params: The ``rrif`` parameter set.

    Returns:
        Boolean mask, ``(n_paths,)``.
    """
    raise NotImplementedError

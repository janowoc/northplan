"""Registered Retirement Savings Plan.

Contributions are deductible against income in the year made; withdrawals are
fully taxable. Contribution room accrues as a fraction of earned income up to
an annual dollar limit, plus unused room carried forward. Both come from
``params/``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.accounts.base import WithdrawalResult
from engine.params.loader import ParamSet


def room_accrued(earned_income: ArrayLike, params: ParamSet) -> NDArray[np.float64]:
    """New contribution room earned by one year of income.

    Args:
        earned_income: Earned income for the year, real dollars, ``(n_paths,)``.
        params: The ``federal`` parameter set, supplying the accrual rate and
            the annual dollar limit.

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

    Args:
        balance: Opening balance, ``(n_paths,)``.
        room: Available contribution room, ``(n_paths,)``.
        requested: Desired contribution, ``(n_paths,)``.

    Returns:
        ``(new_balance, new_room, contributed)``. ``contributed`` is capped at
        ``room``; the excess is not contributed rather than being penalised.
    """
    raise NotImplementedError


def withdraw(balance: ArrayLike, requested: ArrayLike) -> WithdrawalResult:
    """Withdraw from an RRSP. The full amount is taxable income.

    Withholding tax is not modelled: it is a prepayment reconciled at filing,
    and the engine settles tax annually.

    Args:
        balance: Opening balance, ``(n_paths,)``.
        requested: Amount wanted, ``(n_paths,)``.

    Returns:
        A :class:`~engine.accounts.base.WithdrawalResult` with everything in
        ``fully_taxable``.
    """
    raise NotImplementedError


def must_convert_to_rrif(age: ArrayLike, params: ParamSet) -> NDArray[np.bool_]:
    """Whether an RRSP must be converted to a RRIF by the end of this year.

    The conversion age is a parameter, not a literal.

    Args:
        age: Age in whole years at the end of the year, ``(n_paths,)``.
        params: The ``rrif`` parameter set.

    Returns:
        Boolean mask, ``(n_paths,)``.
    """
    raise NotImplementedError

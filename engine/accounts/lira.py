"""Locked-In Retirement Account and the LIF it becomes.

Like an RRSP in tax treatment and unlike it in access: withdrawals are barred
until an unlocking age, and once converted to a LIF there is both a mandatory
minimum and a jurisdiction-specific *maximum* withdrawal. The maximum is the
part that distinguishes this from ``rrif.py`` and it is provincially set, so it
comes from the province's parameter file.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.accounts.base import WithdrawalResult
from engine.params.loader import ParamSet


def maximum_withdrawal(
    opening_balance: ArrayLike,
    age_at_start_of_year: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Jurisdiction-specific maximum LIF withdrawal for the year.

    Args:
        opening_balance: Balance on 1 January, before growth, ``(n_paths,)``.
        age_at_start_of_year: Age in whole years on 1 January.
        params: The province's parameter set.

    Returns:
        Maximum permitted withdrawal, ``(n_paths,)``.
    """
    raise NotImplementedError


def withdraw(
    balance: ArrayLike,
    requested: ArrayLike,
    minimum: ArrayLike,
    maximum: ArrayLike,
) -> WithdrawalResult:
    """Withdraw within the mandatory minimum and permitted maximum. Fully taxable.

    Args:
        balance: Opening balance, ``(n_paths,)``.
        requested: Policy-requested withdrawal, clamped into ``[minimum, maximum]``.
        minimum: RRIF-equivalent minimum for the age.
        maximum: Output of :func:`maximum_withdrawal`.

    Returns:
        A :class:`~engine.accounts.base.WithdrawalResult` with everything in
        ``fully_taxable``.
    """
    raise NotImplementedError

"""Locked-In Retirement Account and the LIF it becomes.

Like an RRSP in tax treatment and unlike it in access: withdrawals are barred
until an unlocking age, and once converted to a LIF there is both a mandatory
minimum and a jurisdiction-specific *maximum* withdrawal. The maximum is the
part that distinguishes this from ``rrif.py`` and it is provincially set, so it
comes from the province's parameter file.

Both the minimum and the maximum are **annual** figures fixed in January from
the 1 January balance, and both are satisfied or consumed across the months of
the year. The maximum is the dangerous one: enforced per month it permits
twelve times what the jurisdiction allows, and nothing in the output looks
wrong. It is enforced against the year-to-date total.
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
    """Jurisdiction-specific maximum LIF withdrawal for the **whole year**.

    Called once, by the January phase of the step, and then drawn down.

    Args:
        opening_balance: Balance on 1 January, before growth, ``(n_paths,)``.
        age_at_start_of_year: Age in whole years on 1 January.
        params: The province's parameter set.

    Returns:
        Maximum permitted withdrawal for the year, ``(n_paths,)``.
    """
    raise NotImplementedError


def withdraw(
    balance: ArrayLike,
    requested: ArrayLike,
    minimum_this_month: ArrayLike,
    maximum_remaining: ArrayLike,
) -> WithdrawalResult:
    """Withdraw this month within the annual bounds. Fully taxable.

    Args:
        balance: Balance at the start of this month, ``(n_paths,)``.
        requested: Policy-requested withdrawal for this month, clamped into
            ``[minimum_this_month, maximum_remaining]``.
        minimum_this_month: RRIF-equivalent floor for this month, from
            ``engine.accounts.rrif.minimum_still_required``.
        maximum_remaining: What is left of the year's maximum, from
            ``engine.accounts.base.remaining_annual_allowance`` applied to
            :func:`maximum_withdrawal`. Not the annual maximum itself.

    Returns:
        A :class:`~engine.accounts.base.WithdrawalResult` with everything in
        ``fully_taxable``.
    """
    raise NotImplementedError

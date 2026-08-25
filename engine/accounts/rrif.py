"""Registered Retirement Income Fund.

A RRIF has a mandatory minimum withdrawal each year, set by a factor that
depends on age. Two things about it are easy to get wrong and are fixed here:

- The factor is selected by **age at the start of the year**, not age at year
  end and not age on the withdrawal date.
- The minimum is computed on the **opening balance**, before that year's
  growth. Computing it on the closing balance overstates every withdrawal.

Factors come from ``params/{year}/rrif.yaml``. There is no formula in this file.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.accounts.base import WithdrawalResult
from engine.params.loader import ParamSet


def minimum_factor(age_at_start_of_year: ArrayLike, params: ParamSet) -> NDArray[np.float64]:
    """Mandatory withdrawal factor for a given age.

    Args:
        age_at_start_of_year: Age in whole years on 1 January, ``(n_paths,)``.
            Not age at year end.
        params: The ``rrif`` parameter set, supplying the factor table and the
            pre-71 basis.

    Returns:
        Factor as a bare fraction, ``(n_paths,)``.
    """
    raise NotImplementedError


def minimum_withdrawal(
    opening_balance: ArrayLike,
    age_at_start_of_year: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Mandatory minimum withdrawal for the year.

    Args:
        opening_balance: Balance on 1 January, before the year's growth,
            ``(n_paths,)``.
        age_at_start_of_year: Age in whole years on 1 January.
        params: The ``rrif`` parameter set.

    Returns:
        Minimum withdrawal in real dollars, ``(n_paths,)``.
    """
    raise NotImplementedError


def withdraw(
    balance: ArrayLike,
    requested: ArrayLike,
    minimum: ArrayLike,
) -> WithdrawalResult:
    """Withdraw at least the minimum. Fully taxable.

    A policy may request less than the minimum; the minimum still comes out.
    That interaction is the reason ``minimum`` is an argument rather than being
    recomputed here.

    Args:
        balance: Opening balance, ``(n_paths,)``.
        requested: Policy-requested withdrawal, ``(n_paths,)``.
        minimum: Output of :func:`minimum_withdrawal`.

    Returns:
        A :class:`~engine.accounts.base.WithdrawalResult` with everything in
        ``fully_taxable``, gross at least ``minimum``.
    """
    raise NotImplementedError


def spousal_rollover(
    balance: ArrayLike,
    surviving_spouse_balance: ArrayLike,
) -> tuple[NDArray[np.float64], ...]:
    """Roll a deceased person's RRIF to their surviving spouse, tax-deferred.

    Without a surviving spouse the balance is instead brought fully into income
    in the year of death; that case is handled by the annual step, not here.

    Args:
        balance: The deceased person's RRIF balance, ``(n_paths,)``.
        surviving_spouse_balance: The survivor's RRIF balance, ``(n_paths,)``.

    Returns:
        ``(deceased_balance, survivor_balance)`` after the rollover.
    """
    raise NotImplementedError

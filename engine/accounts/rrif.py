"""Registered Retirement Income Fund.

A RRIF has a mandatory minimum withdrawal each year, set by a factor that
depends on age. Three things about it are easy to get wrong and are fixed here:

- The factor is selected by **age at the start of the year**, not age at year
  end and not age on the withdrawal date.
- The minimum is computed on the **opening balance**, on 1 January, before that
  year's growth. Computing it on a mid-year balance overstates every
  withdrawal, and on a monthly timestep every month offers a fresh chance to
  make that mistake.
- The minimum is an **annual** obligation with a 31 December deadline, not a
  monthly one. It is fixed in January and satisfied over the months that
  follow, in whatever pattern the policy chooses; whatever is left in December
  is forced out by the year-end close. A per-month minimum equal to the annual
  minimum would take twelve times too much.

Factors come from ``params/{year}/rrif.yaml`` under ``rrif.minimum_factors``.
That file holds one program at two stages of life — the RRSP that accumulates
and the RRIF it becomes — with the conversion age at the top level joining
them. There is no formula in this file: even the pre-table basis is a constant
in the YAML, because ``1 / (C - age)`` puts a ``C`` in a ``.py`` file otherwise.
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
            Not age at year end, and not age in the current month. See
            ``engine.core.timeline.age_at_start_of_year``.
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
    """Mandatory minimum withdrawal for the **whole year**.

    Called once, by the January phase of the step. The result is an annual
    amount that is then drawn down across the year; it is never recomputed
    mid-year, because the balance it is based on no longer exists after January.

    Args:
        opening_balance: Balance on 1 January, before the year's growth and
            before any of the year's withdrawals, ``(n_paths,)``.
        age_at_start_of_year: Age in whole years on 1 January.
        params: The ``rrif`` parameter set.

    Returns:
        Annual minimum withdrawal in real dollars, ``(n_paths,)``.
    """
    raise NotImplementedError


def minimum_still_required(
    annual_minimum: ArrayLike,
    withdrawn_ytd: ArrayLike,
    months_remaining_in_year: int,
) -> NDArray[np.float64]:
    """How much must come out this month for the annual minimum to be met.

    Zero for most of the year: a policy is free to take nothing in January and
    the whole minimum in December. This returns a non-zero floor only when the
    remaining months can no longer accommodate the shortfall, which in practice
    means December.

    Args:
        annual_minimum: The year's minimum, fixed in January, ``(n_paths,)``.
        withdrawn_ytd: Amount already withdrawn this calendar year,
            ``(n_paths,)``.
        months_remaining_in_year: Months left including this one. One in
            December.

    Returns:
        Amount that must be withdrawn this month, ``(n_paths,)``, floored at
        zero.
    """
    raise NotImplementedError


def withdraw(
    balance: ArrayLike,
    requested: ArrayLike,
    minimum_this_month: ArrayLike,
) -> WithdrawalResult:
    """Withdraw at least this month's required floor. Fully taxable.

    A policy may request less than the floor; the floor still comes out. That
    interaction is why ``minimum_this_month`` is an argument rather than being
    recomputed here — and why it is *this month's* floor from
    :func:`minimum_still_required`, not the annual minimum, which would be
    taken twelve times over.

    Args:
        balance: Balance at the start of this month, ``(n_paths,)``.
        requested: Policy-requested withdrawal for this month, ``(n_paths,)``.
        minimum_this_month: Output of :func:`minimum_still_required`.

    Returns:
        A :class:`~engine.accounts.base.WithdrawalResult` with everything in
        ``fully_taxable``, gross at least ``minimum_this_month``.
    """
    raise NotImplementedError


def spousal_rollover(
    balance: ArrayLike,
    surviving_spouse_balance: ArrayLike,
) -> tuple[NDArray[np.float64], ...]:
    """Roll a deceased person's RRIF to their surviving spouse, tax-deferred.

    Without a surviving spouse the balance is instead brought fully into income
    in the year of death; that case is handled by the step, not here. Death is
    resolved monthly, so the rollover happens in the month of death and the
    survivor's own minimum for the year is unaffected — theirs was fixed in
    January from their own opening balance.

    Args:
        balance: The deceased person's RRIF balance, ``(n_paths,)``.
        surviving_spouse_balance: The survivor's RRIF balance, ``(n_paths,)``.

    Returns:
        ``(deceased_balance, survivor_balance)`` after the rollover.
    """
    raise NotImplementedError

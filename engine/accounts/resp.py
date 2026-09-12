# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registered Education Savings Plan — tracked per beneficiary.

**Never one pot.** Grant room, the lifetime contribution limit, and the
withdrawal window are all per-beneficiary and do not aggregate — an EAP paid
for one child cannot draw on another's grant. Every function here takes a
single beneficiary's state; the step iterates.

A withdrawal splits three ways and the split is not the caller's choice:
contributions come out tax-free, while grant and accumulated income come out
as an Educational Assistance Payment taxable in the *student's* hands, not
the subscriber's.

Grant room accrues once a year, in January; the annual grant maximum is
enforced against the year-to-date grant received, not a single month's.
Enrolment begins in a month, not on 1 January, and the EAP cap window is
measured in weeks from the start of enrolment, so the enrolment month is
state the plan carries.

The grant has two tiers: a basic match paid to everyone, and an additional
match on the first dollars of each year's contribution at a rate that steps
down as family income rises and is capped in dollars.

Parameters come from ``params/{year}/resp.yaml``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.accounts.base import WithdrawalResult
from engine.params.loader import ParamSet


class RespState:
    """Per-beneficiary plan state: contributions, grant, and accumulated income.

    The three components are tracked separately because they are taxed
    separately on the way out. Also carries: year-to-date grant received (so
    the annual grant maximum enforces against the year, not a month), the
    month enrolment began (the EAP window is measured from it), and — once
    implemented — year-to-date contributions, needed because the enhanced
    grant tier's eligible window is a dollar amount per year, not per month.

    Defined here as a placeholder; its fields land with the implementation.
    """


def enhanced_grant_rate(
    family_income: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Additional match rate for the first dollars of this year's contribution.

    A step function of family income: the highest rate below the first
    cut-off, a lower rate between cut-offs, and zero above the last — cliffs,
    not a phase-out, so a dollar of income across a cut-off changes the rate
    on every eligible dollar. Cut-offs and rates come from ``params`` under
    ``grant.enhanced``, laid out as a bracket table
    (``len(match_rates) == len(income_edges_annual) + 1``), so the
    branch-free clipping idiom from ``engine.tax.federal.gross_tax`` applies.

    This returns a rate; the dollar cap on eligible contribution is applied
    by :func:`grant_on_contribution`.

    Args:
        family_income: Family income for the governing year, ``(n_paths,)`` —
            summed across the household's living persons, not one person's.
            The governing year is set by
            ``grant.enhanced.income_year_offset`` in ``params`` and applied
            by the caller, not guessed here.
        params: The ``resp`` parameter set.

    Returns:
        Additional match rate as a bare fraction, ``(n_paths,)``, zero above
        the highest cut-off.
    """
    raise NotImplementedError


def grant_on_contribution(
    contribution: ArrayLike,
    grant_room: ArrayLike,
    grant_received_ytd: ArrayLike,
    contributed_ytd: ArrayLike,
    lifetime_grant_paid: ArrayLike,
    family_income: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Grant matched to this month's contribution, for one beneficiary.

    Both tiers, summed: the basic match at a flat rate on every eligible
    dollar, and the additional match at :func:`enhanced_grant_rate` on the
    first ``grant.enhanced.eligible_contribution_annual`` dollars contributed
    this year. Bounded by both match rates, the annual grant room (which
    carries forward), the annual grant maximum against ``grant_received_ytd``,
    and the lifetime grant maximum — every bound from ``params``.

    Args:
        contribution: This month's contribution, ``(n_paths,)``.
        grant_room: Unused annual grant room, including carry-forward.
        grant_received_ytd: Grant received this calendar year, ``(n_paths,)``.
        contributed_ytd: Contributions made this calendar year,
            ``(n_paths,)``, excluding this month's — bounds the enhanced
            tier's eligible window.
        lifetime_grant_paid: Grant received over all years.
        family_income: Family income for the year that governs the enhanced
            rate; see :func:`enhanced_grant_rate`.
        params: The ``resp`` parameter set.

    Returns:
        Grant paid this month, both tiers combined, ``(n_paths,)``.
    """
    raise NotImplementedError


def contribute(
    state: RespState,
    requested: ArrayLike,
    family_income: ArrayLike,
    params: ParamSet,
) -> RespState:
    """Contribute for one beneficiary this month and receive the matching grant.

    Args:
        state: That beneficiary's plan state.
        requested: Desired contribution this month, ``(n_paths,)``. Capped at
            the lifetime contribution maximum, which is per beneficiary and
            does not aggregate across children.
        family_income: Family income for the year that governs the enhanced
            grant rate; see :func:`enhanced_grant_rate`.
        params: The ``resp`` parameter set.

    Returns:
        Updated state, with the contribution, both grant tiers, and the
        year-to-date contribution and grant totals applied.
    """
    raise NotImplementedError


def withdraw(
    state: RespState,
    requested: ArrayLike,
    is_eligible_student: ArrayLike,
    months_since_enrolment: ArrayLike,
    params: ParamSet,
) -> tuple[RespState, WithdrawalResult]:
    """Withdraw for one beneficiary this month.

    The tax-free and taxable portions are determined by the plan's
    composition, not chosen by the caller. EAP withdrawals require the
    beneficiary to be enrolled and are capped for an initial window measured
    in months since enrolment began.

    Args:
        state: That beneficiary's plan state.
        requested: Amount wanted this month, ``(n_paths,)``.
        is_eligible_student: Whether the beneficiary is enrolled this month,
            ``(n_paths,)``. Determines whether an EAP is permitted at all.
        months_since_enrolment: Months since enrolment began, ``(n_paths,)``,
            compared against the cap window from ``params``.
        params: The ``resp`` parameter set, supplying the cap and its window.

    Returns:
        ``(updated_state, result)``. The result's ``fully_taxable`` portion is
        taxable to the student, and the step must attribute it there.
    """
    raise NotImplementedError

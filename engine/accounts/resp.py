# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registered Education Savings Plan — tracked per beneficiary.

**Never one pot.** Grant room, the lifetime contribution limit, and the
withdrawal window are all per-beneficiary and do not aggregate. Two children
with one plan between them still have two independent grant entitlements, and
an EAP paid for one child cannot draw on the other's grant. Every function here
takes a single beneficiary's state; the step iterates.

A withdrawal splits three ways and the split is not the caller's choice:
contributions come out tax-free, while grant and accumulated income come out as
an Educational Assistance Payment taxable in the *student's* hands, not the
subscriber's.

Timing on a monthly step. Grant room accrues once a year, in January, and the
annual grant maximum is an annual cap enforced against the year-to-date grant
received — enforced per month it would pay twelve times the maximum. Enrolment
begins in a month, not on 1 January, and the cap on EAP withdrawals applies to
a window measured in weeks from the start of enrolment, so the month of
enrolment is state the plan has to carry.

The grant has two tiers. The basic match is paid at one rate to everyone; an
additional match is paid on the first dollars of each year's contribution at a
rate that steps down as family income rises, and is capped in dollars rather
than being a pure percentage. Leaving the additional tier out understates the
grant precisely for the households it matters most to, so it is modelled.

The optimizer should discover on its own that contributing up to the annual
grant maximum dominates almost every alternative. If it does not, the model is
wrong — a guaranteed match is hard to beat, and that makes this a free oracle.

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
    separately on the way out. It also carries the year-to-date grant received,
    so the annual grant maximum can be enforced against the year rather than
    against a month, and the month enrolment began, which the EAP window is
    measured from.

    The additional grant tier needs one more year-to-date figure: the
    contribution already made this calendar year, because the enhanced rate
    applies only to the first dollars of each year's contribution and a month
    cannot know how much of that window is left without it. Defined here as a
    placeholder; its fields land with the implementation.
    """


def enhanced_grant_rate(
    family_income: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Additional match rate for the first dollars of this year's contribution.

    A step function of family income: the highest rate below the first cut-off,
    a lower rate between the cut-offs, and zero above the last. The cut-offs
    and rates are a table in ``params`` under ``grant.enhanced``, laid out like
    a bracket table — ``len(match_rates) == len(income_edges_annual) + 1`` — so
    the same branch-free clipping idiom as ``engine.tax.federal.gross_tax``
    applies. The steps are cliffs rather than a phase-out: a dollar of income
    across a cut-off changes the rate on every eligible dollar.

    This returns a *rate*. The dollar cap on the eligible contribution is
    applied by :func:`grant_on_contribution`, which is the only caller that
    knows how much of this year's eligible window is already used.

    Args:
        family_income: Family income for the governing year, ``(n_paths,)``.
            Which year governs is a statutory rule, not this function's guess:
            the offset is in ``params`` under
            ``grant.enhanced.income_year_offset`` and is applied by the caller,
            the way the OAS recovery tax reaches back through the benefit-year
            rule in ``params/<year>/oas.yaml`` —
            ``benefit_year.start_month`` and
            ``benefit_year.income_year_offset`` — applied by
            ``engine/benefits/oas.py``. Passing the current year's income when
            the rule says otherwise overstates the grant for a household whose
            income is rising, and understates it for one drawing down.
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

    Both tiers, summed. The basic match is paid at a flat rate on every
    eligible dollar; the additional match is paid at
    :func:`enhanced_grant_rate` on the first
    ``grant.enhanced.eligible_contribution_annual`` dollars contributed in the
    calendar year, which is what ``contributed_ytd`` measures.

    Bounded five ways: the two match rates, the annual grant room (which
    carries forward), the annual grant maximum measured against what has
    already been received this calendar year, and the lifetime grant maximum.
    Every bound comes from ``params``.

    Two arguments exist solely to keep annual limits annual.
    ``grant_received_ytd`` is what makes the annual maximum an annual maximum —
    without it, twelve monthly contributions each collect a full year's grant.
    ``contributed_ytd`` does the same job for the enhanced tier's eligible
    window, which is a dollar amount per year and not per month.

    Args:
        contribution: This month's contribution for this beneficiary,
            ``(n_paths,)``.
        grant_room: Unused annual grant room, including carry-forward.
        grant_received_ytd: Grant already received for this beneficiary in this
            calendar year, ``(n_paths,)``.
        contributed_ytd: Contributions already made for this beneficiary in
            this calendar year, ``(n_paths,)``, excluding this month's.
        lifetime_grant_paid: Grant already received by this beneficiary over
            all years.
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

    The tax-free and taxable portions are determined by the plan's composition,
    not chosen by the caller. EAP withdrawals require the beneficiary to be
    enrolled and are capped for an initial window measured from the start of
    enrolment — which is why the window, in months since enrolment, is an
    argument. On an annual timestep that cap was invisible; on a monthly one it
    binds, and a plan drawn down too fast in the first months of study is a
    real outcome the model should be able to produce.

    Args:
        state: That beneficiary's plan state.
        requested: Amount wanted this month, ``(n_paths,)``.
        is_eligible_student: Whether the beneficiary is enrolled this month,
            ``(n_paths,)``. Determines whether an EAP is permitted at all.
        months_since_enrolment: Months since enrolment began, ``(n_paths,)``.
            Compared against the cap window from ``params``.
        params: The ``resp`` parameter set, supplying the cap and its window.

    Returns:
        ``(updated_state, result)``. The result's ``fully_taxable`` portion is
        taxable to the student, and the step must attribute it there.
    """
    raise NotImplementedError

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

The optimizer should discover on its own that contributing up to the annual
grant maximum dominates almost every alternative. If it does not, the model is
wrong — a guaranteed match is hard to beat, and that makes this a free oracle.
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
    measured from. Defined here as a placeholder; its fields land with the
    implementation.
    """


def grant_on_contribution(
    contribution: ArrayLike,
    grant_room: ArrayLike,
    grant_received_ytd: ArrayLike,
    lifetime_grant_paid: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Grant matched to this month's contribution, for one beneficiary.

    Bounded four ways: the match rate, the annual grant room (which carries
    forward), the annual grant maximum measured against what has already been
    received this calendar year, and the lifetime grant maximum. All of the
    bounds come from ``params``.

    ``grant_received_ytd`` is the argument that makes the annual maximum an
    annual maximum. Without it, twelve monthly contributions each collect the
    full year's grant.

    Args:
        contribution: This month's contribution for this beneficiary,
            ``(n_paths,)``.
        grant_room: Unused annual grant room, including carry-forward.
        grant_received_ytd: Grant already received for this beneficiary in this
            calendar year, ``(n_paths,)``.
        lifetime_grant_paid: Grant already received by this beneficiary over
            all years.
        params: The ``federal`` parameter set.

    Returns:
        Grant paid this month, ``(n_paths,)``.
    """
    raise NotImplementedError


def contribute(
    state: RespState,
    requested: ArrayLike,
    params: ParamSet,
) -> RespState:
    """Contribute for one beneficiary this month and receive the matching grant.

    Args:
        state: That beneficiary's plan state.
        requested: Desired contribution this month, ``(n_paths,)``.
        params: The ``federal`` parameter set.

    Returns:
        Updated state, with the contribution, any grant, and the year-to-date
        grant total applied.
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
        params: The ``federal`` parameter set, supplying the cap and its window.

    Returns:
        ``(updated_state, result)``. The result's ``fully_taxable`` portion is
        taxable to the student, and the step must attribute it there.
    """
    raise NotImplementedError

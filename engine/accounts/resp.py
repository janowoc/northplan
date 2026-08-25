"""Registered Education Savings Plan — tracked per beneficiary.

**Never one pot.** Grant room, the lifetime contribution limit, and the
withdrawal window are all per-beneficiary and do not aggregate. Two children
with one plan between them still have two independent grant entitlements, and
an EAP paid for one child cannot draw on the other's grant. Every function here
takes a single beneficiary's state; the annual step iterates.

A withdrawal splits three ways and the split is not the caller's choice:
contributions come out tax-free, while grant and accumulated income come out as
an Educational Assistance Payment taxable in the *student's* hands, not the
subscriber's.

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
    separately on the way out. Defined here as a placeholder; its fields land
    with the implementation.
    """


def grant_on_contribution(
    contribution: ArrayLike,
    grant_room: ArrayLike,
    lifetime_grant_paid: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Grant matched to a contribution for one beneficiary.

    Bounded three ways: the match rate, the annual grant room (which carries
    forward), and the lifetime grant maximum. All three from ``params``.

    Args:
        contribution: This year's contribution for this beneficiary,
            ``(n_paths,)``.
        grant_room: Unused annual grant room, including carry-forward.
        lifetime_grant_paid: Grant already received by this beneficiary.
        params: The ``federal`` parameter set.

    Returns:
        Grant paid, ``(n_paths,)``.
    """
    raise NotImplementedError


def contribute(
    state: RespState,
    requested: ArrayLike,
    params: ParamSet,
) -> RespState:
    """Contribute for one beneficiary and receive the matching grant.

    Args:
        state: That beneficiary's plan state.
        requested: Desired contribution, ``(n_paths,)``.
        params: The ``federal`` parameter set.

    Returns:
        Updated state, with the contribution and any grant applied.
    """
    raise NotImplementedError


def withdraw(
    state: RespState,
    requested: ArrayLike,
    is_eligible_student: ArrayLike,
) -> tuple[RespState, WithdrawalResult]:
    """Withdraw for one beneficiary.

    The tax-free and taxable portions are determined by the plan's composition,
    not chosen by the caller. EAP withdrawals require the beneficiary to be
    enrolled and are subject to an early-withdrawal cap in the first months of
    enrolment.

    Args:
        state: That beneficiary's plan state.
        requested: Amount wanted, ``(n_paths,)``.
        is_eligible_student: Whether the beneficiary is currently enrolled,
            ``(n_paths,)``. Determines whether an EAP is permitted at all.

    Returns:
        ``(updated_state, result)``. The result's ``fully_taxable`` portion is
        taxable to the student, and the annual step must attribute it there.
    """
    raise NotImplementedError

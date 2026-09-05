# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The policy interface and the information a policy is allowed to see."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from engine.core.state import HouseholdState
from engine.params.loader import ParamYear


@dataclass(frozen=True, slots=True)
class Decision:
    """What a policy decided for one **month**, for all paths.

    Contributions and withdrawals are this month's amounts, not the year's. A
    policy that returns an annual figure here has it applied twelve times.

    The benefit start ages are the exception: they are ages, in months, at
    which a benefit begins, and they do not change from month to month. They
    are returned every month because the policy is stateless and there is
    nowhere else to put them; the step reads them once, in the month the
    decision binds, and ignores them afterwards.

    Attributes:
        contributions: Amount to contribute *this month* per person per account
            type, real dollars. Keys are ``(person_id, account)``; values
            ``(n_paths,)``.
        withdrawals: Discretionary withdrawals *this month*, same keying.
            Mandatory minimums are applied by the step on top of these, not by
            the policy.
        cpp_start_age_months: Age in months at which each person starts CPP.
        oas_start_age_months: Age in months at which each person starts OAS.
    """

    contributions: dict[tuple[str, str], NDArray[np.float64]]
    withdrawals: dict[tuple[str, str], NDArray[np.float64]]
    cpp_start_age_months: dict[str, NDArray[np.float64]]
    oas_start_age_months: dict[str, NDArray[np.float64]]


class Policy(Protocol):
    """A decision rule the optimizer can evaluate.

    Implementations are pure and stateless: everything they need arrives in
    ``decide``. State kept on the object between months would be a channel for
    information the policy is not entitled to — and with twelve calls a year
    instead of one, it is also twelve times as easy to accumulate by accident.
    Anything a policy needs to remember from an earlier month is already in
    ``HouseholdState``: year-to-date income, room consumed, minimums still
    outstanding.
    """

    def decide(
        self,
        state: HouseholdState,
        params: ParamYear,
    ) -> Decision:
        """Choose this month's contributions and withdrawals.

        Args:
            state: Opening state for the current month, carrying ``year`` and
                ``month``. This, and the current tax year's ``params``, are the
                *only* inputs. Anything else is clairvoyance.

                What the state legitimately offers, and what a monthly policy
                must be careful with: income accumulated *so far* this year is
                knowable, and the year's eventual total is not. A policy that
                reasons about "this year's income" must mean the former.
            params: Parameters for the current tax year.

        Returns:
            The month's :class:`Decision`.
        """
        ...

    def free_parameters(self) -> dict[str, float]:
        """The numbers the optimizer is searching over, by name.

        Returns:
            Parameter name to current value. The optimizer varies these and
            rebuilds the policy; it never mutates a policy in place.
        """
        ...

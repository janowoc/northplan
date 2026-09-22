# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The policy interface and the information a policy is allowed to see."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from engine.core.indexation import RealParamYear
from engine.core.state import HouseholdState


@dataclass(frozen=True, slots=True)
class Decision:
    """What a policy decided for one **month**, for all paths.

    Contributions and withdrawals are this month's amounts, not the year's. A
    policy that returns an annual figure here has it applied twelve times.

    The benefit start ages are the exception: they are ages, in months, at
    which a benefit begins, and they do not change from month to month. They
    are returned every month; the step reads them once, in the month the
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
    ``decide``. Anything a policy needs to remember from an earlier month is
    already in ``HouseholdState``: year-to-date income, room consumed,
    minimums still outstanding.
    """

    def decide(
        self,
        state: HouseholdState,
        real_params: RealParamYear,
    ) -> Decision:
        """Choose this month's contributions and withdrawals.

        Args:
            state: Opening state for the current month, carrying ``year`` and
                ``month``. This, and the current tax year's ``real_params``,
                are the only legitimate inputs. Income accumulated *so far*
                this year is knowable; the year's eventual total is not.
            real_params: Parameters for the current tax year, in the
                scenario's real-dollar view.

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

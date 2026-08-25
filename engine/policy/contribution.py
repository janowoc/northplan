"""Accumulation policies: how to split a savings budget across accounts.

The free variables the optimizer searches for the accumulation use case are the
split weights across RRSP, RESP, TFSA, and taxable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from engine.core.state import HouseholdState
from engine.params.loader import ParamYear
from engine.policy.base import Decision


@dataclass(frozen=True, slots=True)
class SplitContributionPolicy:
    """Split an annual savings budget across accounts by fixed weights.

    Weights are the optimizer's free variables. They are normalised to sum to
    one, and a contribution that exceeds available room spills to the next
    account in priority order rather than being lost — otherwise the optimizer
    sees a cliff where room runs out and the search behaves badly around it.

    Attributes:
        rrsp_weight: Share of the budget directed to RRSP.
        resp_weight: Share directed to RESP, divided across beneficiaries.
        tfsa_weight: Share directed to TFSA.
        taxable_weight: Share directed to the non-registered account.
        spill_order: Account order for redirecting contributions that exceed
            available room.
    """

    rrsp_weight: float
    resp_weight: float
    tfsa_weight: float
    taxable_weight: float
    spill_order: tuple[str, ...]

    def decide(self, state: HouseholdState, params: ParamYear) -> Decision:
        """Split this year's savings budget according to the weights.

        Reads opening balances, contribution room, and the current year's
        parameters. Nothing else.
        """
        raise NotImplementedError

    def free_parameters(self) -> dict[str, float]:
        """The four weights, by name."""
        raise NotImplementedError


def annual_savings_budget(
    state: HouseholdState,
    target_spending: NDArray[np.float64],
) -> NDArray[np.float64]:
    """After-tax income left over for saving this year.

    Args:
        state: Opening state for the current year.
        target_spending: Desired real spending this year, ``(n_paths,)``.

    Returns:
        Amount available to contribute, floored at zero, ``(n_paths,)``.
    """
    raise NotImplementedError

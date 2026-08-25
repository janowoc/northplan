"""Accumulation policies: how to split a savings budget across accounts.

The free variables the optimizer searches for the accumulation use case are the
split weights across RRSP, RESP, TFSA, and taxable. The weights are the same
every month; what changes month to month is the budget they are applied to and
the room still available.
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
    """Split a monthly savings budget across accounts by fixed weights.

    Weights are the optimizer's free variables. They are normalised to sum to
    one, and a contribution that exceeds available room spills to the next
    account in priority order rather than being lost — otherwise the optimizer
    sees a cliff where room runs out and the search behaves badly around it.

    Room is granted annually, in January, and consumed over the months. So the
    spill is not a rare year-end event: an aggressive weight fills a TFSA by
    March and spills for the remaining nine months, and the policy has to
    behave sensibly the whole way.

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
        """Split this month's savings budget according to the weights.

        Reads opening balances, the contribution room *still available* at this
        point in the year, and the current year's parameters. Nothing else.
        """
        raise NotImplementedError

    def free_parameters(self) -> dict[str, float]:
        """The four weights, by name."""
        raise NotImplementedError


def monthly_savings_budget(
    state: HouseholdState,
    monthly_target_spending: NDArray[np.float64],
) -> NDArray[np.float64]:
    """After-tax income left over for saving **this month**.

    Both arguments are monthly. A budget computed from annual income against
    annual spending and then contributed every month over-contributes by a
    factor of twelve, and the resulting plan looks merely optimistic rather
    than impossible.

    The month matters beyond scale. Income is not level across the year: the
    filing month takes a balance owing out in cash, a bonus lands in one month,
    and a benefit that starts mid-year starts mid-year. A household can have a
    negative surplus in one month and a large one in the next.

    Args:
        state: Opening state for the current month.
        monthly_target_spending: Desired real spending this month,
            ``(n_paths,)``.

    Returns:
        Amount available to contribute this month, floored at zero,
        ``(n_paths,)``.
    """
    raise NotImplementedError

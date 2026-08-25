"""Decumulation policies: withdrawal order, thresholds, and benefit start ages.

The free variables the optimizer searches for the decumulation use case are the
account withdrawal order, the income thresholds that govern how much to realize
each year, and the CPP and OAS start ages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.core.state import HouseholdState
from engine.params.loader import ParamYear
from engine.policy.base import Decision


@dataclass(frozen=True, slots=True)
class OrderedWithdrawalPolicy:
    """Draw accounts in a fixed order, up to a taxable-income ceiling.

    The classic parameterization: take mandatory minimums, then fill taxable
    income up to a target bracket edge from registered accounts, then meet any
    remaining need from the accounts in ``order``.

    The ceiling is expressed as an index into the current year's bracket edges
    rather than a dollar amount, so it stays meaningful in real terms across
    years and across parameter updates.

    Attributes:
        order: Account types in withdrawal priority order.
        taxable_ceiling_bracket: Index into the federal bracket edges to fill
            taxable income up to.
        cpp_start_age_months: CPP start age in months, per person.
        oas_start_age_months: OAS start age in months, per person.
    """

    order: tuple[str, ...]
    taxable_ceiling_bracket: int
    cpp_start_age_months: dict[str, int]
    oas_start_age_months: dict[str, int]

    def decide(self, state: HouseholdState, params: ParamYear) -> Decision:
        """Choose this year's withdrawals.

        Reads opening balances, current ages, realized history, and the current
        year's brackets. It must not read this year's return, any later year's
        return, or any terminal value.
        """
        raise NotImplementedError

    def free_parameters(self) -> dict[str, float]:
        """The ceiling bracket index and the benefit start ages, by name."""
        raise NotImplementedError


def fill_to_bracket(
    current_taxable_income: ArrayLike,
    bracket_edges: tuple[float, ...],
    ceiling_index: int,
    available: ArrayLike,
) -> NDArray[np.float64]:
    """Amount to withdraw to fill taxable income up to a bracket edge.

    Args:
        current_taxable_income: Taxable income before this withdrawal,
            ``(n_paths,)``.
        bracket_edges: Current year's bracket edges, from ``params/``.
        ceiling_index: Which edge to fill to.
        available: Balance available to withdraw, ``(n_paths,)``.

    Returns:
        Withdrawal amount, non-negative and capped by ``available``. Zero where
        income already exceeds the ceiling.
    """
    raise NotImplementedError

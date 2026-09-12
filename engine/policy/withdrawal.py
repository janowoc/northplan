# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decumulation policies: withdrawal order, thresholds, and benefit start ages.

The free variables the optimizer searches for the decumulation use case are
the account withdrawal order, the income thresholds that govern how much to
realize each year, and the CPP and OAS start ages.

**The bracket-filling trap.** "Withdraw up to the top of a bracket" is an
annual instruction; filling to the ceiling every month withdraws roughly
twelve times the intended amount. Every threshold here is therefore measured
against year-to-date taxable income and what remains under the ceiling for
the rest of the year — never against the month alone. This is why
:func:`fill_to_bracket` takes the year to date rather than a month's income.
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
    """Draw accounts in a fixed order, up to a taxable-income ceiling for the year.

    The classic parameterization: take mandatory minimums, fill taxable income
    up to a target bracket edge from registered accounts, then meet any
    remaining need from the accounts in ``order``.

    The ceiling is an index into the current year's bracket edges rather than
    a dollar amount, so it stays meaningful in real terms across years. It
    bounds the year, and the policy spreads the room under it across the
    months that remain — how, is an explicit implementation choice
    (front-loaded vs. an even twelfth changes what is invested for how long,
    what a mid-year death brings into income, and how much room survives a
    market fall), stated and tested, not an accident of iteration order.

    Attributes:
        order: Account types in withdrawal priority order.
        taxable_ceiling_bracket: Index into the federal bracket edges to fill
            year-to-date taxable income up to, over the course of the year.
        cpp_start_age_months: CPP start age in months, per person.
        oas_start_age_months: OAS start age in months, per person.
    """

    order: tuple[str, ...]
    taxable_ceiling_bracket: int
    cpp_start_age_months: dict[str, int]
    oas_start_age_months: dict[str, int]

    def decide(self, state: HouseholdState, params: ParamYear) -> Decision:
        """Choose this month's withdrawals.

        Reads opening balances, current ages, income accumulated so far this
        year, realized history, and the current year's brackets. It must not
        read this month's return, any later month's return, or any terminal
        value — and it must not assume the year's remaining income is known.
        """
        raise NotImplementedError

    def free_parameters(self) -> dict[str, float]:
        """The ceiling bracket index and the benefit start ages, by name."""
        raise NotImplementedError


def fill_to_bracket(
    taxable_income_ytd: ArrayLike,
    bracket_edges: tuple[float, ...],
    ceiling_index: int,
    available: ArrayLike,
    months_remaining_in_year: int,
) -> NDArray[np.float64]:
    """Amount to withdraw **this month** toward filling the year to a bracket edge.

    The primitive behind "withdraw up to the top of this bracket" policies, and
    the place the annual-to-monthly conversion is made safe. The headroom is
    computed from income already recognised this year, then apportioned over
    the months that remain, so that twelve calls over a year fill the bracket
    once rather than twelve times.

    Args:
        taxable_income_ytd: Taxable income recognised so far this calendar
            year, before this withdrawal, ``(n_paths,)``. Year to date — not
            this month's income, and not a projection of the year's total.
        bracket_edges: Current year's bracket edges, from ``params/``.
        ceiling_index: Which edge to fill to.
        available: Balance available to withdraw this month, ``(n_paths,)``.
        months_remaining_in_year: Months left including this one, from twelve
            in January down to one in December. What the year's remaining
            headroom is divided over.

    Returns:
        This month's withdrawal amount, non-negative and capped by
        ``available``. Zero where year-to-date income already exceeds the
        ceiling.
    """
    raise NotImplementedError

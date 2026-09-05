# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The path loop: run every month for all paths, under one policy.

This calls ``engine.core.step.advance_month`` in a loop over months. It
contains no financial logic of its own and must never grow any.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from engine.core.state import HouseholdState
from engine.mc.returns import RandomDraws
from engine.params.loader import ParamYear
from engine.policy.base import Policy


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Per-year, per-path output of one policy evaluation.

    The simulation steps monthly; the result is recorded **annually**, at each
    December close. Two reasons, and both are deliberate:

    - A monthly trace is twelve times the memory, and at a hundred thousand
      paths over a forty-year horizon that is gigabytes per policy evaluated.
      The optimizer evaluates many.
    - The quantities worth reporting are annual anyway. Tax is assessed on a
      year. Net worth at a month end is noise around net worth at a year end.

    Anything that needs sub-annual detail — checking that a benefit started in
    the right month, that a death mid-year stopped OAS when it should — is
    inspected on a single path with a debug trace, not carried for every path.

    All dollar amounts are real. Conversion to nominal happens at display, in
    ``api/`` or ``cli/``, never here.

    Attributes:
        years: Calendar years simulated, ``(n_years,)``.
        net_worth: Real household net worth at each 31 December,
            ``(n_years, n_paths)``.
        spending: Real after-tax spending achieved over each year, summed from
            the twelve months, ``(n_years, n_paths)``.
        tax_assessed: Real household tax *assessed* on each year's income,
            ``(n_years, n_paths)``. Assessed, not paid: the cash for it leaves
            in the following year's filing month, and the two are a year apart.
            The cash timing lives in the state ledger and is reflected in
            ``net_worth``.
        depleted: Whether the household ran out of money by each year end,
            ``(n_years, n_paths)``. Monotone in year once true. Depletion is
            detected in the month it happens, then reported at the year that
            contains it.
        seed: The seed of the draws used, for reproducibility.
    """

    years: NDArray[np.int64]
    net_worth: NDArray[np.float64]
    spending: NDArray[np.float64]
    tax_assessed: NDArray[np.float64]
    depleted: NDArray[np.bool_]
    seed: int


def run(
    initial_state: HouseholdState,
    policy: Policy,
    draws: RandomDraws,
    params_by_year: dict[int, ParamYear],
    n_years: int,
) -> SimulationResult:
    """Simulate ``n_years`` for every path under one policy, one month at a time.

    The loop is over ``12 * n_years`` months. ``initial_state`` need not open in
    January — a scenario that begins mid-year begins mid-year, and the first
    tax year is a short one. What the loop must not do is skip the year-opening
    phase for that first partial year or double-count it.

    Args:
        initial_state: Opening state for the first simulated month.
        policy: The policy being evaluated.
        draws: Common random numbers, monthly, generated once and shared across
            every policy. Never regenerate inside this function.
        params_by_year: Loaded parameters keyed by tax year. Indexed by the
            calendar year the current month falls in.
        n_years: Horizon in years; the month count is twelve times it.

    Returns:
        A :class:`SimulationResult`, recorded per year.

    Raises:
        ValueError: If ``draws.n_months`` is shorter than the horizon.
    """
    raise NotImplementedError

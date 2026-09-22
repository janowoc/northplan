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

from engine.core.indexation import RealParamYear
from engine.core.state import HouseholdState
from engine.mc.market import MarketInputs
from engine.mc.returns import RandomDraws
from engine.policy.base import Policy


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Per-year, per-path output of one policy evaluation.

    Recorded annually, at each December close, not monthly: a monthly trace is
    twelve times the memory at scale, and the quantities worth reporting (tax,
    net worth) are annual anyway. Sub-annual detail is inspected on a single
    path with a debug trace, not carried for every path.

    All dollar amounts are real; conversion to nominal happens at display, in
    ``api/`` or ``cli/``, never here.

    Attributes:
        years: Calendar years simulated, ``(n_years,)``.
        net_worth: Real household net worth at each 31 December,
            ``(n_years, n_paths)``.
        spending_achieved: Real after-tax spending achieved over each year, summed
            from the twelve months, ``(n_years, n_paths)``.
        tax_assessed: Real household tax *assessed* on each year's income,
            ``(n_years, n_paths)`` -- not paid; the cash leaves in the
            following year's filing month, reflected in ``net_worth``.
        depleted: Whether the household ran out of money by each year end,
            ``(n_years, n_paths)``. Monotone in year once true; detected in
            the month it happens, reported at the year that contains it.
        seed: The seed of the draws used, for reproducibility.
    """

    years: NDArray[np.int64]
    net_worth: NDArray[np.float64]
    spending_achieved: NDArray[np.float64]
    tax_assessed: NDArray[np.float64]
    depleted: NDArray[np.bool_]
    seed: int


def run(
    initial_state: HouseholdState,
    policy: Policy,
    draws: RandomDraws,
    market: MarketInputs,
    real_params: RealParamYear,
) -> SimulationResult:
    """Simulate every path under one policy, one month at a time.

    The loop is over ``draws.n_months`` months, a count the caller derived
    from the longest survival curve in the household
    (:func:`engine.core.build.build_draws`). The run opens on 1 January of
    the scenario's start year and has no separate horizon: it runs to the
    second death (``docs/limitations.md`` L4, L10).

    Args:
        initial_state: Opening state for the first simulated month.
        policy: The policy being evaluated.
        draws: Common random numbers, monthly, generated once and shared across
            every policy. Never regenerate inside this function.
        market: The asset-class allocation and yield mix each account's
            balance grows under.
        real_params: Parameters for the tax year the run opens in, in the
            scenario's real-dollar view.

    Returns:
        A :class:`SimulationResult`, recorded per year.

    Raises:
        ValueError: A state whose ``death_month_index`` is still
            :data:`engine.core.state.DEATH_NOT_DRAWN` on any person is
            refused — :func:`engine.core.build.draw_deaths` runs before
            ``run``, and a half-built opening state must fail loudly. Also
            refused: ``draws.n_paths`` disagreeing with ``initial_state.n_paths``,
            and ``draws.n_months`` less than or equal to the largest
            ``death_month_index`` in the state, which means the draws are too
            short to reach the last death.
    """
    raise NotImplementedError

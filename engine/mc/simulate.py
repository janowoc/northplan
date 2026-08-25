"""The path loop: run every year for all paths, under one policy.

This calls ``engine.core.step.advance_year`` in a loop over years. It contains
no financial logic of its own and must never grow any.
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

    All dollar amounts are real. Conversion to nominal happens at display, in
    ``api/`` or ``cli/``, never here.

    Attributes:
        years: Calendar years simulated, ``(n_years,)``.
        net_worth: Real household net worth at each year end,
            ``(n_years, n_paths)``.
        spending: Real after-tax spending achieved each year,
            ``(n_years, n_paths)``.
        tax_paid: Real total household tax each year, ``(n_years, n_paths)``.
        depleted: Whether the household ran out of money by each year,
            ``(n_years, n_paths)``. Monotone in year once true.
        seed: The seed of the draws used, for reproducibility.
    """

    years: NDArray[np.int64]
    net_worth: NDArray[np.float64]
    spending: NDArray[np.float64]
    tax_paid: NDArray[np.float64]
    depleted: NDArray[np.bool_]
    seed: int


def run(
    initial_state: HouseholdState,
    policy: Policy,
    draws: RandomDraws,
    params_by_year: dict[int, ParamYear],
    n_years: int,
) -> SimulationResult:
    """Simulate ``n_years`` for every path under one policy.

    Args:
        initial_state: Opening state for the first simulated year.
        policy: The policy being evaluated.
        draws: Common random numbers, generated once and shared across every
            policy. Never regenerate inside this function.
        params_by_year: Loaded parameters keyed by tax year.
        n_years: Number of years to simulate.

    Returns:
        A :class:`SimulationResult`.
    """
    raise NotImplementedError

"""Brute-force grid search over policy parameters.

Obviously correct and, over a simulator vectorized across paths, fast enough.
This is the reference every other search method is validated against.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from engine.core.state import HouseholdState
from engine.mc.returns import RandomDraws
from engine.mc.simulate import SimulationResult
from engine.params.loader import ParamYear
from engine.policy.base import Policy


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The outcome of a policy search.

    Attributes:
        best_policy: The highest-scoring policy evaluated.
        best_score: Its objective value.
        evaluated: Every ``(parameters, score)`` pair, in evaluation order.
            Kept so the objective surface can be inspected — a flat surface or
            a best point on the edge of the grid both mean the answer should
            not be trusted yet.
        seed: Seed of the draws every candidate was evaluated against.
    """

    best_policy: Policy
    best_score: float
    evaluated: tuple[tuple[dict[str, float], float], ...]
    seed: int


def search(
    build_policy: Callable[[dict[str, float]], Policy],
    grid: dict[str, Sequence[float]],
    initial_state: HouseholdState,
    draws: RandomDraws,
    params_by_year: dict[int, ParamYear],
    n_years: int,
    objective: Callable[[SimulationResult], float],
) -> SearchResult:
    """Evaluate every point on the grid and return the best.

    Every candidate is run against the same ``draws``. This function must not
    generate random numbers.

    Args:
        build_policy: Turns a parameter dict into a policy. The optimizer never
            mutates a policy in place.
        grid: Parameter name to the values to try. The full Cartesian product
            is evaluated.
        initial_state: Opening state, shared by every candidate.
        draws: Common random numbers.
        params_by_year: Loaded parameters keyed by tax year.
        n_years: Horizon in years. The simulator steps ``12 * n_years``
            months internally and reports per year.
        objective: Reduces a :class:`SimulationResult` to a score to maximise.

    Returns:
        A :class:`SearchResult`.
    """
    raise NotImplementedError

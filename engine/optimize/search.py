# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Brute-force evaluation of an already-expanded set of policy candidates.

This is the reference every other search method is validated against.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from engine.core.indexation import RealParamYear
from engine.mc.returns import RandomDraws
from engine.mc.simulate import SimulationResult
from engine.policy.base import Policy
from engine.scenario import Scenario


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The outcome of a policy search.

    Attributes:
        best_policy: The highest-scoring policy evaluated.
        best_score: Its objective value.
        evaluated: Every ``(parameters, score)`` pair, in evaluation order,
            ``parameters`` being the winning policy's own
            ``free_parameters()``. Kept so the objective surface can be
            inspected — a flat surface or a best point on the edge of the
            grid both mean the answer should not be trusted yet.
        seed: Seed of the draws every candidate was evaluated against.
    """

    best_policy: Policy
    best_score: float
    evaluated: tuple[tuple[dict[str, float], float], ...]
    seed: int


def search(
    scenario: Scenario,
    n_paths: int,
    policies: Sequence[Policy],
    draws: RandomDraws,
    real_params: RealParamYear,
    objective: Callable[[SimulationResult], float],
) -> SearchResult:
    """Evaluate every candidate policy and return the best.

    Every candidate is run against the same ``draws``. This function must not
    generate random numbers. The caller hands in the already-built
    candidates, and this rebuilds the opening state per candidate via
    :func:`engine.core.build.build_initial_state` — elections live in the
    state, and that function is their only writer, so a policy's elections
    (CPP/OAS start ages, RRIF conversion, pension-credit fill) take effect
    only through a state rebuilt for it. Each rebuilt state is then passed
    through :func:`engine.core.build.draw_deaths` with this same ``draws``,
    reached as ``real_params.raw["mortality"]``, which keeps mortality
    identical across candidates — the whole point of common random numbers.
    The :class:`~engine.mc.market.MarketInputs` every candidate's run needs
    is derived once, from ``engine.core.build.build_market_inputs(scenario.assumptions)``.

    Args:
        scenario: The scenario each candidate's opening state is built from.
        n_paths: Monte Carlo paths, passed through to
            :func:`~engine.core.build.build_initial_state` for every
            candidate.
        policies: The candidate policies to evaluate, already built.
        draws: Common random numbers.
        real_params: Parameters for the tax year the run opens in, in the
            scenario's real-dollar view.
        objective: Reduces a :class:`SimulationResult` to a score to maximise.

    Returns:
        A :class:`SearchResult`.
    """
    raise NotImplementedError

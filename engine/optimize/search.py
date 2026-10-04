# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Brute-force evaluation of an already-expanded set of policy candidates.

This is the reference every other search method is validated against. It owns
the report of each candidate and the choice of the best; it owns no random
numbers, which come from the :class:`~engine.mc.prepare.PreparedRun`.

Invariant: every candidate runs against ``prepared.draws``, the same object,
so scores differ only by policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from engine.mc.prepare import PreparedRun, evaluate
from engine.mc.simulate import SimulationResult
from engine.optimize.objective import gis_exposure, median_estate_after_tax, success_probability
from engine.policy.build import build_policy
from engine.scenario.schema import PolicySpec


@dataclass(frozen=True, slots=True)
class CandidateReport:
    """One evaluated candidate.

    Attributes:
        name: The candidate's expanded policy name.
        parameters: Its own ``free_parameters()``, read-only.
        score: The searched objective's value.
        median_estate_after_tax: Median real after-tax estate over paths.
        success_probability: Fraction of paths never depleted.
        gis_exposure: Pooled share of living person-years in the GIS band.
    """

    name: str
    parameters: Mapping[str, float]
    score: float
    median_estate_after_tax: float
    success_probability: float
    gis_exposure: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The outcome of a policy search.

    Attributes:
        best: The first candidate with the highest score.
        best_score: Its objective value.
        evaluated: Every candidate's report, in ``prepared.scenario.policies``
            order, so the objective surface can be inspected -- a flat surface
            or a best point on the edge of the grid both mean the answer should
            not be trusted yet.
        seed: Seed of the draws every candidate was evaluated against.
    """

    best: PolicySpec
    best_score: float
    evaluated: tuple[CandidateReport, ...]
    seed: int


def search(prepared: PreparedRun, objective: Callable[[SimulationResult], float]) -> SearchResult:
    """Evaluate every expanded policy against the shared draws and return the best.

    Every candidate runs against ``prepared.draws``; no random numbers are
    drawn here. Ties keep the earlier candidate.

    Args:
        prepared: The checked inputs from :func:`~engine.mc.prepare.prepare_run`.
        objective: Reduces a :class:`~engine.mc.simulate.SimulationResult` to
            a score to maximise.

    Returns:
        A :class:`SearchResult`.

    Raises:
        ValueError: An objective value is not finite; names the candidate and
            the value.
    """
    reports: list[CandidateReport] = []
    best: PolicySpec | None = None
    best_score = -np.inf
    for spec in prepared.scenario.policies:
        result = evaluate(prepared, spec)
        score = float(objective(result))
        if not np.isfinite(score):
            raise ValueError(f"search: the objective is {score!r} for policy {spec.name!r}.")
        policy = build_policy(spec, prepared.scenario.household)
        reports.append(
            CandidateReport(
                name=spec.name,
                parameters=MappingProxyType(dict(policy.free_parameters())),
                score=score,
                median_estate_after_tax=median_estate_after_tax(result),
                success_probability=success_probability(result),
                gis_exposure=gis_exposure(result),
            )
        )
        if best is None or score > best_score:
            best, best_score = spec, score
    assert best is not None
    return SearchResult(
        best=best, best_score=best_score, evaluated=tuple(reports), seed=prepared.draws.seed
    )

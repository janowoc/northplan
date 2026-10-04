# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``engine.optimize.search.search`` against the committed example scenario.

The example's grid expands to three candidates. One small prepared run is
built per module and every test reads from it without mutating anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from engine.mc import prepare as prepare_module
from engine.mc.prepare import PreparedRun, evaluate, prepare_run
from engine.mc.simulate import SimulationResult
from engine.optimize.objective import (
    gis_exposure,
    median_estate_after_tax,
    success_probability,
)
from engine.optimize.search import search
from engine.scenario import load_scenario

EXAMPLE = Path(__file__).resolve().parents[2] / "scenarios" / "example.yaml"


@pytest.fixture(scope="module")
def prepared() -> PreparedRun:
    return prepare_run(load_scenario(EXAMPLE), n_paths=8)


def objective(result: SimulationResult) -> float:
    return float(np.mean(result.estate_after_tax) - 0.5 * np.std(result.estate_after_tax))


def test_evaluated_lists_every_expanded_policy_in_order_with_its_own_parameters(
    prepared: PreparedRun,
) -> None:
    outcome = search(prepared, objective)

    assert [r.name for r in outcome.evaluated] == [p.name for p in prepared.scenario.policies]
    assert len(outcome.evaluated) == 3
    ages = [r.parameters["elections.cpp_start_age_years.a"] for r in outcome.evaluated]
    assert ages == [60, 65, 70]


def test_each_score_and_metric_is_recomputed_independently(prepared: PreparedRun) -> None:
    outcome = search(prepared, objective)

    for spec, report in zip(prepared.scenario.policies, outcome.evaluated, strict=True):
        result = evaluate(prepared, spec)
        assert report.score == objective(result)
        assert report.median_estate_after_tax == median_estate_after_tax(result)
        assert report.success_probability == success_probability(result)
        assert report.gis_exposure == gis_exposure(result)


def test_best_is_the_argmax(prepared: PreparedRun) -> None:
    outcome = search(prepared, objective)

    scores = [r.score for r in outcome.evaluated]
    assert len(set(scores)) > 1
    index = scores.index(max(scores))
    assert outcome.best == prepared.scenario.policies[index]
    assert outcome.best_score == max(scores)


def test_a_tie_keeps_the_first_candidate(prepared: PreparedRun) -> None:
    outcome = search(prepared, lambda _result: 1.0)

    assert outcome.best == prepared.scenario.policies[0]
    assert outcome.best_score == 1.0


@pytest.mark.parametrize("bad", [float("nan"), float("-inf"), float("inf")])
def test_a_non_finite_objective_raises_naming_the_first_candidate(
    prepared: PreparedRun, bad: float
) -> None:
    first = prepared.scenario.policies[0].name
    with pytest.raises(ValueError) as caught:
        search(prepared, lambda _result: bad)
    assert first in str(caught.value)
    assert repr(bad) in str(caught.value)


def test_the_seed_is_the_draws_seed(prepared: PreparedRun) -> None:
    assert search(prepared, objective).seed == prepared.draws.seed


def test_every_candidate_runs_against_the_prepared_draws_object(
    prepared: PreparedRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    real_run = prepare_module.run

    def spy(*args: Any, **kwargs: Any) -> SimulationResult:
        seen.append(args[2])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(prepare_module, "run", spy)

    search(prepared, objective)

    assert len(seen) == len(prepared.scenario.policies)
    assert all(draws is prepared.draws for draws in seen)


def test_search_draws_no_random_numbers(
    prepared: PreparedRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("search drew random numbers")

    for target, name in (
        ("engine.mc.returns", "generate"),
        ("engine.core.build", "generate"),
        ("engine.core.build", "deterministic"),
        ("engine.core.build", "build_draws"),
        ("engine.core.build", "build_deterministic_draws"),
        ("engine.mc.prepare", "build_draws"),
        ("engine.mc.prepare", "build_deterministic_draws"),
    ):
        monkeypatch.setattr(f"{target}.{name}", refuse)

    assert len(search(prepared, objective).evaluated) == 3


def test_each_report_field_comes_from_its_own_metric(
    prepared: PreparedRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, list[SimulationResult]] = {
        "median": [],
        "success": [],
        "gis": [],
        "objective": [],
    }

    def stub(key: str, value: float) -> Any:
        def metric(result: SimulationResult) -> float:
            seen[key].append(result)
            return value

        return metric

    monkeypatch.setattr("engine.optimize.search.median_estate_after_tax", stub("median", 101.0))
    monkeypatch.setattr("engine.optimize.search.success_probability", stub("success", 0.202))
    monkeypatch.setattr("engine.optimize.search.gis_exposure", stub("gis", 0.303))

    outcome = search(prepared, stub("objective", 404.0))

    count = len(prepared.scenario.policies)
    assert len(outcome.evaluated) == count
    for report in outcome.evaluated:
        assert report.score == 404.0
        assert report.median_estate_after_tax == 101.0
        assert report.success_probability == 0.202
        assert report.gis_exposure == 0.303
    for results in seen.values():
        assert len(results) == count
    for i, result in enumerate(seen["objective"]):
        assert seen["median"][i] is result
        assert seen["success"][i] is result
        assert seen["gis"][i] is result
    for i, result in enumerate(seen["objective"]):
        for other in seen["objective"][i + 1 :]:
            assert result is not other


def test_parameters_are_read_only(prepared: PreparedRun) -> None:
    report = search(prepared, objective).evaluated[0]

    with pytest.raises(TypeError):
        report.parameters["elections.cpp_start_age_years.a"] = 1.0  # type: ignore[index]

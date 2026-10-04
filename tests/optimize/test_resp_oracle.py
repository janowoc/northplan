# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The RESP oracle: the optimizer must find what the grant rules imply.

Contributions up to the grant-eligible amount should beat smaller ones whenever
the plan is spent on education, because the match is guaranteed. Contributions
beyond it should lose, for two distinct reasons, each checked in its own
scenario with its own guard:

* front-loaded, they reach the lifetime contribution maximum early and forfeit
  the grant room later years would have used (the education is paid in full, so
  nothing is wound up and the wind-up cannot be the cause); and
* whatever education does not use is wound up, and its income is charged the
  special tax (the plan is not spent at all, so the wind-up is the cause).

A modest excess that the education costs do use is not penalised in this model:
it grows sheltered and is paid out tax-free (``docs/limitations.md`` L30, L31),
so no test asserts that it loses.

The last test pins ``docs/limitations.md`` L34: the RESP is excluded from the
estate at the final death.

Every RESP parameter is read from ``load_year`` of the fixture's ``start_year``; the
fixture is ``resp_oracle.yaml`` beside this file. A deterministic run lives to
the life table's terminal age, so every run is built once per module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from engine.core.indexation import real_year
from engine.mc.prepare import PreparedRun, evaluate, prepare_run
from engine.mc.simulate import SimulationResult
from engine.optimize.objective import select_objective
from engine.optimize.search import SearchResult, search
from engine.params.loader import load_year
from engine.scenario import Scenario

FIXTURE = Path(__file__).resolve().parent / "resp_oracle.yaml"


def _values() -> dict[str, Any]:
    values = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(values, dict)
    return values


def _start_year() -> int:
    return int(_values()["start_year"])


def _resp_params() -> Any:
    return real_year(load_year(_start_year()), 0.0).resp


def _scenario(
    *,
    annual_cost: float | None = None,
    weights: dict[str, float] | None = None,
    opening_contributions: float | None = None,
) -> Scenario:
    """The fixture with the opening grant room set from params/.

    ``weights`` maps policy name to its RESP weight; each becomes a copy of
    ``no-resp`` with ``resp: k, taxable: 1 - k``. ``None`` keeps only
    ``no-resp``; otherwise the result holds exactly those policies.
    """
    values = _values()
    child = values["household"]["beneficiaries"][0]
    child["resp"]["grant_room_carried"] = _resp_params().annual_amount("grant.room_annual", 0)
    if annual_cost is not None:
        child["education"]["annual_cost"] = annual_cost
    if opening_contributions is not None:
        child["resp"]["contributions"] = opening_contributions
    if weights is not None:
        base = values["policies"][0]
        policies = []
        for name, k in weights.items():
            policy = yaml.safe_load(yaml.safe_dump(base))
            policy["name"] = name
            policy["contribution"]["weights"] = {"resp": k, "taxable": 1.0 - k}
            policies.append(policy)
        values["policies"] = policies
    return Scenario.model_validate(values)


def _median_objective() -> Any:
    return select_objective(
        "median_estate_after_tax", risk_aversion=None, estate_utility_shift=None
    )


@pytest.fixture(scope="module")
def budget() -> dict[int, float]:
    """Taxable contributions of the ``no-resp`` run summed per calendar year: the annual budget."""
    prepared = prepare_run(_scenario(), deterministic=True)
    result = evaluate(prepared, prepared.scenario.policies[0], trace_path=0)
    totals: dict[int, float] = {}
    for record in result.trace:
        year = record.context.year
        totals[year] = totals.get(year, 0.0) + float(record.contributions[0].taxable[0])
    assert totals[_start_year()] > 0.0
    return totals


@pytest.fixture(scope="module")
def weights(budget: dict[int, float]) -> dict[str, float]:
    """Contribution weights set from the start year's budget."""
    resp = _resp_params()
    room = resp.annual_amount("grant.room_annual", 0)
    rate = resp.number("grant.match_rate")
    w = (room / rate) / budget[_start_year()]
    return {"below": 0.5 * w, "at": w, "above": 2.0 * w}


class Case:
    """A prepared three-policy run, its search, and a traced run per policy."""

    def __init__(self, weights: dict[str, float], annual_cost: float) -> None:
        self.prepared: PreparedRun = prepare_run(
            _scenario(annual_cost=annual_cost, weights=weights), deterministic=True
        )
        self.outcome: SearchResult = search(self.prepared, _median_objective())
        self.traced: dict[str, SimulationResult] = {
            spec.name: evaluate(self.prepared, spec, trace_path=0)
            for spec in self.prepared.scenario.policies
        }

    def score(self, name: str) -> float:
        return next(r.score for r in self.outcome.evaluated if r.name == name)


@pytest.fixture(scope="module")
def spent(weights: dict[str, float]) -> Case:
    """Education costs as written in the fixture: the plan is consumed."""
    return Case(
        weights, annual_cost=_values()["household"]["beneficiaries"][0]["education"]["annual_cost"]
    )


@pytest.fixture(scope="module")
def unspent(weights: dict[str, float]) -> Case:
    """No education cost: the plan is wound up."""
    return Case(weights, annual_cost=0.0)


def _grant_years() -> list[int]:
    resp = _resp_params()
    birth_year = _values()["household"]["beneficiaries"][0]["birth_year"]
    last = birth_year + int(resp.number("grant.cessation_age_years"))
    return list(range(_start_year(), last + 1))


# --- premises ---------------------------------------------------------------


def test_grant_years_fit_under_the_lifetime_maximum() -> None:
    resp = _resp_params()
    room = resp.annual_amount("grant.room_annual", 0)
    lifetime = resp.annual_amount("grant.maximum_lifetime", 0)

    assert len(_grant_years()) * room <= lifetime


def test_at_collects_the_full_basic_grant_every_grant_year(spent: Case) -> None:
    room = _resp_params().annual_amount("grant.room_annual", 0)
    trace = spent.traced["at"].trace
    years = _grant_years()

    for year in years:
        december = [r for r in trace if r.context.year == year and r.context.month == 12]
        assert len(december) == 1
        received = float(december[0].beneficiaries[0].resp.grant_received_ytd[0])
        assert received == pytest.approx(room, abs=1e-6), year

    last = next(r for r in trace if r.context.year == years[-1] and r.context.month == 12)
    assert float(last.beneficiaries[0].resp.grants_lifetime[0]) == pytest.approx(
        len(years) * room, abs=1e-6
    )


def test_at_contributes_the_grant_eligible_amount_scaled_by_each_years_budget(
    spent: Case, budget: dict[int, float]
) -> None:
    resp = _resp_params()
    eligible = resp.annual_amount("grant.room_annual", 0) / resp.number("grant.match_rate")
    start = _start_year()
    trace = spent.traced["at"].trace

    for year in _grant_years():
        contribution = sum(
            float(r.resp_contributions[0][0]) for r in trace if r.context.year == year
        )
        assert contribution >= eligible * (1 - 1e-9), year
        assert contribution <= eligible * budget[year] / budget[start] * (1 + 1e-9), year

    # The guard that the upper bound is not vacuous: some later year's budget is larger.
    assert any(budget[year] > budget[start] for year in _grant_years())


# --- oracle (a): the plan is spent on education -----------------------------


def test_contributing_to_the_grant_maximum_beats_less_and_more_when_education_uses_the_plan(
    spent: Case,
) -> None:
    assert spent.score("at") > spent.score("below")
    assert spent.score("above") < spent.score("at")
    assert spent.outcome.best.name == "at"


def test_no_policy_winds_up_a_plan_the_education_consumes(spent: Case) -> None:
    for name, result in spent.traced.items():
        for record in result.trace:
            assert float(record.wind_up_withholding[0][0]) == 0.0, name
            assert float(record.wind_up_to_cash[0][0]) == 0.0, name


def test_front_loading_forfeits_lifetime_grant(spent: Case) -> None:
    def peak(name: str) -> float:
        return max(
            float(r.beneficiaries[0].resp.grants_lifetime[0]) for r in spent.traced[name].trace
        )

    assert peak("above") < peak("at")


# --- oracle (b): the plan is wound up ---------------------------------------


def test_contributions_lose_when_the_plan_is_wound_up_and_lose_more_the_larger_they_are(
    unspent: Case,
) -> None:
    assert unspent.score("below") > unspent.score("at") > unspent.score("above")


def test_the_wind_up_charge_grows_with_the_contribution(unspent: Case) -> None:
    charge = {
        name: sum(float(r.wind_up_withholding[0][0]) for r in result.trace)
        for name, result in unspent.traced.items()
    }

    assert charge["below"] > 0.0
    assert charge["below"] < charge["at"] < charge["above"]


# --- L34 ----------------------------------------------------------------------


def test_resp_is_excluded_from_the_estate_at_the_final_death() -> None:
    estates = []
    closes = []
    for opening in (0.0, 20000.0):
        prepared = prepare_run(_scenario(opening_contributions=opening), deterministic=True)
        result = evaluate(
            prepared,
            prepared.scenario.policies[0],
            trace_path=0,
            death_months=[100],
        )
        estates.append(result.estate_after_tax)
        closes.append(
            (float(result.trace[99].resp_close[0][0]), float(result.trace[100].resp_close[0][0]))
        )

    # The guard that the test reaches its branch: the plan existed, then was dropped.
    assert closes[1][0] > 0.0
    assert closes[1][1] == 0.0
    np.testing.assert_array_equal(estates[0], estates[1])

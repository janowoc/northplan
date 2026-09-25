# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``engine.mc.simulate.run``: the one loop over time, exercised against the example."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from engine.core.build import (
    build_deterministic_draws,
    build_draws,
    build_initial_state,
    build_market_inputs,
    draw_deaths,
)
from engine.core.indexation import real_year
from engine.mc.market import DEFAULT_KIND, MarketInputs
from engine.mc.returns import RandomDraws
from engine.mc.simulate import run
from engine.params.loader import load_year
from engine.scenario import INVESTABLE_KINDS, load_scenario

from .conftest import walk
from .policies import DoNothingPolicy, RecordingPolicy

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "scenarios" / "example.yaml"


@pytest.fixture
def scenario():
    return load_scenario(EXAMPLE)


@pytest.fixture
def mortality(scenario):
    return load_year(scenario.start_year)["mortality"]


@pytest.fixture
def market(scenario):
    return build_market_inputs(scenario.assumptions)


@pytest.fixture
def real_params(scenario):
    return real_year(load_year(scenario.start_year), scenario.assumptions.inflation)


@pytest.fixture
def draws(scenario, market, mortality):
    return build_deterministic_draws(scenario, market, mortality)


@pytest.fixture
def opening_state(scenario, draws, mortality):
    state = build_initial_state(scenario, n_paths=draws.n_paths)
    return draw_deaths(state, draws, mortality)


@pytest.fixture
def withdrawal_order(scenario):
    return scenario.policies[0].withdrawal.order


class TestResultRows:
    """Requirement 12: a row per December close, and the run steps every month."""

    def test_full_run_row_and_month_counts(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        assert draws.n_months == 604

        recorder = RecordingPolicy(DoNothingPolicy(opening_state.elections, withdrawal_order))
        result = run(opening_state, recorder, draws, market, real_params, trace_path=0)

        assert len(result.years) == 50
        assert int(result.years[-1]) == 2075
        assert len(result.trace) == draws.n_months
        assert len(recorder.calls) == draws.n_months


class TestRunRefusals:
    """Requirement 13: every documented refusal actually refuses."""

    def test_death_not_drawn_is_refused(self, scenario, draws, market, real_params, withdrawal_order):
        state = build_initial_state(scenario, n_paths=1)  # draw_deaths never ran
        policy = DoNothingPolicy(state.elections, withdrawal_order)
        with pytest.raises(ValueError, match="DEATH_NOT_DRAWN"):
            run(state, policy, draws, market, real_params)

    def test_elections_mismatch_is_refused(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        mismatched_elections = dataclasses.replace(
            opening_state.elections,
            fill_pension_credit=not opening_state.elections.fill_pension_credit,
        )
        policy = DoNothingPolicy(mismatched_elections, withdrawal_order)
        with pytest.raises(ValueError, match="elections"):
            run(opening_state, policy, draws, market, real_params)

    def test_n_paths_mismatch_is_refused(
        self, scenario, market, mortality, real_params, withdrawal_order
    ):
        good_draws = build_draws(scenario, market, n_paths=2, mortality=mortality)
        state = build_initial_state(scenario, n_paths=2)
        state = draw_deaths(state, good_draws, mortality)

        mismatched_draws = build_draws(scenario, market, n_paths=3, mortality=mortality)
        policy = DoNothingPolicy(state.elections, withdrawal_order)
        with pytest.raises(ValueError, match="n_paths"):
            run(state, policy, mismatched_draws, market, real_params)

    def test_draws_too_short_is_refused(self, opening_state, draws, market, real_params, withdrawal_order):
        short_months = 5
        short_draws = RandomDraws(
            seed=draws.seed,
            real_returns=draws.real_returns[:short_months],
            mortality=draws.mortality,
            n_months=short_months,
            n_paths=draws.n_paths,
        )
        policy = DoNothingPolicy(opening_state.elections, withdrawal_order)
        with pytest.raises(ValueError, match="n_months"):
            run(opening_state, policy, short_draws, market, real_params)

    def test_asset_count_mismatch_is_refused(
        self, opening_state, draws, real_params, withdrawal_order
    ):
        mismatched_market = MarketInputs(
            asset_class_names=("equity",),
            annual_means=np.array([0.05]),
            annual_covariance=np.array([[0.0256]]),
            interest_yields=np.array([0.0]),
            dividend_yields=np.array([0.02]),
            distributed_gains_yields=np.array([0.005]),
            weights_by_kind={DEFAULT_KIND: np.array([1.0])},
            investable_kinds=INVESTABLE_KINDS,
        )
        policy = DoNothingPolicy(opening_state.elections, withdrawal_order)
        with pytest.raises(ValueError, match="asset_class_names"):
            run(opening_state, policy, draws, mismatched_market, real_params)

    def test_trace_path_out_of_range_is_refused(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        policy = DoNothingPolicy(opening_state.elections, withdrawal_order)
        with pytest.raises(ValueError, match="trace_path"):
            run(opening_state, policy, draws, market, real_params, trace_path=99)


class TestShapes:
    """Requirement 15: the walker passes and SimulationResult carries the documented shapes."""

    def test_walker_passes_and_result_arrays_have_documented_shapes(
        self, scenario, market, mortality, withdrawal_order
    ):
        n_paths = 3
        draws = build_draws(scenario, market, n_paths=n_paths, mortality=mortality)
        assert draws.seed == scenario.seed

        state = build_initial_state(scenario, n_paths=n_paths)
        state = draw_deaths(state, draws, mortality)
        real_params = real_year(load_year(scenario.start_year), scenario.assumptions.inflation)

        recorder = RecordingPolicy(DoNothingPolicy(state.elections, withdrawal_order))
        result = run(state, recorder, draws, market, real_params)

        assert recorder.calls, "the policy was never called; the walker check would be vacuous"
        first_state, _first_context = recorder.calls[0]
        last_state, _last_context = recorder.calls[-1]
        assert walk(first_state, n_paths) > 0
        assert walk(last_state, n_paths) > 0

        n_years = len(result.years)
        assert result.years.shape == (n_years,)
        assert result.net_worth.shape == (n_years, n_paths)
        assert result.spending_achieved.shape == (n_years, n_paths)
        assert result.tax_assessed.shape == (n_years, n_paths)
        assert result.depleted.shape == (n_years, n_paths)

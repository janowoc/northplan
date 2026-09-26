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
from engine.core.state import updated
from engine.mc.market import DEFAULT_KIND, MarketInputs
from engine.mc.returns import RandomDraws
from engine.mc.simulate import run
from engine.params.loader import load_year
from engine.scenario import INVESTABLE_KINDS, load_scenario

from .conftest import walk
from .policies import DoNothingPolicy, RecordingPolicy

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "scenarios" / "example.yaml"
COUPLE = REPO_ROOT / "scenarios" / "late_life_couple.yaml"


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
def couple_scenario():
    return load_scenario(COUPLE)


@pytest.fixture
def couple_mortality(couple_scenario):
    return load_year(couple_scenario.start_year)["mortality"]


@pytest.fixture
def couple_market(couple_scenario):
    return build_market_inputs(couple_scenario.assumptions)


@pytest.fixture
def couple_real_params(couple_scenario):
    return real_year(load_year(couple_scenario.start_year), couple_scenario.assumptions.inflation)


@pytest.fixture
def couple_withdrawal_order(couple_scenario):
    return couple_scenario.policies[0].withdrawal.order


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

    def test_death_not_drawn_is_refused(
        self, scenario, draws, market, real_params, withdrawal_order
    ):
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

    def test_draws_too_short_is_refused(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
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


# =============================================================================
# #36: SimulationResult's new fields, and run's early stop
# =============================================================================


class TestDeathYear:
    """Requirement 11: ``death_year`` matches the forced/drawn death month, per person."""

    def test_death_year_matches_forced_death_months(
        self, scenario, market, mortality, real_params, withdrawal_order
    ):
        n_paths = 3
        forced = np.array([5, 20, 35], dtype=np.int64)
        draws = build_draws(scenario, market, n_paths=n_paths, mortality=mortality)
        state = build_initial_state(scenario, n_paths=n_paths)
        state = draw_deaths(state, draws, mortality)
        person = updated(state.persons[0], death_month_index=forced)
        state = updated(state, persons=(person,))

        policy = DoNothingPolicy(state.elections, withdrawal_order)
        result = run(state, policy, draws, market, real_params)

        expected = state.year + forced // 12
        np.testing.assert_array_equal(result.death_year[0], expected)
        assert result.death_year.shape == (1, n_paths)


class TestRunStopsAtTheFinalDeath:
    """Requirement 11 (D-A): the run stops at the first December on or after the final
    death, or at ``n_months``, whichever comes first -- not at the death month itself.
    """

    FORCED_DEATH_MONTH_INDEX = 15

    def _forced_state(self, scenario, market, mortality, n_paths=1):
        draws = build_draws(scenario, market, n_paths=n_paths, mortality=mortality)
        state = build_initial_state(scenario, n_paths=n_paths)
        state = draw_deaths(state, draws, mortality)
        person = updated(
            state.persons[0],
            death_month_index=np.full(n_paths, self.FORCED_DEATH_MONTH_INDEX, dtype=np.int64),
        )
        state = updated(state, persons=(person,))
        return state, draws

    def test_run_stops_at_the_next_december(
        self, scenario, market, mortality, real_params, withdrawal_order
    ):
        state, draws = self._forced_state(scenario, market, mortality)
        expected_last_month = min(
            (self.FORCED_DEATH_MONTH_INDEX // 12) * 12 + 11, draws.n_months - 1
        )
        assert expected_last_month > self.FORCED_DEATH_MONTH_INDEX, (
            "the forced death must be well before the next December for this to test "
            "anything about waiting for it rather than stopping at the death month"
        )

        recorder = RecordingPolicy(DoNothingPolicy(state.elections, withdrawal_order))
        result = run(state, recorder, draws, market, real_params, trace_path=0)

        assert len(result.trace) == expected_last_month + 1
        assert len(recorder.calls) == expected_last_month + 1
        assert np.all(np.isfinite(result.estate_after_tax))

    def test_the_final_death_years_own_row_matches_the_trace(
        self, scenario, market, mortality, real_params, withdrawal_order
    ):
        """D-A's missing leg of requirement 12: the death year's own December is reached
        by the draws here, so it gets a row, and that row sums the trace exactly.
        """
        state, draws = self._forced_state(scenario, market, mortality)
        recorder = RecordingPolicy(DoNothingPolicy(state.elections, withdrawal_order))
        result = run(state, recorder, draws, market, real_params, trace_path=0)

        death_year = scenario.start_year + self.FORCED_DEATH_MONTH_INDEX // 12
        assert death_year in result.years.tolist()  # guard
        year_index = int(np.flatnonzero(result.years == death_year)[0])

        year_start = (death_year - scenario.start_year) * 12
        year_end = year_start + 12
        monthly = [
            r.context.spending - r.spending_cut
            for r in result.trace
            if year_start <= r.context.month_index < year_end
        ]
        np.testing.assert_allclose(sum(monthly), result.spending_achieved[year_index], atol=0.005)


class TestPersonYearCounts:
    """Requirement 11 / R5: ``gis_band_person_years`` and ``living_person_years``."""

    def test_on_the_seeded_couple(
        self,
        couple_scenario,
        couple_market,
        couple_mortality,
        couple_real_params,
        couple_withdrawal_order,
    ):
        n_paths = 64
        draws = build_draws(
            couple_scenario, couple_market, n_paths=n_paths, mortality=couple_mortality
        )
        state = build_initial_state(couple_scenario, n_paths=n_paths)
        state = draw_deaths(state, draws, couple_mortality)

        recorder = RecordingPolicy(DoNothingPolicy(state.elections, couple_withdrawal_order))
        result = run(state, recorder, draws, couple_market, couple_real_params)

        assert result.gis_band_person_years.sum() > 0  # guard (R5)

        # living_person_years is derived purely from death_month_index and result.years,
        # both known exactly regardless of when the run stopped -- no missing data, for
        # every path.
        expected_living_person_years = np.zeros(n_paths, dtype=np.int64)
        for person in state.persons:
            for year in result.years:
                december_index = (int(year) - state.year) * 12 + 11
                expected_living_person_years += (person.death_month_index > december_index).astype(
                    np.int64
                )
        np.testing.assert_array_equal(result.living_person_years, expected_living_person_years)

        # gis_band_person_years, by contrast, needs each closed year's actual gis_band,
        # which this test can only read back from what RecordingPolicy saw -- and that is
        # missing exactly the household's very last closed year (its own December close
        # runs after the last recorded call). Restrict the hand-sum to paths whose own
        # second death falls before that last year, for which the comparison is exact.
        last_year = int(result.years[-1])
        per_path_second_death_year = np.maximum(result.death_year[0], result.death_year[1])
        safe = per_path_second_death_year < last_year
        assert np.any(safe)  # guard: at least one path finished well before the run did

        history = recorder.calls[-1][0].history
        expected_gis_band_person_years = np.zeros(n_paths, dtype=np.int64)
        for year_record in history:
            for band in year_record.gis_band:
                expected_gis_band_person_years += band.astype(np.int64)
        np.testing.assert_array_equal(
            result.gis_band_person_years[safe], expected_gis_band_person_years[safe]
        )

    def test_a_december_final_death_does_not_count_that_december(
        self, scenario, market, mortality, real_params, withdrawal_order
    ):
        """R5, corrected (round 3, T4): k = a December index + 1 cannot distinguish
        ``>`` from ``>=`` (both agree there). Force the death to k = 11, December of
        the start year itself: ``alive_new = death_month_index > m`` is false that
        same December, so it must not count as a living year. The same run also
        covers D-A's own stopping rule when the final death falls in December.
        """
        n_paths = 1
        draws = build_draws(scenario, market, n_paths=n_paths, mortality=mortality)
        state = build_initial_state(scenario, n_paths=n_paths)
        state = draw_deaths(state, draws, mortality)

        forced_month_index = 11  # December of the start year
        person = updated(
            state.persons[0],
            death_month_index=np.full(n_paths, forced_month_index, dtype=np.int64),
        )
        state = updated(state, persons=(person,))

        policy = DoNothingPolicy(state.elections, withdrawal_order)
        result = run(state, policy, draws, market, real_params, trace_path=0)

        np.testing.assert_array_equal(result.years, [state.year])
        assert len(result.trace) == 12  # stops at its own December close
        assert int(result.living_person_years[0]) == 0

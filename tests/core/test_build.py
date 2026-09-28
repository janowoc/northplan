# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The scenario-to-state builder, exercised against the committed example.

Spot-checks rather than an exhaustive mapping test: the class-level shape,
dtype, and immutability invariants are already covered, exactly once, by
``conftest.py``'s :func:`~tests.core.conftest.walk`, which this module
imports rather than re-implementing.

A second group of tests below builds small variants of the example scenario
— via ``.model_copy(update=...)`` on the frozen pydantic models the schema
already validated, never a hand-rolled dict — to exercise branches the
committed example does not: a benefit already in pay, a date before the run,
a spending schedule with a different shape. ``.model_copy`` does not re-run
the schema's validators, which is exactly right here: the scenario-level
rejections (a bridge that would never pay, among them) are the schema's own
tests, in ``tests/scenario/test_schema.py``, not this module's.
"""

from __future__ import annotations

import enum
from pathlib import Path

import numpy as np
import pytest

from engine.core.build import (
    _month_offset,
    build_deterministic_draws,
    build_draws,
    build_elections,
    build_initial_state,
    build_market_inputs,
    draw_deaths,
    with_death_months,
)
from engine.core.indexation import as_filed_to_real_factor
from engine.core.mortality import death_month_index, months_to_terminal, survival_curve
from engine.core.state import DEATH_NOT_DRAWN, updated
from engine.mc.market import DEFAULT_KIND
from engine.mc.returns import generate
from engine.params.loader import ParamSet, load_year
from engine.scenario import (
    DEFAULT_ALLOCATION,
    Assumptions,
    LiraAccount,
    OasEntitlement,
    SpendingBand,
    load_scenario,
)

from .conftest import walk

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "scenarios" / "example.yaml"
COUPLE = REPO_ROOT / "scenarios" / "late_life_couple.yaml"

N_PATHS = 3


class _Month(enum.IntEnum):
    APRIL = 27


#: The example scenario currently builds exactly 53 arrays at n_paths=3.
#: Close to that count, not merely "not zero": at 30 this floor passed with
#: the entire RESP subtree gone (44) or the entire year-to-date ledger gone
#: (38) — a wrong tree, not an empty one, and the floor never noticed. Left
#: with room for a field or two, not a whole subtree.
MINIMUM_ARRAYS = 50


@pytest.fixture
def scenario():
    return load_scenario(EXAMPLE)


#: The mortality table used by ``TestBuildDraws`` and ``TestDrawDeaths`` below — a
#: constant annual ``q`` at every age, one constant per sex, with the terminal row at
#: ``1.0`` the engine's own convention requires, mirroring
#: ``tests/core/test_mortality.py``'s fixture. Never presented as a real value.
#: Comfortably above the example person's age at start (60 in 2026) and above the
#: second, deliberately younger person used below (11 in 2026), so neither curve
#: degenerates to the length-1 "already past terminal age" case.
TERMINAL_AGE = 80
SYNTHETIC_Q = 0.05


@pytest.fixture
def mortality(tmp_path: Path, scenario) -> ParamSet:
    year_dir = tmp_path / str(scenario.start_year)
    year_dir.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"    {age}: {SYNTHETIC_Q}" for age in range(TERMINAL_AGE))
    text = f"""\
# SYNTHETIC TEST FIXTURE -- flat q(x) per sex, not a real mortality curve.
terminal_age_years: {TERMINAL_AGE}
q_x:
  f:
{rows}
    {TERMINAL_AGE}: 1.0
  m:
{rows}
    {TERMINAL_AGE}: 1.0
"""
    (year_dir / "mortality.yaml").write_text(text, encoding="utf-8")
    return load_year(scenario.start_year, tmp_path)["mortality"]


@pytest.fixture
def two_person_scenario(scenario):
    """A household whose two persons' ``months_to_terminal`` differ, and whose sex differs too.

    Person ``a`` is the example's 1966-born ``f`` adult; person ``b`` is a deliberately
    much younger ``m`` copy, so the household's month count is not the first person's —
    a build_draws that only read persons[0] would compute a month count too small for
    ``b`` and this fixture is built specifically to make that failure visible. The sex
    difference is harmless and mirrors a real household, but it does no work here:
    ``SYNTHETIC_Q`` is identical for ``"f"`` and ``"m"``. What makes swapping the two
    persons' rows change the answer is the birth-year difference together with the two
    independent ``draws.mortality`` rows, so a ``draw_deaths`` that wrote
    ``draws.mortality[0]`` for every person cannot pass a test built against this
    fixture by coincidence.
    """
    person_a = scenario.household.persons[0]
    person_b = person_a.model_copy(
        update={"id": "b", "sex": "m", "birth_year": 2015, "birth_month": 1}
    )
    new_household = scenario.household.model_copy(update={"persons": (person_a, person_b)})

    elections = scenario.policies[0].elections
    new_elections = elections.model_copy(
        update={
            "cpp_start_age_years": {**elections.cpp_start_age_years, "b": 65},
            "oas_start_age_years": {**elections.oas_start_age_years, "b": 65},
        }
    )
    new_policy = scenario.policies[0].model_copy(update={"elections": new_elections})
    return scenario.model_copy(update={"household": new_household, "policies": (new_policy,)})


class TestBuildAgainstTheExample:
    def test_builds_and_passes_the_walker(self, scenario) -> None:
        state = build_initial_state(scenario, n_paths=N_PATHS)
        count = walk(state, N_PATHS)
        assert count >= MINIMUM_ARRAYS, (
            f"walked only {count} arrays building the example scenario; the "
            "walk is vacuous if it can pass over an empty tree."
        )

    def test_opening_position(self, scenario) -> None:
        state = build_initial_state(scenario, n_paths=N_PATHS)
        assert (state.year, state.month, state.month_index) == (2026, 1, 0)
        assert state.province == "ab"

    def test_rrsp_balance(self, scenario) -> None:
        state = build_initial_state(scenario, n_paths=N_PATHS)
        person_a = state.persons[0]
        assert (person_a.rrsp.balance == 400_000.0).all()
        assert (person_a.rrsp.room == 25_000.0).all()

    def test_resp_buckets_and_subscriber_index(self, scenario) -> None:
        state = build_initial_state(scenario, n_paths=N_PATHS)
        resp = state.beneficiaries[0].resp
        assert (resp.contributions == 24_000.0).all()
        assert (resp.grants == 4_800.0).all()
        assert (resp.income == 1_200.0).all()
        assert (resp.contributions_lifetime == 24_000.0).all()
        assert (resp.grants_lifetime == 4_800.0).all()
        assert resp.subscriber_index == 0

    def test_db_pension_start_and_indexation(self, scenario) -> None:
        state = build_initial_state(scenario, n_paths=N_PATHS)
        pension = state.persons[0].pensions[0]
        # 2031-04, start_year 2026: (2031-2026)*12 + (4-1) = 63.
        assert pension.start_month_index == 63
        assert pension.indexed is False

    def test_employment_band(self, scenario) -> None:
        state = build_initial_state(scenario, n_paths=N_PATHS)
        band = state.persons[0].employment[0]
        # 2026-01: (2026-2026)*12 + 0 = 0. 2031-12: (2031-2026)*12 + 11 = 71.
        assert band.from_month_index == 0
        assert band.to_month_index == 71
        assert band.monthly_amount == pytest.approx(95_000.0 / 12.0)

    def test_cpp_contributory_history(self, scenario) -> None:
        state = build_initial_state(scenario, n_paths=N_PATHS)
        assert state.persons[0].cpp.contributory_history == pytest.approx(0.85)
        assert state.persons[0].cpp.in_pay_monthly is None
        assert state.persons[0].oas.contributory_history is None

    def test_oas_start_age(self, scenario) -> None:
        state = build_initial_state(scenario, n_paths=N_PATHS)
        assert state.persons[0].oas.start_age_months == 65 * 12
        assert state.elections.oas_start_age_months == (65 * 12,)

    def test_household_cash_opens_at_the_example_balance(self, scenario) -> None:
        state = build_initial_state(scenario, n_paths=N_PATHS)
        assert (state.cash.balance == 20_000.0).all()

    def test_lira_jurisdiction_and_balance(self, scenario) -> None:
        state = build_initial_state(scenario, n_paths=N_PATHS)
        assert state.persons[0].lira.jurisdiction == "ab"
        assert (state.persons[0].lira.balance == 80_000.0).all()

    def test_example_lif_opens_empty(self, scenario) -> None:
        state = build_initial_state(scenario, n_paths=N_PATHS)
        lif = state.persons[0].lif
        assert (lif.balance == 0.0).all()
        assert lif.jurisdiction == ""
        assert lif.opened_year is None

    def test_prior_year_net_income_is_restated_from_the_scenario(self, scenario) -> None:
        """The example states 92000 as filed for 2025 at 2% inflation. Restated to January
        2026 dollars from mid-2025, half a year earlier: 92000 * 1.02**0.5 (L5).
        """
        state = build_initial_state(scenario, n_paths=N_PATHS)
        expected = 92_000.0 * 1.02**0.5  # 92915.45
        assert state.persons[0].prior_year_net_income == pytest.approx(
            np.full(N_PATHS, expected), rel=1e-12
        )

    def test_the_two_years_of_net_income_are_restated_and_not_crossed(self, scenario) -> None:
        """The example's two figures (92000, 88000) differ on purpose (brief-49 s8): a
        test that only checked one field could pass with the two swapped in build.py.
        Each is restated from the middle of its own year: half a year for 2025, a year
        and a half for 2024 (L5).
        """
        state = build_initial_state(scenario, n_paths=N_PATHS)
        expected_prior = 92_000.0 * 1.02**0.5  # 92915.45
        expected_two_prior = 88_000.0 * 1.02**1.5  # 90653.16
        assert state.persons[0].prior_year_net_income == pytest.approx(
            np.full(N_PATHS, expected_prior), rel=1e-12
        )
        assert state.persons[0].net_income_two_years_prior == pytest.approx(
            np.full(N_PATHS, expected_two_prior), rel=1e-12
        )

    def test_education_start_month_index(self, scenario) -> None:
        state = build_initial_state(scenario, n_paths=N_PATHS)
        resp = state.beneficiaries[0].resp
        # 2033-09, start_year 2026: (2033-2026)*12 + (9-1) = 92.
        assert resp.education_start_month_index == 92

    def test_spending_schedule_and_monthly(self, scenario) -> None:
        state = build_initial_state(scenario, n_paths=N_PATHS)
        assert [(level.from_year, level.monthly_level) for level in state.spending_schedule] == [
            (2026, pytest.approx(80_000.0 / 12.0)),
            (2032, pytest.approx(65_000.0 / 12.0)),
        ]
        # First band, {from_year: 2026, annual: 80000}, is in effect at start_year.
        assert state.spending_monthly == pytest.approx(80_000.0 / 12.0)

    def test_estate_after_tax_is_all_nan(self, scenario) -> None:
        state = build_initial_state(scenario, n_paths=N_PATHS)
        assert state.estate_after_tax.shape == (N_PATHS,)
        assert all(value != value for value in state.estate_after_tax)  # NaN != NaN

    def test_death_month_index_is_all_sentinel(self, scenario) -> None:
        state = build_initial_state(scenario, n_paths=N_PATHS)
        for person in state.persons:
            assert (person.death_month_index == DEATH_NOT_DRAWN).all()

    def test_the_floor_actually_bites_on_a_truncated_state(self, scenario) -> None:
        """Shows ``MINIMUM_ARRAYS`` can fail, not just that it currently passes."""
        state = build_initial_state(scenario, n_paths=N_PATHS)
        truncated = updated(state, beneficiaries=())
        assert walk(truncated, N_PATHS) < MINIMUM_ARRAYS


class TestAsFiledRestatementFactor:
    """The opening pair equals the scenario figure times the restatement factor."""

    def test_the_built_pair_equals_the_scenario_figure_times_the_factor(self, scenario) -> None:
        inflation = scenario.assumptions.inflation
        state = build_initial_state(scenario, n_paths=N_PATHS)
        person = scenario.household.persons[0]
        expected_prior = person.prior_year_net_income * as_filed_to_real_factor(inflation, 1)
        expected_two_prior = person.net_income_two_years_prior * as_filed_to_real_factor(
            inflation, 2
        )
        assert state.persons[0].prior_year_net_income == pytest.approx(
            np.full(N_PATHS, expected_prior)
        )
        assert state.persons[0].net_income_two_years_prior == pytest.approx(
            np.full(N_PATHS, expected_two_prior)
        )

    def test_zero_inflation_leaves_the_pair_equal_to_the_scenario_figure(self, scenario) -> None:
        zero_inflation_assumptions = scenario.assumptions.model_copy(update={"inflation": 0.0})
        zero_inflation_scenario = scenario.model_copy(
            update={"assumptions": zero_inflation_assumptions}
        )
        state = build_initial_state(zero_inflation_scenario, n_paths=N_PATHS)
        person = zero_inflation_scenario.household.persons[0]
        assert state.persons[0].prior_year_net_income == pytest.approx(
            np.full(N_PATHS, person.prior_year_net_income)
        )
        assert state.persons[0].net_income_two_years_prior == pytest.approx(
            np.full(N_PATHS, person.net_income_two_years_prior)
        )


class TestBuildMarketInputs:
    def test_asset_class_names_and_means_and_yields_and_covariance(self, scenario) -> None:
        market = build_market_inputs(scenario.assumptions)
        names = scenario.assumptions.asset_class_names
        classes = [scenario.assumptions.asset_classes[name] for name in names]

        assert market.asset_class_names == names
        np.testing.assert_allclose(market.annual_means, [c.real_mean for c in classes])
        np.testing.assert_allclose(market.interest_yields, [c.interest_yield for c in classes])
        np.testing.assert_allclose(market.dividend_yields, [c.dividend_yield for c in classes])
        np.testing.assert_allclose(
            market.distributed_gains_yields, [c.distributed_gains_yield for c in classes]
        )

        vols = [c.vol for c in classes]
        n = len(names)
        expected_covariance = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                expected_covariance[i, j] = (
                    scenario.assumptions.correlation[i][j] * vols[i] * vols[j]
                )
        np.testing.assert_allclose(market.annual_covariance, expected_covariance)

    def test_weights_for_rrsp_and_default_and_resp(self, scenario) -> None:
        market = build_market_inputs(scenario.assumptions)
        names = scenario.assumptions.asset_class_names
        default_weights = [
            scenario.assumptions.allocations["default"].get(name, 0.0) for name in names
        ]
        resp_weights = [scenario.assumptions.allocations["resp"].get(name, 0.0) for name in names]

        np.testing.assert_allclose(market.weights("default"), default_weights)
        np.testing.assert_allclose(market.weights("rrsp"), default_weights)
        np.testing.assert_allclose(market.weights("resp"), resp_weights)

    def test_every_array_and_weight_vector_is_read_only(self, scenario) -> None:
        market = build_market_inputs(scenario.assumptions)
        for array in (
            market.annual_means,
            market.annual_covariance,
            market.interest_yields,
            market.dividend_yields,
            market.distributed_gains_yields,
        ):
            assert not array.flags.writeable
        for weights in market.weights_by_kind.values():
            assert not weights.flags.writeable

    def test_a_class_absent_from_an_allocation_gets_zero_weight(self, scenario) -> None:
        assumptions = scenario.assumptions
        new_allocations = dict(assumptions.allocations)
        new_allocations["resp"] = {"equity": 1.0}
        new_assumptions = Assumptions.model_validate(
            {
                "inflation": assumptions.inflation,
                "asset_classes": {
                    name: assumptions.asset_classes[name].model_dump()
                    for name in assumptions.asset_class_names
                },
                "correlation": assumptions.correlation,
                "allocations": new_allocations,
            }
        )

        market = build_market_inputs(new_assumptions)

        bonds_index = new_assumptions.asset_class_names.index("bonds")
        assert market.weights("resp")[bonds_index] == 0.0

    def test_weights_follow_asset_class_order_not_allocation_order(self, scenario) -> None:
        assumptions = scenario.assumptions
        new_allocations = {
            "default": {"bonds": 0.4, "equity": 0.6},
            "resp": {"bonds": 0.7, "equity": 0.3},
        }
        new_assumptions = Assumptions.model_validate(
            {
                "inflation": assumptions.inflation,
                "asset_classes": {
                    name: assumptions.asset_classes[name].model_dump()
                    for name in assumptions.asset_class_names
                },
                "correlation": assumptions.correlation,
                "allocations": new_allocations,
            }
        )

        market = build_market_inputs(new_assumptions)

        names = new_assumptions.asset_class_names
        assert names == ("equity", "bonds")

        equity_index = names.index("equity")
        bonds_index = names.index("bonds")

        default_weights = market.weights("default")
        assert default_weights[equity_index] == pytest.approx(0.6)
        assert default_weights[bonds_index] == pytest.approx(0.4)

        resp_weights = market.weights("resp")
        assert resp_weights[equity_index] == pytest.approx(0.3)
        assert resp_weights[bonds_index] == pytest.approx(0.7)

    def test_weights_for_cash_raises(self, scenario) -> None:
        market = build_market_inputs(scenario.assumptions)
        with pytest.raises(ValueError):
            market.weights("cash")

    def test_default_kind_matches_the_scenario_default_allocation(self) -> None:
        assert DEFAULT_KIND == DEFAULT_ALLOCATION

    def test_generate_accepts_the_builder_s_output(self, scenario) -> None:
        market = build_market_inputs(scenario.assumptions)
        generate(1, 3, 5, market.annual_means, market.annual_covariance, n_persons=1)


class TestHouseholdCash:
    def test_a_second_person_s_cash_is_added_to_the_household_balance(self, scenario) -> None:
        person_a = scenario.household.persons[0]
        person_b = person_a.model_copy(
            update={
                "id": "b",
                "accounts": person_a.accounts.model_copy(
                    update={"cash": person_a.accounts.cash.model_copy(update={"balance": 5000.0})}
                ),
            }
        )
        new_household = scenario.household.model_copy(update={"persons": (person_a, person_b)})

        elections = scenario.policies[0].elections
        new_elections = elections.model_copy(
            update={
                "cpp_start_age_years": {**elections.cpp_start_age_years, "b": 65},
                "oas_start_age_years": {**elections.oas_start_age_years, "b": 65},
            }
        )
        new_policy = scenario.policies[0].model_copy(update={"elections": new_elections})
        new_scenario = scenario.model_copy(
            update={"household": new_household, "policies": (new_policy,)}
        )

        state = build_initial_state(new_scenario, n_paths=N_PATHS)
        assert (state.cash.balance == 25_000.0).all()


class TestLifAlreadyConverted:
    def test_a_person_already_converted_opens_the_lif_with_last_year_s_year(self, scenario) -> None:
        person_a = scenario.household.persons[0]
        new_accounts = person_a.accounts.model_copy(
            update={
                "lif": person_a.accounts.lif.model_copy(
                    update={"balance": 50_000.0, "jurisdiction": "ab"}
                )
            }
        )
        new_person = person_a.model_copy(update={"accounts": new_accounts})
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})
        new_scenario = scenario.model_copy(update={"household": new_household})

        state = build_initial_state(new_scenario, n_paths=N_PATHS)
        lif = state.persons[0].lif
        assert (lif.balance == 50_000.0).all()
        assert lif.jurisdiction == "ab"
        assert lif.opened_year == 2025

    def test_a_person_holding_only_a_lif_has_an_empty_lira_jurisdiction(self, scenario) -> None:
        person_a = scenario.household.persons[0]
        new_accounts = person_a.accounts.model_copy(
            update={
                "lira": LiraAccount(),
                "lif": person_a.accounts.lif.model_copy(
                    update={"balance": 50_000.0, "jurisdiction": "ab"}
                ),
            }
        )
        new_person = person_a.model_copy(update={"accounts": new_accounts})
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})
        new_scenario = scenario.model_copy(update={"household": new_household})

        state = build_initial_state(new_scenario, n_paths=N_PATHS)
        lira = state.persons[0].lira
        lif = state.persons[0].lif
        assert lira.jurisdiction == ""
        assert (lira.balance == 0.0).all()
        assert lif.jurisdiction == "ab"
        assert (lif.balance == 50_000.0).all()
        assert lif.opened_year == 2025


class TestPolicySelection:
    def test_no_policy_argument_and_one_policy_works(self, scenario) -> None:
        build_initial_state(scenario, n_paths=N_PATHS)

    def test_no_policy_argument_and_two_policies_raises(self, scenario) -> None:
        second = scenario.policies[0].model_copy(update={"name": "second"})
        two_policies = scenario.model_copy(update={"policies": (scenario.policies[0], second)})

        with pytest.raises(ValueError, match="second"):
            build_initial_state(two_policies, n_paths=N_PATHS)

    def test_naming_one_of_two_policies_works(self, scenario) -> None:
        second = scenario.policies[0].model_copy(update={"name": "second"})
        two_policies = scenario.model_copy(update={"policies": (scenario.policies[0], second)})

        state = build_initial_state(two_policies, n_paths=N_PATHS, policy=second)
        assert walk(state, N_PATHS) > 0


class TestBuildElections:
    def test_matches_build_initial_state_on_the_example(self, scenario) -> None:
        chosen = scenario.policies[0]
        elections = build_elections(scenario.household, chosen)
        state = build_initial_state(scenario, n_paths=N_PATHS)
        assert elections == state.elections

    def test_both_persons_already_in_pay_gives_no_start_ages(self) -> None:
        couple = load_scenario(COUPLE)
        chosen = couple.policies[0]
        elections = build_elections(couple.household, chosen)
        assert elections.cpp_start_age_months == (None, None)
        assert elections.oas_start_age_months == (None, None)


class TestNPaths:
    def test_zero_n_paths_is_rejected(self, scenario) -> None:
        with pytest.raises(ValueError, match="n_paths"):
            build_initial_state(scenario, n_paths=0)

    def test_negative_n_paths_is_rejected(self, scenario) -> None:
        with pytest.raises(ValueError, match="n_paths"):
            build_initial_state(scenario, n_paths=-1)


class TestMonthOffset:
    def test_january_of_the_base_year_is_zero(self) -> None:
        assert _month_offset(2026, 2026, 1) == 0

    def test_a_later_month_counts_forward(self) -> None:
        assert _month_offset(2026, 2031, 4) == 63

    def test_rejects_a_month_outside_1_to_12(self) -> None:
        with pytest.raises(ValueError, match=r"1\.\.12"):
            _month_offset(2026, 2026, 13)

        with pytest.raises(ValueError, match=r"1\.\.12"):
            _month_offset(2026, 2026, 0)

    def test_a_date_before_the_base_year_is_a_negative_offset_not_an_error(self) -> None:
        assert _month_offset(2026, 2025, 12) == -1
        assert _month_offset(2026, 2020, 1) == -72


class TestAlreadyInProgress:
    """Section B: a negative month index means 'already under way', not an error."""

    def test_employment_band_starting_before_the_run(self, scenario) -> None:
        person_a = scenario.household.persons[0]
        earlier_band = person_a.employment[0].model_copy(update={"from_year": 2015})
        new_person = person_a.model_copy(update={"employment": (earlier_band,)})
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})
        new_scenario = scenario.model_copy(update={"household": new_household})

        state = build_initial_state(new_scenario, n_paths=N_PATHS)
        band = state.persons[0].employment[0]
        assert band.from_month_index == _month_offset(2026, 2015, 1)
        assert band.from_month_index < 0

    def test_a_single_year_employment_band_is_twelve_months(self, scenario) -> None:
        person_a = scenario.household.persons[0]
        one_year_band = person_a.employment[0].model_copy(
            update={"from_year": 2026, "to_year": 2026}
        )
        new_person = person_a.model_copy(update={"employment": (one_year_band,)})
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})
        new_scenario = scenario.model_copy(update={"household": new_household})

        state = build_initial_state(new_scenario, n_paths=N_PATHS)
        band = state.persons[0].employment[0]
        assert (band.from_month_index, band.to_month_index) == (0, 11)
        assert band.to_month_index - band.from_month_index + 1 == 12

    def test_a_wholly_past_employment_band_is_carried_not_dropped(self, scenario) -> None:
        person_a = scenario.household.persons[0]
        past_band = person_a.employment[0].model_copy(update={"from_year": 2010, "to_year": 2015})
        new_person = person_a.model_copy(update={"employment": (past_band,)})
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})
        new_scenario = scenario.model_copy(update={"household": new_household})

        state = build_initial_state(new_scenario, n_paths=N_PATHS)
        band = state.persons[0].employment[0]
        assert band.from_month_index == _month_offset(2026, 2010, 1)
        assert band.to_month_index == _month_offset(2026, 2015, 12)
        assert band.from_month_index < 0
        assert band.to_month_index < 0

    def test_a_bridge_that_pays_has_its_end_computed_from_the_birth_date(self, scenario) -> None:
        """The only birth-date arithmetic in the builder, and the only test that runs it.

        Every other scenario here carries ``bridge_annual: 0``, so
        ``bridge_end_month_index`` is ``None`` and ``_build_pension``'s age
        branch executes nowhere — an error in the age-to-index conversion
        would ship unnoticed. The example person reaches 65 in 2031-03, so
        starting the pension in that same month is a legal one-month bridge;
        ``tests/scenario/test_schema.py`` carries the matching acceptance
        case proving a real YAML file can say this.
        """
        person_a = scenario.household.persons[0]
        bridged = person_a.db_pensions[0].model_copy(
            update={"bridge_annual": 8000.0, "start_month": 3}
        )
        new_person = person_a.model_copy(update={"db_pensions": (bridged,)})
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})
        new_scenario = scenario.model_copy(update={"household": new_household})

        pension = build_initial_state(new_scenario, n_paths=N_PATHS).persons[0].pensions[0]

        # Born 1966-03, bridging to 65: the bridge's last month is 2031-03.
        assert pension.bridge_end_month_index == _month_offset(2026, 2031, 3)
        assert pension.bridge_end_month_index == 62
        assert pension.start_month_index == pension.bridge_end_month_index, (
            "A bridge ending the month the pension starts is the boundary the "
            "schema accepts; it pays for exactly that one month."
        )
        assert pension.bridge_monthly[0] == pytest.approx(8000.0 / 12.0)

    def test_pension_already_in_payment(self, scenario) -> None:
        person_a = scenario.household.persons[0]
        earlier_pension = person_a.db_pensions[0].model_copy(
            update={"start_year": 2020, "start_month": 1}
        )
        new_person = person_a.model_copy(update={"db_pensions": (earlier_pension,)})
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})
        new_scenario = scenario.model_copy(update={"household": new_household})

        state = build_initial_state(new_scenario, n_paths=N_PATHS)
        pension = state.persons[0].pensions[0]
        assert pension.start_month_index == _month_offset(2026, 2020, 1)
        assert pension.start_month_index < 0

    def test_education_window_already_under_way(self, scenario) -> None:
        beneficiary = scenario.household.beneficiaries[0]
        earlier_education = beneficiary.education.model_copy(
            update={"start_year": 2024, "start_month": 9}
        )
        new_beneficiary = beneficiary.model_copy(update={"education": earlier_education})
        new_household = scenario.household.model_copy(update={"beneficiaries": (new_beneficiary,)})
        new_scenario = scenario.model_copy(update={"household": new_household})

        state = build_initial_state(new_scenario, n_paths=N_PATHS)
        resp = state.beneficiaries[0].resp
        assert resp.education_start_month_index == _month_offset(2026, 2024, 9)
        assert resp.education_start_month_index < 0

    def test_cpp_already_in_pay(self, scenario) -> None:
        person_a = scenario.household.persons[0]
        in_pay_cpp = person_a.cpp.model_copy(
            update={"contributory_history": None, "in_pay_monthly": 1200.0}
        )
        new_person = person_a.model_copy(update={"cpp": in_pay_cpp})
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})
        new_scenario = scenario.model_copy(update={"household": new_household})

        state = build_initial_state(new_scenario, n_paths=N_PATHS)
        cpp = state.persons[0].cpp
        assert cpp.start_age_months is None
        assert cpp.contributory_history is None
        assert cpp.in_pay_monthly.shape == (N_PATHS,)
        assert cpp.in_pay_monthly.dtype.name == "float64"
        assert not cpp.in_pay_monthly.flags.writeable
        assert (cpp.in_pay_monthly == 1200.0).all()
        assert state.elections.cpp_start_age_months == (None,)

    def test_oas_already_in_pay(self, scenario) -> None:
        person_a = scenario.household.persons[0]
        in_pay_oas = person_a.oas.model_copy(update={"in_pay_monthly": 800.0})
        new_person = person_a.model_copy(update={"oas": in_pay_oas})
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})
        new_scenario = scenario.model_copy(update={"household": new_household})

        state = build_initial_state(new_scenario, n_paths=N_PATHS)
        oas = state.persons[0].oas
        assert oas.start_age_months is None
        assert oas.contributory_history is None
        assert oas.in_pay_monthly.shape == (N_PATHS,)
        assert oas.in_pay_monthly.dtype.name == "float64"
        assert not oas.in_pay_monthly.flags.writeable
        assert (oas.in_pay_monthly == 800.0).all()
        assert state.elections.oas_start_age_months == (None,)

    def test_oas_already_in_pay_with_no_election_at_all(self, scenario) -> None:
        """Proves the builder never looks up an OAS election for a person already in pay."""
        person_a = scenario.household.persons[0]
        in_pay_oas = person_a.oas.model_copy(update={"in_pay_monthly": 800.0})
        new_person = person_a.model_copy(update={"oas": in_pay_oas})
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})

        elections = scenario.policies[0].elections
        new_elections = elections.model_copy(update={"oas_start_age_years": {}})
        new_policy = scenario.policies[0].model_copy(update={"elections": new_elections})
        new_scenario = scenario.model_copy(
            update={"household": new_household, "policies": (new_policy,)}
        )

        state = build_initial_state(new_scenario, n_paths=N_PATHS)
        oas = state.persons[0].oas
        assert oas.start_age_months is None
        assert (oas.in_pay_monthly == 800.0).all()
        assert state.elections.oas_start_age_months == (None,)

    def test_oas_election_looked_up_only_for_the_person_not_yet_in_pay(self, scenario) -> None:
        """Two persons, only one in pay: the builder must key the election by id."""
        person_a = scenario.household.persons[0]
        in_pay_oas = person_a.oas.model_copy(update={"in_pay_monthly": 800.0})
        person_a = person_a.model_copy(update={"oas": in_pay_oas})
        person_b = person_a.model_copy(update={"id": "b", "oas": OasEntitlement()})
        new_household = scenario.household.model_copy(update={"persons": (person_a, person_b)})

        elections = scenario.policies[0].elections
        new_elections = elections.model_copy(
            update={
                "cpp_start_age_years": {**elections.cpp_start_age_years, "b": 65},
                "oas_start_age_years": {"b": 65},
            }
        )
        new_policy = scenario.policies[0].model_copy(update={"elections": new_elections})
        new_scenario = scenario.model_copy(
            update={"household": new_household, "policies": (new_policy,)}
        )

        state = build_initial_state(new_scenario, n_paths=N_PATHS)
        assert state.elections.oas_start_age_months == (None, 780)
        assert state.persons[0].oas.start_age_months is None
        assert state.persons[1].oas.start_age_months == 780
        assert state.persons[1].oas.in_pay_monthly is None


class TestSpendingMonthlySelection:
    def test_start_year_past_the_second_band(self, scenario) -> None:
        new_scenario = scenario.model_copy(update={"start_year": 2033})
        state = build_initial_state(new_scenario, n_paths=N_PATHS)
        assert state.spending_monthly == pytest.approx(65_000.0 / 12.0)

    def test_a_single_band_strictly_before_start_year(self, scenario) -> None:
        new_spending = scenario.spending.model_copy(
            update={"schedule": (SpendingBand(from_year=2000, annual=12_000.0),)}
        )
        new_scenario = scenario.model_copy(update={"spending": new_spending})
        state = build_initial_state(new_scenario, n_paths=N_PATHS)
        assert state.spending_monthly == pytest.approx(1_000.0)


class TestBuildDraws:
    """``build_draws`` and ``build_deterministic_draws``.

    Does not exercise ``draw_deaths`` — see ``TestDrawDeaths`` below, which
    shares this module's ``mortality`` and ``two_person_scenario`` fixtures.
    """

    def test_build_draws_month_count_is_the_household_maximum(
        self, two_person_scenario, mortality: ParamSet
    ) -> None:
        market = build_market_inputs(two_person_scenario.assumptions)
        persons = two_person_scenario.household.persons

        months_a = months_to_terminal(
            persons[0].birth_year,
            persons[0].birth_month,
            persons[0].sex,
            two_person_scenario.start_year,
            mortality,
        )
        months_b = months_to_terminal(
            persons[1].birth_year,
            persons[1].birth_month,
            persons[1].sex,
            two_person_scenario.start_year,
            mortality,
        )
        # The fixture must actually make the two differ, and the younger
        # person (b) must be the larger one, or this test would pass against
        # a first-person implementation by coincidence.
        assert months_b > months_a

        draws = build_draws(two_person_scenario, market, n_paths=4, mortality=mortality)

        assert draws.n_months == months_b
        assert draws.real_returns.shape == (months_b, len(market.asset_class_names), 4)

    def test_build_deterministic_draws_is_one_path_same_month_count(
        self, two_person_scenario, mortality: ParamSet
    ) -> None:
        market = build_market_inputs(two_person_scenario.assumptions)
        stochastic = build_draws(two_person_scenario, market, n_paths=4, mortality=mortality)
        det = build_deterministic_draws(two_person_scenario, market, mortality)

        assert det.n_paths == 1
        assert det.n_months == stochastic.n_months


class TestDrawDeaths:
    #: Large enough that "every path" checks are meaningful, small enough to
    #: run quickly against the ~250-month synthetic curve the module-level
    #: ``mortality`` fixture builds.
    N_PATHS = 3000

    @pytest.fixture
    def opening_state_and_draws(self, scenario, mortality: ParamSet):
        market = build_market_inputs(scenario.assumptions)
        state = build_initial_state(scenario, n_paths=self.N_PATHS)
        draws = build_draws(scenario, market, n_paths=self.N_PATHS, mortality=mortality)
        return state, draws

    def test_every_path_gets_a_death_month_in_range_no_sentinel_left(
        self, opening_state_and_draws, mortality: ParamSet
    ) -> None:
        state, draws = opening_state_and_draws
        result = draw_deaths(state, draws, mortality)

        for person in result.persons:
            assert (person.death_month_index != DEATH_NOT_DRAWN).all()
            assert (person.death_month_index >= 1).all()
            assert (person.death_month_index <= draws.n_months - 1).all()
            # At month zero, alive == (death_month_index > 0), which is true whenever
            # death_month_index >= 1 -- already asserted above -- so the real content
            # left to check is simply that everyone is alive at the opening state.
            assert person.alive.all()

    def test_the_same_draws_give_identical_death_months(
        self, opening_state_and_draws, mortality: ParamSet
    ) -> None:
        state, draws = opening_state_and_draws
        first = draw_deaths(state, draws, mortality)
        second = draw_deaths(state, draws, mortality)

        for person_first, person_second in zip(first.persons, second.persons, strict=True):
            assert (person_first.death_month_index == person_second.death_month_index).all()
            assert (person_first.alive == person_second.alive).all()

    def test_rejects_a_state_not_at_month_index_zero(
        self, opening_state_and_draws, mortality: ParamSet
    ) -> None:
        state, draws = opening_state_and_draws
        later_state = updated(state, month_index=5)

        with pytest.raises(ValueError, match="month_index"):
            draw_deaths(later_state, draws, mortality)

    def test_rejects_a_path_count_mismatch(self, scenario, mortality: ParamSet) -> None:
        market = build_market_inputs(scenario.assumptions)
        state = build_initial_state(scenario, n_paths=5)
        mismatched_draws = build_draws(scenario, market, n_paths=3, mortality=mortality)

        with pytest.raises(ValueError, match="n_paths"):
            draw_deaths(state, mismatched_draws, mortality)

    def test_rejects_a_person_count_mismatch(self, scenario, mortality: ParamSet) -> None:
        market = build_market_inputs(scenario.assumptions)
        state = build_initial_state(scenario, n_paths=4)
        two_person_draws = generate(
            seed=1,
            n_months=10,
            n_paths=4,
            annual_means=market.annual_means,
            annual_covariance=market.annual_covariance,
            n_persons=2,
        )

        # "row" rather than "person": both the person-count message and the
        # survival-curve-length message below begin "person '...':", so
        # matching on "person" would pass here even if the two count checks
        # ran after the loop instead of before it. Only the person-count
        # message says "row(s)".
        with pytest.raises(ValueError, match="row"):
            draw_deaths(state, two_person_draws, mortality)

    def test_rejects_a_survival_curve_longer_than_draws_n_months(
        self, scenario, mortality: ParamSet
    ) -> None:
        market = build_market_inputs(scenario.assumptions)
        state = build_initial_state(scenario, n_paths=4)
        short_draws = generate(
            seed=1,
            n_months=10,
            n_paths=4,
            annual_means=market.annual_means,
            annual_covariance=market.annual_covariance,
            n_persons=1,
        )

        with pytest.raises(ValueError, match="longer than"):
            draw_deaths(state, short_draws, mortality)

    def test_each_person_s_death_month_index_comes_from_their_own_row(
        self, two_person_scenario, mortality: ParamSet
    ) -> None:
        """The alignment brief-51 asked for: person ``i`` gets ``draws.mortality[i]``.

        ``scenarios/example.yaml`` has one person, so every other test in this class
        would pass an implementation that wrote ``draws.mortality[0]`` for every
        person. ``two_person_scenario`` differs in both sex and birth year, so
        swapping the two rows changes the answer.
        """
        market = build_market_inputs(two_person_scenario.assumptions)
        state = build_initial_state(two_person_scenario, n_paths=self.N_PATHS)
        draws = build_draws(two_person_scenario, market, n_paths=self.N_PATHS, mortality=mortality)

        result = draw_deaths(state, draws, mortality)

        persons = two_person_scenario.household.persons
        for index, person_state in enumerate(result.persons):
            curve = survival_curve(
                persons[index].birth_year,
                persons[index].birth_month,
                persons[index].sex,
                two_person_scenario.start_year,
                mortality,
            )
            expected = death_month_index(draws.mortality[index], curve)
            np.testing.assert_array_equal(person_state.death_month_index, expected)

            assert person_state.death_month_index.dtype == np.int64
            assert person_state.death_month_index.shape == (self.N_PATHS,)
            assert not person_state.death_month_index.flags.writeable

            assert person_state.alive.dtype == np.bool_
            assert person_state.alive.shape == (self.N_PATHS,)
            assert not person_state.alive.flags.writeable


class TestWithDeathMonths:
    """``with_death_months``: fixing a drawn death month for a hand check (#63)."""

    N_PATHS = 4

    @pytest.fixture
    def drawn_two_person_state(self, two_person_scenario, mortality: ParamSet):
        market = build_market_inputs(two_person_scenario.assumptions)
        state = build_initial_state(two_person_scenario, n_paths=self.N_PATHS)
        draws = build_draws(two_person_scenario, market, n_paths=self.N_PATHS, mortality=mortality)
        return draw_deaths(state, draws, mortality)

    def test_with_death_months_sets_month_and_alive(self, drawn_two_person_state) -> None:
        original = drawn_two_person_state
        result = with_death_months(original, (5, None))

        person0, person1 = result.persons
        np.testing.assert_array_equal(person0.death_month_index, np.full(self.N_PATHS, 5))
        assert person0.alive.all()  # death_month_index (5) > month_index (0)

        np.testing.assert_array_equal(
            person1.death_month_index, original.persons[1].death_month_index
        )
        np.testing.assert_array_equal(person1.alive, original.persons[1].alive)

    def test_rejects_a_state_not_at_month_index_zero(self, drawn_two_person_state) -> None:
        later_state = updated(drawn_two_person_state, month_index=1)
        with pytest.raises(ValueError, match="requires the opening state"):
            with_death_months(later_state, (5, None))

    def test_rejects_the_wrong_number_of_entries(self, drawn_two_person_state) -> None:
        with pytest.raises(ValueError, match="entries"):
            with_death_months(drawn_two_person_state, (5,))

    def test_rejects_an_undrawn_state(self, two_person_scenario) -> None:
        undrawn = build_initial_state(two_person_scenario, n_paths=self.N_PATHS)
        with pytest.raises(ValueError, match="DEATH_NOT_DRAWN"):
            with_death_months(undrawn, (5, None))

    def test_rejects_a_bool_entry(self, drawn_two_person_state) -> None:
        with pytest.raises(ValueError, match="None nor a plain int"):
            with_death_months(drawn_two_person_state, (True, None))

    def test_rejects_a_float_entry(self, drawn_two_person_state) -> None:
        with pytest.raises(ValueError, match="None nor a plain int"):
            with_death_months(drawn_two_person_state, (27.0, None))

    def test_rejects_a_numpy_integer_entry(self, drawn_two_person_state) -> None:
        with pytest.raises(ValueError, match="None nor a plain int"):
            with_death_months(drawn_two_person_state, (np.int64(27), None))

    def test_rejects_an_int_subclass_entry(self, drawn_two_person_state) -> None:
        with pytest.raises(ValueError, match="None nor a plain int"):
            with_death_months(drawn_two_person_state, (_Month.APRIL, None))

    def test_rejects_zero(self, drawn_two_person_state) -> None:
        with pytest.raises(ValueError, match="less than 1"):
            with_death_months(drawn_two_person_state, (0, None))

    def test_rejects_a_negative_month(self, drawn_two_person_state) -> None:
        with pytest.raises(ValueError, match="less than 1"):
            with_death_months(drawn_two_person_state, (-1, None))

    def test_type_is_checked_before_value_across_entries(self, drawn_two_person_state) -> None:
        """Every entry's type is checked before any entry's value: ``(0, 27.0)`` raises the
        type error for the second entry, not "less than 1" for the first.
        """
        with pytest.raises(ValueError, match="None nor a plain int"):
            with_death_months(drawn_two_person_state, (0, 27.0))

    def test_rejects_the_undrawn_sentinel_as_a_month(self, drawn_two_person_state) -> None:
        with pytest.raises(ValueError, match="not below DEATH_NOT_DRAWN"):
            with_death_months(drawn_two_person_state, (DEATH_NOT_DRAWN, None))

    def test_rejects_a_month_past_int64(self, drawn_two_person_state) -> None:
        with pytest.raises(ValueError, match="not below DEATH_NOT_DRAWN"):
            with_death_months(drawn_two_person_state, (2**63, None))

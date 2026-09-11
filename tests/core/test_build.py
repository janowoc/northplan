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

from pathlib import Path

import numpy as np
import pytest

from engine.core.build import _month_offset, build_initial_state, build_market_inputs
from engine.core.state import DEATH_NOT_DRAWN, updated
from engine.mc.market import DEFAULT_KIND
from engine.mc.returns import generate
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

N_PATHS = 3

#: The example scenario currently builds exactly 53 arrays at n_paths=3.
#: Close to that count, not merely "not zero": at 30 this floor passed with
#: the entire RESP subtree gone (44) or the entire year-to-date ledger gone
#: (38) — a wrong tree, not an empty one, and the floor never noticed. Left
#: with room for a field or two, not a whole subtree.
MINIMUM_ARRAYS = 50


@pytest.fixture
def scenario():
    return load_scenario(EXAMPLE)


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

    def test_prior_year_net_income_is_broadcast_from_the_scenario(self, scenario) -> None:
        state = build_initial_state(scenario, n_paths=N_PATHS)
        assert (state.persons[0].prior_year_net_income == 92_000.0).all()

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

    def test_a_bridge_that_pays_has_its_end_computed_from_the_birth_date(
        self, scenario
    ) -> None:
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

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``engine.core.step``: the monthly step, exercised against the committed example.

Every expected value is computed by calling the engine's own parameter-reading
functions (``tfsa.room_accrued``, ``rrsp.room_accrued``, ``rrif.minimum_withdrawal``,
...), never a literal of a tax value. Variants of the example scenario are built with
``.model_copy``, never a hand-typed new scenario; any synthetic figure is marked as
synthetic in a comment.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from engine.accounts import lif, resp, rrif, rrsp, taxable, tfsa
from engine.core import step as step_module
from engine.core import timeline
from engine.core.build import (
    build_deterministic_draws,
    build_initial_state,
    build_market_inputs,
    draw_deaths,
)
from engine.core.indexation import nominal_carry_factor, real_year
from engine.core.state import updated
from engine.core.step import advance_month, advance_month_traced, close_year, open_year
from engine.mc.returns import RandomDraws
from engine.mc.simulate import run
from engine.params.loader import load_year
from engine.policy.base import Transfer
from engine.scenario import LifAccount, load_scenario
from engine.tax import federal, withholding

from .policies import DoNothingPolicy, RecordingPolicy, ScriptedPolicy

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


def _sum_by_kind(items):
    total = 0.0
    for item in items:
        total = total + item.total()
    return total


def _sum_arrays(items):
    total = 0.0
    for item in items:
        total = total + item
    return total


def _assert_identities_hold(record) -> None:
    ctx = record.context
    lhs1 = ctx.cash_after_flows
    rhs1 = (
        ctx.cash_opening
        + _sum_arrays(inflow.to_cash for inflow in ctx.inflows)
        + _sum_arrays(ctx.education_draws)
        - _sum_arrays(ctx.payroll_withholding)
        - ctx.spending
        - _sum_arrays(ctx.education_costs)
        - _sum_arrays(ctx.tax_settlement)
        + _sum_by_kind(ctx.forced_withdrawals)
        - _sum_arrays(ctx.forced_withholding)
    )
    assert np.all(np.abs(lhs1 - rhs1) < 0.005), (lhs1, rhs1)

    rhs2 = (
        ctx.cash_after_flows
        + _sum_by_kind(record.withdrawals)
        - _sum_arrays(record.withdrawal_withholding)
        - _sum_by_kind(record.contributions)
        - _sum_arrays(record.resp_contributions)
        + _sum_arrays(record.wind_up_to_cash)
        + _sum_by_kind(record.floor_withdrawals)
        + record.depletion_deficit
    )
    assert np.all(np.abs(record.cash_close - rhs2) < 0.005), (record.cash_close, rhs2)


def _w(state) -> np.ndarray:
    """Total household wealth, RESP excluded: cash plus every account balance.

    Takes a plain ``HouseholdState`` directly -- a ``RecordingPolicy`` capture, or the
    original opening state -- and needs no reconstruction: every field this reads is
    already exactly what it names.
    """
    total = state.cash.balance.copy()
    for person in state.persons:
        total = (
            total
            + person.rrsp.balance
            + person.rrif.balance
            + person.lira.balance
            + person.lif.balance
            + person.tfsa.balance
            + person.taxable.balance
        )
    return total


class TestCashIdentityDoNothing:
    """Requirement 1: cash identities hold every month, on the do-nothing policy."""

    def test_identities_hold_and_floor_fires_and_cash_chains(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        recorder = RecordingPolicy(DoNothingPolicy(opening_state.elections, withdrawal_order))
        result = run(opening_state, recorder, draws, market, real_params, trace_path=0)

        assert len(result.trace) == draws.n_months
        assert len(recorder.calls) == draws.n_months

        floor_fired = False
        for record, (seen_state, seen_context) in zip(
            result.trace, recorder.calls, strict=True
        ):
            _assert_identities_hold(record)
            # Requirement 7: the state a policy is shown really does carry cash equal to
            # context.cash_after_flows -- the two are supposed to be the same number by
            # construction (Policy.decide's own docstring says so), not merely close.
            np.testing.assert_array_equal(seen_state.cash.balance, seen_context.cash_after_flows)
            if np.any(_sum_by_kind(record.floor_withdrawals) > 0):
                floor_fired = True

        for m in range(draws.n_months - 1):
            np.testing.assert_allclose(
                result.trace[m + 1].context.cash_opening, result.trace[m].cash_close, atol=0.005
            )

        assert floor_fired, "the cash floor never fired in this run; the test would be vacuous"


class TestCashIdentityScripted:
    """Requirement 2: identities still hold once transfers actually move money."""

    def test_identities_hold_with_a_scripted_policy(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        n_paths = draws.n_paths
        script = {
            3: (
                Transfer(
                    person_index=0,
                    from_kind="rrsp",
                    to_kind="cash",
                    amount=np.full(n_paths, 2_000.0),
                ),
                Transfer(
                    person_index=0,
                    from_kind="tfsa",
                    to_kind="cash",
                    amount=np.full(n_paths, 500.0),
                ),
            ),
            4: (
                Transfer(
                    person_index=0,
                    from_kind="cash",
                    to_kind="tfsa",
                    amount=np.full(n_paths, 300.0),
                ),
                Transfer(
                    person_index=0,
                    from_kind="cash",
                    to_kind="taxable",
                    amount=np.full(n_paths, 300.0),
                ),
                Transfer(
                    person_index=0,
                    from_kind="cash",
                    to_kind="resp",
                    amount=np.full(n_paths, 100.0),
                ),
            ),
        }
        policy = ScriptedPolicy(opening_state.elections, withdrawal_order, script)
        result = run(opening_state, policy, draws, market, real_params, trace_path=0)

        assert len(result.trace) == draws.n_months
        for record in result.trace:
            _assert_identities_hold(record)

        withholding_seen = any(np.any(r.withdrawal_withholding[0] > 0) for r in result.trace)
        contributions_seen = any(np.any(_sum_by_kind(r.contributions) > 0) for r in result.trace)
        resp_contributions_seen = any(
            np.any(_sum_arrays(r.resp_contributions) > 0) for r in result.trace
        )
        assert withholding_seen
        assert contributions_seen
        assert resp_contributions_seen


class TestAnnualGrantTiming:
    """Requirement 3: no grant at month zero, full grant at month twelve."""

    def test_no_grant_at_month_zero_full_grant_at_month_twelve(
        self, scenario, opening_state, draws, market, real_params, withdrawal_order
    ):
        recorder = RecordingPolicy(DoNothingPolicy(opening_state.elections, withdrawal_order))
        run(opening_state, recorder, draws, market, real_params)

        scenario_accounts = scenario.household.persons[0].accounts
        state0 = recorder.calls[0][0]
        assert state0.persons[0].tfsa.room[0] == pytest.approx(scenario_accounts.tfsa.room)
        assert state0.persons[0].rrsp.room[0] == pytest.approx(scenario_accounts.rrsp.room)
        assert state0.beneficiaries[0].resp.grant_room[0] == pytest.approx(
            scenario.household.beneficiaries[0].resp.grant_room_carried
        )

        state11 = recorder.calls[11][0]
        state12 = recorder.calls[12][0]
        carry = nominal_carry_factor(scenario.assumptions.inflation)

        person11 = state11.persons[0]
        person12 = state12.persons[0]
        age_end_tfsa = timeline.age_at_end_of_year(person11.birth_year, person11.birth_month, 2027)
        expected_tfsa_room = (
            person11.tfsa.room * carry
            + person11.tfsa.withdrawn_this_year * carry
            + tfsa.room_accrued(age_end_tfsa, real_params.tfsa, 12)
        )
        np.testing.assert_allclose(person12.tfsa.room, expected_tfsa_room)

        expected_rrsp_room = person11.rrsp.room * carry + rrsp.room_accrued(
            person11.income.employment, real_params.rrif, 12
        )
        np.testing.assert_allclose(person12.rrsp.room, expected_rrsp_room)

        ben11 = state11.beneficiaries[0]
        ben12 = state12.beneficiaries[0]
        age_end_resp = timeline.age_at_end_of_year(ben11.birth_year, ben11.birth_month, 2027)
        expected_grant_room = ben11.resp.grant_room * carry + resp.grant_room_accrued(
            age_end_resp, real_params.resp, 12
        )
        np.testing.assert_allclose(ben12.resp.grant_room, expected_grant_room)


class TestTfsaRestorationLag:
    """Requirement 4: a TFSA withdrawal restores room in the following January, not sooner."""

    def test_room_is_unchanged_until_the_following_january(
        self, scenario, opening_state, draws, market, real_params, withdrawal_order
    ):
        n_paths = draws.n_paths
        withdrawal_amount = 1_000.0
        script = {
            2: (
                Transfer(
                    person_index=0,
                    from_kind="tfsa",
                    to_kind="cash",
                    amount=np.full(n_paths, withdrawal_amount),
                ),
            )
        }
        policy = ScriptedPolicy(opening_state.elections, withdrawal_order, script)
        recorder = RecordingPolicy(policy)
        run(opening_state, recorder, draws, market, real_params)

        opening_room = opening_state.persons[0].tfsa.room
        for m in range(3, 12):
            np.testing.assert_allclose(recorder.calls[m][0].persons[0].tfsa.room, opening_room)

        state11 = recorder.calls[11][0]
        state12 = recorder.calls[12][0]
        person11 = state11.persons[0]
        np.testing.assert_allclose(person11.tfsa.withdrawn_this_year, withdrawal_amount)

        carry = nominal_carry_factor(scenario.assumptions.inflation)
        age_end = timeline.age_at_end_of_year(person11.birth_year, person11.birth_month, 2027)
        expected_room_12 = (
            person11.tfsa.room * carry
            + person11.tfsa.withdrawn_this_year * carry
            + tfsa.room_accrued(age_end, real_params.tfsa, 12)
        )
        np.testing.assert_allclose(state12.persons[0].tfsa.room, expected_room_12)

        state3 = recorder.calls[3][0]
        assert not np.allclose(state3.persons[0].tfsa.room, expected_room_12)


class TestRrifMinimumOutByDecember:
    """Requirement 5: the RRIF minimum is zero all year and forced out by December."""

    def test_forced_out_only_in_december(self, scenario, market, mortality, withdrawal_order):
        rrif_balance = 300_000.0  # SYNTHETIC: exercises the minimum, not a real figure.
        person_a = scenario.household.persons[0]
        new_accounts = person_a.accounts.model_copy(
            update={"rrif": person_a.accounts.rrif.model_copy(update={"balance": rrif_balance})}
        )
        new_person = person_a.model_copy(update={"accounts": new_accounts})
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})
        new_scenario = scenario.model_copy(update={"household": new_household})

        draws = build_deterministic_draws(new_scenario, market, mortality)
        state = build_initial_state(new_scenario, n_paths=1)
        state = draw_deaths(state, draws, mortality)
        real_params = real_year(load_year(new_scenario.start_year), new_scenario.assumptions.inflation)

        policy = DoNothingPolicy(state.elections, withdrawal_order)
        recorder = RecordingPolicy(policy)
        run(state, recorder, draws, market, real_params)

        for m in range(11):
            assert recorder.calls[m][0].persons[0].rrif.withdrawn_ytd[0] == pytest.approx(0.0)

        age_start = timeline.age_at_start_of_year(new_person.birth_year, new_person.birth_month, 2026)
        expected_minimum = rrif.minimum_withdrawal(
            np.array([rrif_balance]), age_start, 2025, 2026, real_params.rrif
        )
        assert expected_minimum[0] > 0.0

        state11 = recorder.calls[11][0]
        np.testing.assert_allclose(state11.persons[0].rrif.withdrawn_ytd, expected_minimum)
        np.testing.assert_allclose(state11.persons[0].rrif.annual_minimum, expected_minimum)


class TestLif:
    """Requirement 5: the LIF's own minimum, maximum, and how the step respects both."""

    # SYNTHETIC balance: within the AB maximum table's age range for person a (age ~60 at
    # the example's start year), and small enough that "requesting more than the maximum
    # every month" and "nothing else to draw on" are both easy to reason about.
    LIF_BALANCE = 200_000.0

    def _lif_scenario(self, scenario):
        person_a = scenario.household.persons[0]
        new_person = person_a.model_copy(
            update={
                "accounts": person_a.accounts.model_copy(
                    update={"lif": LifAccount(balance=self.LIF_BALANCE, jurisdiction="ab")}
                )
            }
        )
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})
        return scenario.model_copy(update={"household": new_household}), new_person

    def test_open_year_fixes_minimum_and_maximum(self, scenario, real_params):
        new_scenario, new_person = self._lif_scenario(scenario)
        state = build_initial_state(new_scenario, n_paths=1)
        opened = open_year(state, real_params)

        age_start = timeline.age_at_start_of_year(
            new_person.birth_year, new_person.birth_month, 2026
        )
        expected_minimum = rrif.minimum_withdrawal(
            np.array([self.LIF_BALANCE]), age_start, 2025, 2026, real_params.rrif
        )
        expected_maximum = lif.maximum_withdrawal(
            np.array([self.LIF_BALANCE]), age_start, real_params.jurisdiction("ab")
        )
        np.testing.assert_allclose(opened.persons[0].lif.annual_minimum, expected_minimum)
        np.testing.assert_allclose(opened.persons[0].lif.annual_maximum, expected_maximum)

    def test_a_scripted_over_request_withdraws_exactly_the_maximum_and_meets_the_minimum(
        self, scenario, mortality, market, real_params, withdrawal_order
    ):
        new_scenario, new_person = self._lif_scenario(scenario)
        draws = build_deterministic_draws(new_scenario, market, mortality)
        state = build_initial_state(new_scenario, n_paths=1)
        state = draw_deaths(state, draws, mortality)

        huge = np.array([1e9])
        script = {
            m: (Transfer(person_index=0, from_kind="lif", to_kind="cash", amount=huge),)
            for m in range(12)
        }
        policy = ScriptedPolicy(state.elections, withdrawal_order, script)
        result = run(state, policy, draws, market, real_params, trace_path=0)

        age_start = timeline.age_at_start_of_year(
            new_person.birth_year, new_person.birth_month, 2026
        )
        expected_maximum = lif.maximum_withdrawal(
            np.array([self.LIF_BALANCE]), age_start, real_params.jurisdiction("ab")
        )
        expected_minimum = rrif.minimum_withdrawal(
            np.array([self.LIF_BALANCE]), age_start, 2025, 2026, real_params.rrif
        )

        # The year's total is the sum of every month's LIF movement across all three
        # phases that can touch it (forced, transferred, floor); none of the three by
        # itself is the year's total.
        total_withdrawn_2026 = np.zeros(1)
        for m in range(12):
            record = result.trace[m]
            total_withdrawn_2026 = (
                total_withdrawn_2026
                + record.context.forced_withdrawals[0].lif
                + record.withdrawals[0].lif
                + record.floor_withdrawals[0].lif
            )
        np.testing.assert_allclose(total_withdrawn_2026, expected_maximum)
        assert total_withdrawn_2026[0] >= expected_minimum[0] - 1e-6

    def test_the_cash_floor_never_exceeds_the_lif_maximum(
        self, scenario, mortality, market, real_params, withdrawal_order
    ):
        new_scenario, new_person = self._lif_scenario(scenario)
        # Strip every other asset and all income so the floor has nothing else to draw on
        # and must lean on the LIF every month it fires (requirement 7's pattern, applied
        # to a household that still holds a LIF).
        accounts = new_person.accounts.model_copy(
            update={
                "cash": new_person.accounts.cash.model_copy(update={"balance": 0.0}),
                "rrsp": new_person.accounts.rrsp.model_copy(update={"balance": 0.0, "room": 0.0}),
                "rrif": new_person.accounts.rrif.model_copy(update={"balance": 0.0}),
                "tfsa": new_person.accounts.tfsa.model_copy(update={"balance": 0.0, "room": 0.0}),
                "taxable": new_person.accounts.taxable.model_copy(
                    update={"balance": 0.0, "acb": 0.0}
                ),
                "lira": new_person.accounts.lira.model_copy(update={"balance": 0.0}),
            }
        )
        broke_person = new_person.model_copy(
            update={
                "accounts": accounts,
                "employment": (),
                "db_pensions": (),
                "cpp": new_person.cpp.model_copy(update={"contributory_history": 0.0}),
            }
        )
        zero_resp = new_scenario.household.beneficiaries[0].resp.model_copy(
            update={"contributions": 0.0, "grants": 0.0, "income": 0.0, "grant_room_carried": 0.0}
        )
        broke_beneficiary = new_scenario.household.beneficiaries[0].model_copy(
            update={"resp": zero_resp}
        )
        broke_household = new_scenario.household.model_copy(
            update={"persons": (broke_person,), "beneficiaries": (broke_beneficiary,)}
        )
        broke_scenario = new_scenario.model_copy(update={"household": broke_household})

        draws = build_deterministic_draws(broke_scenario, market, mortality)
        state = build_initial_state(broke_scenario, n_paths=1)
        state = draw_deaths(state, draws, mortality)

        policy = DoNothingPolicy(state.elections, withdrawal_order)
        recorder = RecordingPolicy(policy)
        run(state, recorder, draws, market, real_params)

        floor_drew_on_the_lif = False
        for m in range(12):
            person_state = recorder.calls[m][0].persons[0]
            assert person_state.lif.withdrawn_ytd[0] <= person_state.lif.annual_maximum[0] + 1e-6
            if person_state.lif.withdrawn_ytd[0] > 0:
                floor_drew_on_the_lif = True
        assert floor_drew_on_the_lif, "the floor never touched the LIF; the test would be vacuous"

    def test_has_maximum_false_gives_an_infinite_maximum_capped_only_by_balance(
        self, scenario, mortality, market, real_params, withdrawal_order, monkeypatch
    ):
        monkeypatch.setattr(lif, "has_maximum", lambda _params: False)

        new_scenario, _new_person = self._lif_scenario(scenario)
        draws = build_deterministic_draws(new_scenario, market, mortality)
        state = build_initial_state(new_scenario, n_paths=1)
        state = draw_deaths(state, draws, mortality)

        opened = open_year(state, real_params)
        assert np.isinf(opened.persons[0].lif.annual_maximum[0])

        huge = np.array([1e9])
        script = {0: (Transfer(person_index=0, from_kind="lif", to_kind="cash", amount=huge),)}
        policy = ScriptedPolicy(state.elections, withdrawal_order, script)
        new_state = advance_month(state, draws.real_returns[0], policy, market, real_params)
        np.testing.assert_allclose(new_state.persons[0].lif.balance, [0.0], atol=1e-6)


class TestPensionCreditFill:
    """Requirement 6, and decision B: the fill nets out what is left of the LIF minimum.

    Three persons: ``young`` (below the eligibility age, never filled), ``old`` (at/above it,
    RRIF only), ``old_lif`` (at/above it, a RRIF *and* an AB LIF whose own minimum is below
    the target -- B's case).
    """

    def _build(self, scenario):
        person_a = scenario.household.persons[0]
        # SYNTHETIC balances: small enough that the ordinary annual minimum stays well below
        # the pension credit target, so the fill logic -- not the mandatory minimum -- is
        # what reaches the target; the LIF balance is small enough its own minimum is below
        # the target too, per decision B.
        young_rrif_balance = 50_000.0
        old_rrif_balance = 10_000.0
        old_lif_lif_balance = 5_000.0

        young = person_a.model_copy(
            update={
                "accounts": person_a.accounts.model_copy(
                    update={
                        "rrif": person_a.accounts.rrif.model_copy(
                            update={"balance": young_rrif_balance}
                        )
                    }
                )
            }
        )
        # SYNTHETIC birth date: old enough to be at/above the eligibility age at year end.
        # db_pensions=() keeps eligible pension income exactly zero before any RRIF/LIF
        # withdrawal, so the "spreading" arithmetic below has no other term to account for.
        old = person_a.model_copy(
            update={
                "id": "old",
                "birth_year": 1950,
                "birth_month": 1,
                "db_pensions": (),
                "accounts": person_a.accounts.model_copy(
                    update={
                        "rrif": person_a.accounts.rrif.model_copy(
                            update={"balance": old_rrif_balance}
                        )
                    }
                ),
            }
        )
        old_lif = person_a.model_copy(
            update={
                "id": "old_lif",
                "birth_year": 1950,
                "birth_month": 1,
                "db_pensions": (),
                "accounts": person_a.accounts.model_copy(
                    update={
                        "rrif": person_a.accounts.rrif.model_copy(
                            update={"balance": old_rrif_balance}
                        ),
                        "lif": LifAccount(balance=old_lif_lif_balance, jurisdiction="ab"),
                    }
                ),
            }
        )
        new_household = scenario.household.model_copy(update={"persons": (young, old, old_lif)})

        policy_spec = scenario.policies[0]
        elections = policy_spec.elections
        new_elections = elections.model_copy(
            update={
                "cpp_start_age_years": {
                    **elections.cpp_start_age_years,
                    "old": 65,
                    "old_lif": 65,
                },
                "oas_start_age_years": {
                    **elections.oas_start_age_years,
                    "old": 65,
                    "old_lif": 65,
                },
            }
        )
        new_policy = policy_spec.model_copy(update={"elections": new_elections})
        new_scenario = scenario.model_copy(
            update={"household": new_household, "policies": (new_policy,)}
        )
        assert new_policy.withdrawal.fill_pension_credit is True
        return new_scenario, young, old, old_lif

    def test_below_age_at_age_and_the_lif_aware_spreading_and_year_end_target(
        self, scenario, market, mortality, withdrawal_order, real_params
    ):
        new_scenario, young, old, old_lif = self._build(scenario)
        fill_age = real_params.federal.number("eligible_pension_income.rrif_minimum_age_years")

        draws = build_deterministic_draws(new_scenario, market, mortality)
        state = build_initial_state(new_scenario, n_paths=1)
        state = draw_deaths(state, draws, mortality)
        new_real_params = real_year(load_year(new_scenario.start_year), new_scenario.assumptions.inflation)

        age_end_young = timeline.age_at_end_of_year(young.birth_year, young.birth_month, 2026)
        age_end_old = timeline.age_at_end_of_year(old.birth_year, old.birth_month, 2026)
        age_end_old_lif = timeline.age_at_end_of_year(old_lif.birth_year, old_lif.birth_month, 2026)
        assert age_end_young < fill_age
        assert age_end_old >= fill_age
        assert age_end_old_lif >= fill_age

        policy = DoNothingPolicy(state.elections, withdrawal_order)
        recorder = RecordingPolicy(policy)
        run(state, recorder, draws, market, new_real_params)

        # Claim 1: below the age, no fill -- zero forced RRIF withdrawals through month 10
        # (December, month 11, forces the ordinary minimum regardless of age; that is not
        # tested here).
        for m in range(11):
            assert recorder.calls[m][0].persons[0].rrif.withdrawn_ytd[0] == pytest.approx(0.0)

        # Claim 2 (spreading): the month-0 forced RRIF withdrawal for old_lif equals
        # max(minimum_still_required, (target - eligible_before_fill - lif_min_remaining) / 12).
        target = max(
            new_real_params.federal.annual_amount("credits.pension_income_amount_annual", 0),
            new_real_params.province(new_scenario.household.province).annual_amount(
                "credits.pension_income_amount_annual", 0
            ),
        )
        state0, context0 = recorder.calls[0]
        old_lif_index = 2
        # eligible_before_fill is exactly this month's DB pension (there is none, and
        # nothing had been withdrawn yet this year), confirmed directly rather than assumed.
        assert context0.inflows[old_lif_index].db_pension[0] == pytest.approx(0.0)
        eligible_before_fill = np.array([0.0])
        months_remaining = 13 - state0.month
        rrif_floor_ordinary = rrif.minimum_still_required(
            state0.persons[old_lif_index].rrif.annual_minimum, np.array([0.0]), months_remaining
        )
        lif_min_remaining = np.clip(state0.persons[old_lif_index].lif.annual_minimum, 0, None)
        fill_month0 = (
            np.clip(target - eligible_before_fill - lif_min_remaining, 0, None) / months_remaining
        )
        expected_rrif_forced_month0 = np.maximum(rrif_floor_ordinary, fill_month0)
        np.testing.assert_allclose(
            context0.forced_withdrawals[old_lif_index].rrif, expected_rrif_forced_month0
        )

        # Claim 3 (B's case, year end): eligible pension income reaches the target within
        # 0.005, and not target + the LIF's own minimum, which is what it would be without
        # netting the LIF minimum out of the fill.
        state11 = recorder.calls[11][0]
        lif_annual_minimum = state11.persons[old_lif_index].lif.annual_minimum[0]
        eligible_old_lif = federal.eligible_pension_income(
            state11.persons[old_lif_index].income, age_end_old_lif, new_real_params.federal
        )
        assert eligible_old_lif[0] == pytest.approx(target, abs=0.005)
        assert eligible_old_lif[0] != pytest.approx(target + lif_annual_minimum, abs=0.005)

        # "old" (RRIF only, no LIF) still reaches the target exactly as before.
        eligible_old = federal.eligible_pension_income(
            state11.persons[1].income, age_end_old, new_real_params.federal
        )
        assert eligible_old[0] == pytest.approx(target, abs=0.005)

        eligible_young = federal.eligible_pension_income(
            state11.persons[0].income, age_end_young, new_real_params.federal
        )
        assert eligible_young[0] == pytest.approx(0.0)


def _broke_scenario(scenario, *, education_active: bool):
    """A copy of the example household with no income and no assets (requirements 7-8)."""
    person_a = scenario.household.persons[0]
    zero_accounts = person_a.accounts.model_copy(
        update={
            "cash": person_a.accounts.cash.model_copy(update={"balance": 0.0}),
            "rrsp": person_a.accounts.rrsp.model_copy(update={"balance": 0.0, "room": 0.0}),
            "rrif": person_a.accounts.rrif.model_copy(update={"balance": 0.0}),
            "tfsa": person_a.accounts.tfsa.model_copy(update={"balance": 0.0, "room": 0.0}),
            "taxable": person_a.accounts.taxable.model_copy(update={"balance": 0.0, "acb": 0.0}),
            "lira": person_a.accounts.lira.model_copy(update={"balance": 0.0}),
        }
    )
    no_income_cpp = person_a.cpp.model_copy(update={"contributory_history": 0.0})
    broke_person = person_a.model_copy(
        update={
            "accounts": zero_accounts,
            "employment": (),
            "db_pensions": (),
            "cpp": no_income_cpp,
        }
    )
    zero_resp = scenario.household.beneficiaries[0].resp.model_copy(
        update={"contributions": 0.0, "grants": 0.0, "income": 0.0, "grant_room_carried": 0.0}
    )
    education = scenario.household.beneficiaries[0].education
    if education_active:
        # SYNTHETIC: moved to already be under way at month index 0.
        education = education.model_copy(update={"start_year": 2024, "start_month": 1})
    broke_beneficiary = scenario.household.beneficiaries[0].model_copy(
        update={"resp": zero_resp, "education": education}
    )
    new_household = scenario.household.model_copy(
        update={"persons": (broke_person,), "beneficiaries": (broke_beneficiary,)}
    )
    return scenario.model_copy(update={"household": new_household})


class TestDepletion:
    """Requirement 7: no income and no assets depletes from month zero."""

    def test_depleted_from_month_zero(self, scenario, market, mortality, withdrawal_order):
        new_scenario = _broke_scenario(scenario, education_active=False)

        draws = build_deterministic_draws(new_scenario, market, mortality)
        state = build_initial_state(new_scenario, n_paths=1)
        state = draw_deaths(state, draws, mortality)
        real_params = real_year(load_year(new_scenario.start_year), new_scenario.assumptions.inflation)

        policy = DoNothingPolicy(state.elections, withdrawal_order)
        result = run(state, policy, draws, market, real_params, trace_path=0)

        month0 = result.trace[0]
        assert month0.spending_cut[0] == pytest.approx(state.spending_monthly)

        assert result.spending_achieved[0, 0] == pytest.approx(0.0)
        assert result.depleted[0, 0]


class TestWriteOffCap:
    """Requirement 8: a deficit larger than the month's spending is only partly written off."""

    def test_spending_cut_is_capped_and_spending_achieved_never_negative(
        self, scenario, market, mortality, withdrawal_order
    ):
        new_scenario = _broke_scenario(scenario, education_active=True)

        draws = build_deterministic_draws(new_scenario, market, mortality)
        state = build_initial_state(new_scenario, n_paths=1)
        state = draw_deaths(state, draws, mortality)
        real_params = real_year(load_year(new_scenario.start_year), new_scenario.assumptions.inflation)

        policy = DoNothingPolicy(state.elections, withdrawal_order)
        result = run(state, policy, draws, market, real_params, trace_path=0)

        month0 = result.trace[0]
        assert month0.spending_cut[0] == pytest.approx(state.spending_monthly)
        assert month0.depletion_deficit[0] > month0.spending_cut[0]
        assert np.all(result.spending_achieved >= 0.0)


class TestTransferRefusal:
    """Requirement 9: every malformed transfer is refused, naming it, before anything moves.

    Every ``match=`` below is a phrase that only ``_validate_transfer`` produces -- not just
    "raises a ValueError", which a completely different bug could also do -- and every message
    the validator raises includes ``repr(transfer)``, so ``match=`` also proves the transfer
    itself was named.
    """

    def _expect_refusal(
        self, opening_state, draws, market, real_params, withdrawal_order, transfer, match
    ):
        policy = ScriptedPolicy(opening_state.elections, withdrawal_order, {0: (transfer,)})
        with pytest.raises(ValueError, match=match) as excinfo:
            advance_month(opening_state, draws.real_returns[0], policy, market, real_params)
        assert repr(transfer) in str(excinfo.value)

    def test_neither_side_is_cash(self, opening_state, draws, market, real_params, withdrawal_order):
        n_paths = draws.n_paths
        t = Transfer(person_index=0, from_kind="rrsp", to_kind="tfsa", amount=np.zeros(n_paths))
        self._expect_refusal(
            opening_state, draws, market, real_params, withdrawal_order, t,
            "exactly one of from_kind/to_kind must be 'cash'",
        )

    def test_both_sides_are_cash(self, opening_state, draws, market, real_params, withdrawal_order):
        n_paths = draws.n_paths
        t = Transfer(person_index=0, from_kind="cash", to_kind="cash", amount=np.zeros(n_paths))
        self._expect_refusal(
            opening_state, draws, market, real_params, withdrawal_order, t,
            "exactly one of from_kind/to_kind must be 'cash'",
        )

    def test_resp_as_a_withdrawal_source(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        n_paths = draws.n_paths
        t = Transfer(person_index=0, from_kind="resp", to_kind="cash", amount=np.zeros(n_paths))
        self._expect_refusal(
            opening_state, draws, market, real_params, withdrawal_order, t,
            "is not in WITHDRAWAL_KINDS",
        )

    def test_rrif_as_a_contribution_target(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        n_paths = draws.n_paths
        t = Transfer(person_index=0, from_kind="cash", to_kind="rrif", amount=np.zeros(n_paths))
        self._expect_refusal(
            opening_state, draws, market, real_params, withdrawal_order, t,
            "is not in CONTRIBUTION_KINDS",
        )

    def test_out_of_range_person_index(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        n_paths = draws.n_paths
        t = Transfer(person_index=7, from_kind="rrsp", to_kind="cash", amount=np.zeros(n_paths))
        self._expect_refusal(
            opening_state, draws, market, real_params, withdrawal_order, t, "out of range",
        )

    def test_negative_amount(self, opening_state, draws, market, real_params, withdrawal_order):
        n_paths = draws.n_paths
        t = Transfer(person_index=0, from_kind="rrsp", to_kind="cash", amount=np.full(n_paths, -1.0))
        self._expect_refusal(
            opening_state, draws, market, real_params, withdrawal_order, t,
            "amount is negative on at least one path",
        )

    def test_nan_amount(self, opening_state, draws, market, real_params, withdrawal_order):
        n_paths = draws.n_paths
        t = Transfer(
            person_index=0, from_kind="rrsp", to_kind="cash", amount=np.full(n_paths, np.nan)
        )
        self._expect_refusal(
            opening_state, draws, market, real_params, withdrawal_order, t,
            "amount is not finite on every path",
        )

    def test_wrong_shape(self, opening_state, draws, market, real_params, withdrawal_order):
        t = Transfer(person_index=0, from_kind="rrsp", to_kind="cash", amount=np.zeros(2))
        self._expect_refusal(
            opening_state, draws, market, real_params, withdrawal_order, t, "amount shape",
        )

    def test_resp_contribution_after_the_education_window_has_ended(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        # Decision C: month_index >= education_start_month_index + education_months refuses
        # an RESP contribution, wound up or not. Scripted for the first month after the
        # window closes, and run through the whole household via run() -- no test may
        # drive advance_month/advance_month_traced in a loop over months.
        resp_state = opening_state.beneficiaries[0].resp
        threshold = resp_state.education_start_month_index + resp_state.education_months
        n_paths = draws.n_paths
        t = Transfer(
            person_index=0, from_kind="cash", to_kind="resp", amount=np.full(n_paths, 10.0)
        )
        policy = ScriptedPolicy(opening_state.elections, withdrawal_order, {threshold: (t,)})
        with pytest.raises(ValueError, match="education window") as excinfo:
            run(opening_state, policy, draws, market, real_params)
        assert repr(t) in str(excinfo.value)


class TestWithdrawalOrderRefusal:
    """Requirement 3 (review): ``policy.withdrawal_order()`` is validated too."""

    def test_a_repeated_kind_is_refused(self, opening_state, draws, market, real_params):
        policy = DoNothingPolicy(opening_state.elections, ("rrsp", "rrsp"))
        with pytest.raises(ValueError, match="is repeated"):
            advance_month(opening_state, draws.real_returns[0], policy, market, real_params)

    def test_lira_is_refused(self, opening_state, draws, market, real_params):
        policy = DoNothingPolicy(opening_state.elections, ("lira",))
        with pytest.raises(ValueError, match="is not in WITHDRAWAL_KINDS"):
            advance_month(opening_state, draws.real_returns[0], policy, market, real_params)

    def test_resp_is_refused(self, opening_state, draws, market, real_params):
        policy = DoNothingPolicy(opening_state.elections, ("resp",))
        with pytest.raises(ValueError, match="is not in WITHDRAWAL_KINDS"):
            advance_month(opening_state, draws.real_returns[0], policy, market, real_params)


class TestContributionCappedAtCash:
    """Requirement 10: a contribution larger than cash contributes only the cash, no floor."""

    def test_contribution_is_capped_at_cash_and_the_floor_does_not_fire(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        n_paths = draws.n_paths
        huge = np.full(n_paths, 1e9)
        script = {
            0: (Transfer(person_index=0, from_kind="cash", to_kind="taxable", amount=huge),)
        }
        policy = ScriptedPolicy(opening_state.elections, withdrawal_order, script)
        _new_state, record = advance_month_traced(
            opening_state, draws.real_returns[0], policy, market, real_params
        )

        available_cash = record.context.cash_after_flows
        np.testing.assert_allclose(record.contributions[0].taxable, available_cash)
        np.testing.assert_allclose(_sum_by_kind(record.floor_withdrawals), 0.0)
        np.testing.assert_allclose(record.cash_close, 0.0, atol=1e-6)


class TestTaxableGrowthExact:
    """Requirement 11: taxable growth and its ledger entries match the account arithmetic."""

    def test_growth_matches_distributions_monthly(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        policy = DoNothingPolicy(opening_state.elections, withdrawal_order)
        opening_balance = opening_state.persons[0].taxable.balance.copy()
        opening_acb = opening_state.persons[0].taxable.acb.copy()

        new_state, record = advance_month_traced(
            opening_state, draws.real_returns[0], policy, market, real_params
        )

        assert np.all(_sum_by_kind(record.withdrawals) == 0.0)
        assert np.all(_sum_by_kind(record.contributions) == 0.0)
        assert np.all(_sum_by_kind(record.floor_withdrawals) == 0.0)

        weighted_yields = market.weighted_yields("taxable")
        interest, dividends, gains = taxable.distributions_monthly(opening_balance, weighted_yields)
        r_taxable = market.weights("taxable") @ draws.real_returns[0]

        expected_balance = opening_balance * (1 + r_taxable)
        expected_acb = opening_acb + interest + dividends + gains

        np.testing.assert_allclose(new_state.persons[0].taxable.balance, expected_balance)
        np.testing.assert_allclose(new_state.persons[0].taxable.acb, expected_acb)
        np.testing.assert_allclose(new_state.persons[0].income.interest, interest)
        np.testing.assert_allclose(new_state.persons[0].income.eligible_dividends, dividends)
        np.testing.assert_allclose(new_state.persons[0].income.capital_gains, gains)


class TestWindUpRunsOnce:
    """Review finding 2 (round 3): ``resp.wind_up`` itself is called exactly once per
    beneficiary, and its cash effect lands only in the month the window closes.

    On the real example, under the deterministic draws, the plan's value drains to
    exactly zero by month 113 -- before the window even closes at month 140 -- so a
    repeated wind-up call would have been invisible to ``wind_up_to_cash`` there (round
    2's version of this test could not have caught that bug: it watched the cash effect,
    which was already zero every month, and the ``wound_up`` flag, which a second no-op
    call would not have disturbed either). This variant shrinks the annual education cost
    so the plan is still funded when the window closes, and wraps ``resp.wind_up`` itself
    with a counting monkeypatch (``engine.core.step`` calls it through the ``resp`` module
    attribute, so patching that attribute is what a call inside the step actually sees).
    """

    def test_wind_up_is_called_exactly_once_and_only_pays_out_at_the_windows_close(
        self, scenario, market, mortality, withdrawal_order, real_params, monkeypatch
    ):
        beneficiary = scenario.household.beneficiaries[0]
        # SYNTHETIC: far below the example's 20000, so the plan (worth roughly 30000 and
        # still growing) is still funded when the window closes, unlike the real example.
        small_education = beneficiary.education.model_copy(update={"annual_cost": 3_000.0})
        new_beneficiary = beneficiary.model_copy(update={"education": small_education})
        new_household = scenario.household.model_copy(
            update={"beneficiaries": (new_beneficiary,)}
        )
        new_scenario = scenario.model_copy(update={"household": new_household})

        draws = build_deterministic_draws(new_scenario, market, mortality)
        state = build_initial_state(new_scenario, n_paths=1)
        state = draw_deaths(state, draws, mortality)
        threshold = (
            state.beneficiaries[0].resp.education_start_month_index
            + state.beneficiaries[0].resp.education_months
        )

        calls: list[object] = []
        original_wind_up = resp.wind_up

        def counting_wind_up(resp_state):
            calls.append(resp_state)
            return original_wind_up(resp_state)

        monkeypatch.setattr(resp, "wind_up", counting_wind_up)

        policy = DoNothingPolicy(state.elections, withdrawal_order)
        result = run(state, policy, draws, market, real_params, trace_path=0)

        assert len(calls) == 1

        for m, record in enumerate(result.trace):
            if m == threshold:
                assert record.wind_up_to_cash[0][0] > 0
            else:
                assert record.wind_up_to_cash[0][0] == 0.0, m


class TestConservation:
    """Requirement 6, round 3's non-circular form: total wealth (RESP excluded) moves
    only by the flows the two brackets name.

    ``W = cash + sum_persons(rrsp + rrif + lira + lif + tfsa + taxable)``, read directly
    from a ``RecordingPolicy`` capture each month -- taken after phase 6, so a forced
    RRIF/LIF withdrawal has already moved money from an account into cash *within that
    same snapshot* and appears in neither bracket below; only its withholding, a real
    leak out of the household, does, in the *following* month's context bracket. A
    forced withdrawal that paid cash without debiting its account, or a transfer that
    did the same, would inflate ``W`` for free and this identity would catch it -- unlike
    the two cash identities above, which are cash-only and blind to exactly this bug.

    The tail of the run's final captured month (phases 7-12, after the last state this
    test reads) is outside this test's reach. ``TestCashIdentityDoNothing`` and
    ``TestCashIdentityScripted`` do cover that tail, but only check that its cash
    movements are recorded -- not that a withdrawal actually debits the account it
    claims to draw from, which is exactly the bug this test exists to catch. That same
    phase 7-12 code runs in every *earlier* month too, though, and there this test does
    check it.
    """

    def test_delta_w_matches_the_flow_identity_under_zero_returns(
        self, scenario, market, mortality, real_params, withdrawal_order
    ):
        person_a = scenario.household.persons[0]
        # SYNTHETIC balances: large enough that both the RRIF and the AB LIF force a
        # non-trivial minimum out in phase 6.
        rrif_balance = 300_000.0
        lif_balance = 200_000.0
        new_person = person_a.model_copy(
            update={
                "accounts": person_a.accounts.model_copy(
                    update={
                        "rrif": person_a.accounts.rrif.model_copy(
                            update={"balance": rrif_balance}
                        ),
                        "lif": LifAccount(balance=lif_balance, jurisdiction="ab"),
                    }
                )
            }
        )
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})
        new_scenario = scenario.model_copy(update={"household": new_household})

        # The LIF's age must fall inside the AB maximum table for lif.maximum_withdrawal
        # not to raise -- read the table's own bounds from params rather than writing
        # them down here.
        ab_params = real_params.jurisdiction("ab")
        table = ab_params.get("lif.maximum_factors.by_age")
        first_age = min(int(age) for age in table)
        terminal_age = int(ab_params.number("lif.maximum_factors.terminal_age_years"))
        age_start = timeline.age_at_start_of_year(
            new_person.birth_year, new_person.birth_month, 2026
        )
        assert first_age <= age_start <= terminal_age

        draws = build_deterministic_draws(new_scenario, market, mortality)
        n_paths = draws.n_paths
        # SYNTHETIC: every real return set to exactly zero, same shape and mortality as
        # the deterministic draws. At r=0, base.grow(balance, 0) is the identity for
        # every registered account, and the taxable account's price return exactly
        # offsets its reinvested distributions (engine.accounts.taxable's own invariant)
        # -- so growth contributes nothing to any balance for the whole run.
        zero_draws = RandomDraws(
            seed=draws.seed,
            real_returns=np.zeros_like(draws.real_returns),
            mortality=draws.mortality,
            n_months=draws.n_months,
            n_paths=draws.n_paths,
        )
        state = build_initial_state(new_scenario, n_paths=n_paths)
        state = draw_deaths(state, zero_draws, mortality)

        script = {
            5: (
                Transfer(
                    person_index=0, from_kind="rrsp", to_kind="cash", amount=np.full(n_paths, 3_000.0)
                ),
            ),
            6: (
                Transfer(
                    person_index=0, from_kind="cash", to_kind="tfsa", amount=np.full(n_paths, 1_000.0)
                ),
                Transfer(
                    person_index=0,
                    from_kind="cash",
                    to_kind="taxable",
                    amount=np.full(n_paths, 1_000.0),
                ),
            ),
            7: (
                Transfer(
                    person_index=0, from_kind="tfsa", to_kind="cash", amount=np.full(n_paths, 500.0)
                ),
            ),
        }
        policy = ScriptedPolicy(state.elections, withdrawal_order, script)
        recorder = RecordingPolicy(policy)
        result = run(state, recorder, zero_draws, market, real_params, trace_path=0)

        forced_fired = False

        # Month 0, from the true opening state -- there is no preceding month's record
        # to bracket, only month 0's own context.
        state0, context0 = recorder.calls[0]
        if np.any(_sum_by_kind(context0.forced_withdrawals) > 0):
            forced_fired = True
        rhs0 = (
            _sum_arrays(inflow.to_cash for inflow in context0.inflows)
            + _sum_arrays(context0.education_draws)
            - _sum_arrays(context0.payroll_withholding)
            - context0.spending
            - _sum_arrays(context0.education_costs)
            - _sum_arrays(context0.tax_settlement)
            - _sum_arrays(context0.forced_withholding)
        )
        np.testing.assert_allclose(_w(state0) - _w(state), rhs0, atol=0.005)

        for m in range(len(recorder.calls) - 1):
            state_m, _context_m = recorder.calls[m]
            state_m1, context_m1 = recorder.calls[m + 1]
            record_m = result.trace[m]
            if np.any(_sum_by_kind(context_m1.forced_withdrawals) > 0):
                forced_fired = True

            lhs = _w(state_m1) - _w(state_m)
            rhs = (
                -_sum_arrays(record_m.withdrawal_withholding)
                + _sum_arrays(record_m.wind_up_to_cash)
                + record_m.depletion_deficit
                + _sum_arrays(inflow.to_cash for inflow in context_m1.inflows)
                + _sum_arrays(context_m1.education_draws)
                - _sum_arrays(context_m1.payroll_withholding)
                - context_m1.spending
                - _sum_arrays(context_m1.education_costs)
                - _sum_arrays(context_m1.tax_settlement)
                - _sum_arrays(context_m1.forced_withholding)
            )
            assert np.all(np.abs(lhs - rhs) < 0.005), m

        assert forced_fired, "phase 6 never forced a withdrawal; the test would be vacuous"


class TestSettlementFailsLoudly:
    """Decision D: phase 5 discards ``settle_tax_balance``'s result, so if #35 ever makes
    it return something other than its argument unchanged, that must fail loudly rather
    than silently disappear.
    """

    def test_a_changed_result_in_the_filing_month_raises(
        self, opening_state, draws, market, real_params, withdrawal_order, monkeypatch
    ):
        filing_month = int(real_params.federal.number("filing_month"))
        filing_state = updated(
            opening_state, month=filing_month, month_index=filing_month - 1
        )
        monkeypatch.setattr(
            step_module, "settle_tax_balance", lambda s, _rp: dataclasses.replace(s)
        )
        policy = DoNothingPolicy(opening_state.elections, withdrawal_order)
        with pytest.raises(AssertionError, match="discards its result"):
            advance_month(filing_state, draws.real_returns[0], policy, market, real_params)

    def test_a_changed_result_in_a_non_filing_month_does_not_raise(
        self, opening_state, draws, market, real_params, withdrawal_order, monkeypatch
    ):
        filing_month = int(real_params.federal.number("filing_month"))
        assert opening_state.month != filing_month
        monkeypatch.setattr(
            step_module, "settle_tax_balance", lambda s, _rp: dataclasses.replace(s)
        )
        policy = DoNothingPolicy(opening_state.elections, withdrawal_order)
        advance_month(opening_state, draws.real_returns[0], policy, market, real_params)


class TestPerPensionPayrollWithholding:
    """Round 3 finding 4: phase 4 withholds on employment and on each pension's own
    monthly amount separately, matching ``withholding.payroll_withholding_monthly``
    called the same way phase 4 itself calls it.
    """

    def test_matches_the_engines_own_function_with_and_without_employment(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        policy = DoNothingPolicy(opening_state.elections, withdrawal_order)
        result = run(opening_state, policy, draws, market, real_params, trace_path=0)

        person = opening_state.persons[0]
        pension = person.pensions[0]
        employment_end = person.employment[0].to_month_index

        # The example's DB pension starts while employment is still active, and
        # employment ends well before the run does -- both cases the example gives.
        with_employment_month = pension.start_month_index
        assert with_employment_month <= employment_end
        without_employment_month = employment_end + 1

        for m in (with_employment_month, without_employment_month):
            context = result.trace[m].context
            january_month_index = context.month_index - (context.month - 1)
            age_end = timeline.age_at_end_of_year(
                person.birth_year, person.birth_month, context.year
            )
            expected = withholding.payroll_withholding_monthly(
                context.inflows[0].employment,
                age_end,
                True,
                opening_state.province,
                real_params,
                january_month_index,
            ) + withholding.payroll_withholding_monthly(
                context.inflows[0].db_pension,
                age_end,
                False,
                opening_state.province,
                real_params,
                january_month_index,
            )
            np.testing.assert_allclose(context.payroll_withholding[0], expected)


class TestCloseYearAssertion:
    """Requirement 14: an unmet RRIF minimum with a non-zero balance raises."""

    def test_unmet_minimum_raises(self, scenario, real_params):
        rrif_balance = 100_000.0  # SYNTHETIC: any non-zero balance with an unmet minimum works.
        person_a = scenario.household.persons[0]
        new_accounts = person_a.accounts.model_copy(
            update={"rrif": person_a.accounts.rrif.model_copy(update={"balance": rrif_balance})}
        )
        new_person = person_a.model_copy(update={"accounts": new_accounts})
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})
        new_scenario = scenario.model_copy(update={"household": new_household})

        state = build_initial_state(new_scenario, n_paths=1)
        opened_state = open_year(state, real_params)
        assert opened_state.persons[0].rrif.annual_minimum[0] > 0.0

        with pytest.raises(AssertionError):
            close_year(opened_state, real_params)

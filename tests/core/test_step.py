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
import re
import shutil
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from engine.accounts import lif, resp, rrif, rrsp, taxable, tfsa
from engine.benefits import cpp as cpp_mod
from engine.benefits.oas import deferral_factor
from engine.core import timeline
from engine.core.build import (
    build_deterministic_draws,
    build_draws,
    build_initial_state,
    build_market_inputs,
    draw_deaths,
)
from engine.core.indexation import nominal_carry_factor, real_year, unindexed_factor
from engine.core.state import (
    DEATH_NOT_DRAWN,
    Assessment,
    BeneficiaryState,
    CashState,
    IncomeLedger,
    RespState,
    select_spending_level,
    updated,
)
from engine.core.step import (
    advance_month,
    advance_month_traced,
    close_year,
    open_year,
    resolve_deaths,
)
from engine.mc.returns import RandomDraws
from engine.mc.simulate import run
from engine.params.loader import load_year
from engine.policy.base import Transfer
from engine.scenario import LifAccount, load_scenario
from engine.scenario.schema import Scenario
from engine.tax import federal, withholding
from engine.tax.combined import household_assessment, person_assessment

from .policies import DoNothingPolicy, RecordingPolicy, ScriptedPolicy

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
def withdrawal_order(scenario):
    return scenario.policies[0].withdrawal.order


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
def couple_draws(couple_scenario, couple_market, couple_mortality):
    return build_deterministic_draws(couple_scenario, couple_market, couple_mortality)


@pytest.fixture
def couple_opening_state(couple_scenario, couple_draws, couple_mortality):
    state = build_initial_state(couple_scenario, n_paths=couple_draws.n_paths)
    return draw_deaths(state, couple_draws, couple_mortality)


@pytest.fixture
def couple_withdrawal_order(couple_scenario):
    return couple_scenario.policies[0].withdrawal.order


def _force_death(state, index: int, month_index: int):
    """``state`` with ``persons[index].death_month_index`` forced to ``month_index`` on every
    path -- ``alive`` at the opening is untouched (stays ``True``, since every forced
    ``month_index`` used below is strictly positive), per the brief's forcing recipe.
    """
    persons = list(state.persons)
    person = persons[index]
    persons[index] = updated(
        person,
        death_month_index=np.full(state.n_paths, month_index, dtype=np.int64),
    )
    return updated(state, persons=tuple(persons))


def _account_amounts_tuple(amounts):
    return (amounts.rrsp, amounts.rrif, amounts.lira, amounts.lif, amounts.tfsa, amounts.taxable)


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
        - ctx.cash_to_estate
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
        - _sum_arrays(record.wind_up_withholding)
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


def _aip_resp_state(
    n_paths: int,
    *,
    income: float,
    contributions: float = 0.0,
    grants: float = 0.0,
    subscriber_index: int = 0,
    education_start_month_index: int = 4,
    education_months: int = 1,
) -> RespState:
    """A synthetic RESP plan, wound up at
    ``education_start_month_index + education_months``, for #57's AIP-withholding tests.
    """
    return RespState(
        contributions=np.full(n_paths, contributions),
        grants=np.full(n_paths, grants),
        income=np.full(n_paths, income),
        contributions_lifetime=np.zeros(n_paths),
        grants_lifetime=np.zeros(n_paths),
        grant_room=np.zeros(n_paths),
        grant_received_ytd=np.zeros(n_paths),
        contributed_ytd=np.zeros(n_paths),
        subscriber_index=subscriber_index,
        education_start_month_index=education_start_month_index,
        education_months=education_months,
        education_monthly_cost=0.0,
        wound_up=np.zeros(n_paths, dtype=bool),
    )


def _swap_beneficiary(state, resp_state: RespState):
    """``state`` with its beneficiaries replaced by a single synthetic one carrying
    ``resp_state``."""
    beneficiary = BeneficiaryState(
        beneficiary_id="child", birth_year=2015, birth_month=1, resp=resp_state
    )
    return updated(state, beneficiaries=(beneficiary,))


def _aip_expected_accumulated(
    market, draws, *, income: float = 5_000.0, wind_up_month_index: int = 5
) -> np.ndarray:
    """Accumulated income :func:`_aip_resp_state`'s plan holds the moment it winds up,
    computed independently of the step under test: the whole opening amount sits in the
    income bucket and compounds at the ``resp`` asset weights for every month strictly
    before ``wind_up_month_index`` (the phase-8 wind-up in that month precedes that
    month's own phase-10 growth -- the same arithmetic as ``engine.accounts.resp.grow``).

    Args:
        market: Supplies ``weights("resp")``.
        draws: Supplies ``real_returns``, ``(n_months, n_assets, n_paths)``, and
            ``n_paths``.
        income: The fixture's opening ``income`` bucket.
        wind_up_month_index: Month index the plan winds up in.

    Returns:
        Accumulated income at the wind-up, ``(n_paths,)``.
    """
    r_resp = market.weights("resp")
    expected = np.full(draws.n_paths, income)
    for m in range(wind_up_month_index):
        expected = expected * (1 + r_resp @ draws.real_returns[m])
    return expected


def _synthetic_resp_params(scenario, real_params, tmp_path, *, rate: float = 0.37):
    """A real param set identical to ``real_params`` except ``resp.yaml``'s
    ``aip.penalty_rate`` is replaced by an obviously synthetic ``rate``.

    Copies ``params/<start_year>`` to ``tmp_path``, substitutes the rate (asserting the
    substitution hit exactly once), then loads and real-values the copy. Asserts the
    loaded rate differs from ``real_params``'s -- the control that proves a caller
    reading the synthetic set would notice a step that (wrongly) reads the real one
    instead.

    Args:
        scenario: Supplies ``start_year`` and ``assumptions.inflation``.
        real_params: The real param set built from the unmodified files, for the
            control comparison.
        tmp_path: A pytest-provided, per-test temporary directory.
        rate: The synthetic ``aip.penalty_rate``, obviously distinct from the real one.

    Returns:
        The synthetic real param set.
    """
    shutil.copytree(
        REPO_ROOT / "params" / str(scenario.start_year), tmp_path / str(scenario.start_year)
    )
    resp_path = tmp_path / str(scenario.start_year) / "resp.yaml"
    original = resp_path.read_text()
    # SYNTHETIC TEST FIXTURE: an obviously fake rate, to prove this reads the copied
    # file rather than a hardcoded value.
    substituted, count = re.subn(
        r"penalty_rate:\s*[0-9.]+[^\n]*",
        f"penalty_rate: {rate}  # SYNTHETIC TEST FIXTURE",
        original,
    )
    assert count == 1
    resp_path.write_text(substituted)

    synthetic_params = real_year(
        load_year(scenario.start_year, tmp_path), scenario.assumptions.inflation
    )
    assert synthetic_params.resp.number("aip.penalty_rate") != real_params.resp.number(
        "aip.penalty_rate"
    )
    return synthetic_params


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
        for record, (seen_state, seen_context) in zip(result.trace, recorder.calls, strict=True):
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
        real_params = real_year(
            load_year(new_scenario.start_year), new_scenario.assumptions.inflation
        )

        policy = DoNothingPolicy(state.elections, withdrawal_order)
        recorder = RecordingPolicy(policy)
        run(state, recorder, draws, market, real_params)

        for m in range(11):
            assert recorder.calls[m][0].persons[0].rrif.withdrawn_ytd[0] == pytest.approx(0.0)

        age_start = timeline.age_at_start_of_year(
            new_person.birth_year, new_person.birth_month, 2026
        )
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

    Split into two households, each a size ``household_assessment`` (and therefore
    ``close_year``, which every ``run()`` below now reaches every December) actually accepts:
    ``young`` alone (below the eligibility age, never filled), and a couple of ``old`` (at/above
    it, RRIF only) and ``old_lif`` (at/above it, a RRIF *and* an AB LIF whose own minimum is
    below the target -- B's case). The three original claims, and every expected expression,
    are unchanged; only which run and which person index each one reads has moved.
    """

    def _build_couple(self, scenario):
        person_a = scenario.household.persons[0]
        # SYNTHETIC balances: small enough that the ordinary annual minimum stays well below
        # the pension credit target, so the fill logic -- not the mandatory minimum -- is
        # what reaches the target; the LIF balance is small enough its own minimum is below
        # the target too, per decision B.
        old_rrif_balance = 10_000.0
        old_lif_lif_balance = 5_000.0

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
        # No beneficiary: the example's one beneficiary subscribes to "a", a person neither
        # of this couple is, and engine.core.build._build_beneficiary raises ValueError for a
        # beneficiary naming no person in the household.
        new_household = scenario.household.model_copy(
            update={"persons": (old, old_lif), "beneficiaries": ()}
        )

        policy_spec = scenario.policies[0]
        elections = policy_spec.elections
        new_elections = elections.model_copy(
            update={
                "cpp_start_age_years": {"old": 65, "old_lif": 65},
                "oas_start_age_years": {"old": 65, "old_lif": 65},
            }
        )
        new_policy = policy_spec.model_copy(update={"elections": new_elections})
        # No grid: the example's grid key "elections.cpp_start_age_years.a" names a person
        # this couple does not have, and run() never expands the grid.
        new_scenario = scenario.model_copy(
            update={
                "household": new_household,
                "policies": (new_policy,),
                "grid": MappingProxyType({}),
            }
        )
        assert new_policy.withdrawal.fill_pension_credit is True
        # model_copy skips validation; the round trip makes the schema accept the household.
        Scenario.model_validate(new_scenario.model_dump(warnings=False))
        return new_scenario, old, old_lif

    def _build_young(self, scenario):
        person_a = scenario.household.persons[0]
        # SYNTHETIC balance: small enough that the ordinary annual minimum stays well below the
        # pension credit target -- irrelevant here since young is below the eligibility age, but
        # kept identical to the original fixture's value.
        young_rrif_balance = 50_000.0

        # id unchanged ("a"): the example's one beneficiary subscribes to "a", and this keeps
        # that subscription valid without touching beneficiaries at all.
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
        new_household = scenario.household.model_copy(update={"persons": (young,)})
        new_scenario = scenario.model_copy(update={"household": new_household})
        assert scenario.policies[0].withdrawal.fill_pension_credit is True
        # model_copy skips validation; the round trip makes the schema accept the household.
        Scenario.model_validate(new_scenario.model_dump(warnings=False))
        return new_scenario, young

    def test_below_age_at_age_and_the_lif_aware_spreading_and_year_end_target(
        self, scenario, market, mortality, withdrawal_order, real_params
    ):
        fill_age = real_params.federal.number("eligible_pension_income.rrif_minimum_age_years")

        # --- The couple: old and old_lif, at/above the eligibility age. ---
        couple_scenario, old, old_lif = self._build_couple(scenario)
        couple_draws = build_deterministic_draws(couple_scenario, market, mortality)
        couple_state = build_initial_state(couple_scenario, n_paths=1)
        couple_state = draw_deaths(couple_state, couple_draws, mortality)
        couple_real_params = real_year(
            load_year(couple_scenario.start_year), couple_scenario.assumptions.inflation
        )

        age_end_old = timeline.age_at_end_of_year(old.birth_year, old.birth_month, 2026)
        age_end_old_lif = timeline.age_at_end_of_year(old_lif.birth_year, old_lif.birth_month, 2026)
        assert age_end_old >= fill_age
        assert age_end_old_lif >= fill_age

        old_index, old_lif_index = 0, 1
        couple_policy = DoNothingPolicy(couple_state.elections, withdrawal_order)
        couple_recorder = RecordingPolicy(couple_policy)
        run(couple_state, couple_recorder, couple_draws, market, couple_real_params)

        # Claim 2 (spreading): the month-0 forced RRIF withdrawal for old_lif equals
        # max(minimum_still_required, (target - eligible_before_fill - lif_min_remaining) / 12).
        target = max(
            couple_real_params.federal.annual_amount("credits.pension_income_amount_annual", 0),
            couple_real_params.province(couple_scenario.household.province).annual_amount(
                "credits.pension_income_amount_annual", 0
            ),
        )
        state0, context0 = couple_recorder.calls[0]
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
        state11 = couple_recorder.calls[11][0]
        lif_annual_minimum = state11.persons[old_lif_index].lif.annual_minimum[0]
        eligible_old_lif = federal.eligible_pension_income(
            state11.persons[old_lif_index].income, age_end_old_lif, couple_real_params.federal
        )
        assert eligible_old_lif[0] == pytest.approx(target, abs=0.005)
        assert eligible_old_lif[0] != pytest.approx(target + lif_annual_minimum, abs=0.005)

        # "old" (RRIF only, no LIF) still reaches the target exactly as before.
        eligible_old = federal.eligible_pension_income(
            state11.persons[old_index].income, age_end_old, couple_real_params.federal
        )
        assert eligible_old[0] == pytest.approx(target, abs=0.005)

        # --- young, alone, below the eligibility age. ---
        young_scenario, young = self._build_young(scenario)
        young_draws = build_deterministic_draws(young_scenario, market, mortality)
        young_state = build_initial_state(young_scenario, n_paths=1)
        young_state = draw_deaths(young_state, young_draws, mortality)
        young_real_params = real_year(
            load_year(young_scenario.start_year), young_scenario.assumptions.inflation
        )

        age_end_young = timeline.age_at_end_of_year(young.birth_year, young.birth_month, 2026)
        assert age_end_young < fill_age

        young_policy = DoNothingPolicy(young_state.elections, withdrawal_order)
        young_recorder = RecordingPolicy(young_policy)
        run(young_state, young_recorder, young_draws, market, young_real_params)

        # Claim 1: below the age, no fill -- zero forced RRIF withdrawals through month 10
        # (December, month 11, forces the ordinary minimum regardless of age; that is not
        # tested here).
        for m in range(11):
            assert young_recorder.calls[m][0].persons[0].rrif.withdrawn_ytd[0] == pytest.approx(0.0)

        state11_young = young_recorder.calls[11][0]
        eligible_young = federal.eligible_pension_income(
            state11_young.persons[0].income, age_end_young, young_real_params.federal
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
        real_params = real_year(
            load_year(new_scenario.start_year), new_scenario.assumptions.inflation
        )

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
        real_params = real_year(
            load_year(new_scenario.start_year), new_scenario.assumptions.inflation
        )

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

    def test_neither_side_is_cash(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        n_paths = draws.n_paths
        t = Transfer(person_index=0, from_kind="rrsp", to_kind="tfsa", amount=np.zeros(n_paths))
        self._expect_refusal(
            opening_state,
            draws,
            market,
            real_params,
            withdrawal_order,
            t,
            "exactly one of from_kind/to_kind must be 'cash'",
        )

    def test_both_sides_are_cash(self, opening_state, draws, market, real_params, withdrawal_order):
        n_paths = draws.n_paths
        t = Transfer(person_index=0, from_kind="cash", to_kind="cash", amount=np.zeros(n_paths))
        self._expect_refusal(
            opening_state,
            draws,
            market,
            real_params,
            withdrawal_order,
            t,
            "exactly one of from_kind/to_kind must be 'cash'",
        )

    def test_resp_as_a_withdrawal_source(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        n_paths = draws.n_paths
        t = Transfer(person_index=0, from_kind="resp", to_kind="cash", amount=np.zeros(n_paths))
        self._expect_refusal(
            opening_state,
            draws,
            market,
            real_params,
            withdrawal_order,
            t,
            "is not in WITHDRAWAL_KINDS",
        )

    def test_rrif_as_a_contribution_target(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        n_paths = draws.n_paths
        t = Transfer(person_index=0, from_kind="cash", to_kind="rrif", amount=np.zeros(n_paths))
        self._expect_refusal(
            opening_state,
            draws,
            market,
            real_params,
            withdrawal_order,
            t,
            "is not in CONTRIBUTION_KINDS",
        )

    def test_out_of_range_person_index(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        n_paths = draws.n_paths
        t = Transfer(person_index=7, from_kind="rrsp", to_kind="cash", amount=np.zeros(n_paths))
        self._expect_refusal(
            opening_state,
            draws,
            market,
            real_params,
            withdrawal_order,
            t,
            "out of range",
        )

    def test_negative_amount(self, opening_state, draws, market, real_params, withdrawal_order):
        n_paths = draws.n_paths
        t = Transfer(
            person_index=0, from_kind="rrsp", to_kind="cash", amount=np.full(n_paths, -1.0)
        )
        self._expect_refusal(
            opening_state,
            draws,
            market,
            real_params,
            withdrawal_order,
            t,
            "amount is negative on at least one path",
        )

    def test_nan_amount(self, opening_state, draws, market, real_params, withdrawal_order):
        n_paths = draws.n_paths
        t = Transfer(
            person_index=0, from_kind="rrsp", to_kind="cash", amount=np.full(n_paths, np.nan)
        )
        self._expect_refusal(
            opening_state,
            draws,
            market,
            real_params,
            withdrawal_order,
            t,
            "amount is not finite on every path",
        )

    def test_wrong_shape(self, opening_state, draws, market, real_params, withdrawal_order):
        t = Transfer(person_index=0, from_kind="rrsp", to_kind="cash", amount=np.zeros(2))
        self._expect_refusal(
            opening_state,
            draws,
            market,
            real_params,
            withdrawal_order,
            t,
            "amount shape",
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
        script = {0: (Transfer(person_index=0, from_kind="cash", to_kind="taxable", amount=huge),)}
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
        new_household = scenario.household.model_copy(update={"beneficiaries": (new_beneficiary,)})
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
                        "rrif": person_a.accounts.rrif.model_copy(update={"balance": rrif_balance}),
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
                    person_index=0,
                    from_kind="rrsp",
                    to_kind="cash",
                    amount=np.full(n_paths, 3_000.0),
                ),
            ),
            6: (
                Transfer(
                    person_index=0,
                    from_kind="cash",
                    to_kind="tfsa",
                    amount=np.full(n_paths, 1_000.0),
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
                - _sum_arrays(record_m.wind_up_withholding)
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


class TestTaxAssessedEqualsSettlementPlusWithholding:
    """The tax identity, end to end -- close_year's tax_assessed for a year equals everything
    withheld during that year plus the following year's filing-month settlement. Mutation checked:
    reverting item 4 of close_year from ``balance_owing = total - remitted`` to ``balance_owing =
    total`` (dropping ``remitted``) breaks this identity, since the settlement would then
    double-count what was already withheld.
    """

    def test_the_identity_holds_for_the_first_year(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        policy = DoNothingPolicy(opening_state.elections, withdrawal_order)
        result = run(opening_state, policy, draws, market, real_params, trace_path=0)

        year = opening_state.year
        filing_month = int(real_params.federal.number("filing_month"))
        assert list(result.years).count(year) == 1
        y_index = list(result.years).index(year)
        tax_assessed_y = float(result.tax_assessed[y_index, 0])
        # Not vacuous: the example has employment income in the start year.
        assert tax_assessed_y > 0.0

        withheld_total = 0.0
        settlement_total = 0.0
        for record in result.trace:
            ctx = record.context
            if ctx.year == year:
                withheld_total += float(sum(w[0] for w in ctx.payroll_withholding))
                withheld_total += float(sum(w[0] for w in ctx.forced_withholding))
                withheld_total += float(sum(w[0] for w in record.withdrawal_withholding))
            if ctx.year == year + 1 and ctx.month == filing_month:
                settlement_total += float(sum(s[0] for s in ctx.tax_settlement))

        assert tax_assessed_y == pytest.approx(withheld_total + settlement_total, abs=0.01)


class TestRefundSettlement:
    """A remittance surplus settles as a refund, a deposit rather than a debit.
    Mutation checked: swapping ``settle_tax_balance``'s ``pay``/``deposit`` calls (paying
    ``max(-owing, 0)`` and depositing ``max(owing, 0)``) makes cash fall by the refund instead
    of rising, which the identity check below catches.
    """

    def test_a_negative_balance_owing_settles_as_a_refund(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        filing_month = int(real_params.federal.number("filing_month"))
        refund = 500.0  # SYNTHETIC: an arbitrary remittance surplus.
        persons = tuple(
            updated(p, balance_owing=np.full(p.balance_owing.shape, -refund))
            for p in opening_state.persons
        )
        filing_state = updated(
            opening_state, persons=persons, month=filing_month, month_index=filing_month - 1
        )
        policy = DoNothingPolicy(opening_state.elections, withdrawal_order)
        new_state, record = advance_month_traced(
            filing_state, draws.real_returns[filing_month - 1], policy, market, real_params
        )

        for settlement in record.context.tax_settlement:
            np.testing.assert_allclose(settlement, -refund)

        _assert_identities_hold(record)

        for person in new_state.persons:
            np.testing.assert_allclose(person.balance_owing, 0.0)


class TestSettlementTimingWindow:
    """The settlement fires only in the filing month, and balance_owing bridges the gap between the
    December close and it. Mutation checked: dropping the ``timeline.is_filing_month`` guard in
    phase 5 (settling every month) makes ``tax_settlement`` nonzero outside the filing month, which
    the first loop below catches.
    """

    def test_tax_settlement_only_fires_in_the_filing_month_and_balance_owing_bridges_the_gap(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        policy = DoNothingPolicy(opening_state.elections, withdrawal_order)
        recorder = RecordingPolicy(policy)
        result = run(opening_state, recorder, draws, market, real_params, trace_path=0)

        filing_month = int(real_params.federal.number("filing_month"))
        december_close_index = 11
        filing_index = 12 + (filing_month - 1)

        # Not vacuous: the first year assesses real tax, so the settlement is nonzero.
        assert result.tax_assessed[0, 0] > 0.0

        # Restricted to the first year's own cycle: tax is owed again in later years too, so
        # a later filing month is expected to settle something nonzero of its own -- checking
        # the whole multi-decade run here would not be "only in the filing month" any more.
        for m, record in enumerate(result.trace[: filing_index + 1]):
            total_settlement = float(sum(s[0] for s in record.context.tax_settlement))
            if m == filing_index:
                assert total_settlement != 0.0
            else:
                assert total_settlement == 0.0

        for m in range(december_close_index + 1, filing_index):
            state_m = recorder.calls[m][0]
            assert np.all(state_m.persons[0].balance_owing != 0.0)

        state_at_filing = recorder.calls[filing_index][0]
        assert np.all(state_at_filing.persons[0].balance_owing == 0.0)


class TestNoIncomeAssessesZero:
    """A person with an empty ledger assesses to zero tax and zero balance owing."""

    def test_empty_ledger_assesses_zero(self, scenario, real_params):
        state = build_initial_state(scenario, n_paths=2)
        opened = open_year(state, real_params)
        closed = close_year(opened, real_params)

        for person in closed.persons:
            np.testing.assert_allclose(person.balance_owing, 0.0)
        np.testing.assert_allclose(closed.history[-1].tax_assessed, 0.0)


class TestBenefitFirstPaymentMonth:
    """On the example scenario, CPP is first paid in the month the elected age is reached and
    OAS in the month after (CPP s.67(3.1); OAS Act s.8(1)), through the step's own ages.
    """

    def _monthly_amounts(
        self, scenario, opening_state, draws, market, real_params, withdrawal_order
    ):
        person = opening_state.persons[0]
        age_at_open = person.age_months(scenario.start_year, 1)
        cpp_start = opening_state.elections.cpp_start_age_months[0]
        oas_start = opening_state.elections.oas_start_age_months[0]
        assert cpp_start is not None and oas_start is not None, "both starts must be elected"
        cpp_month = cpp_start - age_at_open
        oas_month = oas_start - age_at_open
        assert cpp_month > 0, "the CPP start must fall after the opening"
        assert oas_month > 0, "the OAS start must fall after the opening"
        assert person.death_month_index[0] > oas_month + 1, "alive through the months checked"

        policy = DoNothingPolicy(opening_state.elections, withdrawal_order)
        state = opening_state
        amounts = {}
        for month_index in range(oas_month + 2):
            state = advance_month(
                state, draws.real_returns[month_index], policy, market, real_params
            )
            amounts[month_index] = (
                float(state.persons[0].cpp.monthly_amount[0]),
                float(state.persons[0].oas.monthly_amount[0]),
            )
        return amounts, cpp_month, oas_month, oas_start

    def test_cpp_is_first_paid_in_the_month_the_elected_age_is_reached(
        self, scenario, opening_state, draws, market, real_params, withdrawal_order
    ):
        amounts, cpp_month, _, _ = self._monthly_amounts(
            scenario, opening_state, draws, market, real_params, withdrawal_order
        )
        assert amounts[cpp_month - 1][0] == 0.0
        assert amounts[cpp_month][0] > 0.0

    def test_oas_is_first_paid_in_the_month_after_the_elected_age_is_reached(
        self, scenario, opening_state, draws, market, real_params, withdrawal_order
    ):
        amounts, _, oas_month, oas_start = self._monthly_amounts(
            scenario, opening_state, draws, market, real_params, withdrawal_order
        )
        second_band_from = int(real_params.oas.sequence("pension.age_bands")[1]["from_age_months"])
        assert oas_start + 1 < second_band_from, "the first payment must fall in the first band"
        band0 = real_params.oas.amounts("pension.age_bands.*.maximum_monthly", oas_month + 1)[0]
        assert amounts[oas_month][1] == 0.0
        assert amounts[oas_month + 1][1] == pytest.approx(
            band0 * deferral_factor(oas_start, real_params.oas)
        )


class TestRrspAndLiraConversion:
    """The RRSP-to-RRIF and LIRA-to-LIF conversion triggers.

    The example's first person reaches the RRSP-to-RRIF conversion age and their LIRA's
    jurisdiction's LIF conversion deadline in the same year -- a precondition
    :meth:`_statutory` asserts, read from parameters rather than typed by hand. The example's
    own elected conversion age falls in an earlier year, so the two clauses below are
    naturally disjoint without touching the scenario.

    Some tests add a second RRSP holder in memory, since no committed scenario has one: the
    election is one household-wide age compared with each person's own December age, and the
    statutory deadline is each person's own.
    """

    def _lif_age(self, scenario, real_params):
        """The example's LIRA jurisdiction's LIF conversion deadline age, from parameters."""
        person = scenario.household.persons[0]
        return int(
            real_params.jurisdiction(person.accounts.lira.jurisdiction).number(
                "lif.conversion_deadline_age_years"
            )
        )

    def _statutory(self, scenario, real_params):
        """Statutory conversion inputs for the example's first person, from parameters.

        Returns:
            ``(birth_year, statutory_age, statutory_year)``: their birth year, the
            RRSP-to-RRIF conversion age -- asserted equal to their LIRA jurisdiction's LIF
            conversion deadline age -- and the year both fall due.
        """
        person = scenario.household.persons[0]
        birth_year = person.birth_year
        rrif_age = int(real_params.rrif.number("conversion_age_years"))
        lif_age = self._lif_age(scenario, real_params)
        # The statutory test relies on both conversions falling in the same year.
        assert rrif_age == lif_age
        statutory_year = birth_year + rrif_age
        return birth_year, rrif_age, statutory_year

    def _state_at_year(self, scenario, n_paths, year):
        state = build_initial_state(scenario, n_paths=n_paths)
        month_index = 12 * (year - scenario.start_year)
        spending_monthly = select_spending_level(state.spending_schedule, year)
        return updated(
            state, year=year, month=1, month_index=month_index, spending_monthly=spending_monthly
        )

    def _couple_with_election(self, scenario, elected_age, birth_year_offset=4):
        """The example plus a second RRSP holder, with the RRIF election at ``elected_age``.

        Person ``b`` copies the example's person ``a``, born ``birth_year_offset`` years later
        (earlier when negative), and holds half of ``a``'s RRSP and LIRA balances, so a result
        written to the wrong person shows. ``b`` is second, so a close that looked only at the
        first person cannot pass a test in which ``b`` converts. ``b``'s CPP and OAS start ages
        copy ``a``'s.
        """
        person_a = scenario.household.persons[0]
        person_b = person_a.model_copy(
            update={
                "id": "b",
                "birth_year": person_a.birth_year + birth_year_offset,
                "accounts": person_a.accounts.model_copy(
                    update={
                        "rrsp": person_a.accounts.rrsp.model_copy(
                            update={"balance": person_a.accounts.rrsp.balance / 2}
                        ),
                        "lira": person_a.accounts.lira.model_copy(
                            update={"balance": person_a.accounts.lira.balance / 2}
                        ),
                    }
                ),
            }
        )
        new_household = scenario.household.model_copy(update={"persons": (person_a, person_b)})
        elections = scenario.policies[0].elections
        new_elections = elections.model_copy(
            update={
                "cpp_start_age_years": {
                    **elections.cpp_start_age_years,
                    "b": elections.cpp_start_age_years[person_a.id],
                },
                "oas_start_age_years": {
                    **elections.oas_start_age_years,
                    "b": elections.oas_start_age_years[person_a.id],
                },
                "rrif_conversion": elections.rrif_conversion.model_copy(
                    update={"age_years": elected_age}
                ),
            }
        )
        new_policy = scenario.policies[0].model_copy(update={"elections": new_elections})
        return scenario.model_copy(update={"household": new_household, "policies": (new_policy,)})

    def _opened_couple(self, scenario, real_params, elected_age_offset):
        """Open the couple's first year, electing at the younger's December age plus an offset.

        Returns:
            ``(couple, opened)``: the scenario, and its state after ``open_year`` at the start
            year with two paths.
        """
        probe = self._couple_with_election(
            scenario, scenario.policies[0].elections.rrif_conversion.age_years
        )
        december_ages = [
            timeline.age_at_end_of_year(p.birth_year, p.birth_month, scenario.start_year)
            for p in probe.household.persons
        ]
        younger_age = min(december_ages)
        assert december_ages[1] < december_ages[0]
        assert max(december_ages) < real_params.rrif.number("conversion_age_years")
        couple = self._couple_with_election(scenario, younger_age + elected_age_offset)
        state = self._state_at_year(couple, n_paths=2, year=scenario.start_year)
        opened = open_year(state, real_params)
        for person in opened.persons:
            assert np.all(person.rrsp.balance > 0.0)
        return couple, opened

    def test_statutory_conversion_fully_converts_both_accounts(self, scenario, real_params):
        _, _, statutory_year = self._statutory(scenario, real_params)
        state = self._state_at_year(scenario, n_paths=2, year=statutory_year)
        opened = open_year(state, real_params)
        opening_rrsp = opened.persons[0].rrsp.balance.copy()
        assert np.all(opening_rrsp > 0.0)
        opening_lira = opened.persons[0].lira.balance.copy()
        assert np.all(opening_lira > 0.0)
        # Neither account has ever been opened before, so the conversion year's own minimum
        # is zero -- the control this test needs before checking it turns nonzero.
        np.testing.assert_allclose(opened.persons[0].rrif.annual_minimum, 0.0)
        np.testing.assert_allclose(opened.persons[0].lif.annual_minimum, 0.0)

        closed = close_year(opened, real_params)
        person = closed.persons[0]
        np.testing.assert_allclose(person.rrsp.balance, 0.0)
        np.testing.assert_allclose(person.lira.balance, 0.0)
        assert np.all(person.rrif.balance > 0.0)
        assert np.all(person.lif.balance > 0.0)
        np.testing.assert_allclose(person.rrif.balance, opening_rrsp)
        np.testing.assert_allclose(person.lif.balance, opening_lira)
        assert person.rrif.opened_year == statutory_year
        assert person.lif.opened_year == statutory_year

        next_year = statutory_year + 1
        rolled = updated(
            closed, year=next_year, month=1, month_index=12 * (next_year - scenario.start_year)
        )
        opened_next = open_year(rolled, real_params)
        assert np.all(opened_next.persons[0].rrif.annual_minimum > 0.0)
        assert np.all(opened_next.persons[0].lif.annual_minimum > 0.0)

    @pytest.mark.parametrize(
        "birth_year_offset", [-4, 4], ids=["second_person_converts", "first_person_converts"]
    )
    def test_the_statutory_conversion_reaches_only_the_person_at_the_age(
        self, scenario, real_params, birth_year_offset
    ):
        """Each person's own December age decides; the other person's accounts are untouched."""
        _, statutory_age, _ = self._statutory(scenario, real_params)
        elected_age = scenario.policies[0].elections.rrif_conversion.age_years
        couple = self._couple_with_election(scenario, elected_age, birth_year_offset)
        persons = couple.household.persons
        converter = 0 if persons[0].birth_year < persons[1].birth_year else 1
        other = 1 - converter
        year = persons[converter].birth_year + statutory_age
        december_ages = [
            timeline.age_at_end_of_year(p.birth_year, p.birth_month, year) for p in persons
        ]
        assert december_ages[converter] == statutory_age
        assert december_ages[other] < statutory_age
        # Neither person is at the elected age, so only the statutory branch can act.
        assert elected_age not in december_ages

        state = self._state_at_year(couple, n_paths=2, year=year)
        opened = open_year(state, real_params)
        opening_rrsp = [p.rrsp.balance.copy() for p in opened.persons]
        opening_rrif = [p.rrif.balance.copy() for p in opened.persons]
        opening_lira = [p.lira.balance.copy() for p in opened.persons]
        opening_lif = [p.lif.balance.copy() for p in opened.persons]
        for i in (0, 1):
            assert np.all(opening_rrsp[i] > 0.0)
            assert np.all(opening_lira[i] > 0.0)

        closed = close_year(opened, real_params)
        converted = closed.persons[converter]
        np.testing.assert_allclose(converted.rrsp.balance, 0.0)
        np.testing.assert_allclose(converted.rrif.balance, opening_rrsp[converter])
        assert converted.rrif.opened_year == year
        np.testing.assert_allclose(converted.lira.balance, 0.0)
        np.testing.assert_allclose(converted.lif.balance, opening_lira[converter])
        assert converted.lif.opened_year == year
        untouched = closed.persons[other]
        np.testing.assert_allclose(untouched.rrsp.balance, opening_rrsp[other])
        np.testing.assert_allclose(untouched.rrif.balance, opening_rrif[other])
        assert untouched.rrif.opened_year is None
        np.testing.assert_allclose(untouched.lira.balance, opening_lira[other])
        np.testing.assert_allclose(untouched.lif.balance, opening_lif[other])
        assert untouched.lif.opened_year is None

    def test_elected_conversion_moves_only_the_elected_fraction(self, scenario, real_params):
        birth_year, _, statutory_year = self._statutory(scenario, real_params)
        elected_age = scenario.policies[0].elections.rrif_conversion.age_years
        elected_fraction = scenario.policies[0].elections.rrif_conversion.fraction
        year = birth_year + elected_age
        assert year < statutory_year

        lif_age = self._lif_age(scenario, real_params)
        # The elected age falls below the jurisdiction's LIF conversion deadline age.
        assert elected_age < lif_age

        state = self._state_at_year(scenario, n_paths=2, year=year)
        opened = open_year(state, real_params)
        opening_rrsp = opened.persons[0].rrsp.balance.copy()
        assert np.all(opening_rrsp > 0.0)
        opening_lira = opened.persons[0].lira.balance.copy()
        assert np.all(opening_lira > 0.0)

        closed = close_year(opened, real_params)
        person = closed.persons[0]
        np.testing.assert_allclose(person.rrsp.balance, opening_rrsp * (1 - elected_fraction))
        np.testing.assert_allclose(person.rrif.balance, opening_rrsp * elected_fraction)
        np.testing.assert_allclose(person.lira.balance, opening_lira)
        np.testing.assert_allclose(person.lif.balance, 0.0)
        assert person.rrif.opened_year == year

    @pytest.mark.parametrize("offset", [-1, 1])
    def test_the_elected_conversion_fires_in_no_other_year(self, scenario, real_params, offset):
        birth_year, _, statutory_year = self._statutory(scenario, real_params)
        elected_age = scenario.policies[0].elections.rrif_conversion.age_years
        year = birth_year + elected_age + offset
        assert year < statutory_year

        state = self._state_at_year(scenario, n_paths=2, year=year)
        opened = open_year(state, real_params)
        opening_rrsp = opened.persons[0].rrsp.balance.copy()
        assert np.all(opening_rrsp > 0.0)

        closed = close_year(opened, real_params)
        person = closed.persons[0]
        np.testing.assert_allclose(person.rrsp.balance, opening_rrsp)
        assert person.rrif.opened_year is None

    def test_an_election_below_the_start_years_december_age_converts_nothing(
        self, scenario, real_params
    ):
        person = scenario.household.persons[0]
        december_age = timeline.age_at_end_of_year(
            person.birth_year, person.birth_month, scenario.start_year
        )
        elected_age = december_age - 1
        assert elected_age < real_params.rrif.number("conversion_age_years")

        elections = scenario.policies[0].elections
        modified_elections = elections.model_copy(
            update={
                "rrif_conversion": elections.rrif_conversion.model_copy(
                    update={"age_years": elected_age}
                )
            }
        )
        modified_policy = scenario.policies[0].model_copy(update={"elections": modified_elections})
        modified_scenario = scenario.model_copy(update={"policies": (modified_policy,)})

        state = self._state_at_year(modified_scenario, n_paths=2, year=scenario.start_year)
        opened = open_year(state, real_params)
        opening_rrsp = opened.persons[0].rrsp.balance.copy()
        assert np.all(opening_rrsp > 0.0)

        closed = close_year(opened, real_params)
        person_state = closed.persons[0]
        np.testing.assert_allclose(person_state.rrsp.balance, opening_rrsp)
        assert person_state.rrif.opened_year is None

    def test_an_election_at_the_start_years_december_age_converts_at_the_first_close(
        self, scenario, real_params
    ):
        """The control proving the test above reaches the branch that decides."""
        person = scenario.household.persons[0]
        december_age = timeline.age_at_end_of_year(
            person.birth_year, person.birth_month, scenario.start_year
        )
        assert december_age < real_params.rrif.number("conversion_age_years")

        elections = scenario.policies[0].elections
        modified_elections = elections.model_copy(
            update={
                "rrif_conversion": elections.rrif_conversion.model_copy(
                    update={"age_years": december_age}
                )
            }
        )
        modified_policy = scenario.policies[0].model_copy(update={"elections": modified_elections})
        modified_scenario = scenario.model_copy(update={"policies": (modified_policy,)})
        fraction = modified_elections.rrif_conversion.fraction
        assert 0 < fraction < 1

        state = self._state_at_year(modified_scenario, n_paths=2, year=scenario.start_year)
        opened = open_year(state, real_params)
        opening_rrsp = opened.persons[0].rrsp.balance.copy()
        assert np.all(opening_rrsp > 0.0)

        closed = close_year(opened, real_params)
        person_state = closed.persons[0]
        np.testing.assert_allclose(person_state.rrsp.balance, opening_rrsp * (1 - fraction))
        np.testing.assert_allclose(person_state.rrif.balance, opening_rrsp * fraction)
        assert person_state.rrif.opened_year == scenario.start_year

    def test_an_election_below_both_persons_december_ages_converts_for_neither(
        self, scenario, real_params
    ):
        _, opened = self._opened_couple(scenario, real_params, -1)
        opening_rrsp = [p.rrsp.balance.copy() for p in opened.persons]

        closed = close_year(opened, real_params)
        for i in (0, 1):
            np.testing.assert_allclose(closed.persons[i].rrsp.balance, opening_rrsp[i])
            assert closed.persons[i].rrif.opened_year is None

    def test_an_election_at_the_younger_persons_december_age_converts_only_them(
        self, scenario, real_params
    ):
        """The control proving the test above reaches the branch that decides."""
        couple, opened = self._opened_couple(scenario, real_params, 0)
        fraction = couple.policies[0].elections.rrif_conversion.fraction
        assert 0 < fraction < 1
        opening_rrsp = [p.rrsp.balance.copy() for p in opened.persons]
        opening_rrif = [p.rrif.balance.copy() for p in opened.persons]

        closed = close_year(opened, real_params)
        younger = closed.persons[1]
        np.testing.assert_allclose(younger.rrsp.balance, opening_rrsp[1] * (1 - fraction))
        np.testing.assert_allclose(younger.rrif.balance, opening_rrsp[1] * fraction)
        assert younger.rrif.opened_year == scenario.start_year
        older = closed.persons[0]
        np.testing.assert_allclose(older.rrsp.balance, opening_rrsp[0])
        np.testing.assert_allclose(older.rrif.balance, opening_rrif[0])
        assert older.rrif.opened_year is None

    def test_when_both_would_fire_in_the_same_year_only_the_statutory_conversion_applies(
        self, scenario, real_params
    ):
        _, statutory_age, statutory_year = self._statutory(scenario, real_params)
        elections = scenario.policies[0].elections
        conflicting_elections = elections.model_copy(
            update={
                "rrif_conversion": elections.rrif_conversion.model_copy(
                    update={"age_years": statutory_age}
                )
            }
        )
        conflicting_policy = scenario.policies[0].model_copy(
            update={"elections": conflicting_elections}
        )
        conflicting_scenario = scenario.model_copy(update={"policies": (conflicting_policy,)})
        # Elected and statutory now fall on the same age; the elected fraction differs from
        # the statutory one, so which one fired is unambiguous from the balance.
        assert conflicting_elections.rrif_conversion.fraction != 1.0

        state = self._state_at_year(conflicting_scenario, n_paths=2, year=statutory_year)
        opened = open_year(state, real_params)
        opening_rrsp = opened.persons[0].rrsp.balance.copy()
        assert np.all(opening_rrsp > 0.0)

        closed = close_year(opened, real_params)
        person = closed.persons[0]
        np.testing.assert_allclose(person.rrsp.balance, 0.0)
        np.testing.assert_allclose(person.rrif.balance, opening_rrsp)


class TestRrspRoomZeroAfterConversion:
    """No RRSP contribution room, granted or spent, past the statutory conversion deadline --
    both the January accrual (``open_year``) and what phase 8 lets a contribution reach.
    """

    def _statutory_age_and_year(self, scenario, real_params):
        person = scenario.household.persons[0]
        rrif_age = int(real_params.rrif.number("conversion_age_years"))
        return person.birth_year, rrif_age, person.birth_year + rrif_age

    # January-1-of-year state building is exactly TestRrspAndLiraConversion._state_at_year;
    # reused from there rather than copied a second time in this file.
    _state_at_year = TestRrspAndLiraConversion._state_at_year

    def _state_at_year_month(self, scenario, n_paths, year, month):
        state = build_initial_state(scenario, n_paths=n_paths)
        month_index = 12 * (year - scenario.start_year) + (month - 1)
        spending_monthly = select_spending_level(state.spending_schedule, year)
        return updated(
            state,
            year=year,
            month=month,
            month_index=month_index,
            spending_monthly=spending_monthly,
        )

    def test_room_is_zeroed_at_the_statutory_close(self, scenario, real_params):
        _, _, statutory_year = self._statutory_age_and_year(scenario, real_params)
        state = self._state_at_year(scenario, n_paths=2, year=statutory_year)
        opened = open_year(state, real_params)
        closed = close_year(opened, real_params)
        np.testing.assert_allclose(closed.persons[0].rrsp.room, 0.0)

    def test_an_elected_partial_conversion_at_a_younger_age_keeps_room(self, scenario, real_params):
        birth_year, _, statutory_year = self._statutory_age_and_year(scenario, real_params)
        elected_age = scenario.policies[0].elections.rrif_conversion.age_years
        year = birth_year + elected_age
        assert year < statutory_year

        state = self._state_at_year(scenario, n_paths=2, year=year)
        opened = open_year(state, real_params)
        room_before = opened.persons[0].rrsp.room.copy()
        # Preconditions: room is positive before the close, and there is an actual RRIF
        # balance to convert into -- without both, "room unchanged" would pass just as
        # vacuously if the elected conversion had silently failed to fire at all.
        assert np.all(room_before > 0.0)
        assert np.all(opened.persons[0].rrif.balance == 0.0)

        closed = close_year(opened, real_params)
        assert np.all(closed.persons[0].rrif.balance > 0.0), "the elected conversion must fire"
        np.testing.assert_allclose(closed.persons[0].rrsp.room, room_before)
        assert np.all(closed.persons[0].rrsp.room > 0.0)

    def test_room_still_granted_the_january_of_the_conversion_year(self, scenario, real_params):
        """The legal side of ``open_year``'s ``age_end <= conversion_age_years`` boundary.
        ``age_end == conversion_age_years`` is still legal -- the deadline is 31 December of
        that year -- so employment income earned the year before still grants room this
        January. A narrower ``<`` in its place would zero this January's accrual too, one
        year early.
        """
        _, rrif_age, statutory_year = self._statutory_age_and_year(scenario, real_params)
        person_a = scenario.household.persons[0]
        assert (
            timeline.age_at_end_of_year(person_a.birth_year, person_a.birth_month, statutory_year)
            == rrif_age
        )
        extended_employment = person_a.employment[0].model_copy(
            update={"to_year": statutory_year - 1}
        )
        new_person = person_a.model_copy(update={"employment": (extended_employment,)})
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})
        new_scenario = scenario.model_copy(update={"household": new_household})

        state = self._state_at_year(new_scenario, n_paths=1, year=statutory_year)
        person = state.persons[0]
        employment_income = np.full(1, extended_employment.annual)
        state = updated(
            state,
            persons=(updated(person, income=updated(person.income, employment=employment_income)),),
        )
        room_before = state.persons[0].rrsp.room.copy()

        opened = open_year(state, real_params)
        expected_accrual = rrsp.room_accrued(employment_income, real_params.rrif, state.month_index)
        assert np.all(expected_accrual > 0.0), "the accrual must be nonzero to test anything"
        expected_room = room_before * nominal_carry_factor(real_params.inflation_rate) + (
            expected_accrual
        )
        np.testing.assert_allclose(opened.persons[0].rrsp.room, expected_room)

    def test_no_room_granted_the_january_after_conversion_even_with_employment_income(
        self, scenario, real_params
    ):
        _, _, statutory_year = self._statutory_age_and_year(scenario, real_params)
        person_a = scenario.household.persons[0]
        # Employment runs through the statutory conversion year itself, so the January
        # after it would otherwise accrue room on a full year of earnings -- the control
        # this test needs before it can show that no room is granted regardless.
        extended_employment = person_a.employment[0].model_copy(update={"to_year": statutory_year})
        new_person = person_a.model_copy(update={"employment": (extended_employment,)})
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})
        new_scenario = scenario.model_copy(update={"household": new_household})

        next_year = statutory_year + 1
        state = self._state_at_year(new_scenario, n_paths=1, year=next_year)
        # A full year of employment income already sits in the ledger, exactly as it would
        # after twelve months of the statutory conversion year running through the step.
        person = state.persons[0]
        employment_income = np.full(1, extended_employment.annual)
        state = updated(
            state,
            persons=(updated(person, income=updated(person.income, employment=employment_income)),),
        )
        room_before = state.persons[0].rrsp.room.copy()
        assert np.all(room_before > 0.0), "room must start positive for the zeroing to be visible"

        # Room past the conversion age is fixed at zero, not merely left un-accrued on top
        # of whatever it eroded to.
        opened = open_year(state, real_params)
        np.testing.assert_allclose(opened.persons[0].rrsp.room, 0.0)

    def test_phase8_contribution_boundary_both_sides(
        self, scenario, real_params, market, withdrawal_order
    ):
        birth_year, rrif_age, statutory_year = self._statutory_age_and_year(scenario, real_params)
        birth_month = scenario.household.persons[0].birth_month
        assert timeline.age_at_end_of_year(birth_year, birth_month, statutory_year) == rrif_age
        assert (
            timeline.age_at_end_of_year(birth_year, birth_month, statutory_year + 1) == rrif_age + 1
        )

        n_assets = len(market.asset_class_names)
        month_returns = np.zeros((n_assets, 1))
        contribution_amount = 1000.0

        # Side 1: age_end == conversion_age -- still legal, contributed in full.
        state_legal = self._state_at_year_month(scenario, n_paths=1, year=statutory_year, month=6)
        assert np.all(state_legal.persons[0].rrsp.room > 0.0)
        script_legal = {
            state_legal.month_index: (
                Transfer(
                    person_index=0,
                    from_kind="cash",
                    to_kind="rrsp",
                    amount=np.full(1, contribution_amount),
                ),
            )
        }
        policy_legal = ScriptedPolicy(state_legal.elections, withdrawal_order, script_legal)
        _, record_legal = advance_month_traced(
            state_legal, month_returns, policy_legal, market, real_params
        )
        assert record_legal.contributions[0].rrsp[0] == pytest.approx(contribution_amount)

        # Side 2: age_end == conversion_age + 1 -- illegal, contributes zero, even with room
        # forced positive in the state handed to the step.
        state_illegal = self._state_at_year_month(
            scenario, n_paths=1, year=statutory_year + 1, month=6
        )
        person = state_illegal.persons[0]
        person_with_room = updated(person, rrsp=updated(person.rrsp, room=np.full(1, 25000.0)))
        state_illegal = updated(state_illegal, persons=(person_with_room,))
        script_illegal = {
            state_illegal.month_index: (
                Transfer(
                    person_index=0,
                    from_kind="cash",
                    to_kind="rrsp",
                    amount=np.full(1, contribution_amount),
                ),
            )
        }
        policy_illegal = ScriptedPolicy(state_illegal.elections, withdrawal_order, script_illegal)
        _, record_illegal = advance_month_traced(
            state_illegal, month_returns, policy_illegal, market, real_params
        )
        assert record_illegal.contributions[0].rrsp[0] == pytest.approx(0.0)

    def test_room_is_zeroed_at_month_index_zero_for_someone_already_past_conversion(
        self, scenario, real_params
    ):
        """The scenario's own opening room, for someone already past the statutory
        conversion age at the run's start, is illegal state -- it is zeroed by the very
        first ``open_year`` (month index 0), not merely left to stop growing.
        """
        _, rrif_age, _ = self._statutory_age_and_year(scenario, real_params)
        person_a = scenario.household.persons[0]
        shifted_birth_year = scenario.start_year - (rrif_age + 1)
        new_person = person_a.model_copy(update={"birth_year": shifted_birth_year})
        assert (
            timeline.age_at_end_of_year(
                shifted_birth_year, person_a.birth_month, scenario.start_year
            )
            == rrif_age + 1
        )
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})
        new_scenario = scenario.model_copy(update={"household": new_household})

        state = build_initial_state(new_scenario, n_paths=1)
        assert state.month_index == 0
        assert np.all(state.persons[0].rrsp.room > 0.0), "room must start positive to be zeroed"

        opened = open_year(state, real_params)
        np.testing.assert_allclose(opened.persons[0].rrsp.room, 0.0)

    def test_room_kept_at_month_index_zero_for_age_equal_to_conversion_age(
        self, scenario, real_params
    ):
        """The other side of that boundary: age_end == conversion_age is still legal, so a
        scenario opening exactly there keeps its stated room, unchanged, at month index zero.
        """
        _, rrif_age, _ = self._statutory_age_and_year(scenario, real_params)
        person_a = scenario.household.persons[0]
        shifted_birth_year = scenario.start_year - rrif_age
        new_person = person_a.model_copy(update={"birth_year": shifted_birth_year})
        assert (
            timeline.age_at_end_of_year(
                shifted_birth_year, person_a.birth_month, scenario.start_year
            )
            == rrif_age
        )
        new_household = scenario.household.model_copy(update={"persons": (new_person,)})
        new_scenario = scenario.model_copy(update={"household": new_household})

        state = build_initial_state(new_scenario, n_paths=1)
        assert state.month_index == 0
        room_before = state.persons[0].rrsp.room.copy()
        assert np.all(room_before > 0.0)

        opened = open_year(state, real_params)
        np.testing.assert_allclose(opened.persons[0].rrsp.room, room_before)


class TestNetIncomePairShift:
    """The pair shifts at each close from that year's assessed net income."""

    def test_the_pair_shifts_from_the_years_net_income_at_each_close(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        policy = DoNothingPolicy(opening_state.elections, withdrawal_order)
        recorder = RecordingPolicy(policy)
        run(opening_state, recorder, draws, market, real_params)

        state_jan_year0 = recorder.calls[0][0]
        state_jan_year1 = recorder.calls[12][0]

        year0_record = state_jan_year1.history[-1]
        assert year0_record.year == opening_state.year

        person_before = state_jan_year0.persons[0]
        person_after = state_jan_year1.persons[0]

        np.testing.assert_allclose(person_after.prior_year_net_income, year0_record.net_income[0])
        np.testing.assert_allclose(
            person_after.net_income_two_years_prior, person_before.prior_year_net_income
        )

    def test_the_pair_and_the_record_take_net_income_after_the_oas_repayment(
        self, scenario, real_params
    ):
        state = build_initial_state(scenario, n_paths=1)
        person = state.persons[0]
        threshold = real_params.oas.annual_amount("recovery_tax.threshold_annual", 0)
        ledger = updated(
            person.income,
            employment=np.full(1, 2.0 * threshold),
            oas=np.full(1, 5_000.0),  # SYNTHETIC: an arbitrary OAS amount.
        )
        person = updated(person, income=ledger)
        state = updated(state, persons=(person,))

        assessment = household_assessment(state, real_params)[0]
        assert np.all(assessment.oas_repayment > 0)
        assert not np.allclose(assessment.net_income_after_repayment, assessment.net_income)

        closed = close_year(state, real_params)

        np.testing.assert_allclose(
            closed.persons[0].prior_year_net_income, assessment.net_income_after_repayment
        )
        np.testing.assert_allclose(
            closed.history[-1].net_income[0], assessment.net_income_after_repayment
        )


class TestNetIncomePairFreeze:
    """The pair shifts at the close of the last calendar year a person was alive in, and freezes at
    every close after. Three deaths cover the January boundary (a death drawn for January of year Y
    must not shift the pair at Y's close), the ordinary mid-year case, and a December death.

    Mutations checked, each caught by at least one of the three tests: changing the mask's
    ``>`` to ``>=`` makes the close of the death year itself shift the pair for the January
    death, which that test's second assertion catches; comparing against December of the
    closing year (``state.month_index``) instead of January (``january_month_index``) gives
    the same answer as the correct mask for the January death but a different one for
    the mid-year and December deaths, which those two tests' first assertions catch.
    """

    OLD_PRIOR = 1_000.0  # SYNTHETIC: distinguishes "shifted" (line 23600 of an empty ledger,
    OLD_TWO_YEARS = 500.0  # zero) from "frozen" (these two figures, unchanged).

    def _closed_at(self, scenario, real_params, year, death_month_index):
        state = build_initial_state(scenario, n_paths=1)
        month_index = 12 * (year - scenario.start_year) + 11  # December of `year`.
        person = updated(
            state.persons[0],
            death_month_index=np.full(1, death_month_index, dtype=np.int64),
            prior_year_net_income=np.full(1, self.OLD_PRIOR),
            net_income_two_years_prior=np.full(1, self.OLD_TWO_YEARS),
        )
        state = updated(state, persons=(person,), year=year, month=12, month_index=month_index)
        return close_year(state, real_params)

    def test_shifts_in_the_last_year_alive_and_freezes_at_every_close_after(
        self, scenario, real_params
    ):
        # A death drawn for January of 2028: alive throughout 2026-2027, not alive at all in
        # 2028 or after (engine.core.mortality.death_month_index is the first month the
        # person is not alive).
        death_month_index = 12 * (2028 - scenario.start_year)

        closed_2027 = self._closed_at(scenario, real_params, 2027, death_month_index)
        person_2027 = closed_2027.persons[0]
        np.testing.assert_allclose(person_2027.prior_year_net_income, 0.0)
        np.testing.assert_allclose(person_2027.net_income_two_years_prior, self.OLD_PRIOR)

        closed_2028 = self._closed_at(scenario, real_params, 2028, death_month_index)
        person_2028 = closed_2028.persons[0]
        np.testing.assert_allclose(person_2028.prior_year_net_income, self.OLD_PRIOR)
        np.testing.assert_allclose(person_2028.net_income_two_years_prior, self.OLD_TWO_YEARS)

        closed_2029 = self._closed_at(scenario, real_params, 2029, death_month_index)
        person_2029 = closed_2029.persons[0]
        np.testing.assert_allclose(person_2029.prior_year_net_income, self.OLD_PRIOR)
        np.testing.assert_allclose(person_2029.net_income_two_years_prior, self.OLD_TWO_YEARS)

    def test_a_mid_year_death_shifts_at_that_years_close_and_freezes_the_year_after(
        self, scenario, real_params
    ):
        # Alive January through May 2028; June is the first month not alive.
        death_month_index = 12 * (2028 - scenario.start_year) + 5

        closed_2028 = self._closed_at(scenario, real_params, 2028, death_month_index)
        person_2028 = closed_2028.persons[0]
        np.testing.assert_allclose(person_2028.prior_year_net_income, 0.0)
        np.testing.assert_allclose(person_2028.net_income_two_years_prior, self.OLD_PRIOR)

        closed_2029 = self._closed_at(scenario, real_params, 2029, death_month_index)
        person_2029 = closed_2029.persons[0]
        np.testing.assert_allclose(person_2029.prior_year_net_income, self.OLD_PRIOR)
        np.testing.assert_allclose(person_2029.net_income_two_years_prior, self.OLD_TWO_YEARS)

    def test_a_december_death_shifts_at_that_years_close_and_freezes_the_year_after(
        self, scenario, real_params
    ):
        # Alive January through November 2028; December is the first month not alive.
        death_month_index = 12 * (2028 - scenario.start_year) + 11

        closed_2028 = self._closed_at(scenario, real_params, 2028, death_month_index)
        person_2028 = closed_2028.persons[0]
        np.testing.assert_allclose(person_2028.prior_year_net_income, 0.0)
        np.testing.assert_allclose(person_2028.net_income_two_years_prior, self.OLD_PRIOR)

        closed_2029 = self._closed_at(scenario, real_params, 2029, death_month_index)
        person_2029 = closed_2029.persons[0]
        np.testing.assert_allclose(person_2029.prior_year_net_income, self.OLD_PRIOR)
        np.testing.assert_allclose(person_2029.net_income_two_years_prior, self.OLD_TWO_YEARS)


class TestGisBandIndicator:
    """The GIS-band indicator, item 6 of close_year."""

    def test_example_household_never_in_band(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        policy = DoNothingPolicy(opening_state.elections, withdrawal_order)
        recorder = RecordingPolicy(policy)
        run(opening_state, recorder, draws, market, real_params)

        final_state = recorder.calls[-1][0]
        assert len(final_state.history) > 0
        for record in final_state.history:
            for band in record.gis_band:
                assert not np.any(band)

    def _closed_with_income(self, scenario, real_params, *, cpp=0.0, oas=0.0):
        state = build_initial_state(scenario, n_paths=1)
        person = state.persons[0]
        ledger = updated(person.income, cpp=np.full(1, cpp), oas=np.full(1, oas))
        person = updated(person, income=ledger)
        state = updated(state, persons=(person,))
        return close_year(state, real_params)

    def test_a_synthetic_low_income_household_is_in_band_with_an_above_threshold_control(
        self, scenario, real_params
    ):
        threshold = real_params.oas.annual_amount(
            "gis.band_thresholds.single_testable_income_annual", 0
        )
        below = self._closed_with_income(scenario, real_params, cpp=threshold * 0.5, oas=1.0)
        assert below.history[-1].gis_band[0][0]

        above = self._closed_with_income(scenario, real_params, cpp=threshold * 2.0, oas=1.0)
        assert not above.history[-1].gis_band[0][0]

    def test_a_living_person_with_zero_oas_is_false_even_when_the_household_is_in_band(
        self, scenario, real_params
    ):
        threshold = real_params.oas.annual_amount(
            "gis.band_thresholds.single_testable_income_annual", 0
        )
        closed = self._closed_with_income(scenario, real_params, cpp=threshold * 0.5, oas=0.0)
        assert not closed.history[-1].gis_band[0][0]

    def test_has_spouse_is_false_for_a_household_of_one(self, scenario, real_params):
        # The midpoint sits above the single threshold and below the couple one, so only
        # which threshold applies decides the result.
        single = real_params.oas.annual_amount(
            "gis.band_thresholds.single_testable_income_annual", 0
        )
        couple = real_params.oas.annual_amount(
            "gis.band_thresholds.couple_combined_testable_income_annual", 0
        )
        assert single < couple  # precondition

        closed = self._closed_with_income(scenario, real_params, cpp=(single + couple) / 2, oas=1.0)
        assert not closed.history[-1].gis_band[0][0]


class TestGisBandForACouple:
    """The couple threshold applies while both are alive; once one has died,
    the survivor is tested on the single threshold, hand-set via ``alive``.
    """

    @pytest.fixture
    def couple_scenario(self):
        return load_scenario(REPO_ROOT / "scenarios" / "late_life_couple.yaml")

    @pytest.fixture
    def couple_real_params(self, couple_scenario):
        return real_year(
            load_year(couple_scenario.start_year), couple_scenario.assumptions.inflation
        )

    def _closed(self, couple_scenario, couple_real_params, *, both_alive):
        state = build_initial_state(couple_scenario, n_paths=1)
        person_a, person_b = state.persons
        single = couple_real_params.oas.annual_amount(
            "gis.band_thresholds.single_testable_income_annual", 0
        )
        couple = couple_real_params.oas.annual_amount(
            "gis.band_thresholds.couple_combined_testable_income_annual", 0
        )
        assert single < couple  # precondition
        gap = couple - single
        # SYNTHETIC incomes, built from the thresholds themselves rather than hard-coded:
        # person a alone sits above the single threshold, but the couple's combined testable
        # income sits below the couple threshold, so the couple/single distinction is the
        # only thing that changes whether the household is in band.
        person_a = updated(
            person_a,
            income=updated(
                person_a.income, cpp=np.full(1, single + gap / 3), oas=np.full(1, 1_000.0)
            ),
        )
        person_b = updated(
            person_b,
            income=updated(person_b.income, cpp=np.full(1, gap / 3), oas=np.full(1, 800.0)),
            alive=np.full(1, both_alive),
            # alive=False requires a real death_month_index (never DEATH_NOT_DRAWN); the exact
            # month does not matter here, only that person_b is not alive at this close.
            death_month_index=np.full(1, 0 if not both_alive else DEATH_NOT_DRAWN, dtype=np.int64),
        )
        state = updated(state, persons=(person_a, person_b))
        return close_year(state, couple_real_params)

    def test_the_couple_threshold_applies_while_both_are_alive(
        self, couple_scenario, couple_real_params
    ):
        closed = self._closed(couple_scenario, couple_real_params, both_alive=True)
        assert closed.history[-1].gis_band[0][0]
        assert closed.history[-1].gis_band[1][0]

    def test_the_single_threshold_applies_once_the_spouse_has_died(
        self, couple_scenario, couple_real_params
    ):
        closed = self._closed(couple_scenario, couple_real_params, both_alive=False)
        assert not closed.history[-1].gis_band[0][0]
        assert not closed.history[-1].gis_band[1][0]

    def test_a_dead_person_receiving_oas_is_false_even_when_the_survivor_is_in_band(
        self, couple_scenario, couple_real_params
    ):
        """The alive gate, isolated: unlike the case above, the survivor alone is low enough
        income to be in band here, so a mask that dropped the ``alive`` gate would wrongly
        mark the dead spouse (who still has ``oas > 0`` on record) true as well.
        """
        state = build_initial_state(couple_scenario, n_paths=1)
        person_a, person_b = state.persons
        threshold = couple_real_params.oas.annual_amount(
            "gis.band_thresholds.single_testable_income_annual", 0
        )
        person_a = updated(
            person_a,
            income=updated(
                person_a.income, cpp=np.full(1, threshold * 0.5), oas=np.full(1, 1_000.0)
            ),
        )
        person_b = updated(
            person_b,
            income=updated(person_b.income, cpp=np.full(1, threshold * 0.1), oas=np.full(1, 800.0)),
            alive=np.full(1, False),
            death_month_index=np.full(1, 0, dtype=np.int64),
        )
        state = updated(state, persons=(person_a, person_b))
        closed = close_year(state, couple_real_params)

        assert closed.history[-1].gis_band[0][0]
        assert not closed.history[-1].gis_band[1][0]

    def test_the_dead_spouse_drops_out_of_the_income_sum(self, couple_scenario, couple_real_params):
        """Removing the ``np.where(alive, …, 0.0)`` filter from the income sum would pull
        the dead spouse's income back into the household total, pushing the
        survivor-alone testable income -- below the single threshold on its own -- above it.
        """
        single = couple_real_params.oas.annual_amount(
            "gis.band_thresholds.single_testable_income_annual", 0
        )
        state = build_initial_state(couple_scenario, n_paths=1)
        person_a, person_b = state.persons
        person_a = updated(
            person_a,
            income=updated(person_a.income, cpp=np.full(1, 0.9 * single), oas=np.full(1, 1_000.0)),
        )
        person_b = updated(
            person_b,
            income=updated(person_b.income, cpp=np.full(1, 0.2 * single), oas=np.full(1, 800.0)),
            alive=np.full(1, False),
            death_month_index=np.full(1, 0, dtype=np.int64),
        )
        state = updated(state, persons=(person_a, person_b))
        closed = close_year(state, couple_real_params)

        assert closed.history[-1].gis_band[0][0]
        assert not closed.history[-1].gis_band[1][0]

    def test_the_dead_spouse_drops_out_of_the_oas_sum(self, couple_scenario, couple_real_params):
        """Removing the ``np.where(alive, …, 0.0)`` filter from the OAS sum would pull the dead
        spouse's OAS back into the household total, subtracting it from the household's
        combined income and pulling the survivor's testable income from above the single
        threshold to below it.
        """
        single = couple_real_params.oas.annual_amount(
            "gis.band_thresholds.single_testable_income_annual", 0
        )
        state = build_initial_state(couple_scenario, n_paths=1)
        person_a, person_b = state.persons
        person_a = updated(
            person_a,
            income=updated(person_a.income, cpp=np.full(1, 1.1 * single), oas=np.full(1, 1_000.0)),
        )
        person_b = updated(
            person_b,
            income=updated(person_b.income, cpp=np.full(1, 0.0), oas=np.full(1, 0.2 * single)),
            alive=np.full(1, False),
            death_month_index=np.full(1, 0, dtype=np.int64),
        )
        state = updated(state, persons=(person_a, person_b))
        closed = close_year(state, couple_real_params)

        assert not closed.history[-1].gis_band[0][0]
        assert not closed.history[-1].gis_band[1][0]

    def test_a_spouse_who_died_during_the_year_is_out_of_the_household_at_the_close(
        self, couple_scenario, couple_real_params
    ):
        """Pins the human's decision that the GIS band uses ``alive`` at the close, not
        "alive at any point in the year" as the net-income pair does.
        """
        state = build_initial_state(couple_scenario, n_paths=1)
        state = updated(state, month=12, month_index=11)
        person_a, person_b = state.persons
        single = couple_real_params.oas.annual_amount(
            "gis.band_thresholds.single_testable_income_annual", 0
        )
        couple = couple_real_params.oas.annual_amount(
            "gis.band_thresholds.couple_combined_testable_income_annual", 0
        )
        assert 1.4 * single > couple  # precondition: the sum including b clears the threshold
        person_a = updated(
            person_a,
            income=updated(person_a.income, cpp=np.full(1, 0.9 * single), oas=np.full(1, 1_000.0)),
        )
        person_b = updated(
            person_b,
            income=updated(person_b.income, cpp=np.full(1, 0.5 * single), oas=np.full(1, 800.0)),
            alive=np.full(1, False),
            # Died in June of the start year (alive January to May); death_month_index is the
            # first month the person is not alive.
            death_month_index=np.full(1, 5, dtype=np.int64),
        )
        state = updated(state, persons=(person_a, person_b))
        closed = close_year(state, couple_real_params)

        assert closed.history[-1].gis_band[0][0]
        assert not closed.history[-1].gis_band[1][0]

    def test_the_single_threshold_applies_when_the_first_person_has_died(
        self, couple_scenario, couple_real_params
    ):
        """Mirrors ``test_the_single_threshold_applies_once_the_spouse_has_died`` with the
        roles swapped: person a, not person b, is the one who has died.
        """
        state = build_initial_state(couple_scenario, n_paths=1)
        person_a, person_b = state.persons
        single = couple_real_params.oas.annual_amount(
            "gis.band_thresholds.single_testable_income_annual", 0
        )
        couple = couple_real_params.oas.annual_amount(
            "gis.band_thresholds.couple_combined_testable_income_annual", 0
        )
        assert single < couple  # precondition
        gap = couple - single
        person_a = updated(
            person_a,
            income=updated(person_a.income, cpp=np.full(1, gap / 3), oas=np.full(1, 800.0)),
            alive=np.full(1, False),
            death_month_index=np.full(1, 0, dtype=np.int64),
        )
        person_b = updated(
            person_b,
            income=updated(
                person_b.income, cpp=np.full(1, single + gap / 3), oas=np.full(1, 1_000.0)
            ),
        )
        state = updated(state, persons=(person_a, person_b))
        closed = close_year(state, couple_real_params)

        assert not closed.history[-1].gis_band[0][0]
        assert not closed.history[-1].gis_band[1][0]

    def test_a_living_spouse_with_zero_oas_is_false_while_the_other_is_in_band(
        self, couple_scenario, couple_real_params
    ):
        """Isolates the per-person OAS gate: being alive in a household that is in band is
        not enough on its own -- the person must also draw OAS.
        """
        state = build_initial_state(couple_scenario, n_paths=1)
        person_a, person_b = state.persons
        single = couple_real_params.oas.annual_amount(
            "gis.band_thresholds.single_testable_income_annual", 0
        )
        couple = couple_real_params.oas.annual_amount(
            "gis.band_thresholds.couple_combined_testable_income_annual", 0
        )
        assert single < couple  # precondition
        gap = couple - single
        person_a = updated(
            person_a,
            income=updated(person_a.income, cpp=np.full(1, single + gap / 3), oas=np.full(1, 0.0)),
        )
        person_b = updated(
            person_b,
            income=updated(person_b.income, cpp=np.full(1, gap / 3), oas=np.full(1, 800.0)),
        )
        state = updated(state, persons=(person_a, person_b))
        closed = close_year(state, couple_real_params)

        assert not closed.history[-1].gis_band[0][0]
        assert closed.history[-1].gis_band[1][0]

    def test_both_spouses_oas_is_subtracted_from_the_combined_income(
        self, couple_scenario, couple_real_params
    ):
        """Which OAS amounts are subtracted decides the result: keeping only one of the two
        would push the combined testable income above the couple threshold.
        """
        state = build_initial_state(couple_scenario, n_paths=1)
        person_a, person_b = state.persons
        couple = couple_real_params.oas.annual_amount(
            "gis.band_thresholds.couple_combined_testable_income_annual", 0
        )
        person_a = updated(
            person_a,
            income=updated(
                person_a.income, cpp=np.full(1, 0.45 * couple), oas=np.full(1, 0.15 * couple)
            ),
        )
        person_b = updated(
            person_b,
            income=updated(
                person_b.income, cpp=np.full(1, 0.45 * couple), oas=np.full(1, 0.15 * couple)
            ),
        )
        state = updated(state, persons=(person_a, person_b))

        assessments = household_assessment(state, couple_real_params)
        for assessment in assessments:
            np.testing.assert_allclose(assessment.oas_repayment, 0.0)

        closed = close_year(state, couple_real_params)

        assert closed.history[-1].gis_band[0][0]
        assert closed.history[-1].gis_band[1][0]


class TestYearRecordFields:
    """after_tax_net_worth is strictly below net_worth (#36: the terminal-return
    arithmetic, hypothetically, on a living household), and the per-person tuples are
    sized to the household.
    """

    def test_after_tax_net_worth_is_below_net_worth_and_tuples_match_persons(
        self, scenario, real_params
    ):
        state = build_initial_state(scenario, n_paths=2)
        opened = open_year(state, real_params)
        closed = close_year(opened, real_params)

        record = closed.history[-1]
        assert np.all(record.after_tax_net_worth < record.net_worth)
        assert record.after_tax_net_worth is not record.net_worth

        # Recompute #36 section 6's formula independently, via person_assessment
        # directly, rather than trusting close_year's own private helper.
        year = closed.year
        january_month_index = closed.month_index - (closed.month - 1)

        hyp_total = np.zeros(closed.n_paths, dtype=np.float64)
        gross = closed.cash.balance.copy()
        for person in closed.persons:
            deemed = updated(
                person.income,
                rrif_lif_withdrawals=(
                    person.income.rrif_lif_withdrawals
                    + person.rrsp.balance
                    + person.rrif.balance
                    + person.lira.balance
                    + person.lif.balance
                ),
                capital_gains=(
                    person.income.capital_gains + taxable.deemed_disposition(person.taxable)
                ),
            )
            age_end = timeline.age_at_end_of_year(person.birth_year, person.birth_month, year)
            assessment = person_assessment(
                deemed,
                age_end,
                0,
                0,
                closed.province,
                real_params,
                january_month_index,
                died_in_year=True,
            )
            hyp_total = hyp_total + (assessment.total - person.income.remitted)
            gross = (
                gross
                + person.rrsp.balance
                + person.rrif.balance
                + person.lira.balance
                + person.lif.balance
                + person.tfsa.balance
                + person.taxable.balance
            )

        expected = gross - hyp_total
        np.testing.assert_allclose(record.after_tax_net_worth, expected)
        assert len(record.net_income) == len(closed.persons)
        assert len(record.gis_band) == len(closed.persons)


# =============================================================================
# #36: death, the terminal return, and the result object
# =============================================================================


def _recompute_terminal_total(persons, year, january_month_index, province, real_params):
    """Independently recompute the terminal-return arithmetic of ``resolve_deaths`` step 3
    (and ``close_year``'s ``after_tax_net_worth``), from ``person_assessment`` directly.

    Returns ``(household_total, assessments)``.
    """
    n_paths = persons[0].alive.shape[0]
    total = np.zeros(n_paths, dtype=np.float64)
    assessments = []
    for person in persons:
        deemed = updated(
            person.income,
            rrif_lif_withdrawals=(
                person.income.rrif_lif_withdrawals
                + person.rrsp.balance
                + person.rrif.balance
                + person.lira.balance
                + person.lif.balance
            ),
            capital_gains=(
                person.income.capital_gains + taxable.deemed_disposition(person.taxable)
            ),
        )
        age_end = timeline.age_at_end_of_year(person.birth_year, person.birth_month, year)
        died_in_year = person.death_month_index >= january_month_index
        assessment = person_assessment(
            deemed,
            age_end,
            np.zeros(n_paths),
            np.zeros(n_paths),
            province,
            real_params,
            january_month_index,
            died_in_year=died_in_year,
        )
        assessments.append(assessment)
        total = total + assessment.total
    return total, tuple(assessments)


def _gross_wealth(state):
    gross = state.cash.balance.copy()
    for person in state.persons:
        gross = (
            gross
            + person.rrsp.balance
            + person.rrif.balance
            + person.lira.balance
            + person.lif.balance
            + person.tfsa.balance
            + person.taxable.balance
        )
    return gross


class TestFirstDeathOnTheCouple:
    """Requirement 2: the first death's month-k semantics, on the couple."""

    #: April of the second year: (2027 - 2026) * 12 + (4 - 1) -- a full January has
    #: already run by then.
    DEATH_MONTH_INDEX = 15
    MARCH_MONTH_INDEX = 14

    def test_first_death(
        self,
        couple_scenario,
        couple_opening_state,
        couple_draws,
        couple_market,
        couple_real_params,
        couple_withdrawal_order,
    ):
        state = _force_death(couple_opening_state, 0, self.DEATH_MONTH_INDEX)
        policy = DoNothingPolicy(state.elections, couple_withdrawal_order)
        recorder = RecordingPolicy(policy)
        result = run(state, recorder, couple_draws, couple_market, couple_real_params, trace_path=0)

        for record in result.trace:
            _assert_identities_hold(record)

        march = result.trace[self.MARCH_MONTH_INDEX]
        april = result.trace[self.DEATH_MONTH_INDEX]

        # a's OAS: paid in March, stops from April.
        assert march.context.inflows[0].oas[0] > 0.0
        assert april.context.inflows[0].oas[0] == 0.0

        # April's rolled_out[a] equals March's balances_close[a], for every kind.
        march_a_balances = march.balances_close[0]
        april_rolled_a = april.context.rolled_out[0]
        for field in ("rrsp", "rrif", "lira", "lif", "tfsa", "taxable"):
            np.testing.assert_allclose(
                getattr(april_rolled_a, field), getattr(march_a_balances, field)
            )

        # rolled_acb[a]: independently recomputed from March's own taxable
        # distributions, the only source of ACB change between March's close and
        # April's open (no January erosion falls between them).
        march_state_before_growth = recorder.calls[self.MARCH_MONTH_INDEX][0]
        a_taxable_before_march_growth = march_state_before_growth.persons[0].taxable
        weighted_yields = couple_market.weighted_yields("taxable")
        interest, dividends, gains = taxable.distributions_monthly(
            a_taxable_before_march_growth.balance, weighted_yields
        )
        expected_acb_then = a_taxable_before_march_growth.acb + interest + dividends + gains
        np.testing.assert_allclose(april.context.rolled_acb[0], expected_acb_then)

        # After April: a's balances are all zero; b holds a's pre-roll balance plus its
        # own, each grown by April's own return for that kind.
        april_a_close = april.balances_close[0]
        for field in ("rrsp", "rrif", "lira", "lif", "tfsa", "taxable"):
            np.testing.assert_allclose(getattr(april_a_close, field), 0.0)

        march_b_balances = march.balances_close[1]
        april_b_close = april.balances_close[1]
        april_returns = couple_draws.real_returns[self.DEATH_MONTH_INDEX]
        for field in ("rrsp", "rrif", "lira", "lif", "tfsa", "taxable"):
            pre_growth = getattr(march_b_balances, field) + getattr(april_rolled_a, field)
            r = couple_market.weights(field) @ april_returns
            expected = pre_growth * (1 + r)
            np.testing.assert_allclose(getattr(april_b_close, field), expected, rtol=1e-6)

        # Spending: the full level in March, scaled by the survivor share from April.
        np.testing.assert_allclose(march.context.spending, state.spending_monthly)
        np.testing.assert_allclose(
            april.context.spending, state.spending_monthly * state.spending_survivor_share
        )

        # b's DB pension survivor share: 0 in March, survivor_share * a's own from April.
        a_pension = state.persons[0].pensions[0]
        factor = (
            1.0
            if a_pension.indexed
            else unindexed_factor(
                couple_real_params.inflation_rate,
                self.DEATH_MONTH_INDEX - max(a_pension.start_month_index, 0),
            )
        )
        assert march.context.inflows[1].db_pension_survivor[0] == 0.0
        np.testing.assert_allclose(
            april.context.inflows[1].db_pension_survivor,
            a_pension.survivor_share * a_pension.monthly_amount * factor,
        )

        # Requirement 12: the trace sums to the result, year by year, including the
        # first-death year.
        for year_index, year in enumerate(result.years):
            year_start = (int(year) - couple_scenario.start_year) * 12
            year_end = year_start + 12
            monthly = [
                r.context.spending - r.spending_cut
                for r in result.trace
                if year_start <= r.context.month_index < year_end
            ]
            np.testing.assert_allclose(
                sum(monthly), result.spending_achieved[year_index], atol=0.005
            )


#: SYNTHETIC SCENARIO INPUT, not a tax parameter: a made-up DB pension amount, sized so
#: that the survivor share of it clears both the basic personal amount and the age
#: amount at real 2026 params, so payroll withholding on the inherited stream is
#: genuinely non-zero. The couple's own, real pension amount does not do this --
#: withholding.payroll_withholding_monthly on it comes back exactly 0.
SYNTHETIC_PENSION_MONTHLY = 20_000.0


class TestSurvivorStreamWithholding:
    """Payroll withholding really does reach the inherited DB stream, on a synthetic
    pension amount large enough that the withholding is non-zero.
    """

    #: April of the second year, as in TestFirstDeathOnTheCouple.
    DEATH_MONTH_INDEX = 15
    MARCH_MONTH_INDEX = 14

    def test_withholding_on_the_inherited_stream(
        self,
        couple_opening_state,
        couple_draws,
        couple_market,
        couple_real_params,
        couple_withdrawal_order,
    ):
        n_paths = couple_opening_state.n_paths
        person_a = couple_opening_state.persons[0]
        synthetic_pension = updated(
            person_a.pensions[0], monthly_amount=np.full(n_paths, SYNTHETIC_PENSION_MONTHLY)
        )
        person_a = updated(person_a, pensions=(synthetic_pension,))
        state = updated(couple_opening_state, persons=(person_a, couple_opening_state.persons[1]))
        state = _force_death(state, 0, self.DEATH_MONTH_INDEX)

        policy = DoNothingPolicy(state.elections, couple_withdrawal_order)
        result = run(state, policy, couple_draws, couple_market, couple_real_params, trace_path=0)

        march = result.trace[self.MARCH_MONTH_INDEX]
        april = result.trace[self.DEATH_MONTH_INDEX]

        # Guard: b has no employment or pension of their own, so March's withholding
        # (before the first death) is 0.
        assert march.context.payroll_withholding[1][0] == 0.0

        stream = april.context.inflows[1].db_pension_survivor
        age_end_b = timeline.age_at_end_of_year(
            state.persons[1].birth_year, state.persons[1].birth_month, 2027
        )
        january_month_index = 12  # January 2027
        expected = withholding.payroll_withholding_monthly(
            stream, age_end_b, False, state.province, couple_real_params, january_month_index
        )
        assert np.all(expected > 0.0)  # guard: the synthetic amount really clears the credits

        np.testing.assert_allclose(april.context.payroll_withholding[1], expected)


class TestCppSurvivorBothDirections:
    """Requirement 3: the CPP survivor increment's capped and uncapped branches."""

    DEATH_MONTH_INDEX = 15

    def _run_forced(
        self,
        deceased_index,
        couple_opening_state,
        couple_draws,
        couple_market,
        couple_real_params,
        couple_withdrawal_order,
    ):
        state = _force_death(couple_opening_state, deceased_index, self.DEATH_MONTH_INDEX)
        recorder = RecordingPolicy(DoNothingPolicy(state.elections, couple_withdrawal_order))
        result = run(state, recorder, couple_draws, couple_market, couple_real_params, trace_path=0)
        return result, recorder

    def test_a_dies_uncapped_branch(
        self,
        couple_scenario,
        couple_opening_state,
        couple_draws,
        couple_market,
        couple_real_params,
        couple_withdrawal_order,
    ):
        share = couple_real_params.cpp.number("survivor.share_at_65_plus")
        combined_max = couple_real_params.cpp.amount(
            "survivor.combined_maximum_monthly", self.DEATH_MONTH_INDEX
        )
        base_a = float(couple_scenario.household.persons[0].cpp.in_pay_monthly)
        own_b = float(couple_scenario.household.persons[1].cpp.in_pay_monthly)
        assert share * base_a < combined_max - own_b  # guard: the uncapped branch

        result, _recorder = self._run_forced(
            0,
            couple_opening_state,
            couple_draws,
            couple_market,
            couple_real_params,
            couple_withdrawal_order,
        )
        record = result.trace[self.DEATH_MONTH_INDEX]

        np.testing.assert_allclose(record.context.inflows[1].cpp_survivor, share * base_a)
        total = record.context.inflows[1].cpp + record.context.inflows[1].cpp_survivor
        assert np.all(total <= combined_max + 1e-6)

    def test_b_dies_capped_branch(
        self,
        couple_scenario,
        couple_opening_state,
        couple_draws,
        couple_market,
        couple_real_params,
        couple_withdrawal_order,
    ):
        share = couple_real_params.cpp.number("survivor.share_at_65_plus")
        combined_max = couple_real_params.cpp.amount(
            "survivor.combined_maximum_monthly", self.DEATH_MONTH_INDEX
        )
        base_b = float(couple_scenario.household.persons[1].cpp.in_pay_monthly)
        own_a = float(couple_scenario.household.persons[0].cpp.in_pay_monthly)
        assert share * base_b > combined_max - own_a  # guard: the capped branch

        result, recorder = self._run_forced(
            1,
            couple_opening_state,
            couple_draws,
            couple_market,
            couple_real_params,
            couple_withdrawal_order,
        )
        record = result.trace[self.DEATH_MONTH_INDEX]

        total = record.context.inflows[0].cpp + record.context.inflows[0].cpp_survivor
        np.testing.assert_allclose(total, combined_max)

        # The b-to-a direction: a's balances hold b's pre-roll balances, kind by
        # kind, and the ACB, mirroring TestFirstDeathOnTheCouple's a-to-b check.
        march = result.trace[self.DEATH_MONTH_INDEX - 1]
        march_b_balances = march.balances_close[1]
        april_rolled_b = record.context.rolled_out[1]
        for field in ("rrsp", "rrif", "lira", "lif", "tfsa", "taxable"):
            np.testing.assert_allclose(
                getattr(april_rolled_b, field), getattr(march_b_balances, field)
            )

        march_state_before_growth = recorder.calls[self.DEATH_MONTH_INDEX - 1][0]
        b_taxable_before_march_growth = march_state_before_growth.persons[1].taxable
        weighted_yields = couple_market.weighted_yields("taxable")
        interest, dividends, gains = taxable.distributions_monthly(
            b_taxable_before_march_growth.balance, weighted_yields
        )
        expected_acb_then = b_taxable_before_march_growth.acb + interest + dividends + gains
        np.testing.assert_allclose(record.context.rolled_acb[1], expected_acb_then)

        # The check above compares rolled_out against march's balances, but
        # resolve_deaths computes rolled_out *before* the six rollover calls run, so a
        # no-op'd rollover would still pass it. Check the move itself.
        # (a) the deceased's balances stay exactly 0, in the death month and after.
        for later_record in (record, *result.trace[self.DEATH_MONTH_INDEX + 1 :]):
            for field in ("rrsp", "rrif", "lira", "lif", "tfsa", "taxable"):
                np.testing.assert_allclose(getattr(later_record.balances_close[1], field), 0.0)

        assert np.all(record.context.rolled_out[1].rrif > 0.0)  # guard

        # (b) the survivor's balance really absorbed it: march's opening balance plus
        # what rolled in, less this month's own flows, grown by this month's return --
        # exactly _phase10_growth's rule, confirmed for taxable too (balance * (1 + r),
        # #19 decision 5: distributions_monthly's yield component and price_growth's
        # price-only component net out to the account's total return).
        returns_k = couple_draws.real_returns[self.DEATH_MONTH_INDEX]
        for field in ("rrsp", "rrif", "lira", "lif", "tfsa", "taxable"):
            march_a_balance = getattr(march.balances_close[0], field)
            rolled_in = getattr(record.context.rolled_out[1], field)
            if field == "lira":
                # ByKind has no lira field: LIRA takes no withdrawals and no
                # contributions target it, so every flow term is 0.
                zeros = np.zeros_like(march_a_balance)
                forced = withdrawn = floor_swept = contributed = zeros
            else:
                forced = getattr(record.context.forced_withdrawals[0], field)
                withdrawn = getattr(record.withdrawals[0], field)
                floor_swept = getattr(record.floor_withdrawals[0], field)
                contributed = getattr(record.contributions[0], field)
            pre_growth = (
                march_a_balance + rolled_in - forced - withdrawn - floor_swept + contributed
            )
            r = couple_market.weights(field) @ returns_k
            expected = pre_growth * (1 + r)
            np.testing.assert_allclose(
                getattr(record.balances_close[0], field), expected, rtol=1e-6
            )

    def test_survivor_increment_uses_base_pension_monthly_without_an_in_pay_amount(
        self,
        couple_opening_state,
        couple_draws,
        couple_market,
        couple_real_params,
        couple_withdrawal_order,
    ):
        """The deceased's base can also come from ``contributory_history`` rather
        than an ``in_pay_monthly`` amount already in pay.
        """
        contributory_history = 0.8
        person_a = couple_opening_state.persons[0]
        person_a = updated(
            person_a,
            cpp=updated(
                person_a.cpp,
                in_pay_monthly=None,
                contributory_history=contributory_history,
                start_age_months=780,
            ),
        )
        state = updated(couple_opening_state, persons=(person_a, couple_opening_state.persons[1]))
        state = _force_death(state, 0, self.DEATH_MONTH_INDEX)

        policy = DoNothingPolicy(state.elections, couple_withdrawal_order)
        recorder = RecordingPolicy(policy)
        result = run(state, recorder, couple_draws, couple_market, couple_real_params, trace_path=0)
        record = result.trace[self.DEATH_MONTH_INDEX]

        share = couple_real_params.cpp.number("survivor.share_at_65_plus")
        combined_max = couple_real_params.cpp.amount(
            "survivor.combined_maximum_monthly", self.DEATH_MONTH_INDEX
        )
        own_b = couple_opening_state.persons[1].cpp.in_pay_monthly
        base_a = cpp_mod.base_pension_monthly(
            contributory_history, self.DEATH_MONTH_INDEX, couple_real_params.cpp
        )
        assert np.all(share * base_a < combined_max - own_b)  # guard: the uncapped branch

        # The deceased's own monthly_amount, carried into the death month, differs
        # from the unadjusted base -- her actual age at the run's opening already
        # exceeds the CPP start window, so her own pension carries a late-start bonus
        # the survivor formula (L19) does not apply. A mutant reading
        # person_j.cpp.monthly_amount instead of calling base_pension_monthly here
        # would therefore fail the assertion below.
        carried_monthly_amount = (
            recorder.calls[self.DEATH_MONTH_INDEX - 1][0].persons[0].cpp.monthly_amount
        )
        assert not np.allclose(carried_monthly_amount, base_a)

        expected = np.minimum(share * base_a, combined_max - own_b)
        assert np.all(expected > 0.0)  # guard: not vacuously 0
        np.testing.assert_allclose(record.context.inflows[1].cpp_survivor, expected)


class TestSecondDeathTerminalReturn:
    """Requirements 4, 6, 7, 8: the terminal-return arithmetic at the second death,
    exercised directly against ``resolve_deaths`` on hand-built states -- the six
    rollover functions and ``TestFirstDeathOnTheCouple`` already cover the first
    death's own rollover mechanics.
    """

    def test_couple_second_death_a_then_b(self, couple_scenario, couple_real_params):
        n = 1
        state = build_initial_state(couple_scenario, n_paths=n)
        state = updated(state, month=7, month_index=6)
        person_a, person_b = state.persons

        # a died earlier (month 3) and has already rolled fully into b.
        person_a = updated(
            person_a,
            alive=np.zeros(n, dtype=bool),
            death_month_index=np.full(n, 3, dtype=np.int64),
            rrif=updated(person_a.rrif, balance=np.zeros(n)),
            tfsa=updated(person_a.tfsa, balance=np.zeros(n)),
            taxable=updated(person_a.taxable, balance=np.zeros(n), acb=np.zeros(n)),
            lif=updated(person_a.lif, balance=np.zeros(n)),
        )
        person_b = updated(
            person_b,
            alive=np.ones(n, dtype=bool),
            death_month_index=np.full(n, 6, dtype=np.int64),
            income=updated(
                person_b.income,
                rrif_lif_withdrawals=np.full(n, 5_000.0),
                capital_gains=np.full(n, 2_000.0),
                remitted=np.full(n, 1_000.0),
            ),
            balance_owing=np.full(n, 500.0),
        )
        state = updated(
            state, persons=(person_a, person_b), cash=CashState(balance=np.full(n, 3_000.0))
        )

        (
            new_state,
            _rolled_out,
            _rolled_acb,
            terminal_assessment,
            cash_to_estate,
            _terminal_assessments,
        ) = resolve_deaths(state, couple_real_params)

        assert np.all(np.isfinite(new_state.estate_after_tax))
        gross = _gross_wealth(state)
        assert np.all(new_state.estate_after_tax < gross)

        expected_total, _ = _recompute_terminal_total(
            (person_a, person_b), state.year, 0, state.province, couple_real_params
        )
        balance_owing_total = person_a.balance_owing + person_b.balance_owing
        remitted_total = person_a.income.remitted + person_b.income.remitted
        expected_estate = gross - expected_total - balance_owing_total + remitted_total

        np.testing.assert_allclose(terminal_assessment, expected_total)
        np.testing.assert_allclose(new_state.estate_after_tax, expected_estate)
        np.testing.assert_allclose(cash_to_estate, state.cash.balance)
        assert not np.any(new_state.depleted)

        for person in new_state.persons:
            for account in (person.rrsp, person.rrif, person.lira, person.lif, person.tfsa):
                np.testing.assert_allclose(account.balance, 0.0)
            np.testing.assert_allclose(person.taxable.balance, 0.0)
            np.testing.assert_allclose(person.taxable.acb, 0.0)
            np.testing.assert_allclose(person.balance_owing, 0.0)
            for field in dataclasses.fields(person.income):
                np.testing.assert_allclose(getattr(person.income, field.name), 0.0)
        np.testing.assert_allclose(new_state.cash.balance, 0.0)

    def test_terminal_assessments_are_masked_per_path(self, couple_scenario, couple_real_params):
        """#63: ``terminal_assessments`` is masked per path, not just per household --
        the mask that zeroes a finished-elsewhere or still-both-alive path must fire on
        every field of every person's ``Assessment``, not merely on the household total.
        """
        n = 3
        state = build_initial_state(couple_scenario, n_paths=n)
        state = updated(state, month=7, month_index=6)
        person_a, person_b = state.persons
        # path 0: a died earlier, b dies now -> finishes this month.
        # path 1: both alive.
        # path 2: both died earlier and the path already finished.
        person_a = updated(
            person_a,
            alive=np.array([False, True, False]),
            death_month_index=np.array([3, 100, 2], dtype=np.int64),
        )
        person_b = updated(
            person_b,
            alive=np.array([True, True, False]),
            death_month_index=np.array([6, 100, 3], dtype=np.int64),
            income=updated(person_b.income, rrif_lif_withdrawals=np.full(n, 5_000.0)),
        )
        state = updated(
            state, persons=(person_a, person_b), estate_after_tax=np.array([np.nan, np.nan, 0.0])
        )

        (
            _new_state,
            _rolled_out,
            _rolled_acb,
            terminal_assessment,
            _cash_to_estate,
            terminal_assessments,
        ) = resolve_deaths(state, couple_real_params)

        alive_after_step1 = tuple(
            updated(p, alive=p.death_month_index > state.month_index) for p in state.persons
        )
        _oracle_total, oracle = _recompute_terminal_total(
            alive_after_step1, state.year, 0, state.province, couple_real_params
        )

        for person_oracle in oracle:
            assert np.all(person_oracle.total[1:] > 0.0)

        for i, person_oracle in enumerate(oracle):
            for field in dataclasses.fields(Assessment):
                actual = getattr(terminal_assessments[i], field.name)
                expected = getattr(person_oracle, field.name)
                assert actual[0] == expected[0]
                assert actual[1] == 0.0
                assert actual[2] == 0.0

        assert terminal_assessment[1] == 0.0
        assert terminal_assessment[2] == 0.0

    def test_example_only_death(self, scenario, real_params):
        n = 1
        state = build_initial_state(scenario, n_paths=n)
        state = updated(state, month=7, month_index=6)
        person = state.persons[0]
        person = updated(
            person,
            alive=np.ones(n, dtype=bool),
            death_month_index=np.full(n, 6, dtype=np.int64),
            income=updated(
                person.income,
                rrif_lif_withdrawals=np.full(n, 8_000.0),
                capital_gains=np.full(n, 1_500.0),
                remitted=np.full(n, 500.0),
            ),
            balance_owing=np.full(n, 200.0),
        )
        state = updated(state, persons=(person,), cash=CashState(balance=np.full(n, 4_000.0)))

        (
            new_state,
            _rolled_out,
            _rolled_acb,
            terminal_assessment,
            cash_to_estate,
            _terminal_assessments,
        ) = resolve_deaths(state, real_params)

        assert np.all(np.isfinite(new_state.estate_after_tax))
        gross = _gross_wealth(state)
        assert np.all(new_state.estate_after_tax < gross)

        expected_total, _ = _recompute_terminal_total(
            (person,), state.year, 0, state.province, real_params
        )
        expected_estate = gross - expected_total - person.balance_owing + person.income.remitted
        np.testing.assert_allclose(terminal_assessment, expected_total)
        np.testing.assert_allclose(new_state.estate_after_tax, expected_estate)
        np.testing.assert_allclose(cash_to_estate, state.cash.balance)
        assert not np.any(new_state.depleted)

    def test_simultaneous_death(self, couple_scenario, couple_real_params):
        n = 1
        state = build_initial_state(couple_scenario, n_paths=n)
        state = updated(state, month=7, month_index=6)
        person_a, person_b = state.persons
        person_a = updated(
            person_a, alive=np.ones(n, dtype=bool), death_month_index=np.full(n, 6, dtype=np.int64)
        )
        person_b = updated(
            person_b, alive=np.ones(n, dtype=bool), death_month_index=np.full(n, 6, dtype=np.int64)
        )
        state = updated(state, persons=(person_a, person_b))

        (
            new_state,
            rolled_out,
            rolled_acb,
            terminal_assessment,
            cash_to_estate,
            _terminal_assessments,
        ) = resolve_deaths(state, couple_real_params)

        # No rollover: both masks are false when both die in the same month.
        for i in range(2):
            for field in _account_amounts_tuple(rolled_out[i]):
                np.testing.assert_allclose(field, 0.0)
            np.testing.assert_allclose(rolled_acb[i], 0.0)

        # Both persons' own (unrolled) registered balances enter the terminal assessment.
        expected_total, _ = _recompute_terminal_total(
            (person_a, person_b), state.year, 0, state.province, couple_real_params
        )
        np.testing.assert_allclose(terminal_assessment, expected_total)

        gross = _gross_wealth(state)
        balance_owing_total = person_a.balance_owing + person_b.balance_owing
        remitted_total = person_a.income.remitted + person_b.income.remitted
        expected_estate = gross - expected_total - balance_owing_total + remitted_total
        np.testing.assert_allclose(new_state.estate_after_tax, expected_estate)
        np.testing.assert_allclose(cash_to_estate, state.cash.balance)

    def test_same_year_first_death_includes_the_part_year_ledger(
        self, couple_scenario, couple_real_params
    ):
        n = 1
        state = build_initial_state(couple_scenario, n_paths=n)
        state = updated(state, month=9, month_index=8)  # September 2026
        person_a, person_b = state.persons
        # a died in March (month 2) and has already rolled into b; a's own ledger,
        # frozen at death, has not been reset by an intervening January -- both deaths
        # fall in 2026.
        person_a = updated(
            person_a,
            alive=np.zeros(n, dtype=bool),
            death_month_index=np.full(n, 2, dtype=np.int64),
            income=updated(person_a.income, rrif_lif_withdrawals=np.full(n, 50_000.0)),
            rrif=updated(person_a.rrif, balance=np.zeros(n)),
            tfsa=updated(person_a.tfsa, balance=np.zeros(n)),
            taxable=updated(person_a.taxable, balance=np.zeros(n), acb=np.zeros(n)),
            lif=updated(person_a.lif, balance=np.zeros(n)),
        )
        person_b = updated(
            person_b, alive=np.ones(n, dtype=bool), death_month_index=np.full(n, 8, dtype=np.int64)
        )
        state = updated(state, persons=(person_a, person_b))

        (
            _new_state,
            _rolled_out,
            _rolled_acb,
            terminal_assessment,
            _cash_to_estate,
            _terminal_assessments,
        ) = resolve_deaths(state, couple_real_params)

        expected_total, _ = _recompute_terminal_total(
            (person_a, person_b), state.year, 0, state.province, couple_real_params
        )
        np.testing.assert_allclose(terminal_assessment, expected_total)

        # Guard: a's part-year ledger really is included -- zeroing it changes the total.
        zeroed_a = updated(
            person_a, income=updated(person_a.income, rrif_lif_withdrawals=np.zeros(n))
        )
        zeroed_total, _ = _recompute_terminal_total(
            (zeroed_a, person_b), state.year, 0, state.province, couple_real_params
        )
        assert not np.allclose(expected_total, zeroed_total)

    def test_january_edge_uses_an_empty_ledger(self, scenario, real_params):
        n = 1
        state = build_initial_state(scenario, n_paths=n)
        state = updated(state, month=1, month_index=12)  # January 2027
        person = state.persons[0]
        person = updated(
            person, alive=np.ones(n, dtype=bool), death_month_index=np.full(n, 12, dtype=np.int64)
        )
        state = updated(state, persons=(person,))

        (
            _new_state,
            _rolled_out,
            _rolled_acb,
            terminal_assessment,
            _cash_to_estate,
            _terminal_assessments,
        ) = resolve_deaths(state, real_params)

        deemed = updated(
            person.income,
            rrif_lif_withdrawals=(
                person.rrsp.balance + person.rrif.balance + person.lira.balance + person.lif.balance
            ),
            capital_gains=taxable.deemed_disposition(person.taxable),
        )
        age_end = timeline.age_at_end_of_year(person.birth_year, person.birth_month, state.year)
        expected = person_assessment(
            deemed,
            age_end,
            np.zeros(n),
            np.zeros(n),
            state.province,
            real_params,
            12,
            died_in_year=True,
        )
        np.testing.assert_allclose(terminal_assessment, expected.total)


class TestTerminalReturnCapitalLoss:
    """ITA 111(2) at a terminal return and at ``after_tax_net_worth``, where the
    taxable account's deemed gain is negative (``balance < acb``).
    """

    def test_resolve_deaths_applies_the_death_year_deduction(self, scenario, real_params):
        n = 1
        state = build_initial_state(scenario, n_paths=n)
        state = updated(state, month=7, month_index=6)
        person = state.persons[0]
        person = updated(
            person,
            alive=np.ones(n, dtype=bool),
            death_month_index=np.full(n, 6, dtype=np.int64),
            income=updated(person.income, rrif_lif_withdrawals=np.full(n, 60_000.0)),
            taxable=updated(person.taxable, balance=np.full(n, 50_000.0), acb=np.full(n, 90_000.0)),
        )
        state = updated(state, persons=(person,))

        # Guard: the deemed gain really is negative.
        assert np.all(taxable.deemed_disposition(person.taxable) < 0.0)

        (
            _new_state,
            _rolled_out,
            _rolled_acb,
            terminal_assessment,
            _cash_to_estate,
            _terminal_assessments,
        ) = resolve_deaths(state, real_params)

        expected_true, _ = _recompute_terminal_total(
            (person,), state.year, 0, state.province, real_params
        )
        np.testing.assert_allclose(terminal_assessment, expected_true)

        # died_in_year=False (via the same recompute, forced) must give a different total
        # -- the deduction is really reaching the terminal assessment.
        deemed = updated(
            person.income,
            rrif_lif_withdrawals=(
                person.income.rrif_lif_withdrawals
                + person.rrsp.balance
                + person.rrif.balance
                + person.lira.balance
                + person.lif.balance
            ),
            capital_gains=(
                person.income.capital_gains + taxable.deemed_disposition(person.taxable)
            ),
        )
        age_end = timeline.age_at_end_of_year(person.birth_year, person.birth_month, state.year)
        expected_false = person_assessment(
            deemed,
            age_end,
            np.zeros(n),
            np.zeros(n),
            state.province,
            real_params,
            0,
            died_in_year=False,
        )
        assert not np.allclose(terminal_assessment, expected_false.total)

    def test_after_tax_net_worth_applies_the_death_year_deduction(self, scenario, real_params):
        n = 2
        loss_accounts = scenario.household.persons[0].accounts.model_copy(
            update={
                "taxable": scenario.household.persons[0].accounts.taxable.model_copy(
                    update={"balance": 50_000.0, "acb": 90_000.0}
                )
            }
        )
        loss_person = scenario.household.persons[0].model_copy(update={"accounts": loss_accounts})
        household = scenario.household.model_copy(update={"persons": (loss_person,)})
        loss_scenario = scenario.model_copy(update={"household": household})

        state = build_initial_state(loss_scenario, n_paths=n)
        opened = open_year(state, real_params)
        closed = close_year(opened, real_params)
        record = closed.history[-1]
        person = closed.persons[0]

        # Guard: the deemed gain really is negative.
        assert np.all(taxable.deemed_disposition(person.taxable) < 0.0)

        age_end = timeline.age_at_end_of_year(person.birth_year, person.birth_month, closed.year)
        deemed = updated(
            person.income,
            rrif_lif_withdrawals=(
                person.income.rrif_lif_withdrawals
                + person.rrsp.balance
                + person.rrif.balance
                + person.lira.balance
                + person.lif.balance
            ),
            capital_gains=(
                person.income.capital_gains + taxable.deemed_disposition(person.taxable)
            ),
        )
        assessment_true = person_assessment(
            deemed, age_end, 0, 0, closed.province, real_params, 0, died_in_year=True
        )
        assessment_false = person_assessment(
            deemed, age_end, 0, 0, closed.province, real_params, 0, died_in_year=False
        )
        assert not np.allclose(assessment_true.total, assessment_false.total)

        gross = closed.cash.balance + (
            person.rrsp.balance
            + person.rrif.balance
            + person.lira.balance
            + person.lif.balance
            + person.tfsa.balance
            + person.taxable.balance
        )
        expected = gross - (assessment_true.total - person.income.remitted)
        np.testing.assert_allclose(record.after_tax_net_worth, expected)


class TestSecondDeathViaRun:
    """Requirement 13 (second half): cash identities across a full forced two-death run,
    and the run's own timing of ``estate_after_tax``.
    """

    A_DEATH_MONTH_INDEX = 15
    B_DEATH_MONTH_INDEX = 40

    def test_cash_identities_and_estate_timing(
        self,
        couple_opening_state,
        couple_draws,
        couple_market,
        couple_real_params,
        couple_withdrawal_order,
    ):
        state = _force_death(couple_opening_state, 0, self.A_DEATH_MONTH_INDEX)
        state = _force_death(state, 1, self.B_DEATH_MONTH_INDEX)
        recorder = RecordingPolicy(DoNothingPolicy(state.elections, couple_withdrawal_order))
        result = run(state, recorder, couple_draws, couple_market, couple_real_params, trace_path=0)

        # The run stops at the first December on or after the final death, not at
        # the death month itself.
        december_index = (self.B_DEATH_MONTH_INDEX // 12) * 12 + 11
        assert len(result.trace) == december_index + 1

        for record in result.trace:
            _assert_identities_hold(record)

        for m in range(self.B_DEATH_MONTH_INDEX):
            assert np.isnan(result.trace[m].estate_after_tax[0])

        final_record = result.trace[self.B_DEATH_MONTH_INDEX]
        assert np.isfinite(final_record.estate_after_tax[0])
        assert not final_record.depleted[0]
        np.testing.assert_allclose(
            final_record.context.cash_to_estate, final_record.context.cash_opening
        )
        assert np.all(np.isfinite(result.estate_after_tax))


class TestFinishedPathIsInert:
    """Requirement 5: after the final death, everything downstream is exactly zero."""

    #: Within the example beneficiary's education window (which opens well after this).
    FORCED_DEATH_MONTH_INDEX = 100

    def test_finished_path_is_inert(
        self, scenario, market, mortality, real_params, withdrawal_order
    ):
        n_paths = 4
        draws = build_draws(scenario, market, n_paths=n_paths, mortality=mortality)
        state = build_initial_state(scenario, n_paths=n_paths)
        state = draw_deaths(state, draws, mortality)

        person = state.persons[0]
        new_death = np.full(n_paths, draws.n_months - 2, dtype=np.int64)
        new_death[0] = self.FORCED_DEATH_MONTH_INDEX
        person = updated(person, death_month_index=new_death)
        state = updated(state, persons=(person,))

        recorder = RecordingPolicy(DoNothingPolicy(state.elections, withdrawal_order))
        result = run(state, recorder, draws, market, real_params, trace_path=0)

        death_month = self.FORCED_DEATH_MONTH_INDEX
        assert len(result.trace) > death_month + 12, (
            "the run must continue well past the forced death for this test to mean anything"
        )

        for record in result.trace[death_month:]:
            assert record.finished[0]
            assert not record.alive[0][0]
            assert np.all(record.context.spending == 0.0)
            assert np.all(_sum_arrays(record.context.education_costs) == 0.0)
            assert np.all(_sum_arrays(inflow.to_cash for inflow in record.context.inflows) == 0.0)
            assert np.all(_sum_by_kind(record.withdrawals) == 0.0)
            assert np.all(_sum_by_kind(record.contributions) == 0.0)
            assert np.all(_sum_by_kind(record.floor_withdrawals) == 0.0)
            assert np.all(record.depletion_deficit == 0.0)
            assert not record.depleted[0]
            # Every other flow the step can move is also 0 on a finished path.
            assert np.all(_sum_by_kind(record.context.forced_withdrawals) == 0.0)
            assert np.all(_sum_arrays(record.context.payroll_withholding) == 0.0)
            assert np.all(_sum_arrays(record.context.tax_settlement) == 0.0)
            assert np.all(_sum_arrays(record.context.education_draws) == 0.0)
            assert np.all(_sum_arrays(record.resp_contributions) == 0.0)
            assert np.all(_sum_arrays(record.wind_up_to_cash) == 0.0)

        pre = result.trace[death_month]
        for record in result.trace[death_month + 1 :]:
            np.testing.assert_allclose(record.cash_close, pre.cash_close)
            np.testing.assert_allclose(record.estate_after_tax, pre.estate_after_tax)
            for field in ("rrsp", "rrif", "lira", "lif", "tfsa", "taxable"):
                np.testing.assert_allclose(getattr(record.balances_close[0], field), 0.0)

        for resp_close in result.trace[death_month + 1].resp_close:
            np.testing.assert_allclose(resp_close, 0.0)

        death_year = scenario.start_year + death_month // 12
        year_mask = result.years >= death_year
        assert np.any(year_mask)
        np.testing.assert_allclose(result.net_worth[year_mask, 0], 0.0)
        np.testing.assert_allclose(result.tax_assessed[year_mask, 0], 0.0)
        later_year_mask = result.years > death_year
        if np.any(later_year_mask):
            np.testing.assert_allclose(result.spending_achieved[later_year_mask, 0], 0.0)

        # after_tax_net_worth (and, again, net_worth and tax_assessed) is 0 on every
        # YearRecord closed after the death year -- gathered incrementally from what
        # RecordingPolicy saw, since a single final snapshot is missing only the very
        # last year closed for the *other*, still-living paths, far beyond what is
        # checked here.
        history_seen = 0
        checked_any = False
        for state_seen, _context in recorder.calls:
            if len(state_seen.history) > history_seen:
                for year_record in state_seen.history[history_seen:]:
                    if year_record.year > death_year:
                        checked_any = True
                        assert year_record.after_tax_net_worth[0] == pytest.approx(0.0)
                        assert year_record.net_worth[0] == pytest.approx(0.0)
                        assert year_record.tax_assessed[0] == pytest.approx(0.0)
                history_seen = len(state_seen.history)
        assert checked_any  # guard: at least one such year was actually checked

    def test_a_scripted_transfer_on_the_finished_path_is_zero(
        self, scenario, market, mortality, real_params, withdrawal_order
    ):
        n_paths = 4
        draws = build_draws(scenario, market, n_paths=n_paths, mortality=mortality)
        state = build_initial_state(scenario, n_paths=n_paths)
        state = draw_deaths(state, draws, mortality)

        person = state.persons[0]
        new_death = np.full(n_paths, draws.n_months - 2, dtype=np.int64)
        new_death[0] = self.FORCED_DEATH_MONTH_INDEX
        person = updated(person, death_month_index=new_death)
        state = updated(state, persons=(person,))

        attempt_month = self.FORCED_DEATH_MONTH_INDEX + 5
        contribution_amount = np.full(n_paths, 1_000.0)
        # Larger than the contribution: cash can otherwise be negative going into
        # phase 8 (spending already drawn it down for the month), which would cap
        # the contribution at 0 for a reason that has nothing to do with being dead.
        withdrawal_amount = np.full(n_paths, 5_000.0)
        script = {
            attempt_month: (
                Transfer(
                    person_index=0, from_kind="cash", to_kind="tfsa", amount=contribution_amount
                ),
                Transfer(
                    person_index=0, from_kind="rrif", to_kind="cash", amount=withdrawal_amount
                ),
            )
        }
        policy = ScriptedPolicy(state.elections, withdrawal_order, script)
        result = run(state, policy, draws, market, real_params, trace_path=0)

        record = result.trace[attempt_month]
        assert record.finished[0]
        np.testing.assert_allclose(record.contributions[0].tfsa, 0.0)
        np.testing.assert_allclose(record.withdrawals[0].rrif, 0.0)

        # The same script on a live path (path 1, forced to the same long, safe death
        # as every other un-forced path here) really does move money, so the finished
        # path's zeros above are not vacuous.
        control_result = run(state, policy, draws, market, real_params, trace_path=1)
        control_record = control_result.trace[attempt_month]
        assert not control_record.finished[0]  # guard: this path is still alive
        assert np.all(control_record.contributions[0].tfsa > 0.0)
        assert np.all(control_record.withdrawals[0].rrif > 0.0)


class TestAliveInvariantAndEstateNan:
    """Requirements 9 and 10 (success criteria): ``alive`` tracks ``death_month_index``
    exactly, and ``estate_after_tax`` is NaN until, and only until, the second death.
    """

    N_PATHS = 64

    def _run(self, scenario, market, mortality, real_params, withdrawal_order, n_paths):
        draws = build_draws(scenario, market, n_paths=n_paths, mortality=mortality)
        state = build_initial_state(scenario, n_paths=n_paths)
        state = draw_deaths(state, draws, mortality)
        recorder = RecordingPolicy(DoNothingPolicy(state.elections, withdrawal_order))
        result = run(state, recorder, draws, market, real_params)
        return result, recorder

    def test_alive_invariant_on_the_example(
        self, scenario, market, mortality, real_params, withdrawal_order
    ):
        _result, recorder = self._run(
            scenario, market, mortality, real_params, withdrawal_order, n_paths=1
        )
        for state_seen, context in recorder.calls:
            for person in state_seen.persons:
                np.testing.assert_array_equal(
                    person.alive, person.death_month_index > context.month_index
                )

    def test_alive_invariant_and_estate_nan_on_the_seeded_couple(
        self,
        couple_scenario,
        couple_market,
        couple_mortality,
        couple_real_params,
        couple_withdrawal_order,
    ):
        result, recorder = self._run(
            couple_scenario,
            couple_market,
            couple_mortality,
            couple_real_params,
            couple_withdrawal_order,
            n_paths=self.N_PATHS,
        )
        opening_persons = recorder.calls[0][0].persons
        # Guard: at least one path has a first death strictly before a second.
        assert np.any(opening_persons[0].death_month_index != opening_persons[1].death_month_index)

        for state_seen, context in recorder.calls:
            for person in state_seen.persons:
                np.testing.assert_array_equal(
                    person.alive, person.death_month_index > context.month_index
                )
            expected_nan = context.month_index < np.maximum(
                state_seen.persons[0].death_month_index, state_seen.persons[1].death_month_index
            )
            np.testing.assert_array_equal(np.isnan(state_seen.estate_after_tax), expected_nan)

        assert np.all(np.isfinite(result.estate_after_tax))


def _meet_rrif_lif_minimum(state):
    """Withdraw exactly this year's RRIF/LIF minimum for every person, by hand.

    Stands in for phase 6 of ``advance_month``, which this file's other #36 tests reach
    through ``run`` rather than calling directly; here, ``open_year`` is followed straight
    by ``close_year`` with no month in between, so ``close_year``'s own assertion that the
    minimum has been met needs this first.
    """
    new_persons = []
    cash = state.cash
    for person in state.persons:
        minimum_rrif = person.rrif.annual_minimum
        minimum_lif = person.lif.annual_minimum
        new_rrif = updated(
            person.rrif,
            balance=person.rrif.balance - minimum_rrif,
            withdrawn_ytd=minimum_rrif,
        )
        new_lif = updated(
            person.lif, balance=person.lif.balance - minimum_lif, withdrawn_ytd=minimum_lif
        )
        new_income = updated(
            person.income,
            rrif_lif_withdrawals=person.income.rrif_lif_withdrawals + minimum_rrif + minimum_lif,
        )
        cash = CashState(balance=cash.balance + minimum_rrif + minimum_lif)
        new_persons.append(updated(person, rrif=new_rrif, lif=new_lif, income=new_income))
    return updated(state, persons=tuple(new_persons), cash=cash)


class TestAfterTaxNetWorthOnTheCouple:
    """Requirement 15: the couple's first December close matches the #36 section 6 formula."""

    def test_first_close_matches_the_formula(self, couple_scenario, couple_real_params):
        state = build_initial_state(couple_scenario, n_paths=2)
        opened = open_year(state, couple_real_params)
        opened = _meet_rrif_lif_minimum(opened)
        closed = close_year(opened, couple_real_params)

        record = closed.history[-1]
        assert np.all(record.after_tax_net_worth < record.net_worth)

        year = closed.year
        january_month_index = closed.month_index - (closed.month - 1)
        hyp_total = np.zeros(closed.n_paths, dtype=np.float64)
        gross = closed.cash.balance.copy()
        for person in closed.persons:
            deemed = updated(
                person.income,
                rrif_lif_withdrawals=(
                    person.income.rrif_lif_withdrawals
                    + person.rrsp.balance
                    + person.rrif.balance
                    + person.lira.balance
                    + person.lif.balance
                ),
                capital_gains=(
                    person.income.capital_gains + taxable.deemed_disposition(person.taxable)
                ),
            )
            age_end = timeline.age_at_end_of_year(person.birth_year, person.birth_month, year)
            assessment = person_assessment(
                deemed,
                age_end,
                0,
                0,
                closed.province,
                couple_real_params,
                january_month_index,
                died_in_year=True,
            )
            hyp_total = hyp_total + (assessment.total - person.income.remitted)
            gross = (
                gross
                + person.rrsp.balance
                + person.rrif.balance
                + person.lira.balance
                + person.lif.balance
                + person.tfsa.balance
                + person.taxable.balance
            )
        expected = gross - hyp_total
        np.testing.assert_allclose(record.after_tax_net_worth, expected)


class TestDeadPersonStopsParticipating:
    """A dead person's own account never receives a contribution, and an RESP
    wind-up after the subscriber's death credits the living spouse instead.
    """

    DEATH_MONTH_INDEX = 15
    CONTRIBUTION_MONTH_INDEX = 20

    def test_contribution_to_the_dead_person_is_capped_at_zero(
        self,
        couple_opening_state,
        couple_draws,
        couple_market,
        couple_real_params,
        couple_withdrawal_order,
    ):
        state = _force_death(couple_opening_state, 0, self.DEATH_MONTH_INDEX)
        n_paths = state.n_paths
        amount = np.full(n_paths, 500.0)
        script = {
            self.CONTRIBUTION_MONTH_INDEX: (
                Transfer(person_index=0, from_kind="cash", to_kind="tfsa", amount=amount),
                Transfer(person_index=1, from_kind="cash", to_kind="tfsa", amount=amount),
            )
        }
        policy = ScriptedPolicy(state.elections, couple_withdrawal_order, script)
        recorder = RecordingPolicy(policy)
        result = run(state, recorder, couple_draws, couple_market, couple_real_params, trace_path=0)

        record = result.trace[self.CONTRIBUTION_MONTH_INDEX]
        np.testing.assert_allclose(record.contributions[0].tfsa, 0.0)
        # Guard: the identical transfer to the living survivor really did move money.
        assert np.all(record.contributions[1].tfsa > 0.0)

    def test_wind_up_after_the_subscribers_death_credits_the_survivor(
        self,
        couple_opening_state,
        couple_draws,
        couple_market,
        couple_real_params,
        couple_withdrawal_order,
    ):
        n_paths = couple_opening_state.n_paths
        resp_state = RespState(
            contributions=np.zeros(n_paths),
            grants=np.zeros(n_paths),
            income=np.full(n_paths, 5_000.0),
            contributions_lifetime=np.zeros(n_paths),
            grants_lifetime=np.zeros(n_paths),
            grant_room=np.zeros(n_paths),
            grant_received_ytd=np.zeros(n_paths),
            contributed_ytd=np.zeros(n_paths),
            subscriber_index=0,
            education_start_month_index=4,
            education_months=1,
            education_monthly_cost=0.0,
            wound_up=np.zeros(n_paths, dtype=bool),
        )
        beneficiary = BeneficiaryState(
            beneficiary_id="child", birth_year=2015, birth_month=1, resp=resp_state
        )
        state = updated(couple_opening_state, beneficiaries=(beneficiary,))
        state = _force_death(state, 0, 3)  # a dies well before the wind-up (threshold 5)

        policy = DoNothingPolicy(state.elections, couple_withdrawal_order)
        recorder = RecordingPolicy(policy)
        run(state, recorder, couple_draws, couple_market, couple_real_params)

        # Month 6's phase-7 snapshot reflects month 5's wind-up (threshold = 4 + 1 = 5), by
        # which point the plan's value has grown for months 0-4 (contributions and grants
        # stay zero throughout, so growth compounds the income bucket alone -- the same
        # arithmetic as engine.accounts.resp.grow).
        r_resp = couple_market.weights("resp")
        expected_value = 5_000.0
        for m in range(5):
            expected_value *= 1 + r_resp @ couple_draws.real_returns[m]

        after_wind_up = recorder.calls[6][0]
        np.testing.assert_allclose(
            after_wind_up.persons[1].income.resp_accumulated_income, expected_value
        )
        np.testing.assert_allclose(after_wind_up.persons[0].income.resp_accumulated_income, 0.0)

    def test_wind_up_credits_the_subscriber_or_the_spouse_per_path(
        self,
        couple_scenario,
        couple_market,
        couple_mortality,
        couple_real_params,
        couple_withdrawal_order,
    ):
        """Both branches of the wind-up credit rule, side by side on two paths -- the
        subscriber's own death forced before the wind-up on path 0 only, so path 1
        exercises the living-subscriber branch.
        """
        n_paths = 2
        draws = build_draws(
            couple_scenario, couple_market, n_paths=n_paths, mortality=couple_mortality
        )
        state = build_initial_state(couple_scenario, n_paths=n_paths)
        state = draw_deaths(state, draws, couple_mortality)

        resp_state = RespState(
            contributions=np.zeros(n_paths),
            grants=np.zeros(n_paths),
            income=np.full(n_paths, 5_000.0),
            contributions_lifetime=np.zeros(n_paths),
            grants_lifetime=np.zeros(n_paths),
            grant_room=np.zeros(n_paths),
            grant_received_ytd=np.zeros(n_paths),
            contributed_ytd=np.zeros(n_paths),
            subscriber_index=0,
            education_start_month_index=4,
            education_months=1,
            education_monthly_cost=0.0,
            wound_up=np.zeros(n_paths, dtype=bool),
        )
        beneficiary = BeneficiaryState(
            beneficiary_id="child", birth_year=2015, birth_month=1, resp=resp_state
        )
        state = updated(state, beneficiaries=(beneficiary,))

        # a dies at month 3 (before the wind-up, threshold 5) on path 0 only; path 1
        # keeps a alive, forced explicitly (rather than left to the natural draw) so
        # the test cannot flake on an early natural death there.
        person_a = state.persons[0]
        forced_death = np.array([3, draws.n_months - 2], dtype=np.int64)
        person_a = updated(person_a, death_month_index=forced_death)
        state = updated(state, persons=(person_a, state.persons[1]))

        policy = DoNothingPolicy(state.elections, couple_withdrawal_order)
        recorder = RecordingPolicy(policy)
        run(state, recorder, draws, couple_market, couple_real_params)

        r_resp = couple_market.weights("resp")
        expected_value = np.full(n_paths, 5_000.0)
        for m in range(5):
            expected_value = expected_value * (1 + r_resp @ draws.real_returns[m])
        assert np.all(expected_value > 0.0)  # guard

        after_wind_up = recorder.calls[6][0]
        a_income = after_wind_up.persons[0].income.resp_accumulated_income
        b_income = after_wind_up.persons[1].income.resp_accumulated_income

        # Path 0: a is dead -> credited to b, the spouse; a's own ledger gets 0.
        np.testing.assert_allclose(b_income[0], expected_value[0])
        np.testing.assert_allclose(a_income[0], 0.0)
        # Path 1: a is alive -> credited to a, the subscriber; b's ledger gets 0.
        np.testing.assert_allclose(a_income[1], expected_value[1])
        np.testing.assert_allclose(b_income[1], 0.0)


class TestAipWithholdingAtWindUp:
    """#57: the RESP provider withholds the special tax when it pays the AIP.

    ``WIND_UP_MONTH_INDEX`` is the AIP fixture's wind-up month:
    ``education_start_month_index (4) + education_months (1)``.
    ``DECEMBER_CLOSE_INDEX`` is 2026's December close; the run opens 1 January 2026 at
    month index 0.
    """

    WIND_UP_MONTH_INDEX = 5
    DECEMBER_CLOSE_INDEX = 11

    def _filing_index(self, real_params) -> int:
        filing_month = int(real_params.federal.number("filing_month"))
        return 12 + (filing_month - 1)

    def test_wind_up_month_pays_cash_net_of_the_penalty_and_withholds_it_from_remitted(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        n_paths = opening_state.n_paths
        control_state = _swap_beneficiary(opening_state, _aip_resp_state(n_paths, income=0.0))
        aip_state = _swap_beneficiary(opening_state, _aip_resp_state(n_paths, income=5_000.0))

        control_recorder = RecordingPolicy(
            DoNothingPolicy(control_state.elections, withdrawal_order)
        )
        control_result = run(
            control_state, control_recorder, draws, market, real_params, trace_path=0
        )
        aip_recorder = RecordingPolicy(DoNothingPolicy(aip_state.elections, withdrawal_order))
        aip_result = run(aip_state, aip_recorder, draws, market, real_params, trace_path=0)

        # Guard: the control plan's own wind-up really has zero accumulated income.
        control_record = control_result.trace[self.WIND_UP_MONTH_INDEX]
        np.testing.assert_allclose(control_record.wind_up_to_cash[0], 0.0)
        np.testing.assert_allclose(control_record.wind_up_withholding[0], 0.0)

        aip_record = aip_result.trace[self.WIND_UP_MONTH_INDEX]
        accumulated = _aip_expected_accumulated(market, draws)
        assert np.all(accumulated > 0.0)  # guard
        # The field is checked against an independent computation, not trusted.
        np.testing.assert_allclose(aip_record.wind_up_to_cash[0], accumulated)

        rate = real_params.resp.number("aip.penalty_rate")
        penalty = accumulated * rate
        np.testing.assert_allclose(aip_record.wind_up_withholding[0], penalty)

        for m in range(self.WIND_UP_MONTH_INDEX):
            np.testing.assert_allclose(
                aip_result.trace[m].cash_close, control_result.trace[m].cash_close, atol=1e-6
            )
        np.testing.assert_allclose(
            aip_record.cash_close - control_record.cash_close, accumulated - penalty, atol=1e-6
        )

        after_wind_up_aip = aip_recorder.calls[self.WIND_UP_MONTH_INDEX + 1][0]
        after_wind_up_control = control_recorder.calls[self.WIND_UP_MONTH_INDEX + 1][0]
        np.testing.assert_allclose(
            after_wind_up_aip.persons[0].income.remitted
            - after_wind_up_control.persons[0].income.remitted,
            penalty,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            after_wind_up_aip.persons[0].income.resp_accumulated_income
            - after_wind_up_control.persons[0].income.resp_accumulated_income,
            accumulated,
            atol=1e-6,
        )

    def test_who_remits_both_branches_of_the_credit_rule_in_one_run(
        self,
        couple_scenario,
        couple_market,
        couple_mortality,
        couple_real_params,
        couple_withdrawal_order,
    ):
        """Both branches of the who-remits rule, side by side in one run: the
        subscriber (person 0) is forced dead before the wind-up on path 0 only, so
        path 0 exercises the dead-subscriber branch and path 1 the living-subscriber
        branch. Asserts that path 0 credits the spouse and leaves the subscriber's own
        ledger untouched, and path 1 the reverse.
        """
        n_paths = 2
        draws = build_draws(
            couple_scenario, couple_market, n_paths=n_paths, mortality=couple_mortality
        )
        state = build_initial_state(couple_scenario, n_paths=n_paths)
        state = draw_deaths(state, draws, couple_mortality)

        # Subscriber (person 0): dead before the wind-up on path 0, alive through the
        # run on path 1.
        person_a = updated(
            state.persons[0],
            death_month_index=np.array([3, draws.n_months - 2], dtype=np.int64),
        )
        state = updated(state, persons=(person_a, state.persons[1]))
        # Spouse (person 1): alive through the wind-up and the filing month on both
        # paths, so neither branch's credited person happens to be dead too.
        state = _force_death(state, 1, draws.n_months - 2)

        aip_state = _swap_beneficiary(state, _aip_resp_state(n_paths, income=5_000.0))
        control_state = _swap_beneficiary(state, _aip_resp_state(n_paths, income=0.0))

        aip_recorder = RecordingPolicy(
            DoNothingPolicy(aip_state.elections, couple_withdrawal_order)
        )
        run(aip_state, aip_recorder, draws, couple_market, couple_real_params)
        control_recorder = RecordingPolicy(
            DoNothingPolicy(control_state.elections, couple_withdrawal_order)
        )
        run(control_state, control_recorder, draws, couple_market, couple_real_params)

        accumulated = _aip_expected_accumulated(couple_market, draws)
        assert np.all(accumulated > 0.0)  # guard
        rate = couple_real_params.resp.number("aip.penalty_rate")
        penalty = accumulated * rate

        after_aip = aip_recorder.calls[self.WIND_UP_MONTH_INDEX + 1][0]
        after_ctrl = control_recorder.calls[self.WIND_UP_MONTH_INDEX + 1][0]

        # Path 0: the subscriber is dead -> the spouse (person 1) remits the penalty;
        # the subscriber's own ledger is untouched.
        np.testing.assert_allclose(
            after_aip.persons[1].income.remitted[0] - after_ctrl.persons[1].income.remitted[0],
            penalty[0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            after_aip.persons[0].income.remitted[0] - after_ctrl.persons[0].income.remitted[0],
            0.0,
            atol=1e-6,
        )
        # Path 1: the subscriber is alive -> the subscriber remits the penalty; the
        # spouse's ledger is untouched.
        np.testing.assert_allclose(
            after_aip.persons[0].income.remitted[1] - after_ctrl.persons[0].income.remitted[1],
            penalty[1],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            after_aip.persons[1].income.remitted[1] - after_ctrl.persons[1].income.remitted[1],
            0.0,
            atol=1e-6,
        )

    def test_december_assessment_includes_the_penalty_and_april_nets_it_to_zero(
        self, opening_state, draws, market, real_params, scenario, withdrawal_order, tmp_path
    ):
        control_state = _swap_beneficiary(
            opening_state, _aip_resp_state(opening_state.n_paths, income=0.0)
        )
        aip_state = _swap_beneficiary(
            opening_state, _aip_resp_state(opening_state.n_paths, income=5_000.0)
        )
        filing_index = self._filing_index(real_params)

        control_recorder = RecordingPolicy(
            DoNothingPolicy(control_state.elections, withdrawal_order)
        )
        control_result = run(
            control_state, control_recorder, draws, market, real_params, trace_path=0
        )
        aip_recorder = RecordingPolicy(DoNothingPolicy(aip_state.elections, withdrawal_order))
        aip_result = run(aip_state, aip_recorder, draws, market, real_params, trace_path=0)

        accumulated = _aip_expected_accumulated(market, draws)
        assert np.all(accumulated > 0.0)  # guard
        # The field is checked against an independent computation, not trusted.
        np.testing.assert_allclose(
            aip_result.trace[self.WIND_UP_MONTH_INDEX].wind_up_to_cash[0], accumulated
        )
        rate_real = real_params.resp.number("aip.penalty_rate")
        penalty = accumulated * rate_real

        # Neither run's cash floor fires -- a forgiven depletion deficit would move the
        # comparison off the flows this test names.
        for m in range(filing_index + 1):
            assert np.all(_sum_by_kind(control_result.trace[m].floor_withdrawals) == 0.0)
            assert np.all(_sum_by_kind(aip_result.trace[m].floor_withdrawals) == 0.0)

        assert np.all(control_result.trace[filing_index].alive[0])
        assert np.all(aip_result.trace[filing_index].alive[0])

        assert control_result.years[0] == 2026
        assert aip_result.years[0] == 2026
        tax_assessed_ctrl = control_result.tax_assessed[0]
        tax_assessed_with = aip_result.tax_assessed[0]

        ordinary_delta = (tax_assessed_with - penalty) - tax_assessed_ctrl
        assert np.all(ordinary_delta > 0.0)  # guard: the gross really is taxed

        state_after_close_with = aip_recorder.calls[self.DECEMBER_CLOSE_INDEX + 1][0]
        state_after_close_ctrl = control_recorder.calls[self.DECEMBER_CLOSE_INDEX + 1][0]
        balance_owing_with = state_after_close_with.persons[0].balance_owing
        balance_owing_ctrl = state_after_close_ctrl.persons[0].balance_owing
        np.testing.assert_allclose(
            balance_owing_with - balance_owing_ctrl, ordinary_delta, atol=0.005
        )

        settlement_with = aip_result.trace[filing_index].context.tax_settlement[0]
        settlement_ctrl = control_result.trace[filing_index].context.tax_settlement[0]
        np.testing.assert_allclose(settlement_with - settlement_ctrl, ordinary_delta, atol=0.005)

        # A second AIP run, identical to this one except for the penalty rate,
        # isolates the penalty's own contribution to the December assessment: the
        # ordinary tax on the gross income is the same in both runs (same accumulated
        # income, same rate-independent brackets), so the whole difference between the
        # two runs' ``tax_assessed`` is the penalty's rate-times-accumulated delta. A
        # run-level rate comparison catches a step that (wrongly) zeroes the penalty
        # out of ``engine.tax.combined``'s ``total``, which none of this run's own checks
        # above would detect. Because this run's own withholding always matches this run's own
        # assessed penalty (both read the same rate), ``balance_owing`` and the April
        # settlement do not move at all between the two runs, whatever the rate is.
        synthetic_params = _synthetic_resp_params(scenario, real_params, tmp_path)
        rate_synth = synthetic_params.resp.number("aip.penalty_rate")
        synthetic_recorder = RecordingPolicy(DoNothingPolicy(aip_state.elections, withdrawal_order))
        synthetic_result = run(
            aip_state, synthetic_recorder, draws, market, synthetic_params, trace_path=0
        )
        np.testing.assert_allclose(
            synthetic_result.trace[self.WIND_UP_MONTH_INDEX].wind_up_to_cash[0],
            accumulated,
        )

        tax_assessed_synth = synthetic_result.tax_assessed[0]
        np.testing.assert_allclose(
            tax_assessed_synth - tax_assessed_with,
            accumulated * (rate_synth - rate_real),
            atol=0.005,
        )

        state_after_close_synth = synthetic_recorder.calls[self.DECEMBER_CLOSE_INDEX + 1][0]
        balance_owing_synth = state_after_close_synth.persons[0].balance_owing
        np.testing.assert_allclose(balance_owing_synth, balance_owing_with, atol=0.005)

        settlement_synth = synthetic_result.trace[filing_index].context.tax_settlement[0]
        np.testing.assert_allclose(settlement_synth, settlement_with, atol=0.005)

    def test_no_accumulated_income_remits_nothing(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        """A plan holding contributions only, wound up before any growth reaches
        it (``education_start_month_index = education_months = 0``), remits nothing;
        a wholly empty plan wound up the same way moves no cash, and its cash close
        matches a household with no beneficiary at all.
        """
        n_paths = opening_state.n_paths
        contributions_only = _aip_resp_state(
            n_paths,
            income=0.0,
            contributions=5_000.0,
            education_start_month_index=0,
            education_months=0,
        )
        empty = updated(contributions_only, contributions=np.zeros(n_paths))

        contributions_state = _swap_beneficiary(opening_state, contributions_only)
        empty_state = _swap_beneficiary(opening_state, empty)
        no_beneficiary_state = updated(opening_state, beneficiaries=())

        recorder_c = RecordingPolicy(
            DoNothingPolicy(contributions_state.elections, withdrawal_order)
        )
        result_c = run(contributions_state, recorder_c, draws, market, real_params, trace_path=0)
        recorder_e = RecordingPolicy(DoNothingPolicy(empty_state.elections, withdrawal_order))
        result_e = run(empty_state, recorder_e, draws, market, real_params, trace_path=0)
        recorder_n = RecordingPolicy(
            DoNothingPolicy(no_beneficiary_state.elections, withdrawal_order)
        )
        result_n = run(no_beneficiary_state, recorder_n, draws, market, real_params, trace_path=0)

        record_c = result_c.trace[0]
        record_e = result_e.trace[0]
        record_n = result_n.trace[0]

        np.testing.assert_allclose(record_c.wind_up_withholding[0], 0.0)
        np.testing.assert_allclose(record_c.wind_up_to_cash[0], 5_000.0)

        # The contributions-only plan's cash rises by exactly the tax-free
        # contributions, against the empty plan's own wind-up, which moves no cash.
        np.testing.assert_allclose(record_c.cash_close - record_e.cash_close, 5_000.0, atol=1e-6)

        after_c = recorder_c.calls[1][0]
        after_e = recorder_e.calls[1][0]
        # Guard: no accumulated income reaches the ledger.
        np.testing.assert_allclose(
            after_c.persons[0].income.resp_accumulated_income
            - after_e.persons[0].income.resp_accumulated_income,
            0.0,
        )
        np.testing.assert_allclose(
            after_c.persons[0].income.remitted, after_e.persons[0].income.remitted
        )

        # The wholly empty plan's own wind-up moves no cash and remits nothing -- and
        # its cash close matches a household with no beneficiary at all.
        np.testing.assert_allclose(record_e.wind_up_to_cash[0], 0.0)
        np.testing.assert_allclose(record_e.wind_up_withholding[0], 0.0)
        np.testing.assert_allclose(record_e.cash_close, record_n.cash_close, atol=1e-6)

    def test_subscriber_dead_credits_the_survivor_and_household_sums_hold(
        self,
        couple_opening_state,
        couple_draws,
        couple_market,
        couple_real_params,
        couple_withdrawal_order,
    ):
        n_paths = couple_opening_state.n_paths
        aip_resp = _aip_resp_state(n_paths, income=5_000.0)
        control_resp = updated(aip_resp, income=np.zeros(n_paths))

        def _prepare(resp_state):
            state = _swap_beneficiary(couple_opening_state, resp_state)
            state = _force_death(state, 0, 3)  # subscriber dies before the month-5 wind-up
            state = _force_death(state, 1, couple_draws.n_months - 2)  # alive through filing
            return state

        aip_state = _prepare(aip_resp)
        control_state = _prepare(control_resp)

        aip_recorder = RecordingPolicy(
            DoNothingPolicy(aip_state.elections, couple_withdrawal_order)
        )
        aip_result = run(
            aip_state, aip_recorder, couple_draws, couple_market, couple_real_params, trace_path=0
        )
        control_recorder = RecordingPolicy(
            DoNothingPolicy(control_state.elections, couple_withdrawal_order)
        )
        control_result = run(
            control_state,
            control_recorder,
            couple_draws,
            couple_market,
            couple_real_params,
            trace_path=0,
        )

        accumulated = _aip_expected_accumulated(couple_market, couple_draws)
        assert np.all(accumulated > 0.0)  # guard
        # The field is checked against an independent computation, not trusted.
        np.testing.assert_allclose(
            aip_result.trace[self.WIND_UP_MONTH_INDEX].wind_up_to_cash[0], accumulated
        )
        rate = couple_real_params.resp.number("aip.penalty_rate")
        penalty = accumulated * rate

        after_aip = aip_recorder.calls[self.WIND_UP_MONTH_INDEX + 1][0]
        after_ctrl = control_recorder.calls[self.WIND_UP_MONTH_INDEX + 1][0]
        np.testing.assert_allclose(
            after_aip.persons[1].income.remitted - after_ctrl.persons[1].income.remitted,
            penalty,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            after_aip.persons[0].income.remitted - after_ctrl.persons[0].income.remitted,
            0.0,
            atol=1e-6,
        )

        filing_index = self._filing_index(couple_real_params)
        assert np.all(aip_result.trace[filing_index].alive[1])
        assert np.all(control_result.trace[filing_index].alive[1])

        for m in range(filing_index + 1):
            assert np.all(_sum_by_kind(aip_result.trace[m].floor_withdrawals) == 0.0)
            assert np.all(_sum_by_kind(control_result.trace[m].floor_withdrawals) == 0.0)

        assert control_result.years[0] == 2026
        assert aip_result.years[0] == 2026
        tax_assessed_ctrl = control_result.tax_assessed[0]
        tax_assessed_with = aip_result.tax_assessed[0]
        ordinary_delta = (tax_assessed_with - penalty) - tax_assessed_ctrl
        assert np.all(ordinary_delta > 0.0)  # guard: the gross really is taxed

        state_after_close_with = aip_recorder.calls[self.DECEMBER_CLOSE_INDEX + 1][0]
        state_after_close_ctrl = control_recorder.calls[self.DECEMBER_CLOSE_INDEX + 1][0]
        balance_owing_with = sum(p.balance_owing for p in state_after_close_with.persons)
        balance_owing_ctrl = sum(p.balance_owing for p in state_after_close_ctrl.persons)
        np.testing.assert_allclose(
            balance_owing_with - balance_owing_ctrl, ordinary_delta, atol=0.005
        )

        # The April half of the identity, on household sums: the 2027 filing month's
        # settlement moves by exactly the ordinary-tax delta; the penalty, already
        # withheld at the wind-up, nets to zero.
        settlement_with = _sum_arrays(aip_result.trace[filing_index].context.tax_settlement)
        settlement_ctrl = _sum_arrays(control_result.trace[filing_index].context.tax_settlement)
        np.testing.assert_allclose(
            settlement_with[0] - settlement_ctrl[0], ordinary_delta, atol=0.005
        )

    def test_the_withholding_rate_comes_from_params_not_a_hardcoded_value(
        self, opening_state, draws, market, real_params, scenario, withdrawal_order, tmp_path
    ):
        synthetic_params = _synthetic_resp_params(scenario, real_params, tmp_path)
        synthetic_rate = synthetic_params.resp.number("aip.penalty_rate")

        aip_state = _swap_beneficiary(
            opening_state, _aip_resp_state(opening_state.n_paths, income=5_000.0)
        )
        policy = DoNothingPolicy(aip_state.elections, withdrawal_order)
        result = run(aip_state, policy, draws, market, synthetic_params, trace_path=0)

        record = result.trace[self.WIND_UP_MONTH_INDEX]
        accumulated = _aip_expected_accumulated(market, draws)
        assert np.all(accumulated > 0.0)  # guard
        # The field is checked against an independent computation, not trusted.
        np.testing.assert_allclose(record.wind_up_to_cash[0], accumulated)
        np.testing.assert_allclose(record.wind_up_withholding[0], accumulated * synthetic_rate)

    def test_cash_identity_reaches_the_wind_up_withholding_branch(
        self, opening_state, draws, market, real_params, withdrawal_order
    ):
        aip_state = _swap_beneficiary(
            opening_state, _aip_resp_state(opening_state.n_paths, income=5_000.0)
        )
        recorder = RecordingPolicy(DoNothingPolicy(aip_state.elections, withdrawal_order))
        result = run(aip_state, recorder, draws, market, real_params, trace_path=0)

        withheld_somewhere = False
        for record in result.trace:
            _assert_identities_hold(record)
            if np.any(_sum_arrays(record.wind_up_withholding) > 0):
                withheld_somewhere = True

        assert withheld_somewhere, "wind_up_withholding never fired; the test would be vacuous"


class TestJanuaryFinalDeathAtRunLevel:
    """A run-level January final death, at zero inflation so the taxable ACB does
    not erode across the January it dies in. Catches a phase-1/phase-2 swap: if
    ``open_year`` ran after ``resolve_deaths`` instead of before it, the terminal
    return would read last year's ledger, not an empty one.
    """

    DEATH_MONTH_INDEX = 12  # January 2027

    def test_terminal_assessment_matches_decembers_close(
        self, scenario, market, mortality, withdrawal_order
    ):
        zero_inflation_real_params = real_year(load_year(scenario.start_year), 0.0)
        draws = build_draws(scenario, market, n_paths=1, mortality=mortality)
        state = build_initial_state(scenario, n_paths=1)
        state = draw_deaths(state, draws, mortality)
        person = updated(
            state.persons[0],
            death_month_index=np.full(1, self.DEATH_MONTH_INDEX, dtype=np.int64),
        )
        state = updated(state, persons=(person,))

        policy = DoNothingPolicy(state.elections, withdrawal_order)
        recorder = RecordingPolicy(policy)
        result = run(state, recorder, draws, market, zero_inflation_real_params, trace_path=0)

        record = result.trace[self.DEATH_MONTH_INDEX]
        december_balances = result.trace[self.DEATH_MONTH_INDEX - 1].balances_close[0]

        # December's taxable ACB, independently: one month of replayed distributions on
        # top of what RecordingPolicy saw before that month's own growth (the one piece
        # balances_close does not carry).
        december_month_index = self.DEATH_MONTH_INDEX - 1
        pre_growth_taxable = recorder.calls[december_month_index][0].persons[0].taxable
        weighted_yields = market.weighted_yields("taxable")
        interest, dividends, gains = taxable.distributions_monthly(
            pre_growth_taxable.balance, weighted_yields
        )
        december_acb = pre_growth_taxable.acb + interest + dividends + gains

        zero_ledger = IncomeLedger(
            **{f.name: np.zeros(1) for f in dataclasses.fields(IncomeLedger)}
        )
        deemed = updated(
            zero_ledger,
            rrif_lif_withdrawals=(
                december_balances.rrsp
                + december_balances.rrif
                + december_balances.lira
                + december_balances.lif
            ),
            capital_gains=december_balances.taxable - december_acb,
        )
        age_end = timeline.age_at_end_of_year(
            person.birth_year, person.birth_month, state.year + self.DEATH_MONTH_INDEX // 12
        )
        expected = person_assessment(
            deemed,
            age_end,
            np.zeros(1),
            np.zeros(1),
            state.province,
            zero_inflation_real_params,
            self.DEATH_MONTH_INDEX,
            died_in_year=True,
        )
        np.testing.assert_allclose(record.context.terminal_assessment, expected.total)


def _walk_month_record_arrays(value: object, path: str, found: list[str]) -> None:
    """Like ``tests.core.conftest.walk``, but over a :class:`MonthRecord` tree and
    recording every array's dotted path -- this module's fixtures do not build a
    ``HouseholdState``-shaped tree, so the shared walker (keyed by field name alone,
    not by path) is not reused here.
    """
    if isinstance(value, np.ndarray):
        assert value.shape == (1,), path
        assert not value.flags.writeable, path
        found.append(path)
    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            _walk_month_record_arrays(item, f"{path}.{index}" if path else str(index), found)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            child = f"{path}.{field.name}" if path else field.name
            _walk_month_record_arrays(getattr(value, field.name), child, found)


class TestMonthRecordCarriesPersonsBeneficiariesAndYearRecord:
    """#63: ``MonthRecord.persons``/``beneficiaries``/``year_record``, and
    ``MonthContext.terminal_assessments``, survive ``at_path`` fully sliced and frozen.
    """

    def test_a_sliced_december_record_is_fully_sliced_and_frozen(
        self, couple_scenario, couple_real_params, couple_market, couple_withdrawal_order
    ):
        state = build_initial_state(couple_scenario, n_paths=2)
        state = updated(state, month=12, month_index=11)
        policy = DoNothingPolicy(state.elections, couple_withdrawal_order)
        n_assets = len(couple_market.asset_class_names)
        month_returns = np.zeros((n_assets, 2))

        new_state, record = advance_month_traced(
            state, month_returns, policy, couple_market, couple_real_params
        )

        # References, not copies, on the unsliced record.
        assert record.persons is new_state.persons
        assert record.year_record is not None
        assert record.year_record is new_state.history[-1]
        # The couple scenario carries no beneficiaries at all.
        assert record.beneficiaries == ()

        sliced = record.at_path(1)
        found: list[str] = []
        _walk_month_record_arrays(sliced, "", found)

        assert any(path.startswith("persons.") for path in found)
        assert any(path.startswith("year_record.assessments.") for path in found)
        assert any(path.startswith("context.terminal_assessments.") for path in found)

    def test_a_sliced_december_record_reaches_beneficiaries(
        self, scenario, real_params, market, withdrawal_order
    ):
        state = build_initial_state(scenario, n_paths=2)
        state = updated(state, month=12, month_index=11)
        policy = DoNothingPolicy(state.elections, withdrawal_order)
        n_assets = len(market.asset_class_names)
        month_returns = np.zeros((n_assets, 2))

        new_state, record = advance_month_traced(state, month_returns, policy, market, real_params)

        assert record.beneficiaries is new_state.beneficiaries
        assert len(record.beneficiaries) >= 1

        sliced = record.at_path(1)
        found: list[str] = []
        _walk_month_record_arrays(sliced, "", found)

        assert any(path.startswith("beneficiaries.") for path in found)
        assert any(path.startswith("persons.") for path in found)
        assert any(path.startswith("year_record.assessments.") for path in found)

    def test_a_non_december_record_has_no_year_record(
        self, couple_scenario, couple_real_params, couple_market, couple_withdrawal_order
    ):
        state = build_initial_state(couple_scenario, n_paths=2)
        state = updated(state, month=6, month_index=5)
        policy = DoNothingPolicy(state.elections, couple_withdrawal_order)
        n_assets = len(couple_market.asset_class_names)
        month_returns = np.zeros((n_assets, 2))

        _new_state, record = advance_month_traced(
            state, month_returns, policy, couple_market, couple_real_params
        )

        assert record.year_record is None

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``OrderedWithdrawal.transfers``, and the gross-up and same-month budget it feeds into.

``late_life_couple.yaml`` is the fixture throughout: two persons, each already holding a
RRIF and a LIF, which is exactly the shape the fill and the LIF maximum need. Every real
figure (bracket edges, the RRIF minimum/maximum factors, the withholding table) comes from
``params/2026/`` through ``couple_real_params``; nothing here is a hand-typed tax constant.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from engine.core.build import build_initial_state, build_market_inputs
from engine.core.indexation import real_year
from engine.core.state import updated
from engine.core.step import advance_month_traced, open_year
from engine.params.loader import load_year
from engine.policy.build import build_policy
from engine.policy.withdrawal import (
    OrderedWithdrawal,
    _projected_income_rest_of_year,
    fill_to_bracket,
)
from engine.tax import federal as federal_mod
from engine.tax.withholding import registered_withholding


def _opened_couple_state(couple_scenario, couple_real_params):
    """The couple's opening state, through one ``open_year`` -- RRIF/LIF minimum and maximum
    fixed for 2026, year-to-date ledgers at zero.
    """
    state = build_initial_state(couple_scenario, n_paths=1)
    return open_year(state, couple_real_params)


def test_projected_income_rest_of_year_matches_the_projection_formula(
    couple_scenario, couple_real_params, set_person
):
    """``_projected_income_rest_of_year``'s formula, written out here from the fields directly
    rather than by calling the helper twice.

    Path 0 carries a non-zero value in every ingredient the formula reads (employment,
    ``rrsp_withdrawals``, ``rrif_lif_withdrawals``, and a LIF minimum above
    ``withdrawn_ytd``, with a LIF balance ample enough that the cap does not bind). Path 1's
    ``rrsp_deductions`` are large enough to floor ``net_income`` below the registered
    withdrawals it already contains, so the zero clip on ``base`` is exercised for real
    rather than merely never triggering; its LIF balance is also set below its remaining
    minimum, so the cap binds on that path. A RRIF minimum above ``withdrawn_ytd`` is also
    set, on both paths, though the formula no longer reads it -- so that re-adding the RRIF
    term (a mutation this test must catch) actually changes the result.
    """
    state = build_initial_state(couple_scenario, n_paths=2)
    person0 = state.persons[0]
    income = updated(
        person0.income,
        employment=np.array([2000.0, 0.0]),
        rrsp_withdrawals=np.array([1200.0, 5000.0]),
        rrif_lif_withdrawals=np.array([800.0, 3000.0]),
        rrsp_deductions=np.array([0.0, 10000.0]),
    )
    state = set_person(state, 0, income=income)
    state = set_person(
        state,
        0,
        rrif=updated(
            state.persons[0].rrif,
            annual_minimum=np.array([2400.0, 2400.0]),
            withdrawn_ytd=np.array([900.0, 900.0]),
        ),
    )
    state = set_person(
        state,
        0,
        lif=updated(
            state.persons[0].lif,
            annual_minimum=np.array([1800.0, 1800.0]),
            withdrawn_ytd=np.array([600.0, 600.0]),
            balance=np.array([100_000.0, 1000.0]),
        ),
    )
    person = state.persons[0]

    month = 5
    ytd = federal_mod.net_income(person.income, couple_real_params.federal, 0.0, 0.0)
    result = _projected_income_rest_of_year(ytd, person, month)

    base = np.clip(
        ytd - person.income.rrsp_withdrawals - person.income.rrif_lif_withdrawals, 0, None
    )
    run_rate = base / month
    lif_remaining_minimum = np.clip(person.lif.annual_minimum - person.lif.withdrawn_ytd, 0, None)
    expected = run_rate * (12 - month) + np.minimum(lif_remaining_minimum, person.lif.balance)
    assert base[0] > 0.0, "path 0 must have a genuine positive run rate to check against"
    assert base[1] == pytest.approx(0.0), "path 1 must actually exercise the zero clip on base"
    assert lif_remaining_minimum[0] <= person.lif.balance[0], (
        "path 0's LIF cap must not bind, to check the uncapped term"
    )
    assert lif_remaining_minimum[1] > person.lif.balance[1], (
        "path 1's LIF cap must actually bind for this test to exercise it"
    )
    np.testing.assert_allclose(result, expected)


class TestTheFillFiresRegardlessOfNeed:
    def test_the_fill_fires_in_a_surplus_month(
        self, couple_scenario, couple_real_params, make_context
    ):
        state = _opened_couple_state(couple_scenario, couple_real_params)
        withdrawal = OrderedWithdrawal(
            order=couple_scenario.policies[0].withdrawal.order, taxable_ceiling_bracket=1
        )
        context = make_context(
            n_paths=1,
            n_persons=2,
            year=2026,
            month=1,
            month_index=0,
            cash_after_flows=np.array([5000.0]),
        )
        transfers, net = withdrawal.transfers(state, context, couple_real_params)

        assert len(transfers) == 2
        for transfer in transfers:
            assert transfer.from_kind == "rrif"
            assert transfer.to_kind == "cash"
            assert np.all(transfer.amount > 0)
        assert np.all(net > 0)

    def test_a_none_ceiling_draws_nothing_in_a_surplus_month(
        self, couple_scenario, couple_real_params, make_context
    ):
        state = _opened_couple_state(couple_scenario, couple_real_params)
        withdrawal = OrderedWithdrawal(
            order=couple_scenario.policies[0].withdrawal.order, taxable_ceiling_bracket=None
        )
        context = make_context(
            n_paths=1,
            n_persons=2,
            year=2026,
            month=1,
            month_index=0,
            cash_after_flows=np.array([5000.0]),
        )
        transfers, net = withdrawal.transfers(state, context, couple_real_params)

        assert transfers == ()
        np.testing.assert_allclose(net, [0.0])


def test_w1_matches_the_bracket_fill_formula_mid_year(
    couple_scenario, couple_real_params, set_person, make_context
):
    """The fill's exact number, a mid-year month, checked against the formula in
    :class:`OrderedWithdrawal`'s docstring rather than only qualitatively.

    The RRIF balance is set smaller than the expected fill, so the RRSP is left to draw
    exactly the remainder -- proving both halves of the split, not only their sum. Person 0
    keeps ``late_life_couple.yaml``'s own opened LIF minimum (above its ``withdrawn_ytd`` of
    zero, from ``open_year``), so the projection's minimum term is exercised beyond the run
    rate alone -- not only ``rrsp_withdrawals``/``rrif_lif_withdrawals``, which stay zero
    here.
    """
    state = _opened_couple_state(couple_scenario, couple_real_params)
    person0 = state.persons[0]
    state = set_person(state, 0, income=updated(person0.income, employment=np.full(1, 5000.0)))
    small_rrif_balance = 1000.0
    state = set_person(
        state, 0, rrif=updated(state.persons[0].rrif, balance=np.full(1, small_rrif_balance))
    )
    state = set_person(
        state,
        0,
        rrsp=updated(state.persons[0].rrsp, balance=np.full(1, 1_000_000.0), room=np.zeros(1)),
    )
    assert np.all(state.persons[0].lif.annual_minimum > state.persons[0].lif.withdrawn_ytd), (
        "the LIF remaining minimum must actually be positive for the projection to exercise it"
    )

    ceiling = 1
    withdrawal = OrderedWithdrawal(
        order=couple_scenario.policies[0].withdrawal.order, taxable_ceiling_bracket=ceiling
    )
    month = 5
    month_index = 4  # January is month_index 0 for this scenario.
    context = make_context(
        n_paths=1,
        n_persons=2,
        year=2026,
        month=month,
        month_index=month_index,
        cash_after_flows=np.array([0.0]),
    )

    transfers, _net = withdrawal.transfers(state, context, couple_real_params)

    january_month_index = month_index - (month - 1)
    edges = couple_real_params.federal.annual_amounts("brackets.edges_annual", january_month_index)
    edge = edges[ceiling]
    person0 = state.persons[0]
    ytd = federal_mod.net_income(person0.income, couple_real_params.federal, 0.0, 0.0)
    # The projection, written out from the fields rather than by calling the helper.
    base = np.clip(
        ytd - person0.income.rrsp_withdrawals - person0.income.rrif_lif_withdrawals, 0, None
    )
    lif_remaining_minimum = np.clip(person0.lif.annual_minimum - person0.lif.withdrawn_ytd, 0, None)
    projected = base / month * (12 - month) + np.minimum(lif_remaining_minimum, person0.lif.balance)
    expected_fill = (edge - ytd - projected) / (13 - month)
    expected_rrif = np.minimum(expected_fill, small_rrif_balance)
    expected_rrsp = expected_fill - expected_rrif
    assert expected_rrif[0] == pytest.approx(small_rrif_balance), (
        "the RRIF balance must actually bind for this test to exercise the RRSP remainder"
    )

    rrif0 = next(t for t in transfers if t.person_index == 0 and t.from_kind == "rrif")
    rrsp0 = next(t for t in transfers if t.person_index == 0 and t.from_kind == "rrsp")
    np.testing.assert_allclose(rrif0.amount, expected_rrif)
    np.testing.assert_allclose(rrsp0.amount, expected_rrsp)

    # 12 - month would read 12 - 5 = 7 rather than 13 - 5 = 8: a materially different
    # divisor, so this test fails against that mutation rather than passing by accident.
    wrong_divisor_fill = (edge - ytd - projected) / (12 - month)
    assert not np.allclose(expected_fill, wrong_divisor_fill)

    # Reading the edges at context.month_index (4) rather than january_month_index (0) is
    # not checked by comparing values here: engine.core.indexation.RealParamSet.annual_amounts
    # refuses any non-January index outright (see test below), so that mutation is caught by
    # an immediate ValueError from real_params.federal.annual_amounts, not a silently wrong
    # number -- there is no year in which the two reads could coincide by indexation being
    # flat, because the call with a mid-year index never returns at all.


def test_a_mid_year_month_index_is_refused_by_annual_amounts(couple_real_params):
    """The guard :func:`test_w1_matches_the_bracket_fill_formula_mid_year` relies on: reading
    ``brackets.edges_annual`` at ``context.month_index`` instead of ``january_month_index``
    would raise here, not silently return a same-or-different number.
    """
    with pytest.raises(ValueError, match="January"):
        couple_real_params.federal.annual_amounts("brackets.edges_annual", 4)


def test_twelve_months_of_steady_income_land_on_the_edge(
    couple_scenario, couple_real_params, set_person, make_context
):
    """The regression for the overshoot the projection now prevents, at the policy level.

    A steady yearly ``db_pension`` of ``0.6 * edge`` (ceiling 1), run for twelve months
    through ``OrderedWithdrawal.transfers`` alone -- no step -- with a RRIF too large to ever
    bind and RRIF/LIF minimums both zero, so the fill is the only thing moving. The old rule
    (room measured against year to date alone) would fill toward the edge net of *only* the
    ``db_pension`` recognised so far, then have the rest of the year's own ``db_pension``
    still land on top, overshooting the edge; the projection is what keeps the year exactly
    on it.
    """
    state = _opened_couple_state(couple_scenario, couple_real_params)
    january_month_index = 0  # per test_w1_matches_the_bracket_fill_formula_mid_year.
    ceiling = 1
    edges = couple_real_params.federal.annual_amounts("brackets.edges_annual", january_month_index)
    edge = edges[ceiling]
    monthly_db_pension = edge * 0.6 / 12

    zeros1 = np.zeros(1)
    person0 = state.persons[0]
    zeroed_income = updated(
        person0.income,
        employment=zeros1,
        cpp=zeros1,
        oas=zeros1,
        db_pension=zeros1,
        rrsp_withdrawals=zeros1,
        rrif_lif_withdrawals=zeros1,
        interest=zeros1,
        eligible_dividends=zeros1,
        capital_gains=zeros1,
        resp_accumulated_income=zeros1,
        rrsp_deductions=zeros1,
        cpp_base_contributions=zeros1,
        cpp_enhanced_contributions=zeros1,
        ei_premiums=zeros1,
        remitted=zeros1,
    )
    state = set_person(
        state,
        0,
        income=zeroed_income,
        rrif=updated(
            person0.rrif,
            balance=np.full(1, 10_000_000.0),
            annual_minimum=zeros1,
            withdrawn_ytd=zeros1,
        ),
        rrsp=updated(person0.rrsp, balance=zeros1, room=zeros1),
        lif=updated(person0.lif, annual_minimum=zeros1, withdrawn_ytd=zeros1),
    )
    # Person 1 draws nothing towards its own fill: zero RRIF and RRSP balances.
    person1 = state.persons[1]
    state = set_person(
        state,
        1,
        rrif=updated(person1.rrif, balance=zeros1),
        rrsp=updated(person1.rrsp, balance=zeros1, room=zeros1),
    )

    withdrawal = OrderedWithdrawal(
        order=couple_scenario.policies[0].withdrawal.order, taxable_ceiling_bracket=ceiling
    )
    expected_monthly_fill = 0.4 * edge / 12

    for month in range(1, 13):
        person0 = state.persons[0]
        state = set_person(
            state,
            0,
            income=updated(
                person0.income, db_pension=person0.income.db_pension + monthly_db_pension
            ),
        )
        context = make_context(
            n_paths=1,
            n_persons=2,
            year=2026,
            month=month,
            month_index=january_month_index + (month - 1),
            cash_after_flows=np.array([0.0]),
        )
        transfers, _net = withdrawal.transfers(state, context, couple_real_params)
        rrif0 = next(t for t in transfers if t.person_index == 0 and t.from_kind == "rrif")
        np.testing.assert_allclose(rrif0.amount, [expected_monthly_fill], rtol=1e-9)

        person0 = state.persons[0]
        state = set_person(
            state,
            0,
            income=updated(
                person0.income,
                rrif_lif_withdrawals=person0.income.rrif_lif_withdrawals + rrif0.amount,
            ),
            rrif=updated(
                person0.rrif,
                withdrawn_ytd=person0.rrif.withdrawn_ytd + rrif0.amount,
                balance=person0.rrif.balance - rrif0.amount,
            ),
        )

    year_net_income = federal_mod.net_income(
        state.persons[0].income, couple_real_params.federal, 0.0, 0.0
    )
    np.testing.assert_allclose(year_net_income, [edge], rtol=1e-9)


@pytest.mark.parametrize(
    "m_fraction, december_fill_is_full",
    [(0.2, True), (0.5, False)],
    ids=["fill_total_exceeds_m", "fill_total_does_not_exceed_m"],
)
def test_twelve_months_with_a_rrif_minimum_fill_evenly(
    m_fraction, december_fill_is_full, couple_scenario, couple_real_params, set_person, make_context
):
    """The RRIF minimum is not income on top of the fill -- the fill's own draws count towards
    it, so months 1 to 11 fill exactly as in
    ``test_twelve_months_of_steady_income_land_on_the_edge`` regardless of ``M``'s size.
    Built like that test, with a RRIF ``annual_minimum = M``, LIF minimum 0, and a RRIF
    balance too large to ever bind.

    Only when the minimum still owed by the end of November exceeds what the fill has
    already drawn does phase 6's forced December top-up -- simulated here by hand, never
    through the step -- land on top and cut into, or close off, December's own fill.
    ``m_fraction=0.2``: the eleven months' fill total already exceeds ``M``, so there is
    nothing left to force, and December's fill is unaffected. ``m_fraction=0.5``: it does
    not, so the top-up pushes year-to-date net income at or past the edge before December's
    own fill is computed.
    """
    state = _opened_couple_state(couple_scenario, couple_real_params)
    january_month_index = 0
    ceiling = 1
    edges = couple_real_params.federal.annual_amounts("brackets.edges_annual", january_month_index)
    edge = edges[ceiling]
    monthly_db_pension = edge * 0.6 / 12
    m = edge * m_fraction
    expected_monthly_fill = 0.4 * edge / 12
    eleven_month_fill_total = 11 * expected_monthly_fill
    if december_fill_is_full:
        assert eleven_month_fill_total > m, "the fill total must exceed M for this case"
    else:
        assert eleven_month_fill_total <= m, "the fill total must not exceed M for this case"

    zeros1 = np.zeros(1)
    person0 = state.persons[0]
    zeroed_income = updated(
        person0.income,
        employment=zeros1,
        cpp=zeros1,
        oas=zeros1,
        db_pension=zeros1,
        rrsp_withdrawals=zeros1,
        rrif_lif_withdrawals=zeros1,
        interest=zeros1,
        eligible_dividends=zeros1,
        capital_gains=zeros1,
        resp_accumulated_income=zeros1,
        rrsp_deductions=zeros1,
        cpp_base_contributions=zeros1,
        cpp_enhanced_contributions=zeros1,
        ei_premiums=zeros1,
        remitted=zeros1,
    )
    state = set_person(
        state,
        0,
        income=zeroed_income,
        rrif=updated(
            person0.rrif,
            balance=np.full(1, 10_000_000.0),
            annual_minimum=np.full(1, m),
            withdrawn_ytd=zeros1,
        ),
        rrsp=updated(person0.rrsp, balance=zeros1, room=zeros1),
        lif=updated(person0.lif, annual_minimum=zeros1, withdrawn_ytd=zeros1),
    )
    # Person 1 draws nothing towards its own fill: zero RRIF and RRSP balances.
    person1 = state.persons[1]
    state = set_person(
        state,
        1,
        rrif=updated(person1.rrif, balance=zeros1),
        rrsp=updated(person1.rrsp, balance=zeros1, room=zeros1),
    )

    withdrawal = OrderedWithdrawal(
        order=couple_scenario.policies[0].withdrawal.order, taxable_ceiling_bracket=ceiling
    )

    for month in range(1, 13):
        person0 = state.persons[0]
        state = set_person(
            state,
            0,
            income=updated(
                person0.income, db_pension=person0.income.db_pension + monthly_db_pension
            ),
        )
        if month == 12:
            # Phase 6, simulated by hand: force whatever of the minimum the fill's own
            # draws through November have not already met.
            person0 = state.persons[0]
            top_up = np.clip(m - person0.rrif.withdrawn_ytd, 0, None)
            state = set_person(
                state,
                0,
                income=updated(
                    person0.income,
                    rrif_lif_withdrawals=person0.income.rrif_lif_withdrawals + top_up,
                ),
                rrif=updated(
                    person0.rrif,
                    withdrawn_ytd=person0.rrif.withdrawn_ytd + top_up,
                    balance=person0.rrif.balance - top_up,
                ),
            )
        context = make_context(
            n_paths=1,
            n_persons=2,
            year=2026,
            month=month,
            month_index=january_month_index + (month - 1),
            cash_after_flows=np.array([0.0]),
        )
        transfers, _net = withdrawal.transfers(state, context, couple_real_params)
        rrif_transfer = next(
            (t for t in transfers if t.person_index == 0 and t.from_kind == "rrif"), None
        )
        # A December fill of exactly zero (the ``december_fill_is_full=False`` case) emits
        # no transfer at all: ``OrderedWithdrawal`` never emits a zero-amount transfer.
        rrif_amount = rrif_transfer.amount if rrif_transfer is not None else zeros1

        if month < 12 or december_fill_is_full:
            np.testing.assert_allclose(rrif_amount, [expected_monthly_fill], rtol=1e-9)
        else:
            np.testing.assert_allclose(rrif_amount, [0.0], atol=1e-9)

        person0 = state.persons[0]
        state = set_person(
            state,
            0,
            income=updated(
                person0.income,
                rrif_lif_withdrawals=person0.income.rrif_lif_withdrawals + rrif_amount,
            ),
            rrif=updated(
                person0.rrif,
                withdrawn_ytd=person0.rrif.withdrawn_ytd + rrif_amount,
                balance=person0.rrif.balance - rrif_amount,
            ),
        )

    year_net_income = federal_mod.net_income(
        state.persons[0].income, couple_real_params.federal, 0.0, 0.0
    )
    expected_year_net_income = max(edge, 0.6 * edge + m)
    np.testing.assert_allclose(year_net_income, [expected_year_net_income], rtol=1e-9)


def test_kind_major_person_minor_order_on_a_couple(
    couple_scenario, couple_real_params, make_context
):
    state = _opened_couple_state(couple_scenario, couple_real_params)
    order = ("rrif", "lif", "tfsa", "taxable")
    withdrawal = OrderedWithdrawal(order=order, taxable_ceiling_bracket=None)
    context = make_context(
        n_paths=1,
        n_persons=2,
        year=2026,
        month=6,
        month_index=5,
        cash_after_flows=np.array([-500_000.0]),
    )
    transfers, _net = withdrawal.transfers(state, context, couple_real_params)

    assert transfers, "the deficit is far larger than any one account; something must draw"
    kind_rank = {kind: index for index, kind in enumerate(order)}
    ranks = [(kind_rank[t.from_kind], t.person_index) for t in transfers]
    assert ranks == sorted(ranks), ranks
    assert all(t.from_kind != "resp" for t in transfers)

    rrif_persons = {t.person_index for t in transfers if t.from_kind == "rrif"}
    assert rrif_persons == {0, 1}, "the deficit exceeds both RRIFs combined; both must drain"


def test_one_transfer_per_person_kind_when_fill_and_need_hit_one_rrif(
    couple_scenario, couple_real_params, make_context
):
    state = _opened_couple_state(couple_scenario, couple_real_params)
    withdrawal = OrderedWithdrawal(
        order=couple_scenario.policies[0].withdrawal.order, taxable_ceiling_bracket=1
    )

    surplus_context = make_context(
        n_paths=1,
        n_persons=2,
        year=2026,
        month=1,
        month_index=0,
        cash_after_flows=np.array([5000.0]),
    )
    fill_only, _net = withdrawal.transfers(state, surplus_context, couple_real_params)
    fill_only_amount = next(t.amount for t in fill_only if t.person_index == 0)

    deficit_context = make_context(
        n_paths=1,
        n_persons=2,
        year=2026,
        month=1,
        month_index=0,
        cash_after_flows=np.array([-20_000.0]),
    )
    transfers, _net = withdrawal.transfers(state, deficit_context, couple_real_params)
    rrif_for_person_0 = [t for t in transfers if t.person_index == 0 and t.from_kind == "rrif"]

    assert len(rrif_for_person_0) == 1
    assert np.all(rrif_for_person_0[0].amount > fill_only_amount)


def test_lif_is_capped_at_its_remaining_maximum(couple_scenario, couple_real_params, make_context):
    state = _opened_couple_state(couple_scenario, couple_real_params)
    withdrawal = OrderedWithdrawal(order=("lif",), taxable_ceiling_bracket=None)
    context = make_context(
        n_paths=1,
        n_persons=2,
        year=2026,
        month=1,
        month_index=0,
        cash_after_flows=np.array([-1e9]),
    )
    transfers, _net = withdrawal.transfers(state, context, couple_real_params)

    assert len(transfers) == 2
    for index, person in enumerate(state.persons):
        transfer = next(t for t in transfers if t.person_index == index)
        assert transfer.from_kind == "lif"
        np.testing.assert_allclose(transfer.amount, person.lif.annual_maximum)


def test_lif_with_withdrawn_ytd_is_capped_at_what_remains_of_the_maximum(
    couple_scenario, couple_real_params, set_person, make_context
):
    """With some of the year's LIF maximum already drawn, the cap this month is
    ``annual_maximum - withdrawn_ytd``, not ``annual_maximum`` on its own.
    """
    state = _opened_couple_state(couple_scenario, couple_real_params)
    already_withdrawn = state.persons[0].lif.annual_maximum * 0.4
    state = set_person(state, 0, lif=updated(state.persons[0].lif, withdrawn_ytd=already_withdrawn))
    withdrawal = OrderedWithdrawal(order=("lif",), taxable_ceiling_bracket=None)
    context = make_context(
        n_paths=1,
        n_persons=2,
        year=2026,
        month=1,
        month_index=0,
        cash_after_flows=np.array([-1e9]),
    )
    transfers, _net = withdrawal.transfers(state, context, couple_real_params)

    transfer0 = next(t for t in transfers if t.person_index == 0)
    assert transfer0.from_kind == "lif"
    np.testing.assert_allclose(
        transfer0.amount, state.persons[0].lif.annual_maximum - already_withdrawn
    )


def test_keep_g0_when_the_cap_lands_just_past_a_withholding_band_edge(
    scenario, real_params, set_person, make_context
):
    """An RRSP balance of 5,100 with a 4,900 fill already committed for the month. The
    remaining $200 of balance would push the whole withdrawal from $4,900 (below
    ``params/2026/rrif.yaml``'s first ``withholding.edges_each``, so the first
    ``withholding.rates`` band) to $5,100 (the next band), which *nets less* than stopping
    at $4,900 -- so the need-driven walk must keep ``g1 == g0`` rather than draw the rest of
    the balance.

    The fill is produced for real, through step W1, by choosing the person's year-to-date
    employment income so W1's own arithmetic lands on exactly 4,900. At month 1 the
    projection extrapolates that one month's own ytd over the eleven months still to come
    (``run_rate = ytd / 1``, ``projected = run_rate * 11``), so the room below the edge is
    ``edge[1] - 12 * income`` rather than ``edge[1] - income``, and the fill is
    ``edge[1] / 12 - income`` -- not injected directly, since W1's own arithmetic is what
    must produce it.
    """
    state = build_initial_state(scenario, n_paths=1)
    person0 = state.persons[0]
    edge1 = real_params.federal.annual_amounts("brackets.edges_annual", 0)[1]
    income = edge1 / 12 - 4900.0
    state = set_person(state, 0, income=updated(person0.income, employment=np.full(1, income)))
    state = set_person(
        state,
        0,
        rrsp=updated(state.persons[0].rrsp, balance=np.full(1, 5100.0), room=np.zeros(1)),
    )
    state = set_person(state, 0, rrif=updated(state.persons[0].rrif, balance=np.zeros(1)))

    # The person carries no other non-zero income-ledger field and no remaining RRIF/LIF
    # minimum (``build_initial_state`` alone, no ``open_year``, zeroes both): the projection
    # above is driven by ``employment`` alone, so the derivation is not compensating for
    # anything else feeding W1.
    person0 = state.persons[0]
    for field in dataclasses.fields(person0.income):
        if field.name != "employment":
            value = getattr(person0.income, field.name)
            assert np.all(value == 0.0), field.name
    assert np.all(person0.rrif.annual_minimum == 0.0)
    assert np.all(person0.lif.annual_minimum == 0.0)

    # The W1 fill itself, not the need-driven walk, must be what produces 4,900.
    ytd = federal_mod.net_income(person0.income, real_params.federal, 0.0, 0.0)
    projected = _projected_income_rest_of_year(ytd, person0, 1)
    edges = real_params.federal.annual_amounts("brackets.edges_annual", 0)
    available = person0.rrif.balance + person0.rrsp.balance
    w1_fill = fill_to_bracket(ytd, edges, 1, available, 12, projected_income_rest_of_year=projected)
    np.testing.assert_allclose(w1_fill, [4900.0])

    withdrawal = OrderedWithdrawal(order=("rrsp",), taxable_ceiling_bracket=1)
    context = make_context(
        n_paths=1,
        n_persons=1,
        year=2026,
        month=1,
        month_index=0,
        cash_after_flows=np.array([-10_000.0]),
    )

    # Precondition: the cap (5,100) must actually land just past the first withholding
    # band's edge, or this test would not exercise the keep-g0 rule at all.
    edge0 = real_params.rrif.amounts("withholding.edges_each", 0)[0]
    assert 4900.0 <= edge0 < 5100.0, edge0

    transfers, _net = withdrawal.transfers(state, context, real_params)

    rrsp_transfers = [t for t in transfers if t.person_index == 0 and t.from_kind == "rrsp"]
    assert len(rrsp_transfers) == 1
    np.testing.assert_allclose(rrsp_transfers[0].amount, [4900.0])


def test_gross_up_crosses_a_withholding_band_edge_end_to_end(example_values, build_from_values):
    """A single RRSP draw, sized to cross ``params/2026/rrif.yaml``'s first withholding edge,
    through the full monthly step: no floor draw, cash lands at zero.

    Built from the example household stripped to one income-free person with only an RRSP
    (no RRIF minimum to complicate the gross-up: an RRSP's withholding-free floor is always
    zero), and a spending level chosen so the month's whole shortfall -- $4,600 net -- grosses
    up past ``withholding.edges_each[0]``, the table's first edge, into the second band.
    """
    values = example_values()
    person = values["household"]["persons"][0]
    person["cpp"] = {"in_pay_monthly": 0}
    person["oas"] = {"in_pay_monthly": 0}
    person.pop("employment", None)
    person.pop("db_pensions", None)
    person["accounts"] = {
        "cash": {"balance": 0},
        "rrsp": {"balance": 100000, "room": 0},
        "rrif": {"balance": 0},
        "tfsa": {"balance": 0, "room": 0},
        "taxable": {"balance": 0, "acb": 0},
    }
    values["household"]["beneficiaries"] = []
    values["spending"]["schedule"] = [{"from_year": 2026, "annual": 55200.0}]  # 4600/month
    policy = values["policies"][0]
    policy["contribution"] = {
        "weights": {"rrsp": 0.0, "tfsa": 0.0, "taxable": 1.0, "resp": 0.0},
        "spill_order": ["taxable"],
    }
    policy["withdrawal"] = {
        "order": ["rrsp"],
        "taxable_ceiling_bracket": None,
        "fill_pension_credit": False,
    }
    policy["elections"]["cpp_start_age_years"] = {}
    policy["elections"]["oas_start_age_years"] = {}
    del values["grid"]

    scenario, state = build_from_values(values, n_paths=1)

    market = build_market_inputs(scenario.assumptions)
    real_params = real_year(load_year(scenario.start_year), scenario.assumptions.inflation)
    built_policy = build_policy(scenario.policies[0], scenario.household)
    month_returns = np.zeros((len(market.asset_class_names), 1))

    _new_state, record = advance_month_traced(
        state, month_returns, built_policy, market, real_params
    )

    edges = real_params.rrif.amounts("withholding.edges_each", 0)
    assert record.withdrawals[0].rrsp[0] > edges[0], "the draw must actually cross the edge"
    for kind in ("rrsp", "rrif", "lif", "tfsa", "taxable"):
        np.testing.assert_allclose(getattr(record.floor_withdrawals[0], kind), 0.0, atol=1e-9)
    np.testing.assert_allclose(record.cash_close, 0.0, atol=1e-6)


def _sum_bykind(by_kind) -> np.ndarray:
    return sum(getattr(by_kind, kind) for kind in ("rrsp", "rrif", "lif", "tfsa", "taxable"))


def test_same_month_budget_before_room_caps(couple_scenario, couple_real_params):
    """The contribution transfers this month total ``cash_after_flows`` plus this month's own
    withdrawal gross, less its withholding -- the fill fires unconditionally, every month,
    so the first, natural January of ``late_life_couple.yaml`` is already a surplus month
    with a nonzero fill, no manual ``cash_after_flows`` needed.

    Computed end to end through one ``advance_month_traced``, reading the withholding the same
    way phase 8 does (``record.withdrawal_withholding``) rather than re-deriving ``net`` from
    the implementation's own ``OrderedWithdrawal.transfers`` call -- a bug that miscomputes
    ``net`` inside that call would otherwise cancel out of the comparison instead of failing
    it.
    """
    state = build_initial_state(couple_scenario, n_paths=1)
    market = build_market_inputs(couple_scenario.assumptions)
    policy = build_policy(couple_scenario.policies[0], couple_scenario.household)
    month_returns = np.zeros((len(market.asset_class_names), 1))

    _new_state, record = advance_month_traced(
        state, month_returns, policy, market, couple_real_params
    )

    withdrawal_gross_total = sum(_sum_bykind(by_kind) for by_kind in record.withdrawals)
    withholding_total = sum(record.withdrawal_withholding)
    net = withdrawal_gross_total - withholding_total
    expected_budget = np.clip(record.context.cash_after_flows + net, 0, None)

    contribution_total = sum(_sum_bykind(by_kind) for by_kind in record.contributions) + sum(
        record.resp_contributions, np.zeros(1)
    )

    assert np.all(withdrawal_gross_total > 0), "the fill must fire for this to test anything"
    np.testing.assert_allclose(contribution_total, expected_budget)


def test_net_reflects_withholding_on_every_kind(
    couple_scenario, couple_real_params, set_person, make_context
):
    """A test that would pass if ``total_net`` silently summed the gross instead of
    subtracting the withholding: the RRIF fill carries a withholding-free part (its
    remaining minimum), the RRSP fill carries none, and the deficit is large enough that
    ``tfsa`` is also drawn -- untaxed, so it must contribute its full amount, not a
    withheld one.

    ``net`` is computed here from the returned transfers alone, by the withholding rule
    :class:`~engine.policy.withdrawal.OrderedWithdrawal`'s docstring states -- never by
    calling ``net_of`` or anything else private to the module.
    """
    state = _opened_couple_state(couple_scenario, couple_real_params)
    person0 = state.persons[0]
    state = set_person(state, 0, income=updated(person0.income, employment=np.full(1, 5000.0)))
    small_rrif_balance = 1000.0
    state = set_person(
        state, 0, rrif=updated(state.persons[0].rrif, balance=np.full(1, small_rrif_balance))
    )
    state = set_person(
        state,
        0,
        rrsp=updated(state.persons[0].rrsp, balance=np.full(1, 1_000_000.0), room=np.zeros(1)),
    )
    assert np.all(state.persons[0].rrif.annual_minimum > state.persons[0].rrif.withdrawn_ytd), (
        "m must be positive for the RRIF transfer to carry a withholding-free part"
    )

    # rrif/rrsp are excluded from ``order`` on purpose: W1's fill still emits them
    # regardless of ``order``, but the need-driven walk (W2) then leaves their gross
    # exactly at the fill's own planned amount, so the deficit below routes into tfsa.
    withdrawal = OrderedWithdrawal(order=("tfsa", "taxable"), taxable_ceiling_bracket=1)
    context = make_context(
        n_paths=1,
        n_persons=2,
        year=2026,
        month=5,
        month_index=4,
        cash_after_flows=np.array([-1_000_000.0]),
    )

    transfers, net = withdrawal.transfers(state, context, couple_real_params)

    def m_kind(transfer) -> np.ndarray:
        if transfer.from_kind == "rrsp":
            return np.zeros_like(transfer.amount)
        account = getattr(state.persons[transfer.person_index], transfer.from_kind)
        return np.clip(account.annual_minimum - account.withdrawn_ytd, 0, None)

    withholdings: list[np.ndarray] = []
    expected_net = np.zeros(1)
    for transfer in transfers:
        if transfer.from_kind in ("rrsp", "rrif", "lif"):
            above = np.clip(transfer.amount - m_kind(transfer), 0, None)
            withholding = registered_withholding(
                above, couple_real_params.rrif, context.month_index
            )
            withholdings.append(withholding)
            expected_net = expected_net + (transfer.amount - withholding)
        else:
            expected_net = expected_net + transfer.amount

    assert any(np.all(w > 0) for w in withholdings), (
        "at least one transfer must carry positive withholding for this test to exercise it"
    )
    np.testing.assert_allclose(net, expected_net)

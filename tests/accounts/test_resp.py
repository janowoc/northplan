# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registered Education Savings Plan.

Two kinds of test, following ``tests/tax/test_federal.py``: structural tests
against the real 2026 ``resp`` file at zero inflation, and hand-computed
tests against ``SYNTHETIC``, an obviously fake parameter file with round,
wrong numbers chosen so every expected value can be checked by hand.

The cross-year annual-cap test (issue #18's success criterion 5) is a
structural test: it has to run against the real file, since the point is
that the shipped annual grant room, annual grant maximum, and enhanced-tier
eligible-contribution window are each small enough, relative to a year of
aggressive monthly contributions, to bind partway through the year. A
synthetic fixture picked for easy arithmetic would not necessarily reproduce
that; only the real numbers are guaranteed to.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from engine.accounts import resp
from engine.core.indexation import real_year
from engine.core.state import RespState
from engine.params.loader import load_year

JANUARY_YEAR_1 = 0
JANUARY_YEAR_2 = 12


def _state(
    contributions=0.0,
    grants=0.0,
    income=0.0,
    contributions_lifetime=0.0,
    grants_lifetime=0.0,
    grant_room=0.0,
    grant_received_ytd=0.0,
    contributed_ytd=0.0,
) -> RespState:
    return RespState(
        contributions=np.array([contributions], dtype=np.float64),
        grants=np.array([grants], dtype=np.float64),
        income=np.array([income], dtype=np.float64),
        contributions_lifetime=np.array([contributions_lifetime], dtype=np.float64),
        grants_lifetime=np.array([grants_lifetime], dtype=np.float64),
        grant_room=np.array([grant_room], dtype=np.float64),
        grant_received_ytd=np.array([grant_received_ytd], dtype=np.float64),
        contributed_ytd=np.array([contributed_ytd], dtype=np.float64),
        subscriber_index=0,
        education_start_month_index=0,
        education_months=0,
        education_monthly_cost=0.0,
        wound_up=np.array([False]),
    )


# =============================================================================
# Structural tests against the real 2026 file
# =============================================================================


@pytest.fixture
def rp():
    return real_year(load_year(2026), 0.0).resp


def test_grant_room_accrued_zero_after_cessation_age(rp) -> None:
    cessation_age = int(rp.number("grant.cessation_age_years"))
    room_annual = rp.annual_amount("grant.room_annual", JANUARY_YEAR_1)
    assert resp.grant_room_accrued(cessation_age, rp, JANUARY_YEAR_1) == pytest.approx(room_annual)
    assert resp.grant_room_accrued(cessation_age + 1, rp, JANUARY_YEAR_1) == pytest.approx(0.0)


def test_no_grant_in_the_year_after_cessation(rp) -> None:
    cessation_age = int(rp.number("grant.cessation_age_years"))
    state = _state(grant_room=10_000.0, grants_lifetime=0.0, contributions_lifetime=0.0)
    new_state, contributed = resp.contribute(
        state,
        requested=np.array([1000.0]),
        family_income=np.array([0.0]),
        age_at_end_of_year=cessation_age + 1,
        january_month_index=JANUARY_YEAR_1,
        params=rp,
    )
    np.testing.assert_allclose(contributed, [1000.0])  # the contribution itself is unaffected
    np.testing.assert_allclose(new_state.grants, [0.0])
    np.testing.assert_allclose(new_state.grant_received_ytd, [0.0])


def test_annual_cap_binds_partway_through_a_year_of_aggressive_contributions(rp) -> None:
    """Twelve monthly contributions against the real resp.yaml, then a thirteenth month checked.

    The thirteenth month, January of year two, checks what
    :func:`~engine.accounts.resp.grant_on_contribution` would pay without
    actually contributing — enough to show the year boundary reset the
    binding caps.

    A large, constant monthly contribution matches a basic grant every month
    until the annual maximum is exhausted partway through year one; the
    remaining months of that year earn no basic grant; and January of year
    two is matched again because grant_received_ytd resets. The enhanced
    tier's eligible-contribution window is exhausted by the first month's
    contribution alone and is available again in year two because
    contributed_ytd resets.
    """
    monthly_contribution = np.array([500.0])
    family_income = np.array([0.0])  # comfortably below the lowest enhanced cut-off
    age_at_end_of_year = 10  # well below cessation

    state = _state(grant_room=2000.0)  # ample carried-forward room; the annual cap binds first

    basic_history: list[float] = []
    enhanced_history: list[float] = []
    for _ in range(12):
        basic, enhanced = resp.grant_on_contribution(
            monthly_contribution, family_income, state, rp, JANUARY_YEAR_1
        )
        basic_history.append(float(basic[0]))
        enhanced_history.append(float(enhanced[0]))
        state, _ = resp.contribute(
            state, monthly_contribution, family_income, age_at_end_of_year, JANUARY_YEAR_1, rp
        )

    # Basic grant paid every month until the annual maximum is exhausted...
    assert all(amount > 0 for amount in basic_history[:10])
    # ...and nothing in the months after it binds.
    assert basic_history[10] == pytest.approx(0.0)
    assert basic_history[11] == pytest.approx(0.0)

    # The enhanced window is exhausted by the first month's contribution
    # alone (the monthly contribution is at least as large as the window).
    assert enhanced_history[0] > 0.0
    assert all(amount == pytest.approx(0.0) for amount in enhanced_history[1:])

    # January phase of year two: grant_received_ytd and contributed_ytd reset
    # (engine.core.step.open_year's job; simulated here directly).
    state = dataclasses.replace(
        state,
        grant_received_ytd=np.zeros_like(state.grant_received_ytd),
        contributed_ytd=np.zeros_like(state.contributed_ytd),
    )

    basic_year_2, enhanced_year_2 = resp.grant_on_contribution(
        monthly_contribution, family_income, state, rp, JANUARY_YEAR_2
    )
    assert basic_year_2[0] > 0.0  # matched again
    assert enhanced_year_2[0] > 0.0  # available again


# =============================================================================
# Hand-computed tests against a synthetic file
# =============================================================================

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
#: Every value here is deliberately round and wrong. No value in this fixture
#: appears in params/2026/resp.yaml:
#: ``test_the_synthetic_fixture_shares_no_value_with_the_real_file`` enforces
#: it, rather than this comment asserting it.
SYNTHETIC = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  enhanced_income_edges:
    adjustment_months: [1]
    applies_to:
      - grant.enhanced.income_edges_annual
  unindexed:
    adjustment_months: []
    applies_to:
      - contributions.maximum_lifetime
      - grant.room_annual
      - grant.maximum_annual
      - grant.maximum_lifetime
      - grant.enhanced.eligible_contribution_annual

contributions:
  maximum_lifetime: 10000

grant:
  match_rate: 0.5
  room_annual: 200
  maximum_annual: 600
  cessation_age_years: 9
  maximum_lifetime: 3000
  enhanced:
    income_edges_annual: [30000, 60000]
    match_rates: [0.4, 0.2, 0.05]
    eligible_contribution_annual: 100
    income_year_offset: -1

aip:
  penalty_rate: 0.5
"""


def _write(root: Path, name: str, text: str) -> Path:
    year_dir = root / "2026"
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def synth(tmp_path: Path):
    _write(tmp_path, "resp", SYNTHETIC)
    return real_year(load_year(2026, tmp_path), 0.0).resp


def _leaves(value: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    """Every non-mapping, non-sequence value in ``value``, keyed by its path of keys/indices.

    Recurses through nested mappings and sequences (but not strings) so that a table such as
    ``grant.enhanced.match_rates`` yields one leaf per element rather than one leaf holding a
    whole list.
    """
    if isinstance(value, Mapping):
        leaves: dict[tuple[str, ...], Any] = {}
        for key, sub_value in value.items():
            leaves.update(_leaves(sub_value, (*prefix, str(key))))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = {}
        for index, sub_value in enumerate(value):
            leaves.update(_leaves(sub_value, (*prefix, str(index))))
        return leaves
    return {prefix: value}


def test_the_synthetic_fixture_shares_no_value_with_the_real_file(synth) -> None:
    """Enforces the SYNTHETIC docstring's claim in code, not in prose.

    Flattens both files to leaf paths and asserts that no numeric leaf present
    in both has an equal value. The ``indexation`` block is excluded: its
    ``adjustment_months``/``applies_to`` are structural routing the fixture
    must share with the real file to load at all, not a tax parameter that
    could be mistaken for a real one.
    """
    real = real_year(load_year(2026), 0.0).resp
    synthetic_leaves = _leaves(synth.raw.values)
    real_leaves = _leaves(real.raw.values)

    collisions = []
    for path, synthetic_value in synthetic_leaves.items():
        if path[0] == "indexation":
            continue
        if isinstance(synthetic_value, bool) or not isinstance(synthetic_value, (int, float)):
            continue
        if path not in real_leaves:
            continue
        real_value = real_leaves[path]
        if isinstance(real_value, bool) or not isinstance(real_value, (int, float)):
            continue
        if synthetic_value == real_value:
            collisions.append((".".join(path), synthetic_value))

    assert not collisions, (
        f"the synthetic fixture shares a value with params/2026/resp.yaml at: {collisions}. "
        "A synthetic fixture that reproduces a real figure could pass by accident against "
        "the real file; pick a different round number."
    )


def test_synthetic_enhanced_grant_rate_is_a_cliff_table(synth) -> None:
    # edges = [30000, 60000], rates = [0.4, 0.2, 0.05] — a non-zero top band,
    # unlike the real file's, so this test could not pass against
    # params/2026/resp.yaml by accident. A cut-off belongs to the
    # lower-income, higher-rate band: the rate only steps down the dollar
    # AFTER a cut-off, never exactly at it — the opposite convention from a
    # tax bracket edge.
    below = resp.enhanced_grant_rate(np.array([20_000.0]), synth, JANUARY_YEAR_1)
    between = resp.enhanced_grant_rate(np.array([45_000.0]), synth, JANUARY_YEAR_1)
    above = resp.enhanced_grant_rate(np.array([70_000.0]), synth, JANUARY_YEAR_1)
    np.testing.assert_allclose(below, [0.4])
    np.testing.assert_allclose(between, [0.2])
    np.testing.assert_allclose(above, [0.05])

    # All four transitions named in the review: both edges, each exactly at
    # the edge and one dollar above it.
    at_first_edge = resp.enhanced_grant_rate(np.array([30_000.0]), synth, JANUARY_YEAR_1)
    above_first_edge = resp.enhanced_grant_rate(np.array([30_001.0]), synth, JANUARY_YEAR_1)
    at_second_edge = resp.enhanced_grant_rate(np.array([60_000.0]), synth, JANUARY_YEAR_1)
    above_second_edge = resp.enhanced_grant_rate(np.array([60_001.0]), synth, JANUARY_YEAR_1)
    np.testing.assert_allclose(at_first_edge, [0.4])
    np.testing.assert_allclose(above_first_edge, [0.2])
    np.testing.assert_allclose(at_second_edge, [0.2])
    np.testing.assert_allclose(above_second_edge, [0.05])


def test_synthetic_basic_grant_bounded_by_room(synth) -> None:
    # matched = match_rate 0.5 * contribution 1000 = 500, but only 150 of
    # room is left — the smallest of the four bounds.
    state = _state(grant_room=150.0, grants_lifetime=0.0)
    basic = resp.basic_grant(np.array([1000.0]), state, synth, JANUARY_YEAR_1)
    np.testing.assert_allclose(basic, [150.0])


def test_synthetic_basic_grant_bounded_by_annual_maximum(synth) -> None:
    # matched = 0.5 * 1000 = 500; annual bound = maximum_annual 600 -
    # grant_received_ytd 550 = 50, the smallest of the four bounds.
    state = _state(grant_room=10_000.0, grant_received_ytd=550.0)
    basic = resp.basic_grant(np.array([1000.0]), state, synth, JANUARY_YEAR_1)
    np.testing.assert_allclose(basic, [50.0])


def test_synthetic_basic_grant_bounded_by_lifetime_maximum(synth) -> None:
    # matched = 0.5 * 1000 = 500; lifetime bound = maximum_lifetime 3000 -
    # grants_lifetime 2950 = 50, the smallest of the four bounds.
    state = _state(grant_room=10_000.0, grants_lifetime=2950.0)
    basic = resp.basic_grant(np.array([1000.0]), state, synth, JANUARY_YEAR_1)
    np.testing.assert_allclose(basic, [50.0])


def test_synthetic_enhanced_grant_bounded_by_eligible_window(synth) -> None:
    # remaining window = eligible_contribution_annual 100 - contributed_ytd
    # 50 = 50; family income 0 is below the first cut-off, rate 0.4;
    # matched = 0.4 * 50 = 20; the lifetime bound (3000, untouched) does not
    # bind.
    state = _state(contributed_ytd=50.0)
    enhanced = resp.enhanced_grant(
        np.array([1000.0]), np.array([0.0]), state, np.array([0.0]), synth, JANUARY_YEAR_1
    )
    np.testing.assert_allclose(enhanced, [20.0])


def test_synthetic_enhanced_grant_shares_the_lifetime_cap_with_basic(synth) -> None:
    # remaining lifetime = maximum_lifetime 3000 - grants_lifetime 2900 -
    # basic_grant_paid 90 = 10; the matched amount (rate 0.4 * eligible 100
    # = 40) would otherwise be larger.
    state = _state(grants_lifetime=2900.0)
    enhanced = resp.enhanced_grant(
        np.array([1000.0]), np.array([0.0]), state, np.array([90.0]), synth, JANUARY_YEAR_1
    )
    np.testing.assert_allclose(enhanced, [10.0])


def test_synthetic_grant_on_contribution_returns_the_pair_not_the_sum(synth) -> None:
    # contribution 100 (exactly the eligible window): basic = match_rate 0.5
    # * 100 = 50 (room/annual/lifetime bounds all wide open). enhanced:
    # remaining window = 100 - 0 = 100, eligible = min(100, 100) = 100, rate
    # 0.4 (income 0) -> matched = 40, remaining lifetime = 3000 - 0 - 50 =
    # 2950 (not binding) -> enhanced = 40. basic != enhanced != their sum
    # (90), which is the point: the pair is returned, not the total.
    state = _state(grant_room=10_000.0)
    basic, enhanced = resp.grant_on_contribution(
        np.array([100.0]), np.array([0.0]), state, synth, JANUARY_YEAR_1
    )
    np.testing.assert_allclose(basic, [50.0])
    np.testing.assert_allclose(enhanced, [40.0])


def test_synthetic_contribute_caps_at_lifetime_contribution_maximum(synth) -> None:
    # requested 1000 against contributions.maximum_lifetime 10000 -
    # contributions_lifetime 9900 = 100 of room left, so contributed = 100.
    state = _state(contributions_lifetime=9900.0, grant_room=10_000.0)
    new_state, contributed = resp.contribute(
        state,
        requested=np.array([1000.0]),
        family_income=np.array([0.0]),
        age_at_end_of_year=5,  # below the synthetic cessation_age_years of 9
        january_month_index=JANUARY_YEAR_1,
        params=synth,
    )
    np.testing.assert_allclose(contributed, [100.0])
    np.testing.assert_allclose(new_state.contributions_lifetime, [10_000.0])
    # Grants are computed on the capped 100, not the requested 1000: basic =
    # 0.5 * 100 = 50, enhanced = 0.4 * min(100, 100 - 0) = 40 (window and
    # lifetime cap both wide open), total 90.
    np.testing.assert_allclose(new_state.grants, [90.0])


def test_synthetic_contribute_grants_through_the_cessation_age_inclusive(synth) -> None:
    # The mirror of test_no_grant_in_the_year_after_cessation: the guard is
    # age_at_end_of_year > cessation_age, so the beneficiary's final eligible
    # year is the one in which they turn cessation_age, not the year before
    # it. Probes below, at, and above the boundary.
    cessation_age = int(synth.number("grant.cessation_age_years"))
    for age, expect_a_grant in (
        (cessation_age - 1, True),
        (cessation_age, True),
        (cessation_age + 1, False),
    ):
        state = _state(grant_room=10_000.0)
        new_state, _ = resp.contribute(
            state,
            requested=np.array([100.0]),
            family_income=np.array([0.0]),
            age_at_end_of_year=age,
            january_month_index=JANUARY_YEAR_1,
            params=synth,
        )
        if expect_a_grant:
            assert new_state.grants[0] > 0.0, f"age {age}: expected a non-zero grant"
        else:
            np.testing.assert_allclose(new_state.grants, [0.0])


def test_contribute_raises_on_negative_requested(synth) -> None:
    with pytest.raises(ValueError):
        resp.contribute(
            _state(),
            requested=np.array([-1.0]),
            family_income=np.array([0.0]),
            age_at_end_of_year=10,
            january_month_index=JANUARY_YEAR_1,
            params=synth,
        )


def test_grow_accrues_the_whole_return_to_income() -> None:
    state = _state(contributions=1000.0, grants=200.0, income=100.0)
    new_state = resp.grow(state, np.array([0.10]))
    # value = 1300, growth = 130, all into income.
    np.testing.assert_allclose(new_state.income, [230.0])
    np.testing.assert_allclose(new_state.contributions, [1000.0])
    np.testing.assert_allclose(new_state.grants, [200.0])


def test_grow_allows_income_to_go_negative() -> None:
    state = _state(contributions=1000.0, grants=200.0, income=0.0)
    new_state = resp.grow(state, np.array([-0.50]))
    assert new_state.income[0] < 0.0


def test_education_draw_pays_the_scheduled_cost_when_the_plan_can_afford_it() -> None:
    state = dataclasses.replace(
        _state(contributions=1000.0, grants=500.0, income=300.0),
        education_monthly_cost=1000.0,
    )
    new_state, result = resp.education_draw(state)
    np.testing.assert_allclose(result.gross, [1000.0])
    np.testing.assert_allclose(result.shortfall, [0.0])
    # pool = grants + income = 800 (income positive): from_grants = 800*500/800 = 500,
    # from_income = 300, from_contributions = 1000 - 500 - 300 = 200.
    np.testing.assert_allclose(new_state.grants, [0.0])
    np.testing.assert_allclose(new_state.income, [0.0])
    np.testing.assert_allclose(new_state.contributions, [800.0])


def test_education_draw_pays_the_plans_whole_value_when_less_than_the_cost() -> None:
    state = dataclasses.replace(
        _state(contributions=100.0, grants=50.0, income=20.0),
        education_monthly_cost=1000.0,
    )
    _, result = resp.education_draw(state)
    np.testing.assert_allclose(result.gross, [170.0])
    np.testing.assert_allclose(result.shortfall, [830.0])


def test_education_draw_puts_the_whole_payment_in_tax_free() -> None:
    # Verifies docs/limitations.md L30's assumption: the model does not tax
    # the student separately, so the whole EAP is tax-free from the
    # household's point of view and none of it is fully_taxable.
    state = dataclasses.replace(
        _state(contributions=1000.0, grants=500.0, income=300.0),
        education_monthly_cost=400.0,
    )
    _, result = resp.education_draw(state)
    np.testing.assert_allclose(result.tax_free, result.gross)
    np.testing.assert_allclose(result.fully_taxable, [0.0])


def test_education_draw_with_negative_income_draws_from_grant_alone() -> None:
    state = dataclasses.replace(
        _state(contributions=1000.0, grants=300.0, income=-50.0),
        education_monthly_cost=200.0,
    )
    new_state, result = resp.education_draw(state)
    np.testing.assert_allclose(result.gross, [200.0])
    # All 200 comes from grants; income (already negative) is untouched, and
    # contributions are not drawn at all.
    np.testing.assert_allclose(new_state.grants, [100.0])
    np.testing.assert_allclose(new_state.income, [-50.0])
    np.testing.assert_allclose(new_state.contributions, [1000.0])


def test_wind_up_conserves_dollars_including_a_path_with_negative_income() -> None:
    # Path 0: an ordinary year, positive income. Path 1: a loss year, but the
    # plan's total value stays well above the grant balance, so the
    # grants_repaid = min(grants, value) clamp still does not bind — this
    # path alone would pass even if wind_up paid out the whole grant
    # balance regardless of value. Path 2 is the one that actually exercises
    # the clamp: value (200) is below the grant balance (500), so grants_repaid
    # must be capped at value, not at grants.
    #   path 2: contributions=100, grants=500, income=-400 -> value=200.
    #   grants_repaid = min(500, max(200, 0)) = 200.
    #   remaining = 200 - 200 = 0.
    #   to_cash_free = min(100, 0) = 0.
    #   accumulated = 0 - 0 = 0.
    # An unclamped grants_repaid = grants would instead give repaid=500,
    # remaining=-300, to_cash_free=min(100,-300)=-300 — a negative,
    # nonsensical tax-free payment — which is exactly what the clamp exists
    # to prevent.
    state = RespState(
        contributions=np.array([1000.0, 1000.0, 100.0]),
        grants=np.array([500.0, 500.0, 500.0]),
        income=np.array([300.0, -200.0, -400.0]),
        contributions_lifetime=np.array([1000.0, 1000.0, 100.0]),
        grants_lifetime=np.array([500.0, 500.0, 500.0]),
        grant_room=np.array([0.0, 0.0, 0.0]),
        grant_received_ytd=np.array([0.0, 0.0, 0.0]),
        contributed_ytd=np.array([0.0, 0.0, 0.0]),
        subscriber_index=0,
        education_start_month_index=0,
        education_months=0,
        education_monthly_cost=0.0,
        wound_up=np.array([False, False, False]),
    )
    new_state, to_cash_free, accumulated, grants_repaid = resp.wind_up(state)
    value = state.contributions + state.grants + state.income
    np.testing.assert_allclose(to_cash_free + accumulated + grants_repaid, np.clip(value, 0, None))
    assert np.all(accumulated >= 0)

    np.testing.assert_allclose(grants_repaid[2], 200.0)
    np.testing.assert_allclose(to_cash_free[2], 0.0)
    np.testing.assert_allclose(accumulated[2], 0.0)

    np.testing.assert_allclose(new_state.contributions, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(new_state.grants, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(new_state.income, [0.0, 0.0, 0.0])
    assert np.all(new_state.wound_up)


def test_aip_penalty_hand_computed(synth) -> None:
    # penalty_rate 0.5 * accumulated_income 1000 = 500.
    result = resp.aip_penalty(np.array([1000.0]), synth)
    np.testing.assert_allclose(result, [500.0])


# --- erode_nominal -----------------------------------------------------------


def test_erode_nominal_is_identity_at_zero_inflation() -> None:
    state = _state(
        contributions=1000.0,
        grants=500.0,
        income=200.0,
        contributions_lifetime=3000.0,
        grants_lifetime=1500.0,
        grant_room=800.0,
    )
    new_state = resp.erode_nominal(state, 0.0)
    np.testing.assert_allclose(new_state.contributions, [1000.0])
    np.testing.assert_allclose(new_state.grants, [500.0])
    np.testing.assert_allclose(new_state.income, [200.0])
    np.testing.assert_allclose(new_state.contributions_lifetime, [3000.0])
    np.testing.assert_allclose(new_state.grants_lifetime, [1500.0])
    np.testing.assert_allclose(new_state.grant_room, [800.0])


def test_erode_nominal_conserves_the_plans_value() -> None:
    state = _state(
        contributions=1000.0,
        grants=500.0,
        income=200.0,
        contributions_lifetime=3000.0,
        grants_lifetime=1500.0,
        grant_room=800.0,
    )
    before = state.contributions + state.grants + state.income
    new_state = resp.erode_nominal(state, 0.10)
    after = new_state.contributions + new_state.grants + new_state.income
    np.testing.assert_allclose(after, before)
    assert new_state.contributions[0] < 1000.0
    assert new_state.grants[0] < 500.0
    assert new_state.contributions_lifetime[0] < 3000.0
    assert new_state.grants_lifetime[0] < 1500.0
    assert new_state.grant_room[0] < 800.0

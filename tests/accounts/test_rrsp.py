# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registered Retirement Savings Plan.

Two kinds of test, following ``tests/tax/test_federal.py``: structural tests
against the real 2026 ``rrif`` file at zero inflation, where every expected
relationship is read back out of the same ``RealParamSet`` the implementation
reads; and hand-computed tests against ``SYNTHETIC``, an obviously fake
parameter file with round, wrong numbers chosen so every expected value can
be checked by hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.accounts import rrsp
from engine.core.indexation import real_year
from engine.core.state import RrspState
from engine.params.loader import load_year

JANUARY = 0


def _state(balance=0.0, room=0.0, contributed_ytd=0.0, converted=False) -> RrspState:
    return RrspState(
        balance=np.array([balance], dtype=np.float64),
        room=np.array([room], dtype=np.float64),
        contributed_ytd=np.array([contributed_ytd], dtype=np.float64),
        converted_fraction_applied=converted,
    )


# =============================================================================
# Structural tests against the real 2026 file
# =============================================================================


@pytest.fixture
def rr():
    return real_year(load_year(2026), 0.0).rrif


def test_room_accrued_matches_rate_times_income_below_the_dollar_limit(rr) -> None:
    rate = rr.number("rrsp.room.accrual_rate")
    dollar_limit = rr.annual_amount("rrsp.room.dollar_limit_annual", JANUARY)
    income = (dollar_limit / rate) / 2  # comfortably below the cap
    result = rrsp.room_accrued(income, rr, JANUARY)
    np.testing.assert_allclose(result, rate * income)


def test_room_accrued_capped_at_the_dollar_limit(rr) -> None:
    rate = rr.number("rrsp.room.accrual_rate")
    dollar_limit = rr.annual_amount("rrsp.room.dollar_limit_annual", JANUARY)
    huge_income = (dollar_limit / rate) * 10
    result = rrsp.room_accrued(huge_income, rr, JANUARY)
    np.testing.assert_allclose(result, dollar_limit)


def test_room_accrued_floored_at_zero(rr) -> None:
    result = rrsp.room_accrued(-1000.0, rr, JANUARY)
    np.testing.assert_allclose(result, 0.0)


def test_must_convert_at_and_only_at_conversion_age(rr) -> None:
    conversion_age = int(rr.number("conversion_age_years"))
    assert rrsp.must_convert(conversion_age - 1, rr) is False
    assert rrsp.must_convert(conversion_age, rr) is True


# =============================================================================
# Hand-computed tests against a synthetic file
# =============================================================================

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
SYNTHETIC = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  rrsp_limit:
    adjustment_months: [1]
    applies_to:
      - rrsp.room.dollar_limit_annual
  unindexed:
    adjustment_months: []
    applies_to:
      - withholding.edges_each

conversion_age_years: 60

rrsp:
  room:
    accrual_rate: 0.5
    dollar_limit_annual: 1000

withholding:
  edges_each: [100, 200]
  rates: [0.15, 0.25, 0.35]

rrif:
  minimum_factors:
    pre_table:
      formula_constant: 100
    by_age:
      60: 0.05
      61: 0.06
      62: 0.07
    terminal_age_years: 62
"""


def _write(root: Path, name: str, text: str) -> Path:
    year_dir = root / "2026"
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def synth(tmp_path: Path):
    _write(tmp_path, "rrif", SYNTHETIC)
    return real_year(load_year(2026, tmp_path), 0.0).rrif


def test_synthetic_room_accrued_hand_computed(synth) -> None:
    # rate 0.5 * income 800 = 400, below the 1000 cap.
    result = rrsp.room_accrued(800.0, synth, JANUARY)
    np.testing.assert_allclose(result, 400.0)


def test_synthetic_contribute_caps_at_room_not_penalised(synth) -> None:
    state = _state(balance=100.0, room=50.0)
    new_state, contributed = rrsp.contribute(state, np.array([80.0]))
    np.testing.assert_allclose(contributed, [50.0])
    np.testing.assert_allclose(new_state.balance, [150.0])
    np.testing.assert_allclose(new_state.room, [0.0])
    np.testing.assert_allclose(new_state.contributed_ytd, [50.0])


def test_contribute_raises_on_negative_requested() -> None:
    with pytest.raises(ValueError):
        rrsp.contribute(_state(room=100.0), np.array([-5.0]))


def test_synthetic_withdraw_is_fully_taxable(synth) -> None:
    state = _state(balance=500.0)
    new_state, result = rrsp.withdraw(state, np.array([200.0]))
    np.testing.assert_allclose(result.gross, [200.0])
    np.testing.assert_allclose(result.fully_taxable, [200.0])
    np.testing.assert_allclose(result.capital_gain, [0.0])
    np.testing.assert_allclose(result.tax_free, [0.0])
    np.testing.assert_allclose(new_state.balance, [300.0])


def test_withdraw_raises_on_negative_requested() -> None:
    with pytest.raises(ValueError):
        rrsp.withdraw(_state(balance=100.0), np.array([-5.0]))


def test_convert_moves_the_fraction_and_sets_the_flag() -> None:
    state = _state(balance=1000.0, converted=False)
    new_state, moved = rrsp.convert(state, 0.25)
    np.testing.assert_allclose(moved, [250.0])
    np.testing.assert_allclose(new_state.balance, [750.0])
    assert new_state.converted_fraction_applied is True


def test_convert_raises_on_fraction_above_one() -> None:
    with pytest.raises(ValueError, match=r"1\.5"):
        rrsp.convert(_state(balance=1000.0), 1.5)


def test_convert_raises_on_fraction_below_zero() -> None:
    with pytest.raises(ValueError, match=r"-0\.2"):
        rrsp.convert(_state(balance=1000.0), -0.2)


# --- erode_nominal -----------------------------------------------------------


def test_erode_nominal_is_identity_at_zero_inflation() -> None:
    state = _state(room=500.0)
    new_state = rrsp.erode_nominal(state, 0.0)
    np.testing.assert_allclose(new_state.room, [500.0])


def test_erode_nominal_shrinks_room_and_nothing_else_at_positive_inflation() -> None:
    state = _state(balance=1000.0, room=500.0, contributed_ytd=100.0)
    new_state = rrsp.erode_nominal(state, 0.10)
    assert new_state.room[0] < 500.0
    np.testing.assert_allclose(new_state.balance, [1000.0])
    np.testing.assert_allclose(new_state.contributed_ytd, [100.0])


# --- spousal_rollover (#36) ---------------------------------------------------


def _multi(balance, room, contributed_ytd) -> RrspState:
    return RrspState(
        balance=np.array(balance, dtype=np.float64),
        room=np.array(room, dtype=np.float64),
        contributed_ytd=np.array(contributed_ytd, dtype=np.float64),
        converted_fraction_applied=False,
    )


def test_spousal_rollover_moves_balance_and_zeroes_the_deceased() -> None:
    mask = np.array([True, False])
    deceased = _multi([10_000.0, 20_000.0], room=[1000.0, 2000.0], contributed_ytd=[0.0, 0.0])
    survivor = _multi([5_000.0, 6_000.0], room=[500.0, 600.0], contributed_ytd=[0.0, 0.0])

    new_deceased, new_survivor = rrsp.spousal_rollover(deceased, survivor, mask)

    np.testing.assert_allclose(new_deceased.balance, [0.0, 20_000.0])
    np.testing.assert_allclose(new_survivor.balance, [15_000.0, 6_000.0])


def test_spousal_rollover_leaves_room_and_contributed_ytd_untouched_on_both_sides() -> None:
    mask = np.array([True])
    deceased = _multi([10_000.0], room=[1234.0], contributed_ytd=[111.0])
    survivor = _multi([5_000.0], room=[4321.0], contributed_ytd=[222.0])

    new_deceased, new_survivor = rrsp.spousal_rollover(deceased, survivor, mask)

    np.testing.assert_allclose(new_deceased.room, [1234.0])
    np.testing.assert_allclose(new_deceased.contributed_ytd, [111.0])
    np.testing.assert_allclose(new_survivor.room, [4321.0])
    np.testing.assert_allclose(new_survivor.contributed_ytd, [222.0])
    assert new_deceased.converted_fraction_applied is False
    assert new_survivor.converted_fraction_applied is False

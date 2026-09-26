# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tax-Free Savings Account.

Structural tests against the real 2026 file, plus hand-computed tests against
a synthetic one, following ``tests/tax/test_federal.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.accounts import tfsa
from engine.core.indexation import real_year
from engine.core.state import TfsaState
from engine.params.loader import load_year

JANUARY = 0


def _state(balance=0.0, room=0.0, withdrawn_this_year=0.0) -> TfsaState:
    return TfsaState(
        balance=np.array([balance], dtype=np.float64),
        room=np.array([room], dtype=np.float64),
        withdrawn_this_year=np.array([withdrawn_this_year], dtype=np.float64),
    )


# =============================================================================
# Structural tests against the real 2026 file
# =============================================================================


@pytest.fixture
def tf():
    return real_year(load_year(2026), 0.0).tfsa


def test_room_accrued_zero_below_eligibility_age(tf) -> None:
    eligibility_age = int(tf.number("room.eligibility_age_years"))
    assert tfsa.room_accrued(eligibility_age - 1, tf, JANUARY) == pytest.approx(0.0)


def test_room_accrued_the_annual_amount_at_and_above_eligibility_age(tf) -> None:
    eligibility_age = int(tf.number("room.eligibility_age_years"))
    amount = tf.annual_amount("room.amount_annual", JANUARY)
    assert tfsa.room_accrued(eligibility_age, tf, JANUARY) == pytest.approx(amount)
    assert tfsa.room_accrued(eligibility_age + 10, tf, JANUARY) == pytest.approx(amount)


def test_restore_room_lag_is_exactly_one_year(tf) -> None:
    # A single chain, not two disconnected states: a restore_room that added
    # state.balance instead of state.withdrawn_this_year would need both
    # halves changed to be caught, rather than one.
    assert tf.number("recontribution.restored_after_n_year_ends") == pytest.approx(1.0)
    state = _state(balance=1000.0, room=0.0, withdrawn_this_year=0.0)

    withdrawn_state, _ = tfsa.withdraw(state, np.array([300.0]))
    np.testing.assert_allclose(withdrawn_state.room, [0.0])  # no room this year

    restored_state = tfsa.restore_room(withdrawn_state, tf)
    np.testing.assert_allclose(restored_state.room, [300.0])
    np.testing.assert_allclose(restored_state.withdrawn_this_year, [0.0])


# =============================================================================
# Hand-computed tests against a synthetic file
# =============================================================================

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
SYNTHETIC = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  contribution_room:
    adjustment_months: [1]
    applies_to:
      - room.amount_annual

room:
  eligibility_age_years: 20
  amount_annual: 1000

recontribution:
  restored_after_n_year_ends: 1
"""

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
#: A bad lag, to exercise restore_room's guard.
SYNTHETIC_BAD_LAG = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  contribution_room:
    adjustment_months: [1]
    applies_to:
      - room.amount_annual

room:
  eligibility_age_years: 20
  amount_annual: 1000

recontribution:
  restored_after_n_year_ends: 2
"""


def _write(root: Path, name: str, text: str) -> Path:
    year_dir = root / "2026"
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def synth(tmp_path: Path):
    _write(tmp_path, "tfsa", SYNTHETIC)
    return real_year(load_year(2026, tmp_path), 0.0).tfsa


def test_synthetic_room_accrued_hand_computed(synth) -> None:
    assert tfsa.room_accrued(20, synth, JANUARY) == pytest.approx(1000.0)
    assert tfsa.room_accrued(19, synth, JANUARY) == pytest.approx(0.0)


def test_contribute_caps_at_room() -> None:
    state = _state(balance=100.0, room=50.0)
    new_state, contributed = tfsa.contribute(state, np.array([80.0]))
    np.testing.assert_allclose(contributed, [50.0])
    np.testing.assert_allclose(new_state.balance, [150.0])
    np.testing.assert_allclose(new_state.room, [0.0])


def test_contribute_raises_on_negative_requested() -> None:
    with pytest.raises(ValueError):
        tfsa.contribute(_state(room=100.0), np.array([-5.0]))


def test_withdraw_creates_no_room_this_year() -> None:
    state = _state(balance=1000.0, room=0.0, withdrawn_this_year=0.0)
    new_state, result = tfsa.withdraw(state, np.array([300.0]))
    np.testing.assert_allclose(result.tax_free, [300.0])
    np.testing.assert_allclose(result.fully_taxable, [0.0])
    np.testing.assert_allclose(result.capital_gain, [0.0])
    np.testing.assert_allclose(new_state.room, [0.0])
    np.testing.assert_allclose(new_state.withdrawn_this_year, [300.0])


def test_withdraw_raises_on_negative_requested() -> None:
    with pytest.raises(ValueError):
        tfsa.withdraw(_state(balance=100.0), np.array([-5.0]))


def test_restore_room_raises_on_a_lag_other_than_one(tmp_path: Path) -> None:
    _write(tmp_path, "tfsa", SYNTHETIC_BAD_LAG)
    bad = real_year(load_year(2026, tmp_path), 0.0).tfsa
    with pytest.raises(ValueError, match="2"):
        tfsa.restore_room(_state(withdrawn_this_year=100.0), bad)


# --- erode_nominal -----------------------------------------------------------


def test_erode_nominal_is_identity_at_zero_inflation() -> None:
    state = _state(room=500.0, withdrawn_this_year=200.0)
    new_state = tfsa.erode_nominal(state, 0.0)
    np.testing.assert_allclose(new_state.room, [500.0])
    np.testing.assert_allclose(new_state.withdrawn_this_year, [200.0])


def test_erode_nominal_shrinks_room_and_withdrawn_and_nothing_else() -> None:
    state = _state(balance=1000.0, room=500.0, withdrawn_this_year=200.0)
    new_state = tfsa.erode_nominal(state, 0.10)
    assert new_state.room[0] < 500.0
    assert new_state.withdrawn_this_year[0] < 200.0
    np.testing.assert_allclose(new_state.balance, [1000.0])


# =============================================================================
# successor_holder (#36)
# =============================================================================


def _multi(balance, room, withdrawn_this_year) -> TfsaState:
    return TfsaState(
        balance=np.array(balance, dtype=np.float64),
        room=np.array(room, dtype=np.float64),
        withdrawn_this_year=np.array(withdrawn_this_year, dtype=np.float64),
    )


def test_successor_holder_moves_balance_and_zeroes_the_deceased() -> None:
    mask = np.array([True, False])
    deceased = _multi([10_000.0, 20_000.0], room=[1000.0, 2000.0], withdrawn_this_year=[0.0, 0.0])
    survivor = _multi([5_000.0, 6_000.0], room=[500.0, 600.0], withdrawn_this_year=[0.0, 0.0])

    new_deceased, new_survivor = tfsa.successor_holder(deceased, survivor, mask)

    np.testing.assert_allclose(new_deceased.balance, [0.0, 20_000.0])
    np.testing.assert_allclose(new_survivor.balance, [15_000.0, 6_000.0])


def test_successor_holder_leaves_room_and_withdrawn_this_year_untouched_on_both_sides() -> None:
    mask = np.array([True])
    deceased = _multi([10_000.0], room=[1234.0], withdrawn_this_year=[111.0])
    survivor = _multi([5_000.0], room=[4321.0], withdrawn_this_year=[222.0])

    new_deceased, new_survivor = tfsa.successor_holder(deceased, survivor, mask)

    np.testing.assert_allclose(new_deceased.room, [1234.0])
    np.testing.assert_allclose(new_deceased.withdrawn_this_year, [111.0])
    np.testing.assert_allclose(new_survivor.room, [4321.0])
    np.testing.assert_allclose(new_survivor.withdrawn_this_year, [222.0])


def test_successor_holder_untouched_outside_mask() -> None:
    mask = np.array([False])
    deceased = _multi([10_000.0], room=[0.0], withdrawn_this_year=[0.0])
    survivor = _multi([5_000.0], room=[0.0], withdrawn_this_year=[0.0])

    new_deceased, new_survivor = tfsa.successor_holder(deceased, survivor, mask)

    np.testing.assert_allclose(new_deceased.balance, [10_000.0])
    np.testing.assert_allclose(new_survivor.balance, [5_000.0])

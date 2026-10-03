# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registered Retirement Income Fund.

Structural tests against the real 2026 ``rrif`` file, plus hand-computed
tests against a synthetic one, following ``tests/tax/test_federal.py``.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from engine.accounts import rrif
from engine.core.indexation import real_year
from engine.core.state import RrifState
from engine.params.loader import load_year

JANUARY = 0


def _state(
    balance=0.0, annual_minimum=0.0, withdrawn_ytd=0.0, opened_year=2025, inherited_fraction=0.0
) -> RrifState:
    return RrifState(
        balance=np.array([balance], dtype=np.float64),
        annual_minimum=np.array([annual_minimum], dtype=np.float64),
        withdrawn_ytd=np.array([withdrawn_ytd], dtype=np.float64),
        opened_year=opened_year,
        inherited_fraction=np.array([inherited_fraction], dtype=np.float64),
    )


# =============================================================================
# Structural tests against the real 2026 file
# =============================================================================


@pytest.fixture
def rr():
    return real_year(load_year(2026), 0.0).rrif


def test_minimum_factor_at_the_pre_table_boundary(rr) -> None:
    table = rr.get("rrif.minimum_factors.by_age")
    first_age = min(int(age) for age in table)
    constant = rr.number("rrif.minimum_factors.pre_table.formula_constant")
    below = rrif.minimum_factor(first_age - 1, rr)
    assert below == pytest.approx(1 / (constant - (first_age - 1)))


def test_minimum_factor_at_the_first_table_age(rr) -> None:
    table = rr.get("rrif.minimum_factors.by_age")
    first_age = min(int(age) for age in table)
    at_first = rrif.minimum_factor(first_age, rr)
    assert at_first == pytest.approx(float(table[str(first_age)]))


def test_minimum_factor_at_and_above_the_terminal_age(rr) -> None:
    terminal_age = int(rr.number("rrif.minimum_factors.terminal_age_years"))
    table = rr.get("rrif.minimum_factors.by_age")
    terminal_factor = float(table[str(terminal_age)])
    assert rrif.minimum_factor(terminal_age, rr) == pytest.approx(terminal_factor)
    assert rrif.minimum_factor(terminal_age + 5, rr) == pytest.approx(terminal_factor)


def test_minimum_factor_is_non_decreasing_across_the_table(rr) -> None:
    table = rr.get("rrif.minimum_factors.by_age")
    ages = sorted(int(age) for age in table)
    factors = [rrif.minimum_factor(age, rr) for age in ages]
    assert all(b >= a for a, b in itertools.pairwise(factors))


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


def test_synthetic_minimum_factor_hand_computed(synth) -> None:
    assert rrif.minimum_factor(55, synth) == pytest.approx(1 / (100 - 55))
    assert rrif.minimum_factor(61, synth) == pytest.approx(0.06)
    assert rrif.minimum_factor(70, synth) == pytest.approx(0.07)


def test_minimum_withdrawal_zero_when_never_opened(synth) -> None:
    result = rrif.minimum_withdrawal(
        np.array([100_000.0]), 61, opened_year=None, year=2027, params=synth
    )
    np.testing.assert_allclose(result, [0.0])


def test_minimum_withdrawal_zero_in_the_year_opened(synth) -> None:
    result = rrif.minimum_withdrawal(
        np.array([100_000.0]), 61, opened_year=2027, year=2027, params=synth
    )
    np.testing.assert_allclose(result, [0.0])


def test_minimum_withdrawal_non_zero_the_year_after_opening(synth) -> None:
    result = rrif.minimum_withdrawal(
        np.array([100_000.0]), 61, opened_year=2026, year=2027, params=synth
    )
    np.testing.assert_allclose(result, [6_000.0])


def test_minimum_still_required_is_zero_except_in_december() -> None:
    annual_minimum = np.array([1000.0])
    withdrawn_ytd = np.array([200.0])
    for months_remaining in range(2, 13):
        result = rrif.minimum_still_required(annual_minimum, withdrawn_ytd, months_remaining)
        np.testing.assert_allclose(result, [0.0])
    result = rrif.minimum_still_required(annual_minimum, withdrawn_ytd, 1)
    np.testing.assert_allclose(result, [800.0])


def test_minimum_still_required_floors_at_zero_when_already_met() -> None:
    result = rrif.minimum_still_required(np.array([1000.0]), np.array([1200.0]), 1)
    np.testing.assert_allclose(result, [0.0])


def test_withdraw_takes_at_least_the_floor() -> None:
    state = _state(balance=10_000.0, annual_minimum=1000.0, withdrawn_ytd=0.0)
    new_state, result, _ = rrif.withdraw(state, requested=np.array([0.0]), floor=np.array([300.0]))
    np.testing.assert_allclose(result.gross, [300.0])
    np.testing.assert_allclose(result.fully_taxable, [300.0])
    np.testing.assert_allclose(new_state.withdrawn_ytd, [300.0])


def test_withdraw_raises_on_negative_requested() -> None:
    state = _state(balance=10_000.0)
    with pytest.raises(ValueError):
        rrif.withdraw(state, requested=np.array([-1.0]), floor=np.array([0.0]))


def test_above_minimum_is_zero_until_the_annual_minimum_is_satisfied() -> None:
    # The formula in the docstring: max(0, gross - max(0, annual_minimum -
    # withdrawn_ytd)), never gross - floor.
    state = _state(balance=10_000.0, annual_minimum=1000.0, withdrawn_ytd=0.0)
    _, result, above_minimum = rrif.withdraw(
        state, requested=np.array([600.0]), floor=np.array([0.0])
    )
    np.testing.assert_allclose(result.gross, [600.0])
    np.testing.assert_allclose(above_minimum, [0.0])


def test_above_minimum_is_the_whole_gross_once_the_minimum_is_satisfied() -> None:
    state = _state(balance=10_000.0, annual_minimum=1000.0, withdrawn_ytd=1000.0)
    _, _, above_minimum = rrif.withdraw(state, requested=np.array([600.0]), floor=np.array([0.0]))
    np.testing.assert_allclose(above_minimum, [600.0])


def test_above_minimum_straddling_the_annual_minimum() -> None:
    # annual_minimum=1000, withdrawn_ytd=800: 200 left of the minimum.
    # Requesting 500 satisfies it and 300 is above_minimum, not 500 - floor.
    state = _state(balance=10_000.0, annual_minimum=1000.0, withdrawn_ytd=800.0)
    _, result, above_minimum = rrif.withdraw(
        state, requested=np.array([500.0]), floor=np.array([0.0])
    )
    np.testing.assert_allclose(result.gross, [500.0])
    np.testing.assert_allclose(above_minimum, [300.0])


def test_receive_conversion_sets_opened_year_only_when_none() -> None:
    never_opened = _state(balance=0.0, opened_year=None)
    updated = rrif.receive_conversion(never_opened, np.array([5000.0]), year=2030)
    assert updated.opened_year == 2030
    np.testing.assert_allclose(updated.balance, [5000.0])

    already_open = _state(balance=1000.0, opened_year=2020)
    updated = rrif.receive_conversion(already_open, np.array([500.0]), year=2030)
    assert updated.opened_year == 2020
    np.testing.assert_allclose(updated.balance, [1500.0])


def test_there_is_no_erode_nominal() -> None:
    assert not hasattr(rrif, "erode_nominal")


# =============================================================================
# spousal_rollover (#36)
# =============================================================================


def _multi(
    balance, annual_minimum, withdrawn_ytd, opened_year, inherited_fraction=None
) -> RrifState:
    return RrifState(
        balance=np.array(balance, dtype=np.float64),
        annual_minimum=np.array(annual_minimum, dtype=np.float64),
        withdrawn_ytd=np.array(withdrawn_ytd, dtype=np.float64),
        opened_year=opened_year,
        inherited_fraction=(
            np.zeros(len(balance))
            if inherited_fraction is None
            else np.array(inherited_fraction, dtype=np.float64)
        ),
    )


def test_spousal_rollover_moves_balance_and_zeroes_the_deceased() -> None:
    mask = np.array([True, False])
    deceased = _multi([10_000.0, 20_000.0], [500.0, 600.0], [100.0, 200.0], opened_year=2020)
    survivor = _multi([5_000.0, 6_000.0], [50.0, 60.0], [10.0, 20.0], opened_year=2019)

    new_deceased, new_survivor = rrif.spousal_rollover(deceased, survivor, mask)

    np.testing.assert_allclose(new_deceased.balance, [0.0, 20_000.0])
    np.testing.assert_allclose(new_survivor.balance, [15_000.0, 6_000.0])


def test_spousal_rollover_leaves_the_survivors_minimum_and_withdrawn_ytd_untouched() -> None:
    mask = np.array([True])
    deceased = _multi([10_000.0], [500.0], [100.0], opened_year=2020)
    survivor = _multi([5_000.0], [50.0], [10.0], opened_year=2019)

    _, new_survivor = rrif.spousal_rollover(deceased, survivor, mask)

    np.testing.assert_allclose(new_survivor.annual_minimum, [50.0])
    np.testing.assert_allclose(new_survivor.withdrawn_ytd, [10.0])


def test_spousal_rollover_opened_year_none_survivor_takes_the_deceaseds() -> None:
    mask = np.array([True])
    deceased = _multi([10_000.0], [0.0], [0.0], opened_year=2015)
    survivor = _multi([0.0], [0.0], [0.0], opened_year=None)

    _, new_survivor = rrif.spousal_rollover(deceased, survivor, mask)

    assert new_survivor.opened_year == 2015


def test_spousal_rollover_opened_year_survivors_own_is_kept() -> None:
    mask = np.array([True])
    deceased = _multi([10_000.0], [0.0], [0.0], opened_year=2015)
    survivor = _multi([5_000.0], [0.0], [0.0], opened_year=2010)

    _, new_survivor = rrif.spousal_rollover(deceased, survivor, mask)

    assert new_survivor.opened_year == 2010


def test_spousal_rollover_opened_year_unchanged_when_mask_is_all_false() -> None:
    mask = np.array([False])
    deceased = _multi([10_000.0], [0.0], [0.0], opened_year=2015)
    survivor = _multi([0.0], [0.0], [0.0], opened_year=None)

    new_deceased, new_survivor = rrif.spousal_rollover(deceased, survivor, mask)

    assert new_survivor.opened_year is None
    np.testing.assert_allclose(new_deceased.balance, [10_000.0])
    np.testing.assert_allclose(new_survivor.balance, [0.0])


# =============================================================================
# inherited_fraction (#76)
# =============================================================================
# Balances and fractions below are synthetic.


def test_spousal_rollover_sets_the_survivors_inherited_fraction() -> None:
    deceased = _multi([100.0, 100.0], [0.0, 0.0], [0.0, 0.0], 2020)
    survivor = _multi([300.0, 3.0], [0.0, 0.0], [0.0, 0.0], 2021, inherited_fraction=[0.0, 0.1])
    mask = np.array([True, False])

    _, new_survivor = rrif.spousal_rollover(deceased, survivor, mask)

    np.testing.assert_allclose(new_survivor.inherited_fraction[0], 100.0 / 400.0, rtol=1e-15)
    assert new_survivor.inherited_fraction[1] == 0.1  # outside the mask, bit for bit


def test_spousal_rollover_into_an_empty_rrif_makes_it_wholly_inherited() -> None:
    deceased = _multi([250.0], [0.0], [0.0], 2020)
    survivor = _multi([0.0], [0.0], [0.0], None)

    _, new_survivor = rrif.spousal_rollover(deceased, survivor, np.array([True]))

    np.testing.assert_array_equal(new_survivor.inherited_fraction, [1.0])


def test_spousal_rollover_leaves_the_deceaseds_inherited_fraction_unchanged() -> None:
    deceased = _multi([250.0], [0.0], [0.0], 2020, inherited_fraction=[0.3])
    survivor = _multi([100.0], [0.0], [0.0], 2021)

    new_deceased, _ = rrif.spousal_rollover(deceased, survivor, np.array([True]))

    np.testing.assert_array_equal(new_deceased.inherited_fraction, [0.3])


def test_receive_conversion_dilutes_the_inherited_fraction() -> None:
    state = _multi([300.0, 3.0], [0.0, 0.0], [0.0, 0.0], 2021, inherited_fraction=[0.5, 0.1])

    new_state = rrif.receive_conversion(state, np.array([100.0, 0.0]), 2026)

    np.testing.assert_allclose(new_state.inherited_fraction[0], 0.5 * 300.0 / 400.0, rtol=1e-15)
    assert new_state.inherited_fraction[1] == 0.1  # nothing converted, bit for bit


def test_withdraw_leaves_the_inherited_fraction_unchanged() -> None:
    state = _state(balance=1_000.0, opened_year=2020, inherited_fraction=0.4)

    new_state, result, _ = rrif.withdraw(state, np.array([300.0]), np.array([0.0]))

    assert result.gross[0] == 300.0  # guard: something was withdrawn
    np.testing.assert_array_equal(new_state.inherited_fraction, [0.4])

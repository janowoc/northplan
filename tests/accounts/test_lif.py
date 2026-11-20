# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Life Income Fund: the maximum table and the withdraw clamp.

Runs against the real Alberta jurisdiction file — a maximum-factor table is
exactly the kind of thing a synthetic fixture would obscure rather than
clarify, since the required checks (first row, one below it) are about the
table's *shape*, which the real file already has.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.accounts import lif, rrif
from engine.core.indexation import real_year
from engine.core.state import LifState, RrifState
from engine.params.loader import (
    MalformedParamFileError,
    MissingParameterError,
    load_year,
    shown_value,
)


@pytest.fixture
def ab():
    return real_year(load_year(2026), 0.0).jurisdiction("ab")


def _state(
    balance=0.0, annual_minimum=0.0, annual_maximum=0.0, withdrawn_ytd=0.0, inherited_fraction=0.0
) -> LifState:
    return LifState(
        balance=np.array([balance], dtype=np.float64),
        jurisdiction="ab",
        annual_minimum=np.array([annual_minimum], dtype=np.float64),
        annual_maximum=np.array([annual_maximum], dtype=np.float64),
        withdrawn_ytd=np.array([withdrawn_ytd], dtype=np.float64),
        opened_year=2020,
        inherited_fraction=np.array([inherited_fraction], dtype=np.float64),
    )


def test_has_maximum_true_for_alberta(ab) -> None:
    assert lif.has_maximum(ab) is True


def test_maximum_withdrawal_at_the_unlocking_age_is_the_tables_first_row(ab) -> None:
    table = ab.get("lif.maximum_factors.by_age")
    first_age = min(int(age) for age in table)
    unlocking_age = int(ab.number("lif.unlocking_age_years"))
    assert first_age == unlocking_age

    factor = float(table[str(first_age)])
    result = lif.maximum_withdrawal(np.array([100_000.0]), first_age, ab)
    np.testing.assert_allclose(result, [100_000.0 * factor])


def test_maximum_withdrawal_raises_one_age_below_the_first_row(ab) -> None:
    table = ab.get("lif.maximum_factors.by_age")
    first_age = min(int(age) for age in table)
    with pytest.raises(ValueError, match=str(first_age)):
        lif.maximum_withdrawal(np.array([100_000.0]), first_age - 1, ab)


def test_maximum_withdrawal_at_and_above_terminal_age(ab) -> None:
    terminal_age = int(ab.number("lif.maximum_factors.terminal_age_years"))
    table = ab.get("lif.maximum_factors.by_age")
    terminal_factor = float(table[str(terminal_age)])
    at_terminal = lif.maximum_withdrawal(np.array([50_000.0]), terminal_age, ab)
    above_terminal = lif.maximum_withdrawal(np.array([50_000.0]), terminal_age + 3, ab)
    np.testing.assert_allclose(at_terminal, [50_000.0 * terminal_factor])
    np.testing.assert_allclose(above_terminal, [50_000.0 * terminal_factor])


def test_maximum_factor_is_never_read_as_greater_of_factor_and_return(ab) -> None:
    # docs/limitations.md L27: the flag exists in ab.yaml but must not be
    # consulted by this implementation.
    assert ab.get("lif.maximum_is_greater_of_factor_and_prior_year_return") is True
    table = ab.get("lif.maximum_factors.by_age")
    age = min(int(a) for a in table)
    factor = float(table[str(age)])
    result = lif.maximum_withdrawal(np.array([10_000.0]), age, ab)
    np.testing.assert_allclose(result, [10_000.0 * factor])


def test_withdraw_raises_when_floor_exceeds_maximum_remaining() -> None:
    state = _state(balance=100_000.0)
    with pytest.raises(ValueError):
        lif.withdraw(
            state,
            requested=np.array([0.0]),
            floor=np.array([500.0]),
            maximum_remaining=np.array([400.0]),
        )


def test_withdraw_does_not_raise_when_floor_merely_exceeds_the_balance() -> None:
    state = _state(balance=100.0)
    new_state, result, _ = lif.withdraw(
        state,
        requested=np.array([0.0]),
        floor=np.array([500.0]),
        maximum_remaining=np.array([1000.0]),
    )
    np.testing.assert_allclose(result.gross, [100.0])
    np.testing.assert_allclose(result.shortfall, [400.0])
    np.testing.assert_allclose(new_state.balance, [0.0])


def test_withdraw_clamps_requested_into_the_annual_bounds() -> None:
    state = _state(balance=100_000.0)
    _, result, _ = lif.withdraw(
        state,
        requested=np.array([50_000.0]),
        floor=np.array([1000.0]),
        maximum_remaining=np.array([10_000.0]),
    )
    np.testing.assert_allclose(result.gross, [10_000.0])


def test_above_minimum_formula_mirrors_rrif() -> None:
    state = _state(balance=100_000.0, annual_minimum=1000.0, withdrawn_ytd=800.0)
    _, result, above_minimum = lif.withdraw(
        state,
        requested=np.array([500.0]),
        floor=np.array([0.0]),
        maximum_remaining=np.array([100_000.0]),
    )
    np.testing.assert_allclose(result.gross, [500.0])
    np.testing.assert_allclose(above_minimum, [300.0])


def test_withdraw_raises_on_negative_requested() -> None:
    state = _state(balance=1000.0)
    with pytest.raises(ValueError):
        lif.withdraw(
            state,
            requested=np.array([-1.0]),
            floor=np.array([0.0]),
            maximum_remaining=np.array([1000.0]),
        )


def test_there_is_no_lif_minimum_withdrawal() -> None:
    assert not hasattr(lif, "minimum_withdrawal")


def test_there_is_no_erode_nominal() -> None:
    assert not hasattr(lif, "erode_nominal")


# =============================================================================
# pay_to_survivor
# =============================================================================
# Balances, fractions and years below are synthetic.


def _multi(
    balance,
    jurisdiction,
    annual_minimum,
    annual_maximum,
    withdrawn_ytd,
    opened_year,
    inherited_fraction=None,
) -> LifState:
    return LifState(
        balance=np.array(balance, dtype=np.float64),
        jurisdiction=jurisdiction,
        annual_minimum=np.array(annual_minimum, dtype=np.float64),
        annual_maximum=np.array(annual_maximum, dtype=np.float64),
        withdrawn_ytd=np.array(withdrawn_ytd, dtype=np.float64),
        opened_year=opened_year,
        inherited_fraction=(
            np.zeros(len(balance))
            if inherited_fraction is None
            else np.array(inherited_fraction, dtype=np.float64)
        ),
    )


def _rrif(balance, opened_year, inherited_fraction=None, annual_minimum=None, withdrawn_ytd=None):
    n = len(balance)
    return RrifState(
        balance=np.array(balance, dtype=np.float64),
        annual_minimum=np.zeros(n) if annual_minimum is None else np.array(annual_minimum),
        withdrawn_ytd=np.zeros(n) if withdrawn_ytd is None else np.array(withdrawn_ytd),
        opened_year=opened_year,
        inherited_fraction=(
            np.zeros(n) if inherited_fraction is None else np.array(inherited_fraction)
        ),
    )


def _same_rrif(left: RrifState, right: RrifState) -> None:
    np.testing.assert_array_equal(left.balance, right.balance)
    np.testing.assert_array_equal(left.annual_minimum, right.annual_minimum)
    np.testing.assert_array_equal(left.withdrawn_ytd, right.withdrawn_ytd)
    np.testing.assert_array_equal(left.inherited_fraction, right.inherited_fraction)
    assert left.opened_year == right.opened_year


def test_pay_to_survivor_matches_receive_conversion() -> None:
    year = 2030
    deceased = _multi(
        [10_000.0, 20_000.0, 0.0, 7_000.0],
        "ab",
        [500.0, 600.0, 0.0, 10.0],
        [1000.0, 1200.0, 0.0, 20.0],
        [100.0, 200.0, 0.0, 30.0],
        2020,
        inherited_fraction=[0.0, 0.0, 0.0, 0.0],
    )
    masks = [
        np.array([True, True, True, True]),
        np.array([True, False, True, False]),
        np.array([False, False, True, False]),  # only the zero-balance path
    ]
    survivors = [
        _rrif([0.0, 0.0, 0.0, 0.0], None),
        _rrif([5_000.0, 0.0, 9_000.0, 4_000.0], 2015),
        _rrif([5_000.0, 6_000.0, 9_000.0, 4_000.0], 2015, [0.3, 0.0, 0.7, 0.2]),
        _rrif([0.0, 6_000.0, 0.0, 4_000.0], 2012, [0.0, 0.5, 0.0, 0.9]),
    ]
    reached = 0
    for mask in masks:
        for survivor in survivors:
            moved = np.where(mask, deceased.balance, 0.0)
            _, got = lif.pay_to_survivor(deceased, survivor, mask, year)
            if np.any(moved > 0):
                reached += 1
                _same_rrif(got, rrif.receive_conversion(survivor, moved, year))
    assert reached == 8  # two of the three masks move something, against four survivors


def test_pay_to_survivor_with_nothing_moved_leaves_the_rrif_as_it_was() -> None:
    deceased = _multi([0.0, 9_000.0], "ab", [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 2020)
    survivor = _rrif([0.0, 0.0], None)
    for mask in (np.array([False, False]), np.array([True, False])):
        # Guard: receive_conversion given zeros would set opened_year, so this case bites.
        assert rrif.receive_conversion(survivor, np.zeros(2), 2030).opened_year == 2030

        _, got = lif.pay_to_survivor(deceased, survivor, mask, 2030)

        assert got.opened_year is None
        _same_rrif(got, survivor)


def test_pay_to_survivor_moves_the_balance_and_zeroes_the_deceased() -> None:
    mask = np.array([True, False])
    deceased = _multi([10_000.0, 20_000.0], "ab", [0.0, 0.0], [1000.0, 1200.0], [0.0, 0.0], 2020)
    survivor = _rrif([5_000.0, 6_000.0], 2019)

    new_deceased, new_survivor = lif.pay_to_survivor(deceased, survivor, mask, 2026)

    np.testing.assert_array_equal(new_deceased.balance, [0.0, 20_000.0])
    np.testing.assert_array_equal(new_survivor.balance, [15_000.0, 6_000.0])


def test_pay_to_survivor_leaves_paths_outside_the_mask_untouched() -> None:
    mask = np.array([True, False])
    deceased = _multi(
        [10_000.0, 20_000.0], "ab", [1.0, 2.0], [1000.0, 1200.0], [3.0, 4.0], 2020, [0.2, 0.3]
    )
    survivor = _rrif([5_000.0, 6_000.0], 2019, [0.1, 0.4], [7.0, 8.0], [9.0, 10.0])

    new_deceased, new_survivor = lif.pay_to_survivor(deceased, survivor, mask, 2026)

    assert new_deceased.balance[1] == deceased.balance[1]
    assert new_survivor.balance[1] == survivor.balance[1]
    assert new_survivor.inherited_fraction[1] == survivor.inherited_fraction[1]
    assert new_survivor.annual_minimum[1] == survivor.annual_minimum[1]
    assert new_survivor.withdrawn_ytd[1] == survivor.withdrawn_ytd[1]
    assert new_deceased.inherited_fraction[1] == deceased.inherited_fraction[1]
    assert new_survivor.balance[0] != survivor.balance[0]  # guard: the masked path did move


def test_pay_to_survivor_leaves_the_deceaseds_other_fields_unchanged() -> None:
    mask = np.array([True])
    deceased = _multi([10_000.0], "ab", [500.0], [1000.0], [100.0], 2020, [0.3])
    survivor = _rrif([5_000.0], 2019)

    new_deceased, _ = lif.pay_to_survivor(deceased, survivor, mask, 2026)

    assert new_deceased.balance[0] == 0.0  # guard: it did pay
    np.testing.assert_array_equal(new_deceased.annual_minimum, deceased.annual_minimum)
    np.testing.assert_array_equal(new_deceased.annual_maximum, deceased.annual_maximum)
    np.testing.assert_array_equal(new_deceased.withdrawn_ytd, deceased.withdrawn_ytd)
    np.testing.assert_array_equal(new_deceased.inherited_fraction, [0.3])
    assert new_deceased.jurisdiction == "ab"
    assert new_deceased.opened_year == 2020


def test_pay_to_survivor_leaves_the_survivors_minimum_and_withdrawn_ytd_untouched() -> None:
    mask = np.array([True])
    deceased = _multi([10_000.0], "ab", [500.0], [1000.0], [100.0], 2020)
    survivor = _rrif([5_000.0], 2019, annual_minimum=[50.0], withdrawn_ytd=[10.0])

    _, new_survivor = lif.pay_to_survivor(deceased, survivor, mask, 2026)

    assert new_survivor.balance[0] == 15_000.0  # guard: it did pay
    np.testing.assert_array_equal(new_survivor.annual_minimum, [50.0])
    np.testing.assert_array_equal(new_survivor.withdrawn_ytd, [10.0])


def test_pay_to_survivor_credits_the_money_as_the_survivors_own() -> None:
    mask = np.array([True])
    fraction, own, moved = 0.4, 300.0, 100.0
    deceased = _multi([moved], "ab", [0.0], [50.0], [0.0], 2020)
    survivor = _rrif([own], 2019, [fraction])

    _, new_survivor = lif.pay_to_survivor(deceased, survivor, mask, 2026)

    np.testing.assert_allclose(
        new_survivor.inherited_fraction[0], fraction * own / (own + moved), rtol=1e-15
    )
    assert 0.0 < new_survivor.inherited_fraction[0] < fraction


def test_withdraw_leaves_the_inherited_fraction_unchanged() -> None:
    state = _state(balance=1_000.0, annual_maximum=500.0, inherited_fraction=0.4)

    new_state, result, _ = lif.withdraw(
        state, np.array([300.0]), np.array([0.0]), np.array([500.0])
    )

    assert result.gross[0] == 300.0  # guard: something was withdrawn
    np.testing.assert_array_equal(new_state.inherited_fraction, [0.4])


# =============================================================================
# The factor table is read as a table of finite numbers under whole-number keys
# =============================================================================


def _table_params(
    root: Path,
    *,
    terminal: str = "62",
    rows: str = "      60: 0.05\n      61: 0.06\n      62: 0.07\n",
    table: str | None = None,
):
    """A synthetic ``ab`` file with its LIF maximum table spelled out in YAML text."""
    body = table if table is not None else f"    by_age:\n{rows}"
    text = (
        f"lif:\n  has_maximum: true\n  maximum_factors:\n{body}    terminal_age_years: {terminal}\n"
    )
    year_dir = root / "2026"
    year_dir.mkdir(parents=True, exist_ok=True)
    (year_dir / "ab.yaml").write_text(text, encoding="utf-8")
    return real_year(load_year(2026, root), 0.0).jurisdiction("ab")


def test_maximum_withdrawal_refuses_a_fractional_terminal_age(tmp_path: Path) -> None:
    params = _table_params(tmp_path, terminal="62.5")
    with pytest.raises(MalformedParamFileError) as raised:
        lif.maximum_withdrawal(np.array([100.0]), 61, params)
    assert str(raised.value) == (
        f"{params.raw.source}[lif.maximum_factors.terminal_age_years]: "
        f"expected an integer, found 62.5."
    )


def test_maximum_withdrawal_refuses_a_boolean_row(tmp_path: Path) -> None:
    params = _table_params(tmp_path, rows="      60: 0.05\n      61: true\n      62: 0.07\n")
    with pytest.raises(MalformedParamFileError) as raised:
        lif.maximum_withdrawal(np.array([100.0]), 61, params)
    assert str(raised.value) == (
        f"{params.raw.source}[lif.maximum_factors.by_age.61]: expected a number, found bool (True)."
    )


def test_maximum_withdrawal_refuses_a_non_finite_row(tmp_path: Path) -> None:
    params = _table_params(tmp_path, rows="      60: 0.05\n      61: .inf\n      62: 0.07\n")
    with pytest.raises(MalformedParamFileError) as raised:
        lif.maximum_withdrawal(np.array([100.0]), 61, params)
    assert str(raised.value) == (
        f"{params.raw.source}[lif.maximum_factors.by_age.61]: expected a finite number, found inf."
    )


@pytest.mark.parametrize("age", (62, 70))
def test_maximum_withdrawal_refuses_a_boolean_terminal_row(tmp_path: Path, age: int) -> None:
    params = _table_params(tmp_path, rows="      60: 0.05\n      61: 0.06\n      62: true\n")
    with pytest.raises(MalformedParamFileError) as raised:
        lif.maximum_withdrawal(100.0, age, params)
    assert str(raised.value) == (
        f"{params.raw.source}[lif.maximum_factors.by_age.62]: expected a number, found bool (True)."
    )


@pytest.mark.parametrize("age", (62, 70))
def test_maximum_withdrawal_refuses_a_non_finite_terminal_row(tmp_path: Path, age: int) -> None:
    params = _table_params(tmp_path, rows="      60: 0.05\n      61: 0.06\n      62: .nan\n")
    with pytest.raises(MalformedParamFileError) as raised:
        lif.maximum_withdrawal(100.0, age, params)
    assert str(raised.value) == (
        f"{params.raw.source}[lif.maximum_factors.by_age.62]: expected a finite number, found nan."
    )


def test_maximum_withdrawal_raises_missing_for_a_row_absent_inside_the_table(
    tmp_path: Path,
) -> None:
    params = _table_params(tmp_path, rows="      60: 0.05\n      62: 0.07\n")
    with pytest.raises(MissingParameterError, match="'61' is not present"):
        lif.maximum_withdrawal(np.array([100.0]), 61, params)


def test_maximum_withdrawal_refuses_a_key_not_in_plain_decimal(tmp_path: Path) -> None:
    params = _table_params(tmp_path, rows='      60: 0.05\n      "071": 0.06\n      62: 0.07\n')
    with pytest.raises(MalformedParamFileError) as raised:
        lif.maximum_withdrawal(np.array([100.0]), 61, params)
    assert str(raised.value) == (
        f"{params.raw.source}[lif.maximum_factors.by_age]: key '071' is not an integer "
        f"written in plain decimal, such as 71."
    )


def test_maximum_withdrawal_refuses_an_empty_table(tmp_path: Path) -> None:
    params = _table_params(tmp_path, table="    by_age: {}\n")
    with pytest.raises(MalformedParamFileError) as raised:
        lif.maximum_withdrawal(np.array([100.0]), 61, params)
    assert str(raised.value) == (
        f"{params.raw.source}[lif.maximum_factors.by_age]: "
        f"the mapping is empty; a table needs at least one row."
    )


def test_maximum_withdrawal_below_the_first_row_message_is_exact(tmp_path: Path) -> None:
    params = _table_params(tmp_path, rows="      60: 0.05\n      61: 0.06\n      62: 0.07\n")
    with pytest.raises(ValueError) as raised:
        lif.maximum_withdrawal(np.array([1.0]), 59, params)
    assert str(raised.value) == (
        "age 59 is below the LIF maximum table's first row (age 60) "
        f"for jurisdiction {params.name!r}. "
        "The maximum is not extrapolated below the table."
    )


def test_maximum_withdrawal_below_the_first_row_message_bounds_a_long_first_age(
    tmp_path: Path,
) -> None:
    big = "1" + "0" * 999
    assert len(str(int(big))) > 500
    params = _table_params(tmp_path, rows=f"      {big}: 0.05\n")
    with pytest.raises(ValueError) as raised:
        lif.maximum_withdrawal(np.array([1.0]), 60, params)
    message = str(raised.value)
    assert message == (
        "age 60 is below the LIF maximum table's first row "
        f"(age {shown_value(int(big))}) for jurisdiction {params.name!r}. "
        "The maximum is not extrapolated below the table."
    )
    assert "…" in message
    assert len(message) < 1000

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Life Income Fund: the maximum table and the withdraw clamp.

Runs against the real Alberta jurisdiction file — a maximum-factor table is
exactly the kind of thing a synthetic fixture would obscure rather than
clarify, since the required checks (first row, one below it) are about the
table's *shape*, which the real file already has.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.accounts import lif
from engine.core.indexation import real_year
from engine.core.state import LifState
from engine.params.loader import load_year


@pytest.fixture
def ab():
    return real_year(load_year(2026), 0.0).jurisdiction("ab")


def _state(balance=0.0, annual_minimum=0.0, annual_maximum=0.0, withdrawn_ytd=0.0) -> LifState:
    return LifState(
        balance=np.array([balance], dtype=np.float64),
        jurisdiction="ab",
        annual_minimum=np.array([annual_minimum], dtype=np.float64),
        annual_maximum=np.array([annual_maximum], dtype=np.float64),
        withdrawn_ytd=np.array([withdrawn_ytd], dtype=np.float64),
        opened_year=2020,
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

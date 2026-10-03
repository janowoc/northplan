# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared account primitives: growth, withdrawal, and the annual-allowance clip.

Pure arithmetic with no parameters involved, so there is only one section
here — hand-computed cases, not a structural/synthetic split.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from engine.accounts import base


def test_grow_applies_monthly_return() -> None:
    result = base.grow(np.array([1000.0, 2000.0]), np.array([0.01, -0.02]))
    np.testing.assert_allclose(result, [1010.0, 1960.0])


def test_withdraw_takes_the_lesser_of_balance_and_requested() -> None:
    new_balance, withdrawn, shortfall = base.withdraw(
        np.array([100.0, 50.0]), np.array([40.0, 80.0])
    )
    np.testing.assert_allclose(withdrawn, [40.0, 50.0])
    np.testing.assert_allclose(shortfall, [0.0, 30.0])
    np.testing.assert_allclose(new_balance, [60.0, 0.0])


def test_withdraw_invariant_holds_on_every_path() -> None:
    balance = np.array([0.0, 10.0, 1000.0])
    requested = np.array([5.0, 10.0, 500.0])
    _, withdrawn, shortfall = base.withdraw(balance, requested)
    np.testing.assert_allclose(withdrawn + shortfall, requested)


def test_withdraw_raises_on_negative_requested() -> None:
    with pytest.raises(ValueError):
        base.withdraw(np.array([100.0]), np.array([-1.0]))


def test_remaining_annual_allowance_floors_at_zero() -> None:
    result = base.remaining_annual_allowance(np.array([1000.0, 1000.0]), np.array([400.0, 1500.0]))
    np.testing.assert_allclose(result, [600.0, 0.0])


# =============================================================================
# inherited_fraction_after_inflow (#76)
# =============================================================================
# Balances and fractions below are synthetic.


def test_inherited_fraction_blends_an_inherited_inflow() -> None:
    result = base.inherited_fraction_after_inflow(
        np.array([0.25]), np.array([300.0]), np.array([100.0]), np.array([100.0])
    )
    np.testing.assert_allclose(result, [(0.25 * 300.0 + 100.0) / 400.0], rtol=1e-15)


def test_inherited_fraction_is_diluted_by_an_own_inflow() -> None:
    result = base.inherited_fraction_after_inflow(
        np.array([0.5]), np.array([300.0]), np.array([100.0]), 0.0
    )
    np.testing.assert_allclose(result, [0.5 * 300.0 / 400.0], rtol=1e-15)


def test_inherited_fraction_of_an_empty_balance_receiving_only_inherited_money_is_one() -> None:
    result = base.inherited_fraction_after_inflow(
        np.array([0.0]), np.array([0.0]), np.array([123.45]), np.array([123.45])
    )
    np.testing.assert_array_equal(result, [1.0])


def test_inherited_fraction_is_unchanged_bit_for_bit_where_nothing_flows_in() -> None:
    fraction = np.array([0.1, 0.7])
    balance = np.array([3.0, 0.0])
    assert 0.1 * 3.0 / 3.0 != 0.1  # guard: recomputing the blend would drift

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = base.inherited_fraction_after_inflow(fraction, balance, np.zeros(2), np.zeros(2))

    np.testing.assert_array_equal(result, fraction)


def test_inherited_fraction_is_per_path() -> None:
    result = base.inherited_fraction_after_inflow(
        np.array([0.0, 0.5, 0.2]),
        np.array([300.0, 300.0, 50.0]),
        np.array([100.0, 100.0, 0.0]),
        np.array([100.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(result, [0.25, 0.375, 0.2], rtol=1e-15)


def test_inherited_fraction_broadcasts_a_scalar_balance_and_inflow() -> None:
    blended = base.inherited_fraction_after_inflow(np.array([0.3, 0.6]), 5.0, 1.0, 0.0)
    unchanged = base.inherited_fraction_after_inflow(np.array([0.3, 0.6]), 5.0, 0.0, 0.0)

    np.testing.assert_allclose(blended, [0.25, 0.5], rtol=1e-15)
    np.testing.assert_array_equal(unchanged, [0.3, 0.6])


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((0.5, np.array([5.0, 3.0]), 1.0, 0.0), [2.5 / 6.0, 1.5 / 4.0]),
        ((0.5, 3.0, np.array([0.0, 1.0]), 0.0), [0.5, 1.5 / 4.0]),
        ((0.0, 5.0, 1.0, np.array([0.0, 1.0])), [0.0, 1.0 / 6.0]),
    ],
    ids=["balance", "inflow", "inherited-inflow"],
)
def test_inherited_fraction_broadcasts_when_one_other_argument_is_the_array(args, expected) -> None:
    result = base.inherited_fraction_after_inflow(*args)

    assert result.shape == (2,)
    np.testing.assert_allclose(result, expected, rtol=1e-15)

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared account primitives: growth, withdrawal, and the annual-allowance clip.

Pure arithmetic with no parameters involved, so there is only one section
here — hand-computed cases, not a structural/synthetic split.
"""

from __future__ import annotations

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

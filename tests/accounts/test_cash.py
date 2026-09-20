# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cash: deposit, pay, and the one thing that must never happen — flooring pay at zero."""

from __future__ import annotations

import numpy as np
import pytest

from engine.accounts import cash
from engine.core.state import CashState


def _state(balance: float = 100.0) -> CashState:
    return CashState(balance=np.array([balance]))


def test_deposit_raises_balance() -> None:
    new_state = cash.deposit(_state(100.0), np.array([50.0]))
    np.testing.assert_allclose(new_state.balance, [150.0])


def test_pay_lowers_balance() -> None:
    new_state = cash.pay(_state(100.0), np.array([30.0]))
    np.testing.assert_allclose(new_state.balance, [70.0])


def test_pay_may_go_negative() -> None:
    # docs/limitations.md L38: negative cash at month end is a step-level rule,
    # not an invariant cash.pay enforces itself.
    new_state = cash.pay(_state(10.0), np.array([50.0]))
    np.testing.assert_allclose(new_state.balance, [-40.0])
    assert np.all(new_state.balance < 0)


def test_deposit_raises_on_negative_amount() -> None:
    with pytest.raises(ValueError):
        cash.deposit(_state(), np.array([-1.0]))


def test_pay_raises_on_negative_amount() -> None:
    with pytest.raises(ValueError):
        cash.pay(_state(), np.array([-1.0]))


def test_there_is_no_grow() -> None:
    assert not hasattr(cash, "grow")

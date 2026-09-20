# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Non-registered (taxable) investment account.

No parameters are read anywhere in this module — the yields and returns
arrive as plain arguments from the market model — so there is only one
section of hand-computed cases, no structural/synthetic split.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.accounts import taxable
from engine.core.state import TaxableState


def _state(balance=0.0, acb=0.0) -> TaxableState:
    return TaxableState(
        balance=np.array([balance], dtype=np.float64),
        acb=np.array([acb], dtype=np.float64),
    )


def test_distributions_monthly_divides_the_annual_yield_by_twelve() -> None:
    interest, dividends, gains = taxable.distributions_monthly(
        np.array([12_000.0]), (0.03, 0.02, 0.01)
    )
    np.testing.assert_allclose(interest, [30.0])
    np.testing.assert_allclose(dividends, [20.0])
    np.testing.assert_allclose(gains, [10.0])


def test_reinvest_raises_balance_and_acb_by_the_same_total() -> None:
    state = _state(balance=1000.0, acb=1000.0)
    new_state = taxable.reinvest(state, (np.array([10.0]), np.array([20.0]), np.array([5.0])))
    np.testing.assert_allclose(new_state.balance, [1035.0])
    np.testing.assert_allclose(new_state.acb, [1035.0])


def test_price_growth_applies_total_return_less_the_yields_to_balance_only() -> None:
    state = _state(balance=1200.0, acb=900.0)
    # annual yield sum = 0.03 + 0.02 + 0.01 = 0.06 -> 0.005/month
    new_state = taxable.price_growth(state, np.array([0.01]), (0.03, 0.02, 0.01))
    np.testing.assert_allclose(new_state.balance, [1200.0 * (1 + 0.01 - 0.005)])
    np.testing.assert_allclose(new_state.acb, [900.0])


def test_withdraw_realizes_a_proportional_gain() -> None:
    state = _state(balance=1000.0, acb=400.0)
    new_state, result = taxable.withdraw(state, np.array([500.0]))
    # fraction withdrawn = 0.5; gain fraction = (1000-400)/1000 = 0.6
    np.testing.assert_allclose(result.gross, [500.0])
    np.testing.assert_allclose(result.capital_gain, [300.0])
    np.testing.assert_allclose(result.tax_free, [200.0])
    np.testing.assert_allclose(result.fully_taxable, [0.0])
    np.testing.assert_allclose(new_state.balance, [500.0])
    np.testing.assert_allclose(new_state.acb, [200.0])


def test_withdraw_prices_the_gain_against_gross_not_requested() -> None:
    # requested (2000) exceeds balance (1000): gross = min(balance, requested)
    # = 1000. capital_gain = gross * (balance - acb) / balance
    #                       = 1000 * (1000 - 400) / 1000 = 600.
    # tax_free = gross * acb / balance = 1000 * 400 / 1000 = 400.
    # Pricing the gain against the raw requested (2000) instead would give
    # 2000 * 0.6 = 1200, more than was ever actually withdrawn.
    state = _state(balance=1000.0, acb=400.0)
    _, result = taxable.withdraw(state, np.array([2000.0]))
    np.testing.assert_allclose(result.gross, [1000.0])
    np.testing.assert_allclose(result.capital_gain, [600.0])
    np.testing.assert_allclose(result.tax_free, [400.0])
    np.testing.assert_allclose(result.shortfall, [1000.0])
    np.testing.assert_allclose(result.capital_gain + result.tax_free, result.gross)


def test_withdraw_with_acb_above_balance_is_a_negative_gain_and_does_not_raise() -> None:
    # An embedded loss: acb > balance is ordinary, per TaxableState's docstring.
    state = _state(balance=500.0, acb=800.0)
    _, result = taxable.withdraw(state, np.array([200.0]))
    assert result.capital_gain[0] < 0.0
    np.testing.assert_allclose(result.gross, [200.0])


def test_withdraw_at_zero_balance_withdraws_nothing_and_reports_the_shortfall() -> None:
    state = _state(balance=0.0, acb=0.0)
    new_state, result = taxable.withdraw(state, np.array([100.0]))
    np.testing.assert_allclose(result.gross, [0.0])
    np.testing.assert_allclose(result.capital_gain, [0.0])
    np.testing.assert_allclose(result.shortfall, [100.0])
    np.testing.assert_allclose(new_state.acb, [0.0])


def test_withdraw_raises_on_negative_requested() -> None:
    with pytest.raises(ValueError):
        taxable.withdraw(_state(balance=100.0), np.array([-1.0]))


def test_deemed_disposition_returns_the_gain_without_changing_state() -> None:
    state = _state(balance=1000.0, acb=600.0)
    gain = taxable.deemed_disposition(state)
    np.testing.assert_allclose(gain, [400.0])
    np.testing.assert_allclose(state.balance, [1000.0])
    np.testing.assert_allclose(state.acb, [600.0])


# --- erode_nominal -----------------------------------------------------------


def test_erode_nominal_is_identity_at_zero_inflation() -> None:
    state = _state(balance=1000.0, acb=600.0)
    new_state = taxable.erode_nominal(state, 0.0)
    np.testing.assert_allclose(new_state.acb, [600.0])
    np.testing.assert_allclose(new_state.balance, [1000.0])


def test_erode_nominal_shrinks_acb_only() -> None:
    state = _state(balance=1000.0, acb=600.0)
    new_state = taxable.erode_nominal(state, 0.10)
    assert new_state.acb[0] < 600.0
    np.testing.assert_allclose(new_state.balance, [1000.0])

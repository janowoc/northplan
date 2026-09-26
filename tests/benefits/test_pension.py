# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``engine.benefits.pension`` (defined-benefit pensions)."""

from __future__ import annotations

import numpy as np
import pytest

from engine.benefits.pension import db_pension_monthly, survivor_pension_monthly
from engine.core.indexation import unindexed_factor
from engine.core.state import PensionState

N_PATHS = 3


def _pension(
    monthly_amount=1000.0,
    start_month_index=0,
    indexed=True,
    bridge_monthly=0.0,
    bridge_end_month_index=None,
    survivor_share=0.6,
):
    return PensionState(
        name="test",
        monthly_amount=np.full(N_PATHS, monthly_amount, dtype=np.float64),
        start_month_index=start_month_index,
        indexed=indexed,
        bridge_monthly=np.full(N_PATHS, bridge_monthly, dtype=np.float64),
        bridge_end_month_index=bridge_end_month_index,
        survivor_share=survivor_share,
    )


def test_zero_before_start() -> None:
    pension = _pension(start_month_index=12)
    result = db_pension_monthly(pension, 11, np.ones(N_PATHS, dtype=bool), 0.0)
    assert np.all(result == 0.0)


def test_paid_at_start() -> None:
    pension = _pension(monthly_amount=1000.0, start_month_index=12, indexed=True)
    result = db_pension_monthly(pension, 12, np.ones(N_PATHS, dtype=bool), 0.0)
    assert np.all(result == pytest.approx(1000.0))


def test_zero_where_not_alive() -> None:
    pension = _pension(monthly_amount=1000.0, start_month_index=0, indexed=True)
    alive = np.array([True, False, True])
    result = db_pension_monthly(pension, 5, alive, 0.0)
    assert result[1] == 0.0
    assert result[0] == pytest.approx(1000.0)
    assert result[2] == pytest.approx(1000.0)


def test_indexed_constant_at_positive_inflation() -> None:
    pension = _pension(monthly_amount=1000.0, start_month_index=0, indexed=True)
    early = db_pension_monthly(pension, 3, np.ones(N_PATHS, dtype=bool), 0.03)
    later = db_pension_monthly(pension, 36, np.ones(N_PATHS, dtype=bool), 0.03)
    assert np.all(early == pytest.approx(1000.0))
    assert np.all(later == pytest.approx(1000.0))


def test_unindexed_decays_with_future_start() -> None:
    rate = 0.03
    start = 12
    pension = _pension(monthly_amount=1000.0, start_month_index=start, indexed=False)
    month_index = 36
    result = db_pension_monthly(pension, month_index, np.ones(N_PATHS, dtype=bool), rate)
    expected = 1000.0 * unindexed_factor(rate, month_index - max(start, 0))
    assert np.all(result == pytest.approx(expected))


def test_unindexed_decays_with_start_before_run() -> None:
    rate = 0.03
    start = -24  # already in pay before the run opened
    pension = _pension(monthly_amount=1000.0, start_month_index=start, indexed=False)
    month_index = 10
    result = db_pension_monthly(pension, month_index, np.ones(N_PATHS, dtype=bool), rate)
    expected = 1000.0 * unindexed_factor(rate, month_index - max(start, 0))
    assert np.all(result == pytest.approx(expected))


def test_bridge_paid_through_end_month_inclusive_not_after() -> None:
    pension = _pension(
        monthly_amount=1000.0,
        start_month_index=0,
        indexed=True,
        bridge_monthly=200.0,
        bridge_end_month_index=24,
    )
    alive = np.ones(N_PATHS, dtype=bool)
    at_end = db_pension_monthly(pension, 24, alive, 0.0)
    after_end = db_pension_monthly(pension, 25, alive, 0.0)
    assert np.all(at_end == pytest.approx(1200.0))
    assert np.all(after_end == pytest.approx(1000.0))


def test_bridge_with_same_factor_as_base_when_unindexed() -> None:
    rate = 0.02
    pension = _pension(
        monthly_amount=1000.0,
        start_month_index=0,
        indexed=False,
        bridge_monthly=200.0,
        bridge_end_month_index=24,
    )
    alive = np.ones(N_PATHS, dtype=bool)
    month_index = 10
    result = db_pension_monthly(pension, month_index, alive, rate)
    factor = unindexed_factor(rate, month_index)
    expected = (1000.0 + 200.0) * factor
    assert np.all(result == pytest.approx(expected))


# =============================================================================
# survivor_pension_monthly (#36)
# =============================================================================


def test_survivor_zero_before_start() -> None:
    pension = _pension(start_month_index=12, survivor_share=0.6)
    receiving = np.ones(N_PATHS, dtype=bool)
    result = survivor_pension_monthly(pension, 11, receiving, 0.0)
    assert np.all(result == 0.0)


def test_survivor_paid_at_start_no_bridge() -> None:
    pension = _pension(
        monthly_amount=1000.0,
        start_month_index=12,
        indexed=True,
        survivor_share=0.6,
        bridge_monthly=200.0,
        bridge_end_month_index=36,
    )
    receiving = np.ones(N_PATHS, dtype=bool)
    result = survivor_pension_monthly(pension, 12, receiving, 0.0)
    # No bridge in the survivor share, unlike db_pension_monthly for the same pension.
    assert np.all(result == pytest.approx(600.0))


def test_survivor_zero_where_not_receiving() -> None:
    pension = _pension(monthly_amount=1000.0, start_month_index=0, survivor_share=0.5)
    receiving = np.array([True, False, True])
    result = survivor_pension_monthly(pension, 5, receiving, 0.0)
    assert result[1] == 0.0
    assert result[0] == pytest.approx(500.0)
    assert result[2] == pytest.approx(500.0)


def test_survivor_indexed_constant_at_positive_inflation() -> None:
    pension = _pension(monthly_amount=1000.0, start_month_index=0, indexed=True, survivor_share=0.6)
    receiving = np.ones(N_PATHS, dtype=bool)
    early = survivor_pension_monthly(pension, 3, receiving, 0.03)
    later = survivor_pension_monthly(pension, 36, receiving, 0.03)
    assert np.all(early == pytest.approx(600.0))
    assert np.all(later == pytest.approx(600.0))


def test_survivor_unindexed_decays_the_same_way_as_the_members_own() -> None:
    rate = 0.03
    start = 12
    pension = _pension(
        monthly_amount=1000.0, start_month_index=start, indexed=False, survivor_share=0.5
    )
    month_index = 36
    receiving = np.ones(N_PATHS, dtype=bool)
    result = survivor_pension_monthly(pension, month_index, receiving, rate)
    expected = 0.5 * 1000.0 * unindexed_factor(rate, month_index - max(start, 0))
    assert np.all(result == pytest.approx(expected))


def test_survivor_paid_from_start_when_member_dies_before_it_starts() -> None:
    # L59: a member who dies before the pension starts is covered naturally -- the
    # survivor share is paid from the start month as if the pension had been deferred.
    pension = _pension(
        monthly_amount=1000.0, start_month_index=24, indexed=True, survivor_share=0.6
    )
    receiving = np.ones(N_PATHS, dtype=bool)
    before_start = survivor_pension_monthly(pension, 23, receiving, 0.0)
    at_start = survivor_pension_monthly(pension, 24, receiving, 0.0)
    assert np.all(before_start == 0.0)
    assert np.all(at_start == pytest.approx(600.0))

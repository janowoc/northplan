# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Withholding: registered-account lump-sum rates and the payroll approximation.

Two kinds of test, as in ``test_federal.py``: structural tests against the
real 2026 files at zero inflation, and a synthetic-table arithmetic test
against an obviously fake, hand-checkable ``rrif.yaml``. Neither section's
numbers should ever be copied into ``params/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.core.indexation import real_year
from engine.params.loader import load_year
from engine.tax.withholding import payroll_withholding_monthly, registered_withholding

JANUARY = 0
MONTH = 0  # any valid month index; unindexed table at zero inflation ignores it


# =============================================================================
# Structural tests against the real 2026 files
# =============================================================================


@pytest.fixture
def params():
    return real_year(load_year(2026), 0.0)


@pytest.fixture
def rrif(params):
    return params.rrif


def test_registered_withholding_zero_at_zero_and_negative(rrif) -> None:
    assert registered_withholding(0.0, rrif, MONTH) == pytest.approx(0.0)
    assert registered_withholding(-500.0, rrif, MONTH) == pytest.approx(0.0)


def test_registered_withholding_changes_rate_at_each_edge(rrif) -> None:
    edges = rrif.amounts("withholding.edges_each", MONTH)
    rates = rrif.numbers("withholding.rates")
    for i, edge in enumerate(edges):
        just_below = registered_withholding(edge - 1.0, rrif, MONTH)
        at = registered_withholding(edge, rrif, MONTH)
        just_above = registered_withholding(edge + 1.0, rrif, MONTH)

        assert just_below == pytest.approx((edge - 1.0) * rates[i])
        # Exactly at the edge takes the LOWER rate (inclusive upper bound).
        assert at == pytest.approx(edge * rates[i])
        assert just_above == pytest.approx((edge + 1.0) * rates[i + 1])


def test_payroll_withholding_zero_at_zero_income(params) -> None:
    result = payroll_withholding_monthly(0.0, 40, True, "ab", params, JANUARY)
    assert result == pytest.approx(0.0)


def test_payroll_withholding_non_decreasing(params) -> None:
    grid = np.array([0.0, 1000.0, 3000.0, 6000.0, 10_000.0])
    result = payroll_withholding_monthly(grid, 40, True, "ab", params, JANUARY)
    assert np.all(np.diff(result) >= -1e-9)


def test_payroll_withholding_no_larger_for_employment_than_non_employment(params) -> None:
    amount = 5000.0
    employment = payroll_withholding_monthly(amount, 40, True, "ab", params, JANUARY)
    non_employment = payroll_withholding_monthly(amount, 40, False, "ab", params, JANUARY)
    assert employment <= non_employment + 1e-9


# --- shape and dtype contracts ------------------------------------------------


def test_registered_withholding_returns_float64_scalar_and_array(rrif) -> None:
    scalar = registered_withholding(1000.0, rrif, MONTH)
    array = registered_withholding(np.array([1000.0, 20_000.0]), rrif, MONTH)
    assert scalar.dtype == np.float64 and scalar.shape == ()
    assert array.dtype == np.float64 and array.shape == (2,)


def test_payroll_withholding_returns_float64_scalar_and_array(params) -> None:
    scalar = payroll_withholding_monthly(5000.0, 40, True, "ab", params, JANUARY)
    array = payroll_withholding_monthly(
        np.array([5000.0, 6000.0]), np.array([40, 70]), True, "ab", params, JANUARY
    )
    assert scalar.dtype == np.float64 and scalar.shape == ()
    assert array.dtype == np.float64 and array.shape == (2,)


# =============================================================================
# Synthetic-table arithmetic tests
# =============================================================================

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
#: Round, wrong numbers chosen so every expected value below is exact.
SYNTHETIC_RRIF = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  unindexed:
    adjustment_months: []
    applies_to:
      - withholding.edges_each

withholding:
  edges_each: [100, 200]
  rates: [0.1, 0.2, 0.3]
"""


def _write(root: Path, name: str, text: str) -> Path:
    year_dir = root / "2026"
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def synth_rrif(tmp_path: Path):
    _write(tmp_path, "rrif", SYNTHETIC_RRIF)
    return real_year(load_year(2026, tmp_path), 0.0).rrif


def test_synthetic_registered_withholding_hand_computed(synth_rrif) -> None:
    assert registered_withholding(50.0, synth_rrif, MONTH) == pytest.approx(50.0 * 0.1)
    assert registered_withholding(100.0, synth_rrif, MONTH) == pytest.approx(
        100.0 * 0.1
    )  # at edge: lower rate
    assert registered_withholding(150.0, synth_rrif, MONTH) == pytest.approx(150.0 * 0.2)
    assert registered_withholding(200.0, synth_rrif, MONTH) == pytest.approx(
        200.0 * 0.2
    )  # at edge: lower rate
    assert registered_withholding(250.0, synth_rrif, MONTH) == pytest.approx(250.0 * 0.3)


# --- band table validation -----------------------------------------------

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
#: One fewer rate than the invariant requires: len(rates) == len(edges) + 1.
SYNTHETIC_RRIF_WRONG_RATE_COUNT = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  unindexed:
    adjustment_months: []
    applies_to:
      - withholding.edges_each

withholding:
  edges_each: [100, 200]
  rates: [0.1, 0.2]
"""

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
#: Edges given out of order, so they do not strictly ascend.
SYNTHETIC_RRIF_NON_ASCENDING_EDGES = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  unindexed:
    adjustment_months: []
    applies_to:
      - withholding.edges_each

withholding:
  edges_each: [200, 100]
  rates: [0.1, 0.2, 0.3]
"""

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
#: First edge is not positive.
SYNTHETIC_RRIF_NON_POSITIVE_FIRST_EDGE = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  unindexed:
    adjustment_months: []
    applies_to:
      - withholding.edges_each

withholding:
  edges_each: [0, 200]
  rates: [0.1, 0.2, 0.3]
"""


@pytest.fixture
def synth_rrif_wrong_rate_count(tmp_path: Path):
    _write(tmp_path, "rrif", SYNTHETIC_RRIF_WRONG_RATE_COUNT)
    return real_year(load_year(2026, tmp_path), 0.0).rrif


@pytest.fixture
def synth_rrif_non_ascending_edges(tmp_path: Path):
    _write(tmp_path, "rrif", SYNTHETIC_RRIF_NON_ASCENDING_EDGES)
    return real_year(load_year(2026, tmp_path), 0.0).rrif


@pytest.fixture
def synth_rrif_non_positive_first_edge(tmp_path: Path):
    _write(tmp_path, "rrif", SYNTHETIC_RRIF_NON_POSITIVE_FIRST_EDGE)
    return real_year(load_year(2026, tmp_path), 0.0).rrif


def test_registered_withholding_raises_when_rate_count_is_wrong(
    synth_rrif_wrong_rate_count,
) -> None:
    with pytest.raises(ValueError, match="rate"):
        registered_withholding(50.0, synth_rrif_wrong_rate_count, MONTH)


def test_registered_withholding_raises_when_edges_do_not_ascend(
    synth_rrif_non_ascending_edges,
) -> None:
    with pytest.raises(ValueError, match="ascend"):
        registered_withholding(50.0, synth_rrif_non_ascending_edges, MONTH)


def test_registered_withholding_raises_when_first_edge_is_not_positive(
    synth_rrif_non_positive_first_edge,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        registered_withholding(50.0, synth_rrif_non_positive_first_edge, MONTH)

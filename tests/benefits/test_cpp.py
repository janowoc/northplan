# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``engine.benefits.cpp``.

Structural tests against the real 2026 files at zero inflation; every
threshold, age, and rate is read from the loaded params, never typed in.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.benefits.cpp import (
    base_pension_monthly,
    pension_annual,
    pension_monthly,
    start_adjustment_factor,
    survivor_pension_monthly,
)
from engine.core.indexation import erosion_factor, real_year, schedule
from engine.core.state import BenefitState
from engine.params.loader import load_year

JANUARY = 0
MONTH = 0
N_PATHS = 3


@pytest.fixture
def cpp():
    return real_year(load_year(2026), 0.0).cpp


def _benefit_elected(start_age_months, contributory_history=1.0):
    return BenefitState(
        start_age_months=start_age_months,
        in_pay_monthly=None,
        contributory_history=contributory_history,
        monthly_amount=np.zeros(N_PATHS, dtype=np.float64),
    )


def _benefit_in_pay(amount):
    arr = np.full(N_PATHS, amount, dtype=np.float64)
    return BenefitState(
        start_age_months=None,
        in_pay_monthly=arr,
        contributory_history=None,
        monthly_amount=np.zeros(N_PATHS, dtype=np.float64),
    )


# --- start_adjustment_factor -------------------------------------------------


def test_factor_is_one_at_standard_age(cpp) -> None:
    standard = cpp.number("start_age.standard_months")
    assert start_adjustment_factor(standard, cpp) == pytest.approx(1.0)


def test_factor_at_earliest_matches_rate_formula(cpp) -> None:
    standard = cpp.number("start_age.standard_months")
    earliest = cpp.number("start_age.earliest_months")
    early_rate = cpp.number("start_adjustment.early_rate_per_month")
    expected = 1 - early_rate * (standard - earliest)
    assert start_adjustment_factor(earliest, cpp) == pytest.approx(expected)


def test_factor_at_latest_matches_rate_formula(cpp) -> None:
    standard = cpp.number("start_age.standard_months")
    latest = cpp.number("start_age.latest_months")
    late_rate = cpp.number("start_adjustment.late_rate_per_month")
    expected = 1 + late_rate * (latest - standard)
    assert start_adjustment_factor(latest, cpp) == pytest.approx(expected)


def test_factor_below_earliest_clips(cpp) -> None:
    earliest = cpp.number("start_age.earliest_months")
    assert start_adjustment_factor(earliest - 12, cpp) == pytest.approx(
        start_adjustment_factor(earliest, cpp)
    )


def test_factor_above_latest_clips(cpp) -> None:
    latest = cpp.number("start_age.latest_months")
    assert start_adjustment_factor(latest + 12, cpp) == pytest.approx(
        start_adjustment_factor(latest, cpp)
    )


# --- pension_annual -----------------------------------------------------------


def test_pension_annual_at_standard_age_matches_published_maximum(cpp) -> None:
    standard = cpp.number("start_age.standard_months")
    result = pension_annual(1.0, standard, MONTH, cpp)
    expected = cpp.annual_amount("pension.maximum_at_standard_age_annual", JANUARY)
    assert result == pytest.approx(expected)


# --- pension_monthly -----------------------------------------------------------


def test_pension_monthly_zero_before_start_paid_at_start(cpp) -> None:
    standard = int(cpp.number("start_age.standard_months"))
    benefit = _benefit_elected(standard)
    # Run opens well before the standard age; the pension starts partway through.
    age_at_open = standard - 6
    month_of_start = 6
    before = pension_monthly(benefit, age_at_open + month_of_start - 1, month_of_start - 1, cpp)
    at_start = pension_monthly(benefit, age_at_open + month_of_start, month_of_start, cpp)
    assert np.all(before == 0.0)
    expected = base_pension_monthly(1.0, month_of_start, cpp) * start_adjustment_factor(
        standard, cpp
    )
    assert at_start == pytest.approx(expected)


def test_pension_monthly_in_pay_returns_amount_at_any_age_and_month(cpp) -> None:
    latest = int(cpp.number("start_age.latest_months"))
    benefit = _benefit_in_pay(500.0)
    result = pension_monthly(benefit, latest + 60, 37, cpp)
    assert np.all(result == pytest.approx(500.0))


def test_pension_monthly_election_already_past_starts_at_month_zero(cpp) -> None:
    standard = int(cpp.number("start_age.standard_months"))
    elected = standard - 24  # elected two years before standard
    age_at_open = standard  # already past the election when the run opens
    benefit = _benefit_elected(elected)
    result = pension_monthly(benefit, age_at_open, 0, cpp)
    expected = base_pension_monthly(1.0, 0, cpp) * start_adjustment_factor(age_at_open, cpp)
    assert result == pytest.approx(expected)


def test_pension_monthly_past_latest_at_open_not_in_pay_uses_latest_factor(cpp) -> None:
    latest = int(cpp.number("start_age.latest_months"))
    earliest = int(cpp.number("start_age.earliest_months"))
    age_at_open = latest + 24
    benefit = _benefit_elected(earliest)
    result = pension_monthly(benefit, age_at_open, 0, cpp)
    expected = base_pension_monthly(1.0, 0, cpp) * start_adjustment_factor(latest, cpp)
    assert result == pytest.approx(expected)


def test_pension_monthly_election_below_earliest_starts_at_earliest(cpp) -> None:
    earliest = int(cpp.number("start_age.earliest_months"))
    benefit = _benefit_elected(earliest - 60)
    age_at_open = earliest - 60
    before = pension_monthly(benefit, earliest - 1, earliest - 1 - age_at_open, cpp)
    at = pension_monthly(benefit, earliest, earliest - age_at_open, cpp)
    assert np.all(before == 0.0)
    expected = base_pension_monthly(1.0, 0, cpp) * start_adjustment_factor(earliest, cpp)
    assert at == pytest.approx(expected)


def test_pension_monthly_raises_when_neither_source_set(cpp) -> None:
    earliest = int(cpp.number("start_age.earliest_months"))
    benefit = BenefitState(
        start_age_months=None,
        in_pay_monthly=None,
        contributory_history=None,
        monthly_amount=np.zeros(N_PATHS, dtype=np.float64),
    )
    with pytest.raises(ValueError):
        pension_monthly(benefit, earliest, 0, cpp)


# --- survivor_pension_monthly --------------------------------------------------


def test_survivor_increment_zero_when_already_at_combined_maximum(cpp) -> None:
    combined_max = cpp.amount("survivor.combined_maximum_monthly", MONTH)
    result = survivor_pension_monthly(1000.0, combined_max, MONTH, cpp)
    assert result == pytest.approx(0.0)


def test_survivor_share_cap_binds_when_own_is_small(cpp) -> None:
    share = cpp.number("survivor.share_at_65_plus")
    deceased_base = 100.0
    result = survivor_pension_monthly(deceased_base, 0.0, MONTH, cpp)
    assert result == pytest.approx(share * deceased_base)


def test_survivor_increment_floored_at_zero(cpp) -> None:
    combined_max = cpp.amount("survivor.combined_maximum_monthly", MONTH)
    result = survivor_pension_monthly(100.0, combined_max + 1000.0, MONTH, cpp)
    assert result == pytest.approx(0.0)


# --- real view ------------------------------------------------------------


def test_base_pension_monthly_applies_erosion_factor_at_positive_inflation() -> None:
    cpp_real = real_year(load_year(2026), 0.02).cpp
    raw_max = cpp_real.raw.number("pension.maximum_at_standard_age_monthly")
    k, _ = schedule("benefits", cpp_real.raw)
    result = base_pension_monthly(1.0, 1, cpp_real)
    expected = raw_max * erosion_factor(0.02, k)
    assert result == pytest.approx(expected)


def test_pension_monthly_in_pay_identical_at_zero_and_positive_inflation() -> None:
    cpp_zero = real_year(load_year(2026), 0.0).cpp
    cpp_positive = real_year(load_year(2026), 0.02).cpp
    benefit = _benefit_in_pay(500.0)
    at_zero = pension_monthly(benefit, 800, 5, cpp_zero)
    at_positive = pension_monthly(benefit, 800, 5, cpp_positive)
    assert at_zero == pytest.approx(at_positive)

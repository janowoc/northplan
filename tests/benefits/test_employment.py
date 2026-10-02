# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``engine.benefits.employment``.

Structural tests against the real 2026 files at zero inflation; every
threshold, ceiling, and rate is read from the loaded params, never typed in.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.benefits.employment import (
    CppContributions,
    cpp_contributions_monthly,
    ei_premium_monthly,
    employment_income_monthly,
)
from engine.core.indexation import erosion_factor, real_year, schedule, unindexed_factor
from engine.core.state import EmploymentBand
from engine.params.loader import load_year

JANUARY = 0
N_PATHS = 3


@pytest.fixture
def params():
    return real_year(load_year(2026), 0.0)


@pytest.fixture
def cpp(params):
    return params.cpp


@pytest.fixture
def federal(params):
    return params.federal


def _band(from_month, to_month, amount=1000.0):
    return EmploymentBand(
        from_month_index=from_month,
        to_month_index=to_month,
        monthly_amount=np.full(N_PATHS, amount, dtype=np.float64),
    )


def _annual_params(cpp):
    exemption = cpp.annual_amount("contributions.first_tier.basic_exemption_annual", JANUARY)
    tier1_ceiling = cpp.annual_amount("contributions.first_tier.ceiling_annual", JANUARY)
    tier2_ceiling = cpp.annual_amount("contributions.second_tier.ceiling_annual", JANUARY)
    base_rate = cpp.number("contributions.first_tier.base_rate")
    employee_rate = cpp.number("contributions.first_tier.employee_rate")
    tier2_rate = cpp.number("contributions.second_tier.employee_rate")
    return exemption, tier1_ceiling, tier2_ceiling, base_rate, employee_rate, tier2_rate


def _tier1_cumulative(total, exemption, tier1_ceiling, employee_rate):
    return employee_rate * np.clip(total - exemption, 0, tier1_ceiling - exemption)


def _tier2_cumulative(total, tier1_ceiling, tier2_ceiling, tier2_rate):
    return tier2_rate * np.clip(total - tier1_ceiling, 0, tier2_ceiling - tier1_ceiling)


# --- employment_income_monthly -------------------------------------------------


def test_band_edges_inclusive() -> None:
    band = _band(5, 10, amount=1234.0)
    at_start = employment_income_monthly((band,), 5, N_PATHS)
    at_end = employment_income_monthly((band,), 10, N_PATHS)
    before = employment_income_monthly((band,), 4, N_PATHS)
    after = employment_income_monthly((band,), 11, N_PATHS)
    assert np.all(at_start == 1234.0)
    assert np.all(at_end == 1234.0)
    assert np.all(before == 0.0)
    assert np.all(after == 0.0)


def test_empty_bands_gives_zeros_of_shape_n_paths() -> None:
    result = employment_income_monthly((), 3, N_PATHS)
    assert result.shape == (N_PATHS,)
    assert np.all(result == 0.0)


# --- cpp_contributions_monthly: named tuple ------------------------------------


def test_cpp_contributions_monthly_returns_named_tuple_and_unpacks(cpp) -> None:
    result = cpp_contributions_monthly(1000.0, 0.0, JANUARY, cpp)
    assert isinstance(result, CppContributions)
    base, enhanced = result
    assert base == pytest.approx(result.base)
    assert enhanced == pytest.approx(result.enhanced)


# --- cpp_contributions_monthly: closed-form cases ------------------------------


def test_month_crossing_first_ceiling_matches_closed_form(cpp) -> None:
    exemption, tier1_ceiling, tier2_ceiling, _, employee_rate, tier2_rate = _annual_params(cpp)
    # income_ytd below tier1_ceiling; income_ytd + monthly_income lands strictly between
    # tier1_ceiling and tier2_ceiling.
    income_ytd = exemption + 1000.0
    assert income_ytd < tier1_ceiling
    monthly_income = (tier1_ceiling - income_ytd) + 5000.0
    total = income_ytd + monthly_income
    assert tier1_ceiling < total < tier2_ceiling

    base, enhanced = cpp_contributions_monthly(monthly_income, income_ytd, JANUARY, cpp)

    expected_tier1 = employee_rate * (tier1_ceiling - max(income_ytd, exemption))
    expected_tier2 = tier2_rate * (income_ytd + monthly_income - tier1_ceiling)
    expected_base = expected_tier1 * (
        cpp.number("contributions.first_tier.base_rate") / employee_rate
    )
    expected_enhanced = expected_tier1 - expected_base + expected_tier2

    assert float(base) == pytest.approx(expected_base)
    assert float(enhanced) == pytest.approx(expected_enhanced)


def test_first_month_crossing_basic_exemption(cpp) -> None:
    exemption, tier1_ceiling, _, _, employee_rate, _ = _annual_params(cpp)
    income_ytd = 0.0
    monthly_income = exemption + 2000.0
    assert monthly_income < tier1_ceiling

    base, enhanced = cpp_contributions_monthly(monthly_income, income_ytd, JANUARY, cpp)
    expected_total = employee_rate * (monthly_income - exemption)
    assert float(base) + float(enhanced) == pytest.approx(expected_total)


def test_base_plus_enhanced_equals_tier1_plus_tier2_with_tier2_nonzero(cpp) -> None:
    exemption, tier1_ceiling, tier2_ceiling, _, employee_rate, tier2_rate = _annual_params(cpp)
    # income_ytd just below tier1_ceiling, monthly_income pushes the total past it into tier 2.
    income_ytd = tier1_ceiling - 2000.0
    monthly_income = 5000.0
    total = income_ytd + monthly_income
    assert total > tier1_ceiling  # tier 2 must actually be non-zero here

    base, enhanced = cpp_contributions_monthly(monthly_income, income_ytd, JANUARY, cpp)

    expected_tier1_month = _tier1_cumulative(
        total, exemption, tier1_ceiling, employee_rate
    ) - _tier1_cumulative(income_ytd, exemption, tier1_ceiling, employee_rate)
    expected_tier2_month = _tier2_cumulative(
        total, tier1_ceiling, tier2_ceiling, tier2_rate
    ) - _tier2_cumulative(income_ytd, tier1_ceiling, tier2_ceiling, tier2_rate)
    assert expected_tier2_month > 0.0

    assert (float(base) + float(enhanced)) == pytest.approx(
        expected_tier1_month + expected_tier2_month
    )


def test_year_at_tier1_ceiling_sums_to_tier1_maximum_tier2_zero(cpp) -> None:
    exemption, tier1_ceiling, _, _, employee_rate, _ = _annual_params(cpp)
    monthly_income = tier1_ceiling / 12
    income_ytd = 0.0
    total_base = 0.0
    total_enhanced = 0.0
    for _ in range(12):
        base, enhanced = cpp_contributions_monthly(monthly_income, income_ytd, JANUARY, cpp)
        total_base += float(base)
        total_enhanced += float(enhanced)
        income_ytd += monthly_income
    expected = employee_rate * (tier1_ceiling - exemption)
    assert (total_base + total_enhanced) == pytest.approx(expected)


def test_high_income_reaches_ceiling_in_a_later_month_then_zero(cpp) -> None:
    exemption, tier1_ceiling, tier2_ceiling, _, employee_rate, tier2_rate = _annual_params(cpp)
    # Chosen so tier2_ceiling is crossed partway through a later month, not the first.
    monthly_income = tier2_ceiling / 3.5
    income_ytd = 0.0
    crossing_seen = False
    for _ in range(12):
        total = income_ytd + monthly_income
        base, enhanced = cpp_contributions_monthly(monthly_income, income_ytd, JANUARY, cpp)
        contribution = float(base) + float(enhanced)
        expected_tier1 = _tier1_cumulative(
            total, exemption, tier1_ceiling, employee_rate
        ) - _tier1_cumulative(income_ytd, exemption, tier1_ceiling, employee_rate)
        expected_tier2 = _tier2_cumulative(
            total, tier1_ceiling, tier2_ceiling, tier2_rate
        ) - _tier2_cumulative(income_ytd, tier1_ceiling, tier2_ceiling, tier2_rate)
        assert contribution == pytest.approx(expected_tier1 + expected_tier2)
        if income_ytd >= tier2_ceiling:
            assert contribution == pytest.approx(0.0)
        elif total >= tier2_ceiling:
            crossing_seen = True
            assert contribution > 0.0
        income_ytd += monthly_income
    assert crossing_seen


# --- shape ----------------------------------------------------------------


def test_cpp_contributions_monthly_broadcasts_scalar_income_over_array_ytd(cpp) -> None:
    income_ytd = np.array([0.0, 1000.0, 5000.0])
    base, enhanced = cpp_contributions_monthly(500.0, income_ytd, JANUARY, cpp)
    assert base.shape == (3,)
    assert enhanced.shape == (3,)


def test_ei_premium_monthly_broadcasts_scalar_income_over_array_ytd(federal) -> None:
    income_ytd = np.array([0.0, 1000.0, 5000.0])
    premium = ei_premium_monthly(500.0, income_ytd, JANUARY, federal)
    assert premium.shape == (3,)


# --- cross-file consistency -----------------------------------------------


def test_cross_file_base_contribution_matches_federal_credit_maximum(cpp, federal) -> None:
    _, _, tier2_ceiling, _, _, _ = _annual_params(cpp)
    monthly_income = tier2_ceiling
    income_ytd = 0.0
    total_base = 0.0
    for _ in range(12):
        base, _ = cpp_contributions_monthly(monthly_income, income_ytd, JANUARY, cpp)
        total_base += float(base)
        income_ytd += monthly_income
    expected = federal.annual_amount("contribution_credit.cpp_maximum_annual", JANUARY)
    assert total_base == pytest.approx(expected)


def test_cross_file_ei_premium_matches_federal_credit_maximum(federal) -> None:
    maximum = federal.annual_amount("ei.maximum_insurable_earnings_annual", JANUARY)
    monthly_income = maximum
    income_ytd = 0.0
    total = 0.0
    for _ in range(12):
        premium = ei_premium_monthly(monthly_income, income_ytd, JANUARY, federal)
        total += float(premium)
        income_ytd += monthly_income
    expected = federal.annual_amount("contribution_credit.ei_maximum_annual", JANUARY)
    assert total == pytest.approx(expected)


# --- real view: erosion -----------------------------------------------------


def test_cpp_contributions_monthly_unindexed_exemption_decays_with_january_index() -> None:
    cpp_real = real_year(load_year(2026), 0.02).cpp
    raw_exemption = cpp_real.raw.number("contributions.first_tier.basic_exemption_annual")
    employee_rate = cpp_real.raw.number("contributions.first_tier.employee_rate")
    # Above the (eroded) exemption in both years, comfortably inside tier 1.
    monthly_income = raw_exemption * 1.5

    for january_month_index in (0, 12):
        eroded_exemption = (
            raw_exemption * unindexed_factor(0.02, january_month_index) * erosion_factor(0.02, 1)
        )
        expected_total = employee_rate * (monthly_income - eroded_exemption)

        base, enhanced = cpp_contributions_monthly(
            monthly_income, 0.0, january_month_index, cpp_real
        )
        # The income is real, held fixed across the two calls; only the real value of
        # the unindexed exemption changes with january_month_index, so this is a
        # closed-form check of the decay, not a mere direction check.
        assert float(base) + float(enhanced) == pytest.approx(expected_total)


def test_ei_premium_monthly_matches_erosion_factor_at_positive_inflation() -> None:
    federal_zero = real_year(load_year(2026), 0.0).federal
    federal_positive = real_year(load_year(2026), 0.02).federal
    raw_maximum = federal_zero.raw.number("ei.maximum_insurable_earnings_annual")
    rate = federal_zero.raw.number("ei.premium_rate")
    k, _ = schedule("brackets_and_credits", federal_positive.raw)
    expected_maximum = raw_maximum * erosion_factor(0.02, k)

    monthly_income = raw_maximum  # comfortably above the (eroded) maximum in one month
    result = ei_premium_monthly(monthly_income, 0.0, 0, federal_positive)
    assert float(result) == pytest.approx(rate * expected_maximum)

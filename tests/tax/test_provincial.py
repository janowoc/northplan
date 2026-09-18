# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Provincial (Alberta) income tax.

Two kinds of test, as in ``test_federal.py``: structural tests against the
real 2026 files at zero inflation, and synthetic-table arithmetic tests
against an obviously fake, hand-checkable parameter file. Neither section's
numbers should ever be copied into ``params/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.core.indexation import real_year
from engine.params.loader import load_year
from engine.tax import provincial

JANUARY = 0


# =============================================================================
# Structural tests against the real 2026 files
# =============================================================================


@pytest.fixture
def year():
    return real_year(load_year(2026), 0.0)


@pytest.fixture
def ab(year):
    return year.province("ab")


@pytest.fixture
def fed(year):
    return year.federal


def test_gross_tax_zero_at_zero_income(ab) -> None:
    assert provincial.gross_tax(0.0, ab, JANUARY) == pytest.approx(0.0)


def test_gross_tax_non_decreasing(ab) -> None:
    grid = np.array([0, 20_000, 70_000, 160_000, 200_000, 400_000], dtype=np.float64)
    result = provincial.gross_tax(grid, ab, JANUARY)
    assert np.all(np.diff(result) >= 0)


def test_gross_tax_zero_at_negative_income(ab) -> None:
    assert provincial.gross_tax(-500.0, ab, JANUARY) == pytest.approx(0.0)


def test_net_tax_floored_at_zero_when_credits_exceed_gross(ab) -> None:
    gross = provincial.gross_tax(1000.0, ab, JANUARY)
    assert provincial.net_tax(gross, gross + 1_000_000.0) == pytest.approx(0.0)


def test_age_amount_included_only_from_eligibility_age(ab, fed) -> None:
    eligibility_age = ab.number("credits.age_amount.eligibility_age_years")
    amount = ab.annual_amount("credits.age_amount.amount_annual", JANUARY)
    valuation_rate = ab.number("credits.valuation_rate")

    below = provincial.non_refundable_credits(
        0.0, eligibility_age - 1, 0.0, 0.0, 0.0, 0.0, ab, fed, JANUARY
    )
    at = provincial.non_refundable_credits(
        0.0, eligibility_age, 0.0, 0.0, 0.0, 0.0, ab, fed, JANUARY
    )
    assert at - below == pytest.approx(valuation_rate * amount)


def test_age_amount_fully_clawed_back_at_threshold_plus_amount_over_rate(ab, fed) -> None:
    eligibility_age = ab.number("credits.age_amount.eligibility_age_years")
    amount = ab.annual_amount("credits.age_amount.amount_annual", JANUARY)
    threshold = ab.annual_amount("credits.age_amount.reduction_threshold_annual", JANUARY)
    reduction_rate = ab.number("credits.age_amount.reduction_rate")
    basic_personal = ab.annual_amount("credits.basic_personal_amount_annual", JANUARY)
    valuation_rate = ab.number("credits.valuation_rate")

    income = threshold + amount / reduction_rate
    credits = provincial.non_refundable_credits(
        income, eligibility_age, 0.0, 0.0, 0.0, 0.0, ab, fed, JANUARY
    )
    assert credits == pytest.approx(valuation_rate * basic_personal)


def test_pension_income_credit_capped_at_the_amount(ab, fed) -> None:
    pension_amount = ab.annual_amount("credits.pension_income_amount_annual", JANUARY)
    valuation_rate = ab.number("credits.valuation_rate")

    at_zero = provincial.non_refundable_credits(0.0, 40, 0.0, 0.0, 0.0, 0.0, ab, fed, JANUARY)
    at_double = provincial.non_refundable_credits(
        0.0, 40, 2 * pension_amount, 0.0, 0.0, 0.0, ab, fed, JANUARY
    )
    assert at_double - at_zero == pytest.approx(valuation_rate * pension_amount)


def test_cpp_credit_capped_at_federal_maximum(ab, fed) -> None:
    """The ceiling is Canada-wide, read from ``federal_params``, not ``ab.yaml``."""
    cpp_max = fed.annual_amount("contribution_credit.cpp_maximum_annual", JANUARY)
    valuation_rate = ab.number("credits.valuation_rate")

    at_zero = provincial.non_refundable_credits(0.0, 40, 0.0, 0.0, 0.0, 0.0, ab, fed, JANUARY)
    at_double = provincial.non_refundable_credits(
        0.0, 40, 0.0, 2 * cpp_max, 0.0, 0.0, ab, fed, JANUARY
    )
    assert at_double - at_zero == pytest.approx(valuation_rate * cpp_max)


def test_ei_credit_capped_at_federal_maximum(ab, fed) -> None:
    ei_max = fed.annual_amount("contribution_credit.ei_maximum_annual", JANUARY)
    valuation_rate = ab.number("credits.valuation_rate")

    at_zero = provincial.non_refundable_credits(0.0, 40, 0.0, 0.0, 0.0, 0.0, ab, fed, JANUARY)
    at_double = provincial.non_refundable_credits(
        0.0, 40, 0.0, 0.0, 2 * ei_max, 0.0, ab, fed, JANUARY
    )
    assert at_double - at_zero == pytest.approx(valuation_rate * ei_max)


def test_dividend_tax_credit_is_linear_and_not_scaled_by_valuation_rate(ab, fed) -> None:
    gross_up_rate = ab.number("investment_income.eligible_dividend_gross_up_rate")
    credit_rate = ab.number("investment_income.eligible_dividend_credit_rate_of_gross_up")

    at_zero = provincial.non_refundable_credits(0.0, 40, 0.0, 0.0, 0.0, 0.0, ab, fed, JANUARY)
    at_1000 = provincial.non_refundable_credits(0.0, 40, 0.0, 0.0, 0.0, 1000.0, ab, fed, JANUARY)
    delta = at_1000 - at_zero
    assert delta == pytest.approx(1000.0 * gross_up_rate * credit_rate)
    assert delta != pytest.approx(
        1000.0 * gross_up_rate * credit_rate * ab.number("credits.valuation_rate")
    )


def test_provincial_dividend_rates_differ_from_federal(ab, fed) -> None:
    """ab.yaml carries its own base, not the federal one, per the file's own note."""
    assert ab.number("investment_income.eligible_dividend_credit_rate_of_gross_up") != fed.number(
        "investment_income.eligible_dividend_credit_rate_of_gross_up"
    )


# --- shape and dtype contracts ------------------------------------------------


def test_gross_tax_returns_float64_scalar_and_array(ab) -> None:
    scalar = provincial.gross_tax(50_000.0, ab, JANUARY)
    array = provincial.gross_tax(np.array([0.0, 50_000.0]), ab, JANUARY)
    assert scalar.dtype == np.float64 and scalar.shape == ()
    assert array.dtype == np.float64 and array.shape == (2,)


def test_net_tax_returns_float64_scalar_and_array() -> None:
    scalar = provincial.net_tax(100.0, 40.0)
    array = provincial.net_tax(np.array([100.0, 200.0]), np.array([40.0, 300.0]))
    assert scalar.dtype == np.float64 and scalar.shape == ()
    assert array.dtype == np.float64 and array.shape == (2,)


def test_non_refundable_credits_returns_float64_scalar_and_array(ab, fed) -> None:
    scalar = provincial.non_refundable_credits(1000.0, 40, 0.0, 0.0, 0.0, 0.0, ab, fed, JANUARY)
    array = provincial.non_refundable_credits(
        np.array([1000.0, 2000.0]),
        np.array([40, 70]),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        ab,
        fed,
        JANUARY,
    )
    assert scalar.dtype == np.float64 and scalar.shape == ()
    assert array.dtype == np.float64 and array.shape == (2,)


# =============================================================================
# Synthetic-table arithmetic tests
# =============================================================================

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
SYNTHETIC_AB = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  yearly:
    adjustment_months: [1]
    applies_to:
      - brackets.edges_annual
      - credits.basic_personal_amount_annual
      - credits.pension_income_amount_annual
      - credits.age_amount.amount_annual
      - credits.age_amount.reduction_threshold_annual

brackets:
  edges_annual: [500]
  rates: [0.05, 0.25]

credits:
  valuation_rate: 0.1
  basic_personal_amount_annual: 800
  pension_income_amount_annual: 100
  age_amount:
    eligibility_age_years: 65
    amount_annual: 300
    reduction_threshold_annual: 2000
    reduction_rate: 0.2

investment_income:
  eligible_dividend_gross_up_rate: 0.5
  eligible_dividend_credit_rate_of_gross_up: 0.3
"""

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
SYNTHETIC_FEDERAL = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
filing_month: 4
indexation:
  yearly:
    adjustment_months: [1]
    applies_to:
      - contribution_credit.cpp_maximum_annual
      - contribution_credit.ei_maximum_annual

contribution_credit:
  cpp_maximum_annual: 60
  ei_maximum_annual: 30
"""


def _write(root: Path, name: str, text: str) -> Path:
    year_dir = root / "2026"
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def synth(tmp_path: Path):
    _write(tmp_path, "ab", SYNTHETIC_AB)
    _write(tmp_path, "federal", SYNTHETIC_FEDERAL)
    loaded = real_year(load_year(2026, tmp_path), 0.0)
    return loaded.province("ab"), loaded.federal


def test_synthetic_gross_tax_hand_computed(synth) -> None:
    ab, _fed = synth
    # income 800: 500*0.05 + 300*0.25 = 25 + 75 = 100
    assert provincial.gross_tax(800.0, ab, JANUARY) == pytest.approx(100.0)


def test_synthetic_credits_hand_computed(synth) -> None:
    ab, fed = synth
    # income=0, age=65, pension=1000 (capped 100), cpp=1000 (capped 60),
    # ei=1000 (capped 30), dividends=100.
    # valuation_rate * (800 + 300 + 100 + 60 + 30) + 100 * 0.5 * 0.3
    # = 0.1 * 1290 + 15 = 129 + 15 = 144
    credits = provincial.non_refundable_credits(
        0.0, 65, 1000.0, 1000.0, 1000.0, 100.0, ab, fed, JANUARY
    )
    assert credits == pytest.approx(144.0)


def test_synthetic_net_tax_hand_computed(synth) -> None:
    ab, fed = synth
    gross = provincial.gross_tax(800.0, ab, JANUARY)
    credits = provincial.non_refundable_credits(0.0, 40, 0.0, 0.0, 0.0, 0.0, ab, fed, JANUARY)
    # gross=100, credits = 0.1 * 800 = 80 -> net = 20
    assert provincial.net_tax(gross, credits) == pytest.approx(20.0)


def test_synthetic_net_tax_floors_at_zero(synth) -> None:
    ab, fed = synth
    gross = provincial.gross_tax(0.0, ab, JANUARY)
    credits = provincial.non_refundable_credits(0.0, 65, 0.0, 0.0, 0.0, 0.0, ab, fed, JANUARY)
    assert provincial.net_tax(gross, credits) == pytest.approx(0.0)

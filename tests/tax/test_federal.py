# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Federal income tax.

Two kinds of test. The first section runs against the real 2026 parameter
files at zero inflation, so every expected relationship is read back out of
the same ``RealParamSet`` the implementation reads rather than restated as a
literal. The second section runs against ``SYNTHETIC``, an obviously fake
parameter file with round, wrong numbers chosen so every expected value below
can be written and checked by hand. Neither section's numbers should ever be
copied into ``params/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.core.indexation import real_year
from engine.core.state import IncomeLedger
from engine.params.loader import load_year
from engine.tax import federal

JANUARY = 0


def _ledger(n_paths: int = 1, **overrides: object) -> IncomeLedger:
    zeros = np.zeros(n_paths, dtype=np.float64)
    fields: dict[str, np.ndarray] = {
        "employment": zeros,
        "cpp": zeros,
        "oas": zeros,
        "db_pension": zeros,
        "rrsp_withdrawals": zeros,
        "rrif_lif_withdrawals": zeros,
        "interest": zeros,
        "eligible_dividends": zeros,
        "capital_gains": zeros,
        "resp_accumulated_income": zeros,
        "rrsp_deductions": zeros,
        "cpp_base_contributions": zeros,
        "cpp_enhanced_contributions": zeros,
        "ei_premiums": zeros,
        "remitted": zeros,
    }
    for key, value in overrides.items():
        fields[key] = np.full(n_paths, value, dtype=np.float64)
    return IncomeLedger(**fields)


# =============================================================================
# Structural tests against the real 2026 files
# =============================================================================


@pytest.fixture
def fed():
    return real_year(load_year(2026), 0.0).federal


def test_gross_tax_zero_at_zero_income(fed) -> None:
    assert federal.gross_tax(0.0, fed, JANUARY) == pytest.approx(0.0)


def test_gross_tax_non_decreasing(fed) -> None:
    grid = np.array([0, 10_000, 60_000, 120_000, 200_000, 300_000], dtype=np.float64)
    result = federal.gross_tax(grid, fed, JANUARY)
    assert np.all(np.diff(result) >= 0)


def test_gross_tax_zero_at_negative_income(fed) -> None:
    assert federal.gross_tax(-1000.0, fed, JANUARY) == pytest.approx(0.0)


def test_net_tax_floored_at_zero_when_credits_exceed_gross(fed) -> None:
    gross = federal.gross_tax(1000.0, fed, JANUARY)
    huge_credits = gross + 1_000_000.0
    assert federal.net_tax(gross, huge_credits) == pytest.approx(0.0)


def test_age_amount_included_only_from_eligibility_age(fed) -> None:
    eligibility_age = fed.number("credits.age_amount.eligibility_age_years")
    amount = fed.annual_amount("credits.age_amount.amount_annual", JANUARY)
    valuation_rate = fed.number("credits.valuation_rate")

    below = federal.non_refundable_credits(
        0.0, eligibility_age - 1, 0.0, 0.0, 0.0, 0.0, fed, JANUARY
    )
    at = federal.non_refundable_credits(0.0, eligibility_age, 0.0, 0.0, 0.0, 0.0, fed, JANUARY)
    assert at - below == pytest.approx(valuation_rate * amount)


def test_age_amount_fully_clawed_back_at_threshold_plus_amount_over_rate(fed) -> None:
    eligibility_age = fed.number("credits.age_amount.eligibility_age_years")
    amount = fed.annual_amount("credits.age_amount.amount_annual", JANUARY)
    threshold = fed.annual_amount("credits.age_amount.reduction_threshold_annual", JANUARY)
    reduction_rate = fed.number("credits.age_amount.reduction_rate")
    basic_personal = fed.annual_amount("credits.basic_personal_amount_annual", JANUARY)
    valuation_rate = fed.number("credits.valuation_rate")

    income = threshold + amount / reduction_rate
    credits = federal.non_refundable_credits(
        income, eligibility_age, 0.0, 0.0, 0.0, 0.0, fed, JANUARY
    )
    assert credits == pytest.approx(valuation_rate * basic_personal)


def test_pension_income_credit_capped_at_the_amount(fed) -> None:
    pension_amount = fed.annual_amount("credits.pension_income_amount_annual", JANUARY)
    valuation_rate = fed.number("credits.valuation_rate")

    at_zero = federal.non_refundable_credits(0.0, 40, 0.0, 0.0, 0.0, 0.0, fed, JANUARY)
    at_double = federal.non_refundable_credits(
        0.0, 40, 2 * pension_amount, 0.0, 0.0, 0.0, fed, JANUARY
    )
    assert at_double - at_zero == pytest.approx(valuation_rate * pension_amount)


def test_cpp_credit_capped_at_maximum(fed) -> None:
    cpp_max = fed.annual_amount("contribution_credit.cpp_maximum_annual", JANUARY)
    valuation_rate = fed.number("credits.valuation_rate")

    at_zero = federal.non_refundable_credits(0.0, 40, 0.0, 0.0, 0.0, 0.0, fed, JANUARY)
    at_double = federal.non_refundable_credits(0.0, 40, 0.0, 2 * cpp_max, 0.0, 0.0, fed, JANUARY)
    assert at_double - at_zero == pytest.approx(valuation_rate * cpp_max)


def test_ei_credit_capped_at_maximum(fed) -> None:
    ei_max = fed.annual_amount("contribution_credit.ei_maximum_annual", JANUARY)
    valuation_rate = fed.number("credits.valuation_rate")

    at_zero = federal.non_refundable_credits(0.0, 40, 0.0, 0.0, 0.0, 0.0, fed, JANUARY)
    at_double = federal.non_refundable_credits(0.0, 40, 0.0, 0.0, 2 * ei_max, 0.0, fed, JANUARY)
    assert at_double - at_zero == pytest.approx(valuation_rate * ei_max)


def test_dividend_tax_credit_is_linear_and_not_scaled_by_valuation_rate(fed) -> None:
    gross_up_rate = fed.number("investment_income.eligible_dividend_gross_up_rate")
    credit_rate = fed.number("investment_income.eligible_dividend_credit_rate_of_gross_up")

    at_zero = federal.non_refundable_credits(0.0, 40, 0.0, 0.0, 0.0, 0.0, fed, JANUARY)
    at_1000 = federal.non_refundable_credits(0.0, 40, 0.0, 0.0, 0.0, 1000.0, fed, JANUARY)
    delta = at_1000 - at_zero
    assert delta == pytest.approx(1000.0 * gross_up_rate * credit_rate)
    # Not scaled by valuation_rate: that is the error the arithmetic invites.
    assert delta != pytest.approx(
        1000.0 * gross_up_rate * credit_rate * fed.number("credits.valuation_rate")
    )


def test_eligible_pension_income_excludes_rrif_below_minimum_age(fed) -> None:
    min_age = fed.number("eligible_pension_income.rrif_minimum_age_years")
    ledger = _ledger(db_pension=500.0, rrif_lif_withdrawals=700.0)

    below = federal.eligible_pension_income(ledger, min_age - 1, fed)
    at = federal.eligible_pension_income(ledger, min_age, fed)
    np.testing.assert_allclose(below, [500.0])
    np.testing.assert_allclose(at, [1200.0])


def test_eligible_pension_income_never_includes_cpp_oas_or_rrsp(fed) -> None:
    ledger = _ledger(cpp=1000.0, oas=1000.0, rrsp_withdrawals=1000.0)
    result = federal.eligible_pension_income(ledger, 90, fed)
    np.testing.assert_allclose(result, [0.0])


def test_net_income_floored_at_zero(fed) -> None:
    ledger = _ledger(rrsp_deductions=1_000_000.0)
    result = federal.net_income(ledger, fed, 0.0, 0.0)
    np.testing.assert_allclose(result, [0.0])


def test_taxable_income_equals_net_income_at_zero_repayment(fed) -> None:
    ledger = _ledger(employment=50_000.0, rrsp_deductions=5_000.0)
    net = federal.net_income(ledger, fed, 0.0, 0.0)
    np.testing.assert_allclose(federal.taxable_income(net, 0.0), net)


def test_taxable_income_subtracts_the_oas_repayment(fed) -> None:
    ledger = _ledger(employment=50_000.0, rrsp_deductions=5_000.0)
    net = federal.net_income(ledger, fed, 0.0, 0.0)
    np.testing.assert_allclose(federal.taxable_income(net, 1_234.0), net - 1_234.0)


def test_total_income_grosses_up_dividends_and_includes_capital_gains(fed) -> None:
    gross_up_rate = fed.number("investment_income.eligible_dividend_gross_up_rate")
    inclusion_rate = fed.number("investment_income.capital_gains_inclusion_rate")
    ledger = _ledger(eligible_dividends=1000.0, capital_gains=2000.0)
    result = federal.total_income(ledger, fed, 0.0, 0.0)
    expected = 1000.0 * (1 + gross_up_rate) + 2000.0 * inclusion_rate
    np.testing.assert_allclose(result, [expected])


def test_total_income_applies_transfers(fed) -> None:
    ledger = _ledger(employment=10_000.0)
    with_transfer_in = federal.total_income(ledger, fed, 5000.0, 0.0)
    with_transfer_out = federal.total_income(ledger, fed, 0.0, 5000.0)
    np.testing.assert_allclose(with_transfer_in, [15_000.0])
    np.testing.assert_allclose(with_transfer_out, [5_000.0])


def test_a_net_capital_loss_does_not_reduce_total_net_or_taxable_income(fed) -> None:
    loss_ledger = _ledger(employment=60_000.0, capital_gains=-10_526.0)
    zero_ledger = _ledger(employment=60_000.0, capital_gains=0.0)

    loss_total = federal.total_income(loss_ledger, fed, 0.0, 0.0)
    zero_total = federal.total_income(zero_ledger, fed, 0.0, 0.0)
    np.testing.assert_array_equal(loss_total, zero_total)

    loss_net = federal.net_income(loss_ledger, fed, 0.0, 0.0)
    zero_net = federal.net_income(zero_ledger, fed, 0.0, 0.0)
    np.testing.assert_array_equal(loss_net, zero_net)

    np.testing.assert_array_equal(
        federal.taxable_income(loss_net, 0.0), federal.taxable_income(zero_net, 0.0)
    )


def test_the_capital_loss_floor_is_per_path(fed) -> None:
    inclusion_rate = fed.number("investment_income.capital_gains_inclusion_rate")
    ledger = _ledger(n_paths=2, employment=60_000.0, capital_gains=np.array([-10_526.0, 2_000.0]))
    result = federal.total_income(ledger, fed, 0.0, 0.0)
    np.testing.assert_array_equal(result[0], 60_000.0)
    np.testing.assert_allclose(result[1], 60_000.0 + 2_000.0 * inclusion_rate)


# =============================================================================
# death_year_capital_loss_deduction (#36, ITA 111(2))
# =============================================================================


def test_death_year_deduction_zero_where_not_died_in_year(fed) -> None:
    ledger = _ledger(capital_gains=-10_000.0)
    result = federal.death_year_capital_loss_deduction(ledger, fed, False)
    assert result == pytest.approx(0.0)


def test_death_year_deduction_zero_for_a_gain(fed) -> None:
    ledger = _ledger(capital_gains=10_000.0)
    result = federal.death_year_capital_loss_deduction(ledger, fed, True)
    assert result == pytest.approx(0.0)


def test_death_year_deduction_is_inclusion_rate_times_the_loss(fed) -> None:
    inclusion_rate = fed.number("investment_income.capital_gains_inclusion_rate")
    ledger = _ledger(capital_gains=-8_000.0)
    result = federal.death_year_capital_loss_deduction(ledger, fed, True)
    assert result == pytest.approx(inclusion_rate * 8_000.0)


def test_death_year_deduction_is_per_path(fed) -> None:
    inclusion_rate = fed.number("investment_income.capital_gains_inclusion_rate")
    ledger = _ledger(n_paths=2, capital_gains=np.array([-8_000.0, 3_000.0]))
    died_in_year = np.array([True, True])
    result = federal.death_year_capital_loss_deduction(ledger, fed, died_in_year)
    np.testing.assert_allclose(result, [inclusion_rate * 8_000.0, 0.0])


def test_death_year_deduction_died_in_year_is_per_path(fed) -> None:
    inclusion_rate = fed.number("investment_income.capital_gains_inclusion_rate")
    ledger = _ledger(n_paths=2, capital_gains=np.array([-8_000.0, -8_000.0]))
    died_in_year = np.array([True, False])
    result = federal.death_year_capital_loss_deduction(ledger, fed, died_in_year)
    np.testing.assert_allclose(result, [inclusion_rate * 8_000.0, 0.0])


# =============================================================================
# Non-zero inflation: an unindexed credit decays with the tax year; an
# indexed one, and gross_tax, do not
# =============================================================================


@pytest.fixture
def fed_nonzero_inflation():
    return real_year(load_year(2026), 0.02).federal


def test_pension_credit_decays_with_the_tax_year_at_nonzero_inflation(
    fed_nonzero_inflation,
) -> None:
    fed = fed_nonzero_inflation
    valuation_rate = fed.number("credits.valuation_rate")
    amount_at_0 = fed.annual_amount("credits.pension_income_amount_annual", 0)
    amount_at_36 = fed.annual_amount("credits.pension_income_amount_annual", 36)
    assert amount_at_36 < amount_at_0

    # Eligible pension income at or above the larger (earlier-year) amount,
    # so the credit is capped by the annual amount in both years, never by
    # income — isolating the plumbing this test pins.
    eligible_pension_income = amount_at_0
    credits_at_0 = federal.non_refundable_credits(
        0.0, 40, eligible_pension_income, 0.0, 0.0, 0.0, fed, 0
    )
    credits_at_36 = federal.non_refundable_credits(
        0.0, 40, eligible_pension_income, 0.0, 0.0, 0.0, fed, 36
    )

    assert credits_at_36 < credits_at_0
    assert credits_at_0 - credits_at_36 == pytest.approx(
        valuation_rate * (amount_at_0 - amount_at_36)
    )


def test_gross_tax_invariant_across_tax_years_at_nonzero_inflation(fed_nonzero_inflation) -> None:
    fed = fed_nonzero_inflation
    # brackets.edges_annual is on an indexed schedule: its erosion factor is
    # one number per schedule per scenario, not a function of elapsed time,
    # so gross_tax on a fixed income must be identical at every valid
    # january_month_index. An implementation that let it decay under
    # unindexed_factor's rule instead — the obvious wrong reading — would
    # fail this.
    at_0 = federal.gross_tax(100_000.0, fed, 0)
    at_36 = federal.gross_tax(100_000.0, fed, 36)
    at_72 = federal.gross_tax(100_000.0, fed, 72)
    assert at_0 == pytest.approx(at_36)
    assert at_0 == pytest.approx(at_72)


def test_basic_personal_credit_invariant_across_tax_years_at_nonzero_inflation(
    fed_nonzero_inflation,
) -> None:
    fed = fed_nonzero_inflation
    # Isolated from every other credit: age 40 is below eligibility (age
    # amount 0), and eligible pension income/CPP/EI/dividends are all 0, so
    # this is valuation_rate * basic_personal_amount_annual alone — indexed,
    # and so invariant across tax years by the same reasoning as gross_tax.
    at_0 = federal.non_refundable_credits(0.0, 40, 0.0, 0.0, 0.0, 0.0, fed, 0)
    at_36 = federal.non_refundable_credits(0.0, 40, 0.0, 0.0, 0.0, 0.0, fed, 36)
    assert at_0 == pytest.approx(at_36)


# --- shape and dtype contracts ------------------------------------------------


def test_gross_tax_returns_float64_scalar_and_array(fed) -> None:
    scalar = federal.gross_tax(50_000.0, fed, JANUARY)
    array = federal.gross_tax(np.array([0.0, 50_000.0]), fed, JANUARY)
    assert isinstance(scalar, np.ndarray) and scalar.dtype == np.float64 and scalar.shape == ()
    assert array.dtype == np.float64 and array.shape == (2,)


def test_net_tax_returns_float64_scalar_and_array() -> None:
    scalar = federal.net_tax(100.0, 40.0)
    array = federal.net_tax(np.array([100.0, 200.0]), np.array([40.0, 300.0]))
    assert scalar.dtype == np.float64 and scalar.shape == ()
    assert array.dtype == np.float64 and array.shape == (2,)


def test_taxable_income_returns_float64_scalar_and_array() -> None:
    scalar = federal.taxable_income(100.0, 40.0)
    array = federal.taxable_income(np.array([100.0, 200.0]), np.array([40.0, 300.0]))
    assert scalar.dtype == np.float64 and scalar.shape == ()
    assert array.dtype == np.float64 and array.shape == (2,)


def test_non_refundable_credits_returns_float64_scalar_and_array(fed) -> None:
    scalar = federal.non_refundable_credits(1000.0, 40, 0.0, 0.0, 0.0, 0.0, fed, JANUARY)
    array = federal.non_refundable_credits(
        np.array([1000.0, 2000.0]),
        np.array([40, 70]),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        fed,
        JANUARY,
    )
    assert scalar.dtype == np.float64 and scalar.shape == ()
    assert array.dtype == np.float64 and array.shape == (2,)


# =============================================================================
# Synthetic-table arithmetic tests
# =============================================================================

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
#: Round, wrong numbers chosen so every expected value below is exact.
SYNTHETIC = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
filing_month: 4
indexation:
  yearly:
    adjustment_months: [1]
    applies_to:
      - brackets.edges_annual
      - credits.basic_personal_amount_annual
      - credits.age_amount.amount_annual
      - credits.age_amount.reduction_threshold_annual
      - contribution_credit.cpp_maximum_annual
      - contribution_credit.ei_maximum_annual
  unindexed:
    adjustment_months: []
    applies_to:
      - credits.pension_income_amount_annual

brackets:
  edges_annual: [1000, 2000]
  rates: [0.1, 0.2, 0.5]

credits:
  valuation_rate: 0.2
  basic_personal_amount_annual: 1000
  pension_income_amount_annual: 200
  age_amount:
    eligibility_age_years: 65
    amount_annual: 500
    reduction_threshold_annual: 3000
    reduction_rate: 0.1

contribution_credit:
  cpp_maximum_annual: 100
  ei_maximum_annual: 50

investment_income:
  capital_gains_inclusion_rate: 0.5
  eligible_dividend_gross_up_rate: 0.5
  eligible_dividend_credit_rate_of_gross_up: 0.4

pension_splitting:
  maximum_transfer_share: 0.5

eligible_pension_income:
  rrif_minimum_age_years: 65
"""


def _write(root: Path, name: str, text: str) -> Path:
    year_dir = root / "2026"
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def synth(tmp_path: Path):
    _write(tmp_path, "federal", SYNTHETIC)
    return real_year(load_year(2026, tmp_path), 0.0).federal


def test_synthetic_gross_tax_hand_computed(synth) -> None:
    # income 2500: 1000*0.1 + 1000*0.2 + 500*0.5 = 100 + 200 + 250 = 550
    assert federal.gross_tax(2500.0, synth, JANUARY) == pytest.approx(550.0)


def test_synthetic_credits_hand_computed(synth) -> None:
    # income=0, age=65 (>=eligibility), pension=1000 (capped at 200),
    # cpp=1000 (capped at 100), ei=1000 (capped at 50), dividends=100.
    #
    # valuation_rate * (1000 + 500 + 200 + 100 + 50) + 100 * 0.5 * 0.4
    # = 0.2 * 1850 + 20 = 370 + 20 = 390
    credits = federal.non_refundable_credits(0.0, 65, 1000.0, 1000.0, 1000.0, 100.0, synth, JANUARY)
    assert credits == pytest.approx(390.0)


def test_synthetic_credits_age_amount_partial_reduction(synth) -> None:
    # income=4000: excess over 3000 is 1000, reduction = 1000*0.1 = 100.
    # age_amount_allowed = 500 - 100 = 400.
    # valuation_rate * (1000 [basic] + 400 [age]) = 0.2 * 1400 = 280
    credits = federal.non_refundable_credits(4000.0, 65, 0.0, 0.0, 0.0, 0.0, synth, JANUARY)
    assert credits == pytest.approx(280.0)


def test_synthetic_net_tax_hand_computed(synth) -> None:
    gross = federal.gross_tax(2500.0, synth, JANUARY)
    credits = federal.non_refundable_credits(0.0, 40, 0.0, 0.0, 0.0, 0.0, synth, JANUARY)
    # gross=550, credits = 0.2 * 1000 = 200 -> net = 350
    assert federal.net_tax(gross, credits) == pytest.approx(350.0)


def test_synthetic_net_tax_floors_at_zero(synth) -> None:
    gross = federal.gross_tax(0.0, synth, JANUARY)
    credits = federal.non_refundable_credits(0.0, 65, 0.0, 0.0, 0.0, 0.0, synth, JANUARY)
    assert federal.net_tax(gross, credits) == pytest.approx(0.0)


def test_synthetic_total_income_hand_computed(synth) -> None:
    ledger = _ledger(
        employment=1000.0,
        cpp=100.0,
        oas=100.0,
        db_pension=100.0,
        rrsp_withdrawals=100.0,
        rrif_lif_withdrawals=100.0,
        interest=100.0,
        eligible_dividends=1000.0,
        capital_gains=1000.0,
        resp_accumulated_income=100.0,
    )
    # base = 1000+100+100+100+100+100+100 = 1600
    # dividends grossed up: 1000 * 1.5 = 1500
    # capital gains: 1000 * 0.5 = 500
    # resp: 100
    # total = 1600 + 1500 + 500 + 100 = 3700
    result = federal.total_income(ledger, synth, 0.0, 0.0)
    np.testing.assert_allclose(result, [3700.0])

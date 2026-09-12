# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Provincial income tax.

Alberta first; the module is written to take a province's parameter set
rather than hard-coding one, so adding a province is a YAML file plus a
verification row, not new code. Parameters come from
``params/{year}/{province}.yaml``.

Alberta assesses the federal taxable income: this file computes provincial
tax **on the federal income figure**, taken as an argument. It never defines
total, net, taxable, or eligible pension income itself — there is one income
figure per person per year and ``engine.tax.federal`` owns it.

The province's parameter file also carries the LIF maximum withdrawal rules
that ``engine/accounts/lira.py`` reads, since those are provincially set.
Those are annual limits fixed each January, not monthly ones.

They are in the same file but reached by a different route, and the
difference matters. Income tax follows the province of **residence**, so this
module is handed ``RealParamYear.province(household.province)``. A LIF
follows the province its originating pension was **registered** in, so
``lira`` is handed ``RealParamYear.jurisdiction(LifState.jurisdiction)``. For
a household that never moved these resolve to one object; for one that did,
taking the LIF table from this module's parameter set is wrong and nothing
downstream will say so.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.core.indexation import RealParamSet
from engine.tax import brackets


def gross_tax(
    taxable: ArrayLike,
    params: RealParamSet,
    january_month_index: int,
) -> NDArray[np.float64]:
    """Provincial tax before credits, on the federal taxable income.

    Args:
        taxable: Federal taxable income, real dollars, ``(n_paths,)`` or
            scalar.
        params: The province's parameter set for the tax year.
        january_month_index: Month index of January of the tax year.

    Returns:
        Provincial tax before non-refundable credits.
    """
    edges = params.annual_amounts("brackets.edges_annual", january_month_index)
    rates = params.numbers("brackets.rates")
    return brackets.tax_on_income(taxable, edges, rates)


def non_refundable_credits(
    net_income: ArrayLike,
    age_at_end_of_year: ArrayLike,
    eligible_pension_income: ArrayLike,
    cpp_base_contributions: ArrayLike,
    ei_premiums: ArrayLike,
    eligible_dividends: ArrayLike,
    params: RealParamSet,
    federal_params: RealParamSet,
    january_month_index: int,
) -> NDArray[np.float64]:
    """Value of provincial non-refundable credits, in dollars of tax reduced.

    Same structure as ``engine.tax.federal.non_refundable_credits``, with the
    province's own credit amounts, valuation rate, and dividend gross-up and
    credit rates — none of it derivable from the federal file.

    Args:
        net_income: Federal net income, real dollars.
        age_at_end_of_year: Age in whole years on 31 December.
        eligible_pension_income: Eligible pension income for the pension
            income amount.
        cpp_base_contributions: This year's CPP base-tier contributions.
        ei_premiums: This year's EI premiums.
        eligible_dividends: Pre-gross-up eligible dividends.
        params: The province's parameter set for the tax year.
        federal_params: The ``federal`` parameter set, supplying only the
            Income Tax Act ceilings the CPP and EI credits are capped at —
            those are Canada-wide, not provincial.
        january_month_index: Month index of January of the tax year.

    Returns:
        Total credit value, real dollars.
    """
    valuation_rate = params.number("credits.valuation_rate")
    basic_personal_amount = params.annual_amount(
        "credits.basic_personal_amount_annual", january_month_index
    )

    eligibility_age = params.number("credits.age_amount.eligibility_age_years")
    age_amount = params.annual_amount("credits.age_amount.amount_annual", january_month_index)
    reduction_threshold = params.annual_amount(
        "credits.age_amount.reduction_threshold_annual", january_month_index
    )
    reduction_rate = params.number("credits.age_amount.reduction_rate")

    net_income_arr = np.asarray(net_income, dtype=np.float64)
    age_arr = np.asarray(age_at_end_of_year)
    age_amount_allowed = np.where(
        age_arr >= eligibility_age,
        np.clip(
            age_amount - reduction_rate * np.clip(net_income_arr - reduction_threshold, 0, None),
            0,
            None,
        ),
        0.0,
    )

    pension_income_amount = params.annual_amount(
        "credits.pension_income_amount_annual", january_month_index
    )
    pension_amount_allowed = np.clip(
        np.minimum(np.asarray(eligible_pension_income, dtype=np.float64), pension_income_amount),
        0,
        None,
    )

    cpp_maximum = federal_params.annual_amount(
        "contribution_credit.cpp_maximum_annual", january_month_index
    )
    cpp_allowed = np.clip(
        np.minimum(np.asarray(cpp_base_contributions, dtype=np.float64), cpp_maximum), 0, None
    )

    ei_maximum = federal_params.annual_amount(
        "contribution_credit.ei_maximum_annual", january_month_index
    )
    ei_allowed = np.clip(np.minimum(np.asarray(ei_premiums, dtype=np.float64), ei_maximum), 0, None)

    gross_up_rate = params.number("investment_income.eligible_dividend_gross_up_rate")
    credit_rate_of_gross_up = params.number(
        "investment_income.eligible_dividend_credit_rate_of_gross_up"
    )
    dividend_tax_credit = (
        np.asarray(eligible_dividends, dtype=np.float64) * gross_up_rate * credit_rate_of_gross_up
    )

    ordinary = (
        basic_personal_amount
        + age_amount_allowed
        + pension_amount_allowed
        + cpp_allowed
        + ei_allowed
    )
    return np.asarray(valuation_rate * ordinary + dividend_tax_credit, dtype=np.float64)


def net_tax(gross: ArrayLike, credits: ArrayLike) -> NDArray[np.float64]:
    """Provincial tax after non-refundable credits, floored at zero."""
    net = np.asarray(gross, dtype=np.float64) - np.asarray(credits, dtype=np.float64)
    return np.asarray(np.clip(net, 0, None), dtype=np.float64)

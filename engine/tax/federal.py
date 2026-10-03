# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Federal income tax.

Defines five income figures — total income, net income (line 23400), taxable
income (line 23600), qualified pension income and eligible pension income —
and federal tax on an income figure, before and after non-refundable credits.
``engine.tax.provincial`` defines none of its own and is handed a figure.

Two shapes of entry point. ``total_income``, ``net_income``,
``qualified_pension_income`` and ``eligible_pension_income`` build a figure
from an ``engine.core.state.IncomeLedger`` — one person's income components
accumulated over the year — and, all but ``qualified_pension_income``, a
``federal`` ``engine.core.indexation.RealParamSet``. ``taxable_income``, ``gross_tax``,
``non_refundable_credits`` and ``net_tax`` take figures already computed and
never see a ledger. Every income figure here is annual rather than monthly: a
caller holding a monthly figure scales to a year first, and what it hands over
may be an annualised approximation rather than the calendar year's income
(L16).

``net_income`` is line 23400 and is never adjusted downward afterward;
:func:`taxable_income` is line 23600, net income less the OAS repayment.
Line 26000 equals line 23600 except in a person's year of death, when the
ITA 111(2) deduction (:func:`death_year_capital_loss_deduction`,
``docs/limitations.md`` L17) reduces it further; that reduction is applied
by ``engine.tax.combined.person_assessment``, not by this module, since it
never changes net income or the OAS repayment. Parameters come from
``params/{year}/federal.yaml``. Nothing numeric lives in this file.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.core.indexation import RealParamSet
from engine.core.state import IncomeLedger
from engine.tax import brackets


def total_income(
    ledger: IncomeLedger,
    params: RealParamSet,
    transfer_in: ArrayLike,
    transfer_out: ArrayLike,
) -> NDArray[np.float64]:
    """Total income for the year, adjusted for the pension-splitting election.

    Not line 15000 once a split is elected: in reality the transfer is a
    deduction and an inclusion (lines 21000 and 11600), and this nets both into
    the total. The difference never reaches tax, since :func:`net_income` is
    the only consumer of this figure and :func:`taxable_income` derives from
    ``net_income``, but an income-tested rule wanting a pre-election figure
    must not read this.

    A net capital loss for the year (a negative ``ledger.capital_gains``) is
    floored at zero before the inclusion rate, so it never reduces other
    income; the ledger stays signed, so losses still offset gains of the same
    year (L17).

    Args:
        ledger: This person's income components, accumulated over the year.
        params: The ``federal`` parameter set for the tax year.
        transfer_in: Split-eligible pension income received from the other
            spouse, dollars, ``(n_paths,)`` or scalar. 0 for no split.
        transfer_out: Split-eligible pension income given to the other
            spouse, same shape. 0 for no split.

    Returns:
        Total income, real dollars, shape broadcast from the arguments.
    """
    gross_up_rate = params.number("investment_income.eligible_dividend_gross_up_rate")
    inclusion_rate = params.number("investment_income.capital_gains_inclusion_rate")
    total = (
        ledger.employment
        + ledger.cpp
        + ledger.oas
        + ledger.db_pension
        + ledger.rrsp_withdrawals
        + ledger.rrif_lif_withdrawals
        + ledger.interest
        + ledger.eligible_dividends * (1 + gross_up_rate)
        + np.maximum(ledger.capital_gains, 0.0) * inclusion_rate
        + ledger.resp_accumulated_income
        + np.asarray(transfer_in, dtype=np.float64)
        - np.asarray(transfer_out, dtype=np.float64)
    )
    return np.asarray(total, dtype=np.float64)


def net_income(
    ledger: IncomeLedger,
    params: RealParamSet,
    transfer_in: ArrayLike,
    transfer_out: ArrayLike,
) -> NDArray[np.float64]:
    """Net income (line 23400): total income less this model's only deductions (L14).

    Floored at zero once, here, after the pension-splitting transfers — there
    is no second floor anywhere downstream. Nothing is ever subtracted back
    out of this figure: line 23600 (see :func:`taxable_income`) is a separate
    figure derived from this one by subtracting the OAS repayment, not a
    reduction of it.

    Args:
        ledger: This person's income components, accumulated over the year.
        params: The ``federal`` parameter set for the tax year.
        transfer_in: Split-eligible pension income received, see
            :func:`total_income`.
        transfer_out: Split-eligible pension income given, see
            :func:`total_income`.

    Returns:
        Net income, real dollars, non-negative.
    """
    gross = total_income(ledger, params, transfer_in, transfer_out)
    net = gross - ledger.rrsp_deductions - ledger.cpp_enhanced_contributions
    return np.asarray(np.clip(net, 0, None), dtype=np.float64)


def taxable_income(
    net_income: ArrayLike,
    oas_repayment: ArrayLike,
) -> NDArray[np.float64]:
    """Taxable income (line 23600): net income less the OAS repayment.

    This is the one place the subtraction happens. This function applies no
    floor and enforces nothing about how its two arguments relate: the result
    is non-negative in the engine only because of how the one real caller
    pairs them. ``engine.tax.combined.oas_repayment`` returns a fraction below
    one of the excess of net income over a positive threshold, capped at OAS
    received — hence strictly less than net income whenever it is non-zero,
    and the cap only lowers it further — so the subtraction here cannot go
    negative for that caller; :func:`net_income`'s own docstring already
    states there is no second floor downstream. Line 26000 equals this figure
    except in a person's year of death, when
    :func:`death_year_capital_loss_deduction` (L17) reduces it further; that
    reduction is a Division C deduction applied by
    ``engine.tax.combined.person_assessment``, not here, since it must not
    feed back into the OAS repayment or the age amount, both of which are
    tested against this figure.

    Args:
        net_income: Line 23400, real dollars, ``(n_paths,)`` or scalar. Shadows
            the module-level :func:`net_income` inside this body only.
        oas_repayment: The social benefits repayment (line 23500), same shape.

    Returns:
        Net income after the social benefits repayment, real dollars, shape
        broadcast from the arguments.
    """
    return np.asarray(
        np.asarray(net_income, dtype=np.float64) - np.asarray(oas_repayment, dtype=np.float64),
        dtype=np.float64,
    )


def death_year_capital_loss_deduction(
    ledger: IncomeLedger,
    params: RealParamSet,
    died_in_year: ArrayLike,
) -> NDArray[np.float64]:
    """ITA 111(2) deduction for a person's own year-of-death net capital loss.

    In the year of death (and the preceding year, not modelled here — see
    ``docs/limitations.md`` L17), a net capital loss for the year is
    deductible from **taxable income** against any income
    (https://laws-lois.justice.gc.ca/eng/acts/i-3.3/section-111.html). This is
    a Division C deduction: it reduces line 26000 below line 23600 but never
    changes net income (line 23400) or the OAS repayment, which are both
    computed before it applies. The model carries no carried-forward losses
    and no capital gains exemption, so the deduction is only the current
    year's own net loss, at the same inclusion rate :func:`total_income`
    applies to a gain.

    Args:
        ledger: This person's income components, accumulated over the year.
            Only ``capital_gains`` is read; a positive value (a net gain)
            gives a zero deduction.
        params: The ``federal`` parameter set for the tax year.
        died_in_year: Whether this is this person's year of death,
            ``(n_paths,)`` or scalar.

    Returns:
        The deduction, real dollars, non-negative, zero where
        ``died_in_year`` is false or ``ledger.capital_gains`` is non-negative.
    """
    inclusion_rate = params.number("investment_income.capital_gains_inclusion_rate")
    loss = np.maximum(-ledger.capital_gains, 0.0)
    return np.asarray(
        np.where(np.asarray(died_in_year, dtype=bool), inclusion_rate * loss, 0.0),
        dtype=np.float64,
    )


def gross_tax(
    taxable: ArrayLike,
    params: RealParamSet,
    january_month_index: int,
) -> NDArray[np.float64]:
    """Federal tax before credits.

    Args:
        taxable: The income the brackets are applied to — line 23600 in the
            annual assessment, or the annualised single-source approximation
            in the monthly withholding estimate (L16), real dollars,
            ``(n_paths,)`` or scalar.
        params: The ``federal`` parameter set for the tax year.
        january_month_index: Month index of January of the tax year.

    Returns:
        Federal tax before non-refundable credits.
    """
    edges = params.annual_amounts("brackets.edges_annual", january_month_index)
    rates = params.numbers("brackets.rates")
    return brackets.tax_on_income(taxable, edges, rates)


def non_refundable_credits(
    income: ArrayLike,
    age_at_end_of_year: ArrayLike,
    eligible_pension_income: ArrayLike,
    cpp_base_contributions: ArrayLike,
    ei_premiums: ArrayLike,
    eligible_dividends: ArrayLike,
    params: RealParamSet,
    january_month_index: int,
) -> NDArray[np.float64]:
    """Value of federal non-refundable credits, in dollars of tax reduced.

    ``valuation_rate * (basic personal + age + pension + CPP + EI) +
    dividend tax credit``. The dividend tax credit is NOT scaled by
    ``valuation_rate``: ``eligible_dividend_credit_rate_of_gross_up`` is
    already a credit rate in dollars of tax, valued against the gross-up
    amount (``investment_income.eligible_dividend_gross_up_rate`` times the
    dividend), not the dividend or the grossed-up dividend.

    Args:
        income: The income the credits are tested against — line 23600 in the
            annual assessment, or the annualised single-source approximation
            in the monthly withholding estimate (L16), real dollars.
        age_at_end_of_year: Age in whole years on 31 December, from
            ``engine.core.timeline.age_at_end_of_year``.
        eligible_pension_income: Eligible pension income for the pension
            income amount, see :func:`eligible_pension_income`.
        cpp_base_contributions: This year's CPP base-tier contributions.
        ei_premiums: This year's EI premiums.
        eligible_dividends: Pre-gross-up eligible dividends, the same figure
            :func:`total_income` grosses up.
        params: The ``federal`` parameter set for the tax year.
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

    income_arr = np.asarray(income, dtype=np.float64)
    age_arr = np.asarray(age_at_end_of_year)
    age_amount_allowed = np.where(
        age_arr >= eligibility_age,
        np.clip(
            age_amount - reduction_rate * np.clip(income_arr - reduction_threshold, 0, None),
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

    cpp_maximum = params.annual_amount(
        "contribution_credit.cpp_maximum_annual", january_month_index
    )
    cpp_allowed = np.clip(
        np.minimum(np.asarray(cpp_base_contributions, dtype=np.float64), cpp_maximum), 0, None
    )

    ei_maximum = params.annual_amount("contribution_credit.ei_maximum_annual", january_month_index)
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
    """Federal tax after non-refundable credits, floored at zero.

    Args:
        gross: Output of :func:`gross_tax`.
        credits: Output of :func:`non_refundable_credits`.

    Returns:
        Federal tax payable, non-negative.
    """
    net = np.asarray(gross, dtype=np.float64) - np.asarray(credits, dtype=np.float64)
    return np.asarray(np.clip(net, 0, None), dtype=np.float64)


def qualified_pension_income(ledger: IncomeLedger) -> NDArray[np.float64]:
    """Qualified pension income (ITA 118(7)), at any age.

    The DB pension, and the RRIF/LIF withdrawals paid out of a balance rolled
    over from a deceased spouse (``ledger.inherited_rrif_lif_withdrawals``,
    L15).

    Args:
        ledger: This person's income components, accumulated over the year.

    Returns:
        Qualified pension income, real dollars, non-negative.
    """
    return np.asarray(ledger.db_pension + ledger.inherited_rrif_lif_withdrawals, dtype=np.float64)


def eligible_pension_income(
    ledger: IncomeLedger,
    age_at_end_of_year: ArrayLike,
    params: RealParamSet,
) -> NDArray[np.float64]:
    """Income eligible for the pension income amount and for splitting.

    Always :func:`qualified_pension_income`; every RRIF/LIF withdrawal from
    the year the recipient turns
    ``eligible_pension_income.rrif_minimum_age_years`` by year end. Never
    CPP, OAS, or RRSP withdrawals.

    Args:
        ledger: This person's income components, accumulated over the year.
        age_at_end_of_year: Age in whole years on 31 December.
        params: The ``federal`` parameter set for the tax year.

    Returns:
        Eligible pension income, real dollars, non-negative.
    """
    min_age = params.number("eligible_pension_income.rrif_minimum_age_years")
    age_arr = np.asarray(age_at_end_of_year)
    result = ledger.db_pension + np.where(
        age_arr >= min_age, ledger.rrif_lif_withdrawals, ledger.inherited_rrif_lif_withdrawals
    )
    return np.asarray(result, dtype=np.float64)

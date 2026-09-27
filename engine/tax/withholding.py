# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Withholding: a prepayment, remitted in the month, not tax.

Withholding is remitted monthly into ``IncomeLedger.remitted``, reconciled
against ``engine.tax.combined.household_assessment`` at the December close,
and settled — the balance owing or a refund — in the filing month (L16).
Nothing here is an assessment.

Everything here is deliberately an approximation: :func:`registered_withholding`
of the CRA lump-sum withdrawal table, and :func:`payroll_withholding_monthly`
of the payroll deduction tables. The figures each derives are cash-timing
estimates, not the household's actual accrued contributions or its actual
tax liability for the month.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.core.indexation import RealParamSet, RealParamYear
from engine.core.timeline import MONTHS_PER_YEAR
from engine.tax import federal, provincial


def _validated_band_table(
    edges: NDArray[np.float64], rates: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``edges`` and ``rates`` as given, or ``ValueError`` if not a valid band table.

    A lump-sum withholding band table has the same shape invariant as a
    bracket ladder — edges strictly ascending and positive, one more rate than
    edge — validated separately from ``engine.tax.brackets._validated_table``
    rather than by importing it: this checks a withholding table for a
    different purpose, and the two must be free to diverge.
    """
    if edges.size:
        if edges[0] <= 0:
            raise ValueError(f"first withholding edge must be positive, got {edges[0]}")
        if np.any(np.diff(edges) <= 0):
            raise ValueError(f"withholding edges must strictly ascend, got {edges}")
    if rates.size != edges.size + 1:
        raise ValueError(
            f"expected {edges.size + 1} withholding rates for {edges.size} edges "
            f"(len(rates) must be len(edges) + 1), got {rates.size} rates"
        )
    return edges, rates


def registered_withholding(
    gross_withdrawal: ArrayLike,
    params: RealParamSet,
    month_index: int,
) -> NDArray[np.float64]:
    """Tax withheld at source from one RRSP or RRIF-above-minimum withdrawal.

    The rate is selected by the size of this single withdrawal and applied to
    the whole withdrawal, not to the excess over a band edge. An edge is the
    **inclusive upper bound** of its band, so a withdrawal of exactly an edge
    takes the lower rate — the opposite convention from
    ``engine.tax.brackets.marginal_rate``, which puts the next dollar in the
    higher bracket; this uses ``side="left"`` where that uses ``side="right"``.

    Args:
        gross_withdrawal: One withdrawal, real dollars, ``(n_paths,)`` or
            scalar. Zero or negative withholds zero.
        params: The ``rrif`` parameter set for the tax year.
        month_index: Month index of the withdrawal itself — a band edge is
            tested against one month's withdrawal, never a year's income.

    Returns:
        Amount withheld, non-negative.

    Raises:
        ValueError: If the withholding table is not a valid band ladder: the
            edges do not strictly ascend and are not all positive, or there is
            not exactly one more rate than edge.
    """
    edges = np.asarray(params.amounts("withholding.edges_each", month_index), dtype=np.float64)
    rates = np.asarray(params.numbers("withholding.rates"), dtype=np.float64)
    edges, rates = _validated_band_table(edges, rates)
    withdrawal = np.asarray(gross_withdrawal, dtype=np.float64)
    index = np.searchsorted(edges, withdrawal, side="left")
    rate = rates[index]
    return np.asarray(np.where(withdrawal > 0, withdrawal * rate, 0.0), dtype=np.float64)


def _implied_cpp_base_contributions(
    annual_earnings: ArrayLike,
    cpp_params: RealParamSet,
    january_month_index: int,
) -> NDArray[np.float64]:
    """Payroll-table approximation (L16) of the CPP base-tier contribution implied.

    Only the first tier's base rate generates a non-refundable credit; the
    enhanced portion and the second tier are deductions, not credits, and are
    not computed here. Not the household's actual accrued contribution, which
    the step (``engine/core/step.py``) accrues into the ``IncomeLedger``.

    Args:
        annual_earnings: Annualised employment earnings, real dollars.
        cpp_params: The ``cpp`` parameter set for the tax year.
        january_month_index: Month index of January of the tax year.

    Returns:
        Implied CPP base-tier contribution, non-negative.
    """
    base_rate = cpp_params.number("contributions.first_tier.base_rate")
    ceiling = cpp_params.annual_amount(
        "contributions.first_tier.ceiling_annual", january_month_index
    )
    exemption = cpp_params.annual_amount(
        "contributions.first_tier.basic_exemption_annual", january_month_index
    )
    earnings = np.asarray(annual_earnings, dtype=np.float64)
    contribution = base_rate * np.clip(np.minimum(earnings, ceiling) - exemption, 0, None)
    return np.asarray(contribution, dtype=np.float64)


def _implied_ei_premiums(
    annual_earnings: ArrayLike,
    federal_params: RealParamSet,
    january_month_index: int,
) -> NDArray[np.float64]:
    """Payroll-table approximation (L16) of the EI premium implied.

    Not the household's actual accrued premium, which the step
    (``engine/core/step.py``) accrues into the ``IncomeLedger``.

    Args:
        annual_earnings: Annualised employment earnings, real dollars.
        federal_params: The ``federal`` parameter set for the tax year.
        january_month_index: Month index of January of the tax year.

    Returns:
        Implied EI premium, non-negative.
    """
    rate = federal_params.number("ei.premium_rate")
    maximum = federal_params.annual_amount(
        "ei.maximum_insurable_earnings_annual", january_month_index
    )
    earnings = np.asarray(annual_earnings, dtype=np.float64)
    return np.asarray(rate * np.clip(np.minimum(earnings, maximum), 0, None), dtype=np.float64)


def payroll_withholding_monthly(
    monthly_amount: ArrayLike,
    age_at_end_of_year: ArrayLike,
    is_employment: bool,
    province: str,
    params: RealParamYear,
    january_month_index: int,
) -> NDArray[np.float64]:
    """Tax withheld at source from one month of employment or DB pension income.

    Annualises ``monthly_amount`` and runs it through this year's combined
    tax, then divides by twelve. Employment income's implied CPP and EI
    contributions (see the two private helpers above) earn their credits;
    non-employment income earns neither. The estimate never carries the
    pension income amount, which over-withholds slightly on a DB pension —
    cash timing only, reconciled at the December close (L16).

    Args:
        monthly_amount: This month's gross amount, real dollars, ``(n_paths,)``
            or scalar.
        age_at_end_of_year: Age in whole years on 31 December — the only use
            this has is the age amount.
        is_employment: Whether this is employment income, not per path: a
            property of the income source.
        province: Two-letter province code of residence.
        params: Every parameter file for the tax year, in real dollars.
        january_month_index: Month index of January of the tax year.

    Returns:
        Amount withheld this month, real dollars.
    """
    annual = MONTHS_PER_YEAR * np.asarray(monthly_amount, dtype=np.float64)
    fed = params.federal
    prov = params.province(province)

    if is_employment:
        cpp_contributions = _implied_cpp_base_contributions(annual, params.cpp, january_month_index)
        ei_premiums = _implied_ei_premiums(annual, fed, january_month_index)
    else:
        cpp_contributions = np.zeros_like(annual)
        ei_premiums = np.zeros_like(annual)

    zero_dividends = np.zeros_like(annual)
    fed_credits = federal.non_refundable_credits(
        annual,
        age_at_end_of_year,
        0.0,
        cpp_contributions,
        ei_premiums,
        zero_dividends,
        fed,
        january_month_index,
    )
    prov_credits = provincial.non_refundable_credits(
        annual,
        age_at_end_of_year,
        0.0,
        cpp_contributions,
        ei_premiums,
        zero_dividends,
        prov,
        fed,
        january_month_index,
    )
    fed_tax = federal.net_tax(federal.gross_tax(annual, fed, january_month_index), fed_credits)
    prov_tax = provincial.net_tax(
        provincial.gross_tax(annual, prov, january_month_index), prov_credits
    )
    return np.asarray((fed_tax + prov_tax) / MONTHS_PER_YEAR, dtype=np.float64)

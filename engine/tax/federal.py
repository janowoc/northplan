"""Federal income tax.

Annual, and called once per simulated year from the year-end close in
``engine/core/step.py``. Every income argument below is a **full calendar
year's** figure, accumulated over twelve monthly steps in
``engine.core.state.TaxLedger.ytd_income``. Handing one of these a single
month's income yields a small number at a low marginal rate and no error.

Parameters come from ``params/{year}/federal.yaml``. Nothing numeric lives in
this file — including the filing month, which is a statutory rule and is read
from the same place.

That file holds income tax and nothing else. The registered account rules that
once shared it now live one program per file: ``rrif.yaml`` (RRSP and RRIF),
``tfsa.yaml``, ``resp.yaml``. What stays here is what the Income Tax Act sets
Canada-wide — brackets, credits, the treatment of investment income, and the
maximum CPP and EI contributions the contribution credit is capped at.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.params.loader import ParamSet


def taxable_income(
    gross_income: ArrayLike,
    deductions: ArrayLike,
) -> NDArray[np.float64]:
    """Taxable income: gross less deductions, floored at zero.

    Args:
        gross_income: All income sources for the full calendar year, summed,
            real dollars, ``(n_paths,)``.
        deductions: RRSP contributions made over the year and other
            above-the-line deductions.

    Returns:
        Taxable income for the year, non-negative.
    """
    raise NotImplementedError


def net_income(
    gross_income: ArrayLike,
    deductions: ArrayLike,
) -> NDArray[np.float64]:
    """Net income (line 23600), the base for income-tested benefits.

    Distinct from :func:`taxable_income`, which subtracts further amounts.
    OAS recovery tax and GIS are assessed on net income, so keeping the two
    apart matters — conflating them understates the clawback.

    The result outlives the year that produced it. It is stored in
    ``TaxLedger.prior_year_net_income`` and read back one or two years later,
    when the benefit period it governs comes around; see
    ``engine.core.timeline.benefit_year_income_year``.

    Args:
        gross_income: All income sources for the full calendar year, summed,
            real dollars, ``(n_paths,)``.
        deductions: Deductions allowed in arriving at net income.

    Returns:
        Net income for the year, non-negative.
    """
    raise NotImplementedError


def gross_tax(income: ArrayLike, params: ParamSet) -> NDArray[np.float64]:
    """Federal tax before credits.

    Args:
        income: Taxable income, real dollars, ``(n_paths,)`` or scalar.
        params: The ``federal`` parameter set for the tax year.

    Returns:
        Federal tax before non-refundable credits.
    """
    raise NotImplementedError


def non_refundable_credits(
    income: ArrayLike,
    age: ArrayLike,
    pension_income: ArrayLike,
    cpp_ei_contributions: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Value of federal non-refundable credits.

    Credits reduce tax, not income, and are valued at ``credits.valuation_rate``
    from ``params``. That rate is read as its own value rather than as
    ``brackets.rates[0]``: the two coincide today, but they are distinct legal
    rules and either can change without the other.

    Several credits are themselves income-tested — the age amount is clawed
    back — so this takes income rather than being a constant.

    Args:
        income: Net income, real dollars, ``(n_paths,)``.
        age: Age in whole years at the end of the tax year, ``(n_paths,)``,
            from ``engine.core.timeline.age_at_end_of_year``. Not age in the
            month the assessment runs, which is the same thing only for a
            December birthday.
        pension_income: Eligible pension income received over the year, for the
            pension income amount.
        cpp_ei_contributions: CPP and EI contributions actually withheld over
            the year, ``(n_paths,)``, accumulated across the twelve monthly
            steps. Unlike the other credits this one is not a fixed amount: it
            is what the person paid, capped at the statutory maxima in
            ``params`` under ``contribution_credit``. Contributions stop
            partway through the year once a ceiling is reached, so a year's
            figure is not twelve times a month's and the cap belongs here
            rather than in the monthly accrual.
        params: The ``federal`` parameter set for the tax year.

    Returns:
        Total credit value in dollars of tax reduced.
    """
    raise NotImplementedError


def net_tax(
    taxable: ArrayLike,
    credits: ArrayLike,
) -> NDArray[np.float64]:
    """Federal tax after non-refundable credits, floored at zero.

    Args:
        taxable: Output of :func:`gross_tax`.
        credits: Output of :func:`non_refundable_credits`.

    Returns:
        Federal tax payable, non-negative.
    """
    raise NotImplementedError

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Federal income tax.

Annual, and called once per simulated year from the year-end close in
``engine/core/step.py``. Every income argument below is a **full calendar
year's** figure, accumulated over twelve monthly steps in an
``engine.core.state.IncomeLedger``. Handing one of these a single
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
    """Net income (line 23400), the base for income-tested amounts.

    Line 23400 is net income *before* adjustments; line 23600 subtracts the social
    benefits repayment from it. This engine computes only the first and tests
    everything against it (L49) — distinct from :func:`taxable_income`, which
    subtracts further amounts; conflating the two understates the OAS repayment,
    which is assessed on net income.

    The result outlives the year: stored in ``PersonState.prior_year_net_income``,
    read the following year by the RESP enhanced-grant rate
    (``grant.enhanced.income_year_offset``). The OAS repayment reads only the
    current year's figure, never this stored one.

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

    Credits reduce tax, not income, valued at ``credits.valuation_rate`` from
    ``params`` — read as its own value rather than ``brackets.rates[0]``, since the
    two are distinct legal rules that can diverge. Several credits are themselves
    income-tested (the age amount is clawed back), so this takes income rather
    than being a constant.

    Args:
        income: Net income, real dollars, ``(n_paths,)``.
        age: Age in whole years at the end of the tax year, ``(n_paths,)``, from
            ``engine.core.timeline.age_at_end_of_year`` — not age at assessment.
        pension_income: Eligible pension income received over the year, for the
            pension income amount.
        cpp_ei_contributions: CPP and EI contributions actually withheld over the
            year, ``(n_paths,)``. Not a fixed amount: capped at the statutory
            maxima under ``contribution_credit``, since contributions stop once a
            ceiling is reached and a year's figure is not twelve times a month's.
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

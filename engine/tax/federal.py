"""Federal income tax.

Parameters come from ``params/{year}/federal.yaml``. Nothing numeric lives in
this file.
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
        gross_income: All income sources summed, real dollars, ``(n_paths,)``.
        deductions: RRSP contributions and other above-the-line deductions.

    Returns:
        Taxable income, non-negative.
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

    Args:
        gross_income: All income sources summed, real dollars, ``(n_paths,)``.
        deductions: Deductions allowed in arriving at net income.

    Returns:
        Net income, non-negative.
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
    params: ParamSet,
) -> NDArray[np.float64]:
    """Value of federal non-refundable credits.

    Credits reduce tax, not income, and are valued at the lowest bracket rate.
    Several are themselves income-tested (the age amount is clawed back), so
    this takes income rather than being a constant.

    Args:
        income: Net income, real dollars, ``(n_paths,)``.
        age: Age in whole years at the end of the tax year, ``(n_paths,)``.
        pension_income: Eligible pension income for the pension income amount.
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

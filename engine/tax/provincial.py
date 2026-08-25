"""Provincial income tax.

Alberta first; the module is written to take a province's parameter set rather
than hard-coding one, so adding a province is a YAML file plus a verification
row, not new code. Parameters come from ``params/{year}/{province}.yaml``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.params.loader import ParamSet


def gross_tax(income: ArrayLike, params: ParamSet) -> NDArray[np.float64]:
    """Provincial tax before credits.

    Args:
        income: Taxable income, real dollars, ``(n_paths,)`` or scalar.
        params: The province's parameter set for the tax year.

    Returns:
        Provincial tax before non-refundable credits.
    """
    raise NotImplementedError


def non_refundable_credits(
    income: ArrayLike,
    age: ArrayLike,
    pension_income: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Value of provincial non-refundable credits.

    Provincial credit amounts and the rate they are valued at differ from the
    federal ones and are not derivable from them.

    Args:
        income: Net income, real dollars, ``(n_paths,)``.
        age: Age in whole years at the end of the tax year.
        pension_income: Eligible pension income.
        params: The province's parameter set for the tax year.

    Returns:
        Total credit value in dollars of tax reduced.
    """
    raise NotImplementedError


def net_tax(taxable: ArrayLike, credits: ArrayLike) -> NDArray[np.float64]:
    """Provincial tax after non-refundable credits, floored at zero."""
    raise NotImplementedError

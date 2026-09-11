# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Provincial income tax.

Alberta first; the module is written to take a province's parameter set rather
than hard-coding one, so adding a province is a YAML file plus a verification
row, not new code. Parameters come from ``params/{year}/{province}.yaml``.

Annual, like ``federal``, and called once per simulated year from the year-end
close. Every income argument is a full calendar year's figure.

The province's parameter file also carries the LIF maximum withdrawal rules
that ``engine/accounts/lira.py`` reads, since those are provincially set. Those
are annual limits fixed each January, not monthly ones.

They are in the same file but reached by a different route, and the difference
matters. Income tax follows the province of **residence**, so this module is
handed ``ParamYear.province(household.province)``. A LIF follows the province
its originating pension was **registered** in, so ``lira`` is handed
``ParamYear.jurisdiction(LifState.jurisdiction)``. For a household
that never moved these resolve to one object; for one that did, taking the LIF
table from this module's parameter set is wrong and nothing downstream will say
so.
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
    cpp_ei_contributions: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Value of provincial non-refundable credits.

    Provincial credit amounts and the rate they are valued at differ from the
    federal ones and are not derivable from them.

    Args:
        income: Net income, real dollars, ``(n_paths,)``.
        age: Age in whole years at the end of the tax year, from
            ``engine.core.timeline.age_at_end_of_year``.
        pension_income: Eligible pension income received over the year.
        cpp_ei_contributions: CPP and EI contributions actually withheld over
            the year, ``(n_paths,)``. The same figure this year's federal
            credit is computed from, valued here at the provincial rate. The
            statutory maxima it is capped at are Canada-wide and live in
            ``federal.yaml``, not in this file — the province sets the rate the
            credit is worth, not the ceiling on the contribution.
        params: The province's parameter set for the tax year.

    Returns:
        Total credit value in dollars of tax reduced.
    """
    raise NotImplementedError


def net_tax(taxable: ArrayLike, credits: ArrayLike) -> NDArray[np.float64]:
    """Provincial tax after non-refundable credits, floored at zero."""
    raise NotImplementedError

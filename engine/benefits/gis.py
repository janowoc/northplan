"""Guaranteed Income Supplement.

Income-tested, non-taxable, and reduced against a different income base than
the OAS recovery tax uses — notably OAS itself is excluded from that base.
Modelled only for households whose projected income makes it reachable.

GIS follows OAS's benefit period and quarterly adjustment schedule, so
everything said in ``engine/benefits/oas.py`` about the income year changing
partway through a calendar year, and about indexation costing a constant in
real terms, applies here unchanged.

One difference worth stating: GIS entitlement is recalculated at each benefit
year, and a large one-off registered withdrawal in one calendar year cuts the
supplement in the benefit period a year later, not immediately. On a monthly
timeline that delayed consequence is visible and is exactly the kind of thing
the decumulation optimizer should be able to see and avoid.

Parameters from ``params/{year}/oas.yaml``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.params.loader import ParamSet


def countable_income(
    income_year_net_income: ArrayLike,
    oas_received: ArrayLike,
    employment_income: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Income base against which GIS is reduced.

    Not the same as net income: OAS is excluded, and part of employment income
    is exempt. All three arguments are annual figures for the income year that
    governs the current benefit period, not month figures — the reduction is
    computed once per benefit period and then applied to each monthly payment.

    Args:
        income_year_net_income: Net income for the governing income year, from
            ``engine.core.timeline.benefit_year_income_year``, ``(n_paths,)``.
        oas_received: OAS included in that net income, to be excluded here.
        employment_income: Employment and self-employment income in that year,
            for the earnings exemption.
        params: The ``oas`` parameter set.

    Returns:
        Countable income for the benefit period, non-negative.
    """
    raise NotImplementedError


def supplement_monthly(
    countable: ArrayLike,
    has_spouse: ArrayLike,
    spouse_receives_oas: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Monthly GIS payable this month.

    The maximum and the reduction rate both depend on marital status and on
    whether a spouse receives OAS, which is why those flags are arguments
    rather than being folded into a single rate. Both flags are read for *this
    month*: a spouse's death changes them from the month it happens, and the
    surviving-spouse maximum applies from then, not from the next benefit year.

    Args:
        countable: Output of :func:`countable_income`, an annual figure for the
            governing income year.
        has_spouse: Whether the recipient has a spouse or common-law partner
            this month.
        spouse_receives_oas: Whether that spouse receives OAS this month.
        params: The ``oas`` parameter set.

    Returns:
        Monthly GIS in real dollars, non-negative and non-taxable, at the
        published amount before the constant indexation factor.
    """
    raise NotImplementedError

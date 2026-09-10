# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Old Age Security, including the recovery tax (clawback).

Parameters from ``params/{year}/oas.yaml``.

The rule that gets implemented wrong most often, stated up front: **the OAS
recovery tax for a benefit period beginning in a statutory month of year Y is
assessed on net income from calendar year Y-1.** Assessing it against the
current year's income overstates the clawback for someone whose income is
rising and understates it for someone drawing down. Every function here takes
the income year explicitly for that reason, and the benefit-year rule in
``params/<year>/oas.yaml`` — ``benefit_year.start_month`` and
``benefit_year.income_year_offset`` — applied by this module is what works
out which year that is for a given month.

The monthly timestep makes the benefit period visible rather than notional. The
recovery tax is withheld from each monthly payment across the period, and the
period straddles two calendar years — so the withholding applied in the months
before the benefit-year changeover is set by a different income year than the
withholding applied after it, within one calendar year. On an annual step that
distinction collapsed; here it does not, and a benefit period that changes
mid-calendar-year is the normal case, not an edge one.

OAS is adjusted quarterly, so in real dollars OAS averages slightly less than
its published amount — by a constant, every month.
``engine.core.indexation.erosion_factor`` supplies that constant and the step
applies it. The amounts here are the published ones, before it. The
quarter-to-quarter oscillation around that constant is not modelled, and
neither is the CPI lag; see ``engine/core/indexation.py`` for why.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.params.loader import ParamSet


def deferral_factor(start_age_months: ArrayLike, params: ParamSet) -> NDArray[np.float64]:
    """Multiplier for deferring OAS past the earliest eligible age.

    OAS may be deferred but not taken early; the multiplier is 1.0 at the
    earliest eligible age and rises per month of deferral up to the statutory
    maximum. Per month — which the monthly timeline now supplies directly.

    Args:
        start_age_months: Age in months at which OAS starts, ``(n_paths,)``.
        params: The ``oas`` parameter set.

    Returns:
        Multiplier, at least 1.0.
    """
    raise NotImplementedError


def gross_pension_monthly(
    current_age_months: ArrayLike,
    start_age_months: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Monthly OAS before the recovery tax, for this month.

    The maximum monthly pension steps up at an age threshold, so this takes
    current age as well as start age. Both are in months: the step-up happens
    in the month the threshold is reached, not in the following January.

    Args:
        current_age_months: Age in months this month, from
            ``engine.core.timeline.age_in_months``.
        start_age_months: Age in months at which OAS started.
        params: The ``oas`` parameter set.

    Returns:
        Monthly gross OAS in real dollars, zero before the start month, at the
        published amount before the constant indexation factor.
    """
    raise NotImplementedError


def recovery_tax_monthly(
    income_year_net_income: ArrayLike,
    gross_oas_monthly: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Recovery tax withheld from this month's payment, capped at the payment.

    The threshold and rate are annual quantities; the withholding is monthly.
    The conversion happens here, once, and the cap is applied to the *monthly*
    gross so that a month's net OAS can never be negative.

    Args:
        income_year_net_income: Net income from the calendar year that governs
            the benefit period this month falls in — from the benefit-year
            rule in ``params/<year>/oas.yaml`` — ``benefit_year.start_month``
            and ``benefit_year.income_year_offset`` — applied by this module,
            and stored in ``PersonState.prior_year_net_income``. Passing the
            current year's income here is a correctness bug, not an
            approximation. So is passing the income year that governed the
            *previous* benefit period, which is what happens if the mid-year
            changeover is missed.
        gross_oas_monthly: Output of :func:`gross_pension_monthly`; the
            recovery tax cannot exceed it.
        params: The ``oas`` parameter set, supplying the annual threshold and
            rate.

    Returns:
        Recovery tax for this month, in ``[0, gross_oas_monthly]``.
    """
    raise NotImplementedError


def net_pension_monthly(
    gross_oas_monthly: ArrayLike,
    recovery_monthly: ArrayLike,
) -> NDArray[np.float64]:
    """OAS actually received this month: gross less recovery tax, floored at zero."""
    raise NotImplementedError


def gross_pension_annual(
    current_age_months: ArrayLike,
    start_age_months: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Twelve months of gross OAS at the current rate, for golden tests only.

    Published OAS figures include annual maxima, so a golden test needs an
    annual number. The simulation does not call this: it accrues twelve monthly
    payments, which may span two different recovery-tax income years and may
    start or stop partway through the year, and those do not sum to twelve
    times any single month. The indexation factor is not one of the reasons —
    it is constant across the year.
    """
    raise NotImplementedError

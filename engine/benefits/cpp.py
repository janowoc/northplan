# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Canada Pension Plan retirement and survivor pensions.

Parameters from ``params/{year}/cpp.yaml``: the standard/earliest/latest start
ages, the per-month early and late start-adjustment rates, the maximum
pension at the standard age, and the survivor formula. Contribution ceilings
and rates live here too but are read by ``engine/benefits/employment.py``, not
by this module.

Every dollar amount arrives through a :class:`~engine.core.indexation.RealParamSet`,
which already applies the erosion factor for CPP's annual adjustment cycle; a
function here never multiplies by it again. An amount already in pay when the
scenario starts carries no erosion factor at all — it is the January
start-year figure the scenario states, unchanged every month (L52). A start
age outside the statutory window, or already past when the run opens, is
clipped and started at the later of the election and the age at open, with no
back payment (L53).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.core.indexation import RealParamSet
from engine.core.state import BenefitState


def start_adjustment_factor(
    start_age_months: ArrayLike, params: RealParamSet
) -> NDArray[np.float64]:
    """Multiplier applied to the base pension for starting other than at the standard age.

    Early and late adjustments use different per-month rates; the adjustment
    is per *month* away from the standard age, not per year, so a start age
    must not be rounded to whole years before it reaches here.

    Args:
        start_age_months: Age in months at which the pension starts,
            ``(n_paths,)`` or scalar. Clipped to the statutory earliest and
            latest start ages from ``params`` before the rate is applied.
        params: The ``cpp`` parameter set.

    Returns:
        Multiplier, 1.0 at exactly the standard start age, ``float64`` array.
    """
    standard = params.number("start_age.standard_months")
    earliest = params.number("start_age.earliest_months")
    latest = params.number("start_age.latest_months")
    early_rate = params.number("start_adjustment.early_rate_per_month")
    late_rate = params.number("start_adjustment.late_rate_per_month")

    start = np.clip(np.asarray(start_age_months, dtype=np.float64), earliest, latest)
    return np.asarray(
        np.where(
            start < standard,
            1 - early_rate * (standard - start),
            1 + late_rate * (start - standard),
        ),
        dtype=np.float64,
    )


def base_pension_monthly(
    contributory_history: ArrayLike,
    month_index: int,
    params: RealParamSet,
) -> NDArray[np.float64]:
    """Monthly CPP at the standard start age given a contributory earnings history.

    Args:
        contributory_history: Fraction of the maximum the person earned toward,
            in ``[0, 1]``, ``(n_paths,)`` or scalar. A person who always earned
            at or above the YMPE is 1.0.
        month_index: Months since January of the scenario's start year.
        params: The ``cpp`` parameter set.

    Returns:
        Monthly pension in real dollars, at the standard-age rate before any
        start adjustment.
    """
    maximum = params.amount("pension.maximum_at_standard_age_monthly", month_index)
    return np.asarray(
        maximum * np.asarray(contributory_history, dtype=np.float64), dtype=np.float64
    )


def pension_monthly(
    benefit: BenefitState,
    current_age_months: int,
    month_index: int,
    params: RealParamSet,
) -> NDArray[np.float64]:
    """Monthly CPP payable this month.

    Two mutually exclusive sources, mirroring :class:`BenefitState`: a
    pension already in pay is returned unchanged from month 0 (L18, L52), or
    an elected start age and contributory history are turned into a monthly
    amount, scaled by :func:`start_adjustment_factor`.

    Args:
        benefit: This person's CPP standing.
        current_age_months: This person's age in months this month. The age
            at the run's opening is derived from it as ``current_age_months -
            month_index``.
        month_index: Months since January of the scenario's start year.
        params: The ``cpp`` parameter set.

    Returns:
        Monthly pension in real dollars, shape of ``benefit.monthly_amount``,
        zero before the effective start month.

    Raises:
        ValueError: If neither ``in_pay_monthly`` nor both
            ``start_age_months`` and ``contributory_history`` are set.
    """
    if benefit.in_pay_monthly is not None:
        return np.array(benefit.in_pay_monthly, dtype=np.float64)

    missing = [
        name
        for name, value in (
            ("start_age_months", benefit.start_age_months),
            ("contributory_history", benefit.contributory_history),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"benefit.in_pay_monthly is not set, so this pension has not "
            f"started yet, but {', '.join(missing)} is also None. Exactly "
            f"one way of knowing this benefit's amount must be supplied."
        )

    earliest = params.number("start_age.earliest_months")
    latest = params.number("start_age.latest_months")
    age_at_open = current_age_months - month_index
    effective_start = np.clip(max(benefit.start_age_months, age_at_open), earliest, latest)
    base = base_pension_monthly(benefit.contributory_history, month_index, params)
    factor = start_adjustment_factor(effective_start, params)
    started = current_age_months >= effective_start
    result = np.where(started, base * factor, 0.0)
    return np.broadcast_to(result, benefit.monthly_amount.shape).astype(np.float64)


def survivor_pension_monthly(
    deceased_base_monthly: ArrayLike,
    survivor_own_monthly: ArrayLike,
    month_index: int,
    params: RealParamSet,
) -> NDArray[np.float64]:
    """Monthly CPP survivor increment payable to a surviving spouse.

    The 65-and-over formula is applied at every age (L19): a fixed share of
    the deceased's base pension, with the survivor's combined total capped.
    For a deceased already in pay, ``deceased_base_monthly`` is the amount in
    pay; the step supplies whichever applies.

    Args:
        deceased_base_monthly: The deceased's base CPP pension, real dollars.
        survivor_own_monthly: The survivor's own CPP pension, real dollars.
        month_index: Months since January of the scenario's start year.
        params: The ``cpp`` parameter set.

    Returns:
        The survivor increment, real dollars, floored at zero.
    """
    share = params.number("survivor.share_at_65_plus")
    combined_max = params.amount("survivor.combined_maximum_monthly", month_index)
    deceased = np.asarray(deceased_base_monthly, dtype=np.float64)
    own = np.asarray(survivor_own_monthly, dtype=np.float64)
    increment = np.minimum(share * deceased, combined_max - own)
    return np.asarray(np.clip(increment, 0, None), dtype=np.float64)


def pension_annual(
    contributory_history: ArrayLike,
    start_age_months: ArrayLike,
    month_index: int,
    params: RealParamSet,
) -> NDArray[np.float64]:
    """Twelve months of CPP at the current rate, for golden tests only.

    Published CPP figures are annual maxima, so a golden test needs an annual
    number to compare against. The simulation does not call this: it accrues
    twelve monthly payments, and a payment may start or stop partway through a
    year.

    Args:
        contributory_history: Fraction of the maximum, in ``[0, 1]``.
        start_age_months: Age in months at which the pension starts.
        month_index: Months since January of the scenario's start year.
        params: The ``cpp`` parameter set.

    Returns:
        Twelve times the adjusted monthly pension, ``(n_paths,)``.
    """
    base = base_pension_monthly(contributory_history, month_index, params)
    factor = start_adjustment_factor(start_age_months, params)
    return np.asarray(12 * base * factor, dtype=np.float64)

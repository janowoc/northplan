"""Canada Pension Plan retirement benefit.

Parameters from ``params/{year}/cpp.yaml``: YMPE, contribution rates, the
maximum retirement pension, the per-month early and late start adjustment
factors, and the indexation schedule. None of these are ever inlined.

CPP is paid monthly and adjusted once a year. Between adjustments it is fixed
in nominal terms, and the adjustment lags the inflation it is compensating for,
so in real dollars CPP is worth a little less than its published amount — by a
constant, every month, for the life of the plan. The step applies that constant
via ``engine.core.indexation.real_factor``. The amounts here are the published
ones, before it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.params.loader import ParamSet


def start_adjustment_factor(start_age_months: ArrayLike, params: ParamSet) -> NDArray[np.float64]:
    """Multiplier applied to the base pension for starting other than at 65.

    Early and late adjustments use different per-month rates, and the
    adjustment is per *month* away from 65, not per year. On a monthly
    timeline that precision is now available directly from
    ``engine.core.timeline.age_in_months`` and there is no reason to round a
    start age to whole years before it reaches here.

    Args:
        start_age_months: Age in months at which the pension starts,
            ``(n_paths,)`` or scalar. Bounded by the statutory earliest and
            latest start ages from ``params``.
        params: The ``cpp`` parameter set.

    Returns:
        Multiplier, 1.0 at exactly the standard start age. That age, in
        months, comes from ``params`` like everything else; do not write
        the number of months into the implementation.
    """
    raise NotImplementedError


def base_pension_monthly(
    contributory_history: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Monthly CPP at age 65 given a contributory earnings history.

    Args:
        contributory_history: Fraction of the maximum the person earned toward,
            in ``[0, 1]``, ``(n_paths,)`` or scalar. A person who always earned
            at or above the YMPE is 1.0.
        params: The ``cpp`` parameter set.

    Returns:
        Monthly pension in real dollars, at the age-65 rate before any start
        adjustment, as at the last January adjustment.
    """
    raise NotImplementedError


def pension_monthly(
    contributory_history: ArrayLike,
    start_age_months: ArrayLike,
    current_age_months: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Monthly CPP payable this month.

    The base pension scaled by the start adjustment, or zero if the pension has
    not started yet. Unlike the annual version this replaced, the comparison is
    made here rather than by the caller, because the caller now has the current
    age in months and the answer is unambiguous: the pension is payable from
    the month the person reaches ``start_age_months``, not from the January
    after it.

    Args:
        contributory_history: Fraction of the maximum, in ``[0, 1]``.
        start_age_months: Age in months at which the pension starts.
        current_age_months: Age in months this month, from
            ``engine.core.timeline.age_in_months``.
        params: The ``cpp`` parameter set.

    Returns:
        Monthly pension in real dollars, ``(n_paths,)``, zero before the start
        month.
    """
    raise NotImplementedError


def pension_annual(
    contributory_history: ArrayLike,
    start_age_months: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Twelve months of CPP at the current rate, for golden tests only.

    Published CPP figures are annual maxima, so a golden test needs an annual
    number to compare against. The simulation does not call this: it accrues
    twelve monthly payments, and a payment may start or stop partway through a
    year. The indexation factor is constant across the year and so is not a
    reason these differ — the start month is.

    Args:
        contributory_history: Fraction of the maximum, in ``[0, 1]``.
        start_age_months: Age in months at which the pension starts.
        params: The ``cpp`` parameter set.

    Returns:
        Twelve times the adjusted monthly pension, ``(n_paths,)``.
    """
    raise NotImplementedError

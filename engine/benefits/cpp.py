"""Canada Pension Plan retirement benefit.

Parameters from ``params/{year}/cpp.yaml``: YMPE, contribution rates, the
maximum retirement pension, and the per-month early and late start adjustment
factors. None of these are ever inlined.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.params.loader import ParamSet


def start_adjustment_factor(start_age_months: ArrayLike, params: ParamSet) -> NDArray[np.float64]:
    """Multiplier applied to the base pension for starting other than at 65.

    Early and late adjustments use different per-month rates, and the
    adjustment is per *month* away from 65, not per year — a start age given in
    whole years must be converted before it reaches here.

    Args:
        start_age_months: Age in months at which the pension starts,
            ``(n_paths,)`` or scalar. Bounded by the statutory earliest and
            latest start ages from ``params``.
        params: The ``cpp`` parameter set.

    Returns:
        Multiplier, 1.0 at exactly age 65.
    """
    raise NotImplementedError


def base_pension_annual(
    contributory_history: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Annual CPP at age 65 given a contributory earnings history.

    Args:
        contributory_history: Fraction of the maximum the person earned toward,
            in ``[0, 1]``, ``(n_paths,)`` or scalar. A person who always earned
            at or above the YMPE is 1.0.
        params: The ``cpp`` parameter set.

    Returns:
        Annual pension in real dollars, at the age-65 rate before any start
        adjustment.
    """
    raise NotImplementedError


def pension_annual(
    contributory_history: ArrayLike,
    start_age_months: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Annual CPP payable given contributory history and start age.

    The base pension scaled by the start adjustment. Returns zero for years
    before the pension starts; the caller is responsible for supplying the
    current year and the start year, since this function cannot see time.

    Args:
        contributory_history: Fraction of the maximum, in ``[0, 1]``.
        start_age_months: Age in months at which the pension starts.
        params: The ``cpp`` parameter set.

    Returns:
        Annual pension in real dollars, ``(n_paths,)``.
    """
    raise NotImplementedError

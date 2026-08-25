"""Old Age Security, including the recovery tax (clawback).

Parameters from ``params/{year}/oas.yaml``.

The rule that gets implemented wrong most often, stated up front: **the OAS
recovery tax for the benefit period from July of year Y to June of year Y+1 is
assessed on net income from calendar year Y-1.** Assessing it against the
current year's income overstates the clawback for someone whose income is
rising and understates it for someone drawing down. Every function here takes
the income year explicitly for that reason.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.params.loader import ParamSet


def deferral_factor(start_age_months: ArrayLike, params: ParamSet) -> NDArray[np.float64]:
    """Multiplier for deferring OAS past the earliest eligible age.

    OAS may be deferred but not taken early; the multiplier is 1.0 at the
    earliest eligible age and rises per month of deferral up to the statutory
    maximum.

    Args:
        start_age_months: Age in months at which OAS starts, ``(n_paths,)``.
        params: The ``oas`` parameter set.

    Returns:
        Multiplier, at least 1.0.
    """
    raise NotImplementedError


def gross_pension_annual(
    age: ArrayLike,
    start_age_months: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Annual OAS before the recovery tax.

    The maximum monthly pension steps up at an age threshold, so this takes
    current age as well as start age. Monthly parameters are converted to
    annual here, once.

    Args:
        age: Current age in whole years, ``(n_paths,)``.
        start_age_months: Age in months at which OAS started.
        params: The ``oas`` parameter set.

    Returns:
        Annual gross OAS in real dollars, zero before the pension starts.
    """
    raise NotImplementedError


def recovery_tax(
    prior_year_net_income: ArrayLike,
    gross_oas: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """OAS recovery tax, capped at the gross pension.

    Args:
        prior_year_net_income: Net income from the calendar year *before* the
            one the benefit period starts in. Passing the current year's income
            here is a correctness bug, not an approximation.
        gross_oas: Output of :func:`gross_pension_annual`; the recovery tax
            cannot exceed it.
        params: The ``oas`` parameter set, supplying the threshold and rate.

    Returns:
        Recovery tax in real dollars, in ``[0, gross_oas]``.
    """
    raise NotImplementedError


def net_pension_annual(
    gross_oas: ArrayLike,
    recovery: ArrayLike,
) -> NDArray[np.float64]:
    """OAS actually received: gross less recovery tax, floored at zero."""
    raise NotImplementedError

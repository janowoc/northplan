# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Old Age Security, including the repayment (clawback).

Parameters from ``params/{year}/oas.yaml``.

The rule that gets implemented wrong most often, stated up front: **the OAS
repayment is a line on the current year's return.** It is assessed once, at the
December close, on the current calendar year's net income — which includes the
OAS received in that same year — and it is capped at that OAS. It is settled
with the balance owing in the following year's filing month. OAS itself is paid
gross every month; nothing is deducted from a payment.

In reality a recovery amount is withheld from the July-to-June payments that
follow an assessment. That withholding is a refundable prepayment of the
repayment rather than a separate charge, and it is not modelled (L16, L23).

The ``params/<year>/oas.yaml`` block describing that July-to-June period is
read by no module here; it is there for a future GIS implementation. See
``engine/benefits/__init__.py``, which names the two parameters.

OAS is adjusted quarterly; see ``engine/benefits/__init__.py`` for the
indexation convention and its erosion factor.
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


def gross_pension_monthly(
    current_age_months: ArrayLike,
    start_age_months: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Monthly OAS for this month, gross.

    The maximum monthly pension steps up at an age threshold, so this takes
    current age as well as start age. Both are in months: the step-up happens
    in the month the threshold is reached, not in the following January.

    Args:
        current_age_months: Age in months this month, from
            ``engine.core.timeline.age_in_months``.
        start_age_months: Age in months at which OAS started.
        params: The ``oas`` parameter set.

    Returns:
        Monthly OAS in real dollars, zero before the start month, at the
        published amount before the constant indexation factor. Nothing is
        deducted: the repayment is assessed on the year at the December close.
    """
    raise NotImplementedError


def recovery_tax_monthly(
    net_income: ArrayLike,
    gross_oas_monthly: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Obsolete; issue 16 deletes this.

    The repayment is assessed once, at the December close, not withheld from
    each monthly payment.
    """
    raise NotImplementedError


def net_pension_monthly(
    gross_oas_monthly: ArrayLike,
    recovery_monthly: ArrayLike,
) -> NDArray[np.float64]:
    """Obsolete; issue 16 deletes this.

    OAS is paid gross: this month's payment has nothing deducted from it.
    """
    raise NotImplementedError


def gross_pension_annual(
    current_age_months: ArrayLike,
    start_age_months: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Twelve months of gross OAS at the current rate, for golden tests only.

    Published OAS figures include annual maxima, so a golden test needs an
    annual number. The simulation does not call this: it accrues twelve
    monthly payments, which may start or stop partway through the year, and
    those do not sum to twelve times any single month. The indexation factor
    is not one of the reasons — it is constant across the year.
    """
    raise NotImplementedError

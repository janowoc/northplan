# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Old Age Security: the pension. The repayment lives in ``engine/tax/combined.py``.

Parameters from ``params/{year}/oas.yaml``.

The rule that gets implemented wrong most often, stated up front because this
is the module a reader comes to looking for it: **the OAS repayment is a line
on the current year's return.** It is assessed once, at the December close, on
the current calendar year's net income — which includes the OAS received in
that same year — and it is capped at that OAS. It is settled with the balance
owing in the following year's filing month. OAS itself is paid gross every
month; nothing is deducted from a payment. It is computed by
``engine.tax.combined.oas_repayment``, with the household assessment it is a
line of, and not by anything here.

In reality a recovery amount is withheld from the July-to-June payments that
follow an assessment. That withholding is a refundable prepayment of the
repayment rather than a separate charge, and it is not modelled (L16, L23).

The ``params/<year>/oas.yaml`` block describing that July-to-June period is
read by no module here; it is there for a future GIS implementation. See
``engine/benefits/__init__.py``, which names the two parameters.

**What this module owns.** The maximum monthly pension steps up by current
age at a threshold (:func:`gross_pension_monthly`'s band lookup); an amount
already in pay carries that step-up as a ratio, exact because it is the same
step-up that produced the published maxima, and no erosion factor, since the
factor cancels in the ratio (L52); an amount not yet in pay starts at the
later of the elected age and the age when the run opens, clipped to the
statutory window; the election itself was checked at load against the
window and the person's age in whole years
(``engine/scenario/start_ages.py``, L54). Amounts come through the real view
(:class:`~engine.core.indexation.RealParamSet`), which already applies OAS's
quarterly erosion factor (L5, L24, L52).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.core.indexation import RealParamSet
from engine.core.state import BenefitState
from engine.params.loader import MalformedParamFileError


def deferral_factor(start_age_months: ArrayLike, params: RealParamSet) -> NDArray[np.float64]:
    """Multiplier for deferring OAS past the earliest eligible age.

    OAS may be deferred but not taken early; the multiplier is 1.0 at the
    earliest eligible age and rises per month of deferral up to the statutory
    maximum, then stays flat.

    Args:
        start_age_months: Age in months at which OAS starts, ``(n_paths,)``
            or scalar.
        params: The ``oas`` parameter set.

    Returns:
        Multiplier, at least 1.0, ``float64`` array.
    """
    earliest = params.number("start_age.earliest_months")
    increment = params.number("deferral.increment_rate_per_month")
    maximum_months = params.number("deferral.maximum_months")
    deferred = np.clip(np.asarray(start_age_months, dtype=np.float64) - earliest, 0, maximum_months)
    return np.asarray(1 + increment * deferred, dtype=np.float64)


def _band_max(age_months: float, month_index: int, params: RealParamSet) -> float:
    """The maximum monthly pension for the band ``age_months`` falls in.

    The band for an age is the last band whose ``from_age_months`` does not
    exceed it; an age below the first band takes the first band. Reads the
    band's starting age from the raw (unrouted) table and its maximum through
    the routed, real-terms accessor.

    Args:
        age_months: Age in months, a single value (not per path).
        month_index: Months since January of the scenario's start year.
        params: The ``oas`` parameter set.

    Returns:
        The band maximum, real dollars.

    Raises:
        MalformedParamFileError: If a band is not a mapping, a band has no
            ``from_age_months``, the number of band starting ages does not
            match the number of band maxima, or the starting ages do not
            strictly ascend.
    """
    bands = params.sequence("pension.age_bands")
    from_ages = []
    for band_index, band in enumerate(bands):
        if not isinstance(band, Mapping):
            raise MalformedParamFileError(
                f"{params.raw.source}[pension.age_bands.{band_index}]: expected a "
                f"mapping, found {type(band).__name__} ({band!r})."
            )
        if "from_age_months" not in band:
            raise MalformedParamFileError(
                f"{params.raw.source}[pension.age_bands.{band_index}]: has no 'from_age_months'."
            )
        from_ages.append(band["from_age_months"])
    maxima = params.amounts("pension.age_bands.*.maximum_monthly", month_index)
    if len(from_ages) != len(maxima):
        raise MalformedParamFileError(
            f"{params.raw.source}[pension.age_bands]: {len(from_ages)} band starting "
            f"ages but {len(maxima)} maxima; each band needs exactly one of each. "
            f"from_age_months={from_ages!r}, maxima={maxima!r}."
        )
    if any(a >= b for a, b in itertools.pairwise(from_ages)):
        raise MalformedParamFileError(
            f"{params.raw.source}[pension.age_bands]: from_age_months must strictly "
            f"ascend, got {from_ages!r}."
        )
    index = 0
    for i, from_age in enumerate(from_ages):
        if from_age <= age_months:
            index = i
    return maxima[index]


def gross_pension_monthly(
    benefit: BenefitState,
    current_age_months: int,
    month_index: int,
    params: RealParamSet,
) -> NDArray[np.float64]:
    """Monthly OAS for this month, gross.

    Two mutually exclusive sources, mirroring :class:`BenefitState`. In pay:
    the amount in pay scaled by the ratio of this month's band maximum to the
    band maximum at the age the run opened, not eroded since the erosion
    factor cancels out of the ratio (L52). Elected: the band maximum at the
    effective start age (the later of the election and the age at open,
    clipped to the statutory window; the election was checked at load, L54)
    times :func:`deferral_factor`, from the effective start month.

    Args:
        benefit: This person's OAS standing.
        current_age_months: This person's age in months this month; the age
            at the run's opening is ``current_age_months - month_index``.
        month_index: Months since January of the scenario's start year.
        params: The ``oas`` parameter set.

    Returns:
        Monthly OAS in real dollars, shape of ``benefit.monthly_amount``,
        zero before the start month. Nothing is deducted: the repayment is
        assessed at the December close.

    Raises:
        ValueError: If neither ``in_pay_monthly`` nor ``start_age_months`` is set.
    """
    age_at_open = current_age_months - month_index

    if benefit.in_pay_monthly is not None:
        ratio = _band_max(current_age_months, month_index, params) / _band_max(
            age_at_open, month_index, params
        )
        return np.asarray(np.array(benefit.in_pay_monthly, dtype=np.float64) * ratio)

    if benefit.start_age_months is None:
        raise ValueError(
            "neither benefit.in_pay_monthly nor benefit.start_age_months is "
            "set; exactly one way of knowing this benefit's amount must be "
            "supplied."
        )

    earliest = params.number("start_age.earliest_months")
    latest = params.number("start_age.latest_months")
    effective_start = np.clip(max(benefit.start_age_months, age_at_open), earliest, latest)
    max_at_current = _band_max(current_age_months, month_index, params)
    factor = deferral_factor(effective_start, params)
    started = current_age_months >= effective_start
    result = np.where(started, max_at_current * factor, 0.0)
    return np.broadcast_to(result, benefit.monthly_amount.shape).astype(np.float64)


def gross_pension_annual(
    current_age_months: int,
    start_age_months: ArrayLike,
    month_index: int,
    params: RealParamSet,
) -> NDArray[np.float64]:
    """Twelve months of gross OAS at the current rate, for golden tests only.

    Published OAS figures include annual maxima, so a golden test needs an
    annual number. The simulation does not call this: it accrues twelve
    monthly payments, which may start or stop partway through the year, and
    those do not sum to twelve times any single month.

    Args:
        current_age_months: Age in months this month.
        start_age_months: Age in months at which OAS started, clipped to the
            statutory window before the started test, as
            :func:`gross_pension_monthly` clips its effective start.
        month_index: Months since January of the scenario's start year.
        params: The ``oas`` parameter set.

    Returns:
        Twelve times the monthly OAS, ``(n_paths,)``, zero before start.
    """
    earliest = params.number("start_age.earliest_months")
    latest = params.number("start_age.latest_months")
    start = np.clip(np.asarray(start_age_months, dtype=np.float64), earliest, latest)
    max_at_current = _band_max(current_age_months, month_index, params)
    factor = deferral_factor(start, params)
    started = current_age_months >= start
    return np.asarray(np.where(started, 12 * max_at_current * factor, 0.0), dtype=np.float64)

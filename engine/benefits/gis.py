# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guaranteed Income Supplement — **not modelled**; this computes an exposure indicator instead.

GIS is income-tested, non-taxable, and reduced against a different income
base than the OAS recovery tax uses. None of that is populated:
``params/{year}/oas.yaml`` carries ``gis.modelled: false`` and nothing else.

This module computes no supplement. It reports whether a household's testable
income — line 23600 less the OAS received, annual, current year — falls in the
band where an unmodelled GIS would begin to matter, so the caller can report
the fraction of path-years in band (L2). Treating GIS as zero everywhere would
silently understate income exactly where it matters most, and would hide the
marginal rate that makes early registered drawdown attractive to a low-income
household. An earlier version of this module stopped the whole run the moment
any single path reached the band; that meant the lower tail of an ordinary
Monte Carlo run — the part a retirement plan exists to be read for — refused
most modest households outright. Reporting the exposure instead keeps the
answer and makes the exposure visible rather than silencing it.

The band thresholds sit below the published GIS cut-offs on purpose: they are
a conservative band around where GIS would begin to matter, not a GIS
calculation, and must not be read as one.

The income is an approximation on the same terms. Of the five entries under
``gis.income_base`` in ``params/gis_not_implemented.yaml``, three are met: the
OAS pension is excluded because this subtracts it, GIS itself vacuously since
none is modelled, and couples are tested on combined income when the caller
sums the household, as the ``Args`` entry requires. Two are not: the
employment-income exemption, and the timing — GIS tests an earlier year over
the benefit year in ``params/{year}/oas.yaml`` under ``benefit_year``, where
this tests the current calendar year. L2 records the direction of both.
Line 23400 would serve equally: the two differ only by the OAS repayment,
which is zero everywhere the band is reachable.

When GIS is implemented: the schema is already written up in
``params/gis_not_implemented.yaml``; move it under ``params/{year}/``,
populate it, and flip ``modelled`` to true.

Parameters from ``params/{year}/oas.yaml`` under ``gis``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.core.indexation import RealParamSet

__all__ = ["band_threshold_annual", "in_band", "is_modelled"]


def is_modelled(params: RealParamSet) -> bool:
    """Whether GIS is modelled at all for this parameter year.

    Args:
        params: The ``oas`` parameter set.

    Returns:
        The value of ``gis.modelled``.

    Raises:
        MissingParameterError: If the flag is absent. A year whose file does not
            say either way is not "probably not modelled" — it is unknown, and
            the run stops. Same reasoning as ``lif.has_maximum``.
        TypeError: If the flag is present but not a boolean, which usually means
            a YAML ``no``/``off`` was written where a value was intended.
    """
    modelled = params.get("gis.modelled")
    if not isinstance(modelled, bool):
        raise TypeError(
            f"gis.modelled must be a boolean, got {type(modelled).__name__}: {modelled!r}"
        )
    return modelled


def band_threshold_annual(
    has_spouse: ArrayLike, january_month_index: int, params: RealParamSet
) -> NDArray[np.float64]:
    """Annual testable income threshold that bounds the reporting band.

    Two thresholds, selected per path by marital status, because the couple
    figure is measured against *combined* income and is not twice the single
    one.

    Args:
        has_spouse: Whether the recipient has a spouse or common-law partner,
            ``(n_paths,)`` or scalar. Read for the year being checked: a
            spouse's death moves a household onto the single threshold from
            that year.
        january_month_index: Month index of January of the year being checked.
        params: The ``oas`` parameter set.

    Returns:
        Threshold in annual real dollars, broadcast to ``has_spouse``'s shape.

    Raises:
        MissingParameterError: If either threshold is absent.
    """
    single = params.annual_amount(
        "gis.band_thresholds.single_testable_income_annual", january_month_index
    )
    couple = params.annual_amount(
        "gis.band_thresholds.couple_combined_testable_income_annual", january_month_index
    )
    return np.asarray(
        np.where(np.asarray(has_spouse, dtype=bool), couple, single), dtype=np.float64
    )


def in_band(
    net_income_after_repayment: ArrayLike,
    oas_received: ArrayLike,
    has_spouse: ArrayLike,
    january_month_index: int,
    params: RealParamSet,
) -> NDArray[np.bool_]:
    """Whether a household's testable income sits in the unmodelled-GIS band.

    All-false when :func:`is_modelled` is true: the exposure indicator has
    nothing to report once GIS is actually computed elsewhere. Otherwise
    compares ``net_income_after_repayment - oas_received`` to
    :func:`band_threshold_annual`, inclusive at the threshold. The caller is
    expected to count only path-years in which at least one living household
    member receives OAS — L2's "living pensioner" — since a household with no
    one drawing OAS is not exposed to GIS at all.

    Args:
        net_income_after_repayment: Line 23600 for the year, real dollars —
            net income after the OAS repayment, which approximates the GIS
            testable base before OAS is subtracted; see the module docstring
            for what that approximation matches and misses. For a couple the
            caller passes the sum of both persons' figures; this function
            does not sum across a household.
        oas_received: Annual gross OAS received for the year, real dollars,
            on the same combined-or-not basis as
            ``net_income_after_repayment``.
        has_spouse: Whether the recipient has a spouse or common-law partner,
            ``(n_paths,)`` or scalar.
        january_month_index: Month index of January of the year being checked.
        params: The ``oas`` parameter set.

    Returns:
        ``(n_paths,)`` (or the broadcast shape of the inputs) boolean array.
    """
    if is_modelled(params):
        shape = np.broadcast(
            np.asarray(net_income_after_repayment),
            np.asarray(oas_received),
            np.asarray(has_spouse),
        ).shape
        return np.zeros(shape, dtype=np.bool_)

    testable = np.asarray(net_income_after_repayment, dtype=np.float64) - np.asarray(
        oas_received, dtype=np.float64
    )
    threshold = band_threshold_annual(has_spouse, january_month_index, params)
    return np.asarray(testable <= threshold, dtype=np.bool_)

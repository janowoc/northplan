# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guaranteed Income Supplement — **not modelled**, and a tripwire that says so.

GIS is income-tested, non-taxable, and reduced against a different income
base than the OAS recovery tax uses. None of that is populated:
``params/{year}/oas.yaml`` carries ``gis.modelled: false`` and nothing else.

This module computes no supplement. Instead it refuses to answer for a
household whose income falls in the band where an unmodelled GIS would
change the result — treating it as zero would silently understate income
exactly where it matters most, and bias the drawdown question this engine
exists to answer: GIS is clawed back against registered withdrawals, so a
model without it cannot see the marginal rate that makes early RRSP
drawdown attractive.

The refusal thresholds sit below the published GIS cut-offs on purpose: they
are a conservative band around where GIS would begin to matter, not a GIS
calculation, and must not be read as one.

When GIS is implemented: the schema is already written up in
``params/gis_not_implemented.yaml``; move it under ``params/{year}/``,
populate it, flip ``modelled`` to true, and replace
:func:`check_within_scope` with the real calculation.

Parameters from ``params/{year}/oas.yaml`` under ``gis``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.params.loader import ParamSet

__all__ = ["GisWouldApplyError", "check_within_scope", "is_modelled", "refusal_threshold"]


class GisWouldApplyError(RuntimeError):
    """A household's income reaches the band where an unmodelled GIS matters.

    Deliberately not a :class:`~engine.params.loader.ParamError`: nothing is
    missing from ``params/``. The parameters are complete and say GIS is
    unmodelled. What is out of range is the household, not the params.
    """


def is_modelled(params: ParamSet) -> bool:
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


def refusal_threshold(has_spouse: ArrayLike, params: ParamSet) -> NDArray[np.float64]:
    """Annual testable income below which this engine refuses to answer.

    Two thresholds, selected per path by marital status, because the couple
    figure is measured against *combined* income and is not twice the single
    one.

    Args:
        has_spouse: Whether the recipient has a spouse or common-law partner,
            ``(n_paths,)`` or scalar. Read for the month being checked: a
            spouse's death moves a household onto the single threshold from
            that month.
        params: The ``oas`` parameter set.

    Returns:
        Threshold in annual real dollars, broadcast to ``has_spouse``'s shape.

    Raises:
        MissingParameterError: If either threshold is absent.
    """
    single = params.number("gis.band_thresholds.single_testable_income_annual")
    couple = params.number("gis.band_thresholds.couple_combined_testable_income_annual")
    return np.where(np.asarray(has_spouse, dtype=bool), couple, single).astype(np.float64)


def check_within_scope(
    testable_income_annual: ArrayLike,
    has_spouse: ArrayLike,
    params: ParamSet,
) -> None:
    """Stop the run if any path reaches the band where GIS would matter.

    A no-op when GIS is modelled, and when every path sits clear of the
    threshold. Otherwise it raises. **Any path, not most**: a single path in
    the GIS band means the distribution's lower tail — the part a retirement
    plan is read for — is wrong.

    Call this once per benefit year against projected annual income for that
    year, before any policy decision reads the result; calling it monthly
    would compare a monthly figure to an annual threshold and refuse every
    household.

    Args:
        testable_income_annual: Projected annual income for the benefit year, on
            the GIS testable basis — combined across the couple where there is
            one, matching whichever threshold applies. ``(n_paths,)``.
        has_spouse: Marital status per path, ``(n_paths,)`` or scalar.
        params: The ``oas`` parameter set.

    Raises:
        GisWouldApplyError: If any path's income is at or below its threshold.
        MissingParameterError: If ``gis.modelled`` or either threshold is absent.
    """
    if is_modelled(params):
        return

    income, threshold = np.broadcast_arrays(
        np.asarray(testable_income_annual, dtype=np.float64),
        refusal_threshold(has_spouse, params),
    )
    breaching = income <= threshold
    if not breaching.any():
        return

    n_breaching = int(breaching.sum())
    n_paths = int(income.size)
    lowest = float(income[breaching].min())
    raise GisWouldApplyError(
        f"{n_breaching} of {n_paths} paths have testable income at or below the "
        f"GIS refusal threshold (lowest {lowest:,.0f}, threshold "
        f"{float(np.min(threshold[breaching])):,.0f}). GIS is not modelled for this "
        f"parameter year, and for these households it would materially change the "
        f"answer — omitting it understates income and hides the clawback that drives "
        f"the drawdown decision. Populate params/gis_not_implemented.yaml, move it "
        f"under params/{{year}}/, and set gis.modelled: true."
    )

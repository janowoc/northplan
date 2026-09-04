"""Guaranteed Income Supplement — **not modelled**, and a tripwire that says so.

GIS is income-tested, non-taxable, and reduced against a different income base
than the OAS recovery tax uses. Getting it right needs the full benefit tables
by marital status and spousal-OAS status, the earnings exemption, and the
Allowance and Allowance for the Survivor alongside it. That is a large body of
parameters and none of it is populated: ``params/{year}/oas.yaml`` carries
``gis.modelled: false`` and nothing else.

So this module computes no supplement. What it does instead is refuse to let a
household quietly receive an answer that a missing GIS would have changed.

**Why a refusal rather than a zero.** Treating an unmodelled GIS as zero is not
neutral. It understates income for exactly the households that depend on it
most, and it does so silently — the plan comes back looking feasible-but-tight
with nothing on the face of it to say a whole benefit was omitted. The bias also
runs the wrong way for the decumulation question this engine exists to answer:
GIS is clawed back against registered withdrawals, so a model without it cannot
see the marginal rate that makes early RRSP drawdown attractive, and will
recommend deferral it would not otherwise recommend.

**Why the thresholds sit below the published cut-offs.** They are not a GIS
calculation and must not be read as one. They are a band around the region
where GIS would begin to matter, set conservatively on purpose: erring low
widens the band and refuses more often, which is the safe direction for a
tripwire. Raising them to the published cut-offs would narrow the refusal to
households already receiving GIS and miss the ones about to.

When GIS is implemented, the schema it needs is already written up in
``params/gis_not_implemented.yaml`` — including the three ways the headline
maxima mislead. Move it back under ``params/{year}/``, populate it, flip
``modelled`` to true, and replace :func:`check_within_scope` with the real
calculation. The two function stubs that used to sit here for
``countable_income`` and ``supplement_monthly`` live in that file's header
rather than here, so that this module contains no specification it cannot
honour.

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
    missing from ``params/``. The parameters are complete and they say this
    engine does not model GIS. What is out of range is the household.

    Not catchable into a fallback. A caller that swallows this to carry on with
    a zero supplement has reintroduced the silent understatement the refusal
    exists to prevent.
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
    single = params.number("gis.refusal_thresholds.single_testable_income_annual")
    couple = params.number("gis.refusal_thresholds.couple_combined_testable_income_annual")
    return np.where(np.asarray(has_spouse, dtype=bool), couple, single).astype(np.float64)


def check_within_scope(
    testable_income_annual: ArrayLike,
    has_spouse: ArrayLike,
    params: ParamSet,
) -> None:
    """Stop the run if any path reaches the band where GIS would matter.

    A no-op when GIS is modelled, and a no-op when every path sits clear of the
    threshold. Otherwise it raises.

    **Any path, not most paths.** A Monte Carlo path is a scenario the household
    might actually live, so a run in which one path in ten thousand drops into
    the GIS band has produced a distribution whose lower tail is wrong — and the
    lower tail is the part a retirement plan is read for. If a tolerance is ever
    wanted it belongs here as an explicit, defended parameter, never as a
    quietly chosen fraction.

    Call this once per benefit year against projected income for that year,
    before any policy decision reads the result. Calling it monthly against a
    month's income would compare a monthly figure to an annual threshold and
    refuse every household.

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

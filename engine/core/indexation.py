"""The constant real-terms cost of periodic, lagged indexation.

An engine that works in real dollars and steps annually can treat a fully
indexed benefit as constant. An engine that steps monthly could, in principle,
track the fact that it is not — and this module is where that was going to
live. It does something simpler and better instead.

What is actually going on
-------------------------

An indexed amount is fixed in *nominal* terms between adjustment dates, so its
real value falls a little each month and steps back up on the adjustment date.
And because an adjustment is computed from a CPI average over a window that
closes some months before it takes effect, the step back up restores the
inflation of that earlier window rather than of the months just past. Under
steady inflation, the amount never returns to where it started.

That decomposes into two effects with very different characters:

- **An oscillation** around the cycle's own mean. Bounded, and mean-zero by
  construction. At two percent inflation it is worth about a sixth of a percent
  for a quarterly-indexed benefit and under one percent for an annually indexed
  one.
- **A level.** The cycle's mean sits below the adjustment-date value, and the
  lag pushes it lower still. This one is permanent, it does not average out,
  and it is always in the direction that flatters the plan.

**This module models the level and ignores the oscillation.** The level is the
larger term and the biased one; the oscillation is small, bounded, and changes
no decision the optimizer makes. Dropping it means every function here is
independent of the month, which is the point: an indexed benefit is once again
a constant in real dollars, and the monthly loop does not have to thread an
adjustment calendar through anything.

What this is *not* is the assumption that an indexed benefit is worth its
published real value. That is the version that costs roughly one percent of CPP
and OAS income for the life of the plan, compounding into every downstream
decision. The factor is constant, not one.

The adjustment frequency and the lag are statutory and come from ``params/``.
The inflation rate they act on is a scenario assumption, not a tax parameter,
and is supplied by the caller.

Every function here returns a *factor* to multiply a real amount by. It never
returns dollars, so there is no way to apply it twice without the double
application being visible at the call site.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.params.loader import ParamSet


def real_factor(
    inflation_rate: ArrayLike,
    adjustments_per_year: int,
    lag_months: int,
) -> NDArray[np.float64]:
    """Constant real-value factor for a periodically indexed amount.

    Takes no year and no month. The whole simplification is in that absence: an
    indexed benefit is a constant in real dollars, just not the constant its
    published amount suggests. Compute this once per benefit per scenario and
    reuse it for every month of the run.

    Combines the mean erosion across the indexation cycle with the permanent
    shortfall the CPI lag leaves behind, and ignores the oscillation around
    that mean — see the module docstring for why. At most 1.0, and equal to 1.0
    only when inflation is zero.

    Args:
        inflation_rate: Assumed annual inflation as a bare fraction,
            ``(n_paths,)`` or scalar. A scenario input, not a tax parameter.
            The uncertainty in this assumption is an order of magnitude larger
            than the oscillation this function drops, which is the other reason
            modelling the oscillation would have been false precision.
        adjustments_per_year: How many times a year the amount is adjusted.
            Four for a quarterly-indexed benefit, one for an annual one. From
            ``params/`` via :func:`schedule`.
        lag_months: Months between the close of the CPI window and the
            adjustment taking effect, from ``params/``.

    Returns:
        Multiplier in ``(0, 1]``, shape broadcast from ``inflation_rate``.

    Raises:
        ValueError: If ``adjustments_per_year`` is not positive or does not
            divide twelve, or if ``lag_months`` is negative.
    """
    raise NotImplementedError


def unindexed_real_factor(
    months_elapsed: int,
    inflation_rate: ArrayLike,
) -> NDArray[np.float64]:
    """Real-value factor for an amount that is never indexed at all.

    A DB pension with no indexation, a flat benefit whose statute has no
    escalator. This one **does** depend on time and cannot be collapsed to a
    constant, which is exactly the difference between the two cases: an indexed
    amount's real loss is bounded by one indexation cycle, while an unindexed
    amount's grows without limit. Over a thirty-year retirement it is the
    largest single real-terms effect in the model, and approximating it away
    would be a different kind of decision from approximating away the
    oscillation.

    Args:
        months_elapsed: Months since the amount was fixed in nominal terms.
        inflation_rate: Assumed annual inflation as a bare fraction.

    Returns:
        Multiplier in ``(0, 1]``, falling monotonically in ``months_elapsed``.
    """
    raise NotImplementedError


def schedule(name: str, params: ParamSet) -> tuple[int, int]:
    """Look up an amount's indexation schedule.

    ``params/`` records the months an amount is adjusted in, because that is
    the verifiable statutory fact and it is what a human checks against a
    source. This returns the *count* of them, because with the oscillation
    dropped, the count is all the model uses — which month of the quarter the
    adjustment lands in no longer changes any result.

    Args:
        name: The indexed amount, as ``params/`` names it.
        params: The parameter set that owns it.

    Returns:
        ``(adjustments_per_year, lag_months)``.

    Raises:
        MissingParameterError: If the schedule is absent. An amount whose
            indexation rule has not been verified is not indexed "annually by
            default" — it is unknown, and the run stops.
    """
    raise NotImplementedError

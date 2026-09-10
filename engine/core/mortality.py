# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Mortality: one uniform per person per path, mapped through a survival curve.

The mechanism (README design decision 8): a period life table gives an
annual death probability per sex and age; :func:`survival_curve` compounds it
into a monthly probability of being alive at the start of each month of the
run, and :func:`death_month_index` inverts one uniform draw per person per
path through that curve into a death month. Spouses' deaths are independent
draws.

The table publishes annual probabilities; the model needs a monthly one, and
gets it by assuming a constant force of mortality within each year of age —
:func:`monthly_hazard`. Nothing in the source states that assumption; it is a
modelling choice made here (see ``docs/limitations.md`` L9).

Every path dies by the table's terminal age, where the annual probability is
one. The simulation runs each path to the second death and has no separate
horizon (``docs/limitations.md`` L10).

Two further properties belong to the table itself, not to this module:
it is a period table carrying no mortality improvement (``docs/limitations.md``
L8), and one national table is applied to every province, including the one
this engine models (``docs/limitations.md`` L45).

This module is arithmetic over a parameter table. It does not import
``engine.core.state`` and has no dependency on the state tree.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.core.timeline import MONTHS_PER_YEAR, age_in_years, year_month
from engine.params.loader import ParamSet

__all__ = ["death_month_index", "monthly_hazard", "survival_curve"]


def monthly_hazard(q_annual: ArrayLike) -> NDArray[np.float64] | float:
    """Monthly hazard implied by an annual death probability ``q_annual``.

    ``1 - (1 - q_annual) ** (1 / MONTHS_PER_YEAR)``, elementwise, accepting a
    float or an array and returning the same shape.

    Deriving a monthly figure from an annual one requires an assumption the
    source table does not make: this assumes a constant force of mortality
    within the year of age, which is a modelling choice, not a fact the table
    states (``docs/limitations.md`` L9).

    No validation of the input range: ``q`` is bounded to ``(0, 1]`` by
    ``tests/params/test_param_file_structure.py``, which owns that invariant.
    A second copy of the check here would be a second place to keep in step
    with it.

    Args:
        q_annual: Annual death probability, a bare fraction, scalar or array.

    Returns:
        The equivalent monthly hazard, same shape as ``q_annual``.
    """
    return 1 - (1 - q_annual) ** (1 / MONTHS_PER_YEAR)


def survival_curve(
    birth_year: int,
    birth_month: int,
    sex: str,
    start_year: int,
    table: ParamSet,
) -> NDArray[np.float64]:
    """Probability of being alive at the start of each month of the run.

    ``curve[i]`` is the probability the person is alive at the start of month
    index ``i``, where month index 0 is January of ``start_year`` — the same
    index :func:`engine.core.timeline.month_index` produces. ``curve[0]`` is
    always ``1.0``.

    Construction: for each month index ``i`` in turn, take the person's age in
    whole years at that month (:func:`engine.core.timeline.age_in_years` on
    :func:`engine.core.timeline.year_month` of ``i``), look up ``q`` for that
    age, and set ``curve[i + 1] = curve[i] * (1 - monthly_hazard(q))``.

    Let ``M`` be the first month index at which the person's age in whole
    years equals ``terminal_age_years``. ``q`` there is 1, so
    ``monthly_hazard`` is 1 and ``curve[M + 1]`` is exactly ``0.0``. The curve
    stops there: valid indices are ``0`` through ``M + 1``, length ``M + 2``,
    final entry ``0.0``. The person lives through the first month of their
    terminal-age year and not past it — ``curve[M]`` itself is not zero,
    because the table does not say the person is already dead at the start of
    that year.

    If the person's age in whole years at month index 0 is strictly greater
    than ``terminal_age_years``, there is no ``q`` row for that age and one is
    not invented: this returns ``np.array([0.0])``, a length-1 curve. Age
    exactly equal to ``terminal_age_years`` at month 0 is *not* this case; it
    goes through the ordinary construction above and yields ``[1.0, 0.0]``.

    Args:
        birth_year: Calendar year of birth.
        birth_month: Month of birth, ``1..12``.
        sex: ``"f"`` or ``"m"``, selecting the life table.
        start_year: The simulation's first calendar year — month index 0.
        table: The ``mortality`` parameter set (``params["mortality"]``), not
            a :class:`~engine.core.indexation.RealParamSet`: this file holds
            no dollars and declares no indexation schedule, so the real-terms
            view has nothing to do here. Reached through
            :meth:`~engine.params.loader.ParamSet.number`, which raises
            :class:`~engine.params.loader.MissingParameterError` naming the
            file and the missing age row rather than silently filling one in.

    Returns:
        A read-only array. The curve is computed once per person and handed
        to every path, so it is frozen here rather than left for a caller to
        remember not to mutate.
    """
    terminal_age = int(table.number("terminal_age_years"))
    age_at_start = age_in_years(birth_year, birth_month, start_year, 1)
    if age_at_start > terminal_age:
        curve = np.array([0.0], dtype=np.float64)
        curve.flags.writeable = False
        return curve

    curve = [1.0]
    index = 0
    while True:
        year, month = year_month(index, start_year)
        age = age_in_years(birth_year, birth_month, year, month)
        q = table.number(f"q_x.{sex}.{age}")
        curve.append(curve[index] * (1 - monthly_hazard(q)))
        if age == terminal_age:
            break
        index += 1

    array = np.array(curve, dtype=np.float64)
    array.flags.writeable = False
    return array


def death_month_index(u: NDArray[np.float64], curve: NDArray[np.float64]) -> NDArray[np.int64]:
    """First month index, per path, at which ``curve`` falls below ``u``.

    ``u`` is ``(n_paths,)`` float64, each value in the **open** interval
    ``(0, 1)``. Returns ``(n_paths,)`` int64: for each path, the first index
    ``i`` at which ``curve[i] < u``.

    The returned index is the first month in which the person is **not**
    alive, so ``alive`` at month ``m`` is exactly
    ``death_month_index > m`` — the invariant
    ``engine/core/state.py`` writes down beside
    ``PersonState.death_month_index`` and says it cannot check without a
    month index; this function is what makes it true. It follows that
    ``P(alive at month i) == curve[i]``, which is what makes the
    distribution test in ``tests/core/test_mortality.py`` meaningful rather
    than a tautology.

    This function never returns :data:`engine.core.state.DEATH_NOT_DRAWN`.
    That sentinel is for a state whose draw has not happened yet; every path
    handed to this function gets a real death month back.

    The inversion is done with :func:`numpy.searchsorted` on the reversed
    curve rather than a broadcast comparison. ``curve`` is non-increasing, so
    ``curve[::-1]`` is non-decreasing and
    ``len(curve) - np.searchsorted(curve[::-1], u, side="left")`` is the
    count of indices whose survival probability is ``>= u``, which is
    exactly the first index below it. A ``(n_paths, n_months)`` boolean
    comparison would give the same answer while allocating memory this does
    not need to.

    Args:
        u: Per-path uniform draws, each strictly inside ``(0, 1)``.
        curve: A survival curve from :func:`survival_curve`.

    Returns:
        Per-path death month index, ``(n_paths,)``, dtype ``int64``.

    Raises:
        ValueError: If any element of ``u`` is outside the open interval
            ``(0, 1)``. Both endpoints are genuinely broken and neither fails
            visibly without this guard: at ``u == 0`` no index satisfies
            ``curve[i] < u`` even though the curve ends at 0, and the natural
            implementations return an index one past the end of the curve —
            an out-of-range month that looks like an ordinary answer. At
            ``u == 1`` every person with any hazard at all dies in month 1.
    """
    u = np.asarray(u, dtype=np.float64)
    offending = (u <= 0) | (u >= 1)
    if np.any(offending):
        count = int(np.count_nonzero(offending))
        example = float(u[offending][0])
        raise ValueError(
            f"death_month_index requires every u to be strictly inside (0, 1); "
            f"{count} of {u.size} value(s) are not, for example {example!r}. "
            f"u == 0 and u == 1 are both genuinely broken rather than ordinary "
            f"edge cases — see this function's docstring."
        )
    return (len(curve) - np.searchsorted(curve[::-1], u, side="left")).astype(np.int64)

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Mortality: one uniform per person per path, mapped through a survival curve.

:func:`survival_curve` turns a life table's annual probability into a monthly curve that
:func:`death_month_index` inverts, per path, into a death month. Modelling choices the table
itself does not make (``docs/limitations.md``): constant mortality force within a year of age
(L9); a period table with no improvement (L8) applied nationally to every province (L45); dying
by the terminal age, since the run goes to the second death with no separate horizon (L10).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.core.timeline import MONTHS_PER_YEAR, age_in_years, year_month
from engine.params.loader import ParamSet

__all__ = ["death_month_index", "monthly_hazard", "survival_curve"]


def monthly_hazard(q_annual: ArrayLike) -> NDArray[np.float64] | float:
    """Monthly hazard implied by annual death probability ``q_annual``: ``1 - (1 - q_annual) **
    (1 / MONTHS_PER_YEAR)``, elementwise. Scalar or array in, same shape out; not range-checked
    here (owned by ``tests/params/test_param_file_structure.py``).
    """
    return 1 - (1 - q_annual) ** (1 / MONTHS_PER_YEAR)


def survival_curve(
    birth_year: int,
    birth_month: int,
    sex: str,
    start_year: int,
    table: ParamSet,
) -> NDArray[np.float64]:
    """Probability of being alive at the start of each month of the run, for one person (birth
    date, ``sex`` ``"f"``/``"m"``) against the ``mortality`` parameter set ``table``.

    ``curve[i]`` is that probability at month index ``i`` (aligned with
    :func:`engine.core.timeline.month_index`); ``curve[0]`` is ``1.0``. Let ``M`` be the first
    month index at which the person's age in whole years reaches ``terminal_age_years``: the
    curve ends there, length ``M + 2``, with a final entry of exactly ``0.0``. If already past
    that age at month 0, this returns the length-1 ``[0.0]`` instead. Read-only: shared across
    every path.

    Args:
        birth_year: Calendar year of birth.
        birth_month: Month of birth, ``1..12``.
        sex: ``"f"`` or ``"m"``, selecting the life table.
        start_year: The simulation's first calendar year — month index 0.
        table: The ``mortality`` parameter set (``params["mortality"]``).

    Returns:
        The survival curve described above, dtype ``float64``.

    Raises:
        MissingParameterError: If ``terminal_age_years`` is absent, or ``table`` has no
            ``q_x`` row for the age reached at some month, naming the file and the missing
            key.
        MalformedParamFileError: If that row, or ``terminal_age_years``, is present but not
            numeric.
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
    """First month index, per path, at which ``curve`` falls below ``u``: the first month the
    person is **not** alive, so ``alive`` at month ``m`` is exactly ``death_month_index > m``.
    ``u`` and the returned int64 array share the same ``(n_paths,)`` shape. This function never
    returns :data:`engine.core.state.DEATH_NOT_DRAWN`; that sentinel is only for a state whose
    draw has not happened yet.

    Raises:
        ValueError: If ``u`` has a value outside the open ``(0, 1)`` — ``0`` matches no index
            even though the curve ends at 0, and ``1`` would put every hazard in month 1.
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

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Defined-benefit pensions: a scenario input, not a parameter.

``PensionState.monthly_amount`` is stated by the scenario, in January dollars
of the start year, not read from ``params/``. This is therefore the one place
in ``engine/benefits/`` that applies a factor by hand rather than through
:class:`~engine.core.indexation.RealParamSet` (L5): a fully indexed pension is
a real-dollar constant and gets no erosion factor at all (L52); an unindexed
one decays under :func:`~engine.core.indexation.unindexed_factor`, counted
from month 0 even for a pension already in pay when the run opens, since
``monthly_amount`` is stated as at that January.

The survivor share (``PensionState.survivor_share``) is computed here, by
:func:`survivor_pension_monthly`, and applied by the step from the month the
first death takes effect (L40, L41) -- including a member who dies before
the pension's start month, paid from that start month as if the pension had
been deferred (L59).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.core.indexation import unindexed_factor
from engine.core.state import PensionState


def db_pension_monthly(
    pension: PensionState,
    month_index: int,
    alive: ArrayLike,
    inflation_rate: float,
) -> NDArray[np.float64]:
    """Monthly defined-benefit pension payable this month, including any bridge.

    Args:
        pension: The pension's standing.
        month_index: Months since January of the scenario's start year.
        alive: Whether the recipient is alive this month, ``(n_paths,)`` or
            scalar.
        inflation_rate: Assumed annual inflation as a bare fraction; used only
            when ``pension.indexed`` is ``False``.

    Returns:
        Monthly amount in real dollars, shape of ``pension.monthly_amount``,
        zero before ``pension.start_month_index`` and wherever not alive.
    """
    shape = pension.monthly_amount.shape
    if month_index < pension.start_month_index:
        return np.zeros(shape, dtype=np.float64)

    months_since_start = month_index - max(pension.start_month_index, 0)
    factor = 1.0 if pension.indexed else unindexed_factor(inflation_rate, months_since_start)

    amount = pension.monthly_amount * factor
    if pension.bridge_end_month_index is not None and month_index <= pension.bridge_end_month_index:
        amount = amount + pension.bridge_monthly * factor

    return np.asarray(np.where(np.asarray(alive, dtype=bool), amount, 0.0), dtype=np.float64)


def survivor_pension_monthly(
    pension: PensionState,
    month_index: int,
    receiving: ArrayLike,
    inflation_rate: float,
) -> NDArray[np.float64]:
    """The survivor's share of a defined-benefit pension, for one month.

    ``pension.survivor_share * pension.monthly_amount * factor``, where
    ``factor`` follows exactly the rule :func:`db_pension_monthly` applies to
    the member's own pension: ``1.0`` if indexed, else
    :func:`~engine.core.indexation.unindexed_factor` counted from
    ``pension.start_month_index`` (or from month 0 if that is negative, a
    pension already in pay when the run opens). **No bridge** — the bridge is
    the member's own, never paid to a survivor.

    Zero before ``pension.start_month_index``, which is what covers a member
    who dies before the pension starts: the survivor share is paid from the
    start month, as if the pension had been deferred to it
    (``docs/limitations.md`` L59). Zero wherever ``receiving`` is false.

    Args:
        pension: The pension's standing.
        month_index: Months since January of the scenario's start year.
        receiving: Whether the survivor is entitled to this pension's share
            this month, ``(n_paths,)`` or scalar — the member is not alive
            and the survivor is.
        inflation_rate: Assumed annual inflation as a bare fraction; used only
            when ``pension.indexed`` is ``False``.

    Returns:
        Monthly survivor share in real dollars, shape of
        ``pension.monthly_amount``, zero before the start month and wherever
        not receiving.
    """
    shape = pension.monthly_amount.shape
    if month_index < pension.start_month_index:
        return np.zeros(shape, dtype=np.float64)

    months_since_start = month_index - max(pension.start_month_index, 0)
    factor = 1.0 if pension.indexed else unindexed_factor(inflation_rate, months_since_start)
    amount = pension.survivor_share * pension.monthly_amount * factor
    return np.asarray(np.where(np.asarray(receiving, dtype=bool), amount, 0.0), dtype=np.float64)

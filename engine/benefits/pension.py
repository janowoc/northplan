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

The survivor share (``PensionState.survivor_share``) is applied by the step
at first death (L41), not here.
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

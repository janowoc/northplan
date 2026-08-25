"""The annual step. One year, all paths, one implementation.

``advance_year`` is the only place a simulated year happens. Monte Carlo calls
it in a loop over years; the optimizer calls Monte Carlo. If a second function
in this repository starts to look like a year, that is a bug — say so rather
than writing it.

Ordering within the year is a correctness decision, not a style one, and it is
fixed here so that no account module can quietly disagree with it. The order is
stated in :func:`advance_year`'s docstring and every change to it needs a
verification row.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from engine.core.state import HouseholdState
from engine.params.loader import ParamYear
from engine.policy.base import Policy


def advance_year(
    state: HouseholdState,
    real_returns: NDArray[np.float64],
    policy: Policy,
    params: ParamYear,
) -> HouseholdState:
    """Advance the household by one year, for all paths at once.

    Order of operations within the year, which the implementation must follow
    exactly:

    1. Ages advance; deaths are resolved. A death changes what follows in the
       same year — OAS ceases, a RRIF may roll over to the survivor.
    2. RRIF and LIF minimums are computed from **opening** balances, before any
       growth is applied.
    3. Income the household receives regardless of policy: employment, DB
       pension, CPP, gross OAS.
    4. The policy chooses contributions and discretionary withdrawals, reading
       only opening balances, current age, and current-year parameters.
    5. Withdrawals are taken, at least the mandatory minimums.
    6. Contributions are made against room available at the start of the year;
       room is updated afterwards, never before.
    7. Growth is applied to closing balances.
    8. Tax is assessed on the year's income, including the OAS recovery tax
       computed against the **prior** year's net income, not this year's.
    9. The year is recorded in history.

    Args:
        state: Opening state for ``state.year``.
        real_returns: Real return for the year per path and asset class, shape
            ``(n_assets, n_paths)``. Real, not nominal.
        policy: The decision rules being evaluated. Given only the information
            available at this point in the simulation.
        params: Parameters for this tax year.

    Returns:
        Opening state for ``state.year + 1``.
    """
    raise NotImplementedError


def resolve_deaths(
    state: HouseholdState,
    mortality_draw: NDArray[np.float64],
    params: ParamYear,
) -> HouseholdState:
    """Apply mortality for the year and its immediate consequences.

    Consequences that land in the same year: OAS and GIS cease for the
    deceased, a RRIF rolls to a surviving spouse tax-deferred, and with no
    surviving spouse the registered balance is brought fully into income.

    Args:
        state: Opening state.
        mortality_draw: Uniform draws, ``(n_persons, n_paths)``, compared
            against the year's mortality rate. From the common random number
            stream, so mortality is identical across policies.
        params: Parameters for this tax year.

    Returns:
        State with ``alive`` and any rollover applied.
    """
    raise NotImplementedError

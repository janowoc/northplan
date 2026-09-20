# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The monthly step. One month, all paths, one implementation.

``advance_month`` is the only place simulated time passes; Monte Carlo calls it in a loop over
months, and the optimizer calls Monte Carlo.

The annual events are *phases* ``advance_month`` invokes when due: ``open_year`` in January,
``close_year`` in December, ``settle_tax_balance`` in the filing month. Nothing outside this
module may call them directly.

Ordering is a correctness decision, fixed here so no account module can quietly disagree with
it; the order is stated in each function's docstring and every change to it needs a verification
row.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from engine.core.state import HouseholdState
from engine.params.loader import ParamYear
from engine.policy.base import Policy


def advance_month(
    state: HouseholdState,
    real_returns: NDArray[np.float64],
    policy: Policy,
    params: ParamYear,
) -> HouseholdState:
    """Advance the household by one month, across every path. Order of operations:

    1. January: run :func:`open_year`.
    2. Resolve deaths for the month; may stop OAS/GIS, roll a RRIF to a survivor.
    3. Filing month: run :func:`settle_tax_balance`.
    4. Receive this month's employment, DB pension, CPP, and gross OAS/GIS income.
       Employment is already a real monthly figure, no factor. CPP and gross OAS
       each take the constant erosion factor from
       ``engine.core.indexation.erosion_factor`` for their own schedule. A DB pension
       takes no factor if fully indexed (already a real-dollar constant, on no
       schedule) or ``engine.core.indexation.unindexed_factor`` if not indexed.
    5. Remit tax withheld at source into ``remitted_ytd``, a prepayment only.
    6. Policy picks contributions/withdrawals from opening state, not this return.
    7. Take withdrawals; force the RRIF/LIF minimum only once the year is running
       out of months, so it is fully out by December.
    8. Make contributions against room available at the start of the month; room
       updates after.
    9. Apply this month's real return, once, to closing balances.
    10. Add this month's income to the year-to-date ledger.
    11. December: run :func:`close_year`.
    12. Advance the month, rolling the year and recomputing ``spending_monthly``;
        the roll happens one call before January's phases run.

    Args:
        state: Opening state for ``state.year``/``state.month``.
        real_returns: This month's real return per path and asset class,
            ``(n_assets, n_paths)``, monthly and real.
        policy: Decision rules, given only information available at this point.
        params: Parameters for the tax year ``state.year`` falls in.

    Returns:
        Opening state for the following month.
    """
    raise NotImplementedError


def open_year(state: HouseholdState, params: ParamYear) -> HouseholdState:
    """January phase: grant room, fix the year's annual limits, reset the ledger.

    Called by :func:`advance_month`, never directly.

    1. Grant TFSA room; restore room for the *prior* year's withdrawals.
    2. Grant RRSP room accrued on the prior year's earned income.
    3. Accrue RESP grant room, per beneficiary.
    4. Fix the RRIF/LIF minimum from the **opening** 1 January balance and age at
       the start of the year.
    5. Fix the LIF maximum the same way, from
       ``params.jurisdiction(LifState.jurisdiction)``; where
       ``engine.accounts.lif.has_maximum`` is false, none is stored.
    6. Reset the year-to-date income ledger and ``remitted_ytd`` (not
       ``balance_owing``, still owed until filing); reset per-beneficiary RESP
       grant-received and contributed-ytd.

    In the start year's January, items 1-3 grant nothing; the opening state already
    includes it.

    Args:
        state: Opening state for January.
        params: Parameters for the new tax year.

    Returns:
        State with the year's annual quantities established.
    """
    raise NotImplementedError


def close_year(state: HouseholdState, params: ParamYear) -> HouseholdState:
    """December phase: assess the year, record it, carry the balance forward.

    Called by :func:`advance_month`, never directly.

    1. Force out any unwithdrawn RRIF/LIF minimum for the year.
    2. Convert to a RRIF an RRSP whose owner reaches conversion age this year.
    3. Assess the year in one joint step, on this year's brackets:
       ``engine.tax.combined.household_assessment`` elects pension income
       splitting and assesses every person at the elected split together, since
       the election is chosen by minimising the assessment it is part of. The
       OAS repayment is a line within that assessment, computed on this year's
       net income before the repayment (line 23400, which itself includes this
       year's OAS), and capped at the OAS received this year.
    4. Assessment less ``remitted_ytd`` becomes ``balance_owing`` (or a refund),
       settled in next year's filing month.
    5. Shift the pair of net-income fields: ``net_income_two_years_prior``
       takes the value ``prior_year_net_income`` held all year, and
       ``prior_year_net_income`` takes this year's net income after the
       social benefits repayment (line 23600,
       ``Assessment.net_income_after_repayment``) — written for every person,
       including one who has died (#35). ``net_income_two_years_prior`` is
       the one next year's RESP enhanced-grant rate reads; the newly written
       ``prior_year_net_income`` is not read until the December close after
       this one shifts it into that older field. The OAS repayment is not
       stored here, it is assessed in item 3.
    6. Append the year to ``history``.

    Args:
        state: State at the end of December, with twelve months accumulated.
        params: Parameters for the tax year being closed.

    Returns:
        State with the year assessed and recorded.
    """
    raise NotImplementedError


def settle_tax_balance(state: HouseholdState, params: ParamYear) -> HouseholdState:
    """Filing-month phase: pay the prior year's balance owing in cash.

    Called by :func:`advance_month`, never directly. The month comes from
    ``params``; it is not written into the code.

    This is the step that makes tax a cash flow rather than an accrual. The
    money leaves the household's accounts here, months after the income that
    generated it, and where it comes from is a withdrawal like any other — it
    can realize a capital gain, and that gain is income of the *current* year.
    A refund arrives the same way, in this month, and is not income at all.

    Args:
        state: State in the filing month, carrying a non-zero balance owing.
        params: Parameters for the current tax year, supplying the filing
            month.

    Returns:
        State with the balance discharged and the cash effect applied.
    """
    raise NotImplementedError


def resolve_deaths(
    state: HouseholdState,
    mortality_draw: NDArray[np.float64],
    params: ParamYear,
) -> HouseholdState:
    """Apply mortality for this month and its immediate consequences.

    Monthly, not annual: a death in March stops three quarters of a year of OAS that an annual
    step would pay in full, and the final return covers only the part-year to death.

    Consequences in the same month: OAS/GIS cease for the deceased from the following month, a
    RRIF rolls to a surviving spouse tax-deferred, and with no survivor the registered balance
    comes fully into income in the year of death.

    Args:
        state: Opening state for the month.
        mortality_draw: Uniform draws, ``(n_persons, n_paths)``, from the common random number
            stream, so mortality is identical across policies.
        params: Parameters for the current tax year.

    Returns:
        State with ``alive`` and any rollover applied.
    """
    raise NotImplementedError

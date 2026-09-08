# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The monthly step. One month, all paths, one implementation.

``advance_month`` is the only place simulated time passes. Monte Carlo calls it
in a loop over months; the optimizer calls Monte Carlo. If a second function in
this repository starts to look like a step through time, that is a bug — say so
rather than writing it.

The annual events have not gone away; they have become *phases* that
``advance_month`` invokes in the months that call for them. ``open_year`` runs
in January, ``close_year`` in December, ``settle_tax_balance`` in the filing
month. They are called from inside the single loop and are not loops
themselves. Nothing outside this module may call them.

Ordering is a correctness decision, not a style one, and it is fixed here so
that no account module can quietly disagree with it. The order is stated in
each function's docstring and every change to it needs a verification row.
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
    """Advance the household by one month, for all paths at once.

    Order of operations within the month, which the implementation must follow
    exactly:

    1. If this is January, run :func:`open_year`. Ages advance on birthdays,
       not on 1 January, so this is about the *tax* year: room is granted, the
       annual RRIF minimum and LIF maximum are fixed from opening balances, and
       the year-to-date ledger is reset.
    2. Deaths are resolved for this month. A death changes what follows in the
       same month — OAS and GIS stop, a RRIF may roll over to the survivor.
    3. If this is the filing month, run :func:`settle_tax_balance`: the prior
       year's balance owing is paid in cash out of the household's accounts.
       This is a full year after the income that caused it.
    4. Income the household receives this month regardless of policy: one
       month of employment, one month of DB pension with its explicit and
       growing real decay, one month of CPP, one month of gross OAS and GIS.
       Benefit amounts are the published monthly amounts times the constant
       factor from ``engine.core.indexation.erosion_factor``, which accounts
       for an indexed benefit averaging below its published real value.
       That factor is the same every month and is computed once per scenario,
       so this step reads it rather than recomputing it.
    5. Tax withheld at source on that income is remitted and added to
       ``remitted_ytd``. It is a prepayment, not an assessment.
    6. The policy chooses this month's contributions and discretionary
       withdrawals, reading only opening balances, current age, income
       accumulated so far this year, and the current year's parameters. It may
       not read this month's return, which has not been applied yet.
    7. Withdrawals are taken. The remaining RRIF minimum for the year is
       tracked but not forced this month unless the year is running out of
       months to take it in; by December it must have come out in full.
    8. Contributions are made against room available at the start of the month;
       room is updated afterwards, never before.
    9. Growth is applied to closing balances, using this month's real return.
       One month, not one twelfth of a year applied twelve times to the opening
       balance — compounding within the year is the point of stepping monthly.
    10. This month's income is added to the year-to-date ledger.
    11. If this is December, run :func:`close_year`.
    12. The month advances, rolling the year over after December.

    Args:
        state: Opening state for ``state.year``/``state.month``.
        real_returns: Real return for *this month* per path and asset class,
            shape ``(n_assets, n_paths)``. Real, not nominal, and monthly, not
            annual — passing an annual figure here overstates growth by roughly
            a factor of twelve and will not fail loudly.
        policy: The decision rules being evaluated. Given only the information
            available at this point in the simulation.
        params: Parameters for the tax year ``state.year`` falls in.

    Returns:
        Opening state for the following month.
    """
    raise NotImplementedError


def open_year(state: HouseholdState, params: ParamYear) -> HouseholdState:
    """January phase: grant room, fix the year's annual limits, reset the ledger.

    Called by :func:`advance_month`, never directly. What happens here happens
    once a year and is then drawn down over the eleven months that follow:

    1. TFSA room is granted for the year, and room for the *prior* year's
       withdrawals is restored — not the current year's, which is the whole
       point of the restoration lag.
    2. RRSP room accrued on the prior year's earned income is granted.
    3. RESP grant room accrues, per beneficiary.
    4. The RRIF and LIF minimum for the year is computed from the **opening**
       balance on 1 January, before any of this year's growth, using age at the
       start of the year. It is stored as an annual amount with a running
       "still to be withdrawn" figure, because the withdrawal itself happens
       across the months.
    5. The LIF *maximum* for the year is fixed the same way, and read from the
       parameter set of the jurisdiction each locked-in account is registered
       in — ``params.jurisdiction(terms.registration_jurisdiction)``, not
       ``params.province(household.province)``. Where
       ``engine.accounts.lira.has_maximum`` is false the jurisdiction imposes
       no ceiling and none is stored; that is a rule, not a missing table.
    6. The year-to-date income ledger is reset to zero and ``remitted_ytd`` with
       it. ``balance_owing`` is *not* reset: it is still owed until the filing
       month. The per-beneficiary RESP year-to-date figures reset here too:
       both the grant received and the contributions made, the second because
       the additional grant tier's eligible window is a dollar amount per
       calendar year.

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

    1. Any unwithdrawn RRIF or LIF minimum for the year is forced out now. The
       minimum is a statutory obligation with a 31 December deadline, so a
       policy that under-withdrew all year has the balance taken from it here.
    2. An RRSP belonging to a person who reaches the conversion age this year
       becomes a RRIF, effective for next January's minimum.
    3. Tax is assessed on the full year's accumulated income — one assessment,
       on twelve months of accrued income, using this year's brackets. The OAS
       recovery tax within it is computed against the net income of the year
       :func:`engine.core.timeline.benefit_year_income_year` names, not this
       year's.
    4. Pension income splitting is elected for the year, jointly across the
       household. It is a year-end election and cannot be made monthly.
    5. The assessment less ``remitted_ytd`` becomes ``balance_owing``, payable
       in next year's filing month. A negative balance is a refund and is
       received in that same month, not immediately.
    6. This year's net income is stored in ``prior_year_net_income`` for the
       benefit years that will be assessed against it.
    7. The year is appended to ``history``.

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

    Monthly, not annual: a death in March stops three quarters of a year of OAS
    that an annual step would have paid in full, and the deceased's final
    return covers only the part-year to the date of death.

    The draw is compared against a *monthly* hazard derived from the annual
    mortality table. The conversion from an annual ``q_x`` to a monthly rate
    assumes a constant force of mortality within the year; that assumption is
    stated here because it is a modelling choice rather than a sourced value,
    and it belongs in a comment beside the mortality table it is applied to as
    well as here.

    Consequences that land in the same month: OAS and GIS cease for the
    deceased from the following month, a RRIF rolls to a surviving spouse
    tax-deferred, and with no surviving spouse the registered balance is
    brought fully into income in the year of death.

    Args:
        state: Opening state for the month.
        mortality_draw: Uniform draws, ``(n_persons, n_paths)``, compared
            against this month's mortality hazard. From the common random
            number stream, so mortality is identical across policies.
        params: Parameters for the current tax year.

    Returns:
        State with ``alive`` and any rollover applied.
    """
    raise NotImplementedError

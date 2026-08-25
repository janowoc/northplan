"""Household state carried from one simulated month to the next.

Everything here is a frozen dataclass. The monthly step returns a new state
rather than mutating the old one, which keeps a path's history inspectable and
makes it impossible for a policy to write into state it should only read.

Shape convention: scalar fields (a person's birth year) are plain Python
values, shared across all paths. Anything that varies by path — every balance,
every realized return — is an array of shape ``(n_paths,)``. Per-person arrays
are ``(n_persons, n_paths)`` and are indexed by position in
``Household.persons``.

Because the timestep is a month and the tax year is a year, some of this state
exists purely to bridge the two: year-to-date income accumulates over twelve
steps before a single assessment consumes it, and the balance that assessment
produces sits in the ledger until the filing month. See :class:`TaxLedger`.

Field lists here are deliberately minimal. They grow as the modules that need
them land, in the order given in the README's build order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Person:
    """One member of the household.

    A household is a list of these from day one, even when only one person is
    modelled, because pension splitting, survivor benefits, the RRIF spousal
    rollover, OAS ceasing at first death, and two mortality timelines all
    require the second slot to exist.

    Attributes:
        person_id: Stable identifier, unique within the household.
        birth_year: Calendar year of birth.
        birth_month: Month of birth, ``1..12``. Required, not optional: on a
            monthly timeline the month a person turns 65 decides the month
            their pension can start, and CPP's adjustment is defined per month
            away from 65. A birth year alone cannot answer that.
    """

    person_id: str
    birth_year: int
    birth_month: int

    def age_months(self, year: int, month: int) -> int:
        """Age in whole months at the start of ``(year, month)``.

        Delegates to :mod:`engine.core.timeline`. Age is derived from the birth
        date on every call and never stored, so the two cannot drift apart.
        """
        raise NotImplementedError

    def age_years(self, year: int, month: int) -> int:
        """Age in whole years at the start of ``(year, month)``.

        Age on the birthday. Rules that key off age at the *start* or *end* of
        a calendar year are different quantities and have their own helpers in
        :mod:`engine.core.timeline`; a call site must say which it means.
        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Beneficiary:
    """An RESP beneficiary.

    Tracked individually and never pooled: grant room, the lifetime
    contribution limit, and the withdrawal window are all per-beneficiary and
    do not aggregate across children.

    Attributes:
        beneficiary_id: Stable identifier, unique within the household.
        birth_year: Calendar year of birth; drives grant eligibility and the
            year post-secondary withdrawals may begin.
        birth_month: Month of birth, ``1..12``. Enrolment starts in a month,
            not on 1 January, and the EAP cap applies to the first months of
            enrolment.
    """

    beneficiary_id: str
    birth_year: int
    birth_month: int


@dataclass(frozen=True, slots=True)
class Household:
    """The people being simulated.

    Attributes:
        persons: One or two adults. Ordered; order is stable across the run and
            is what per-person arrays are indexed by.
        beneficiaries: RESP beneficiaries, possibly empty.
        province: Two-letter province code selecting the provincial parameter
            file, e.g. ``"ab"``.
    """

    persons: tuple[Person, ...]
    beneficiaries: tuple[Beneficiary, ...]
    province: str

    def person(self, person_id: str) -> Person:
        """Return the person with that id.

        Raises:
            KeyError: If no person in the household has that id.
        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PersonIncome:
    """One person's income components, accumulated year to date.

    Not a month's income: these are running totals for the current calendar
    year, added to by each monthly step and consumed once by the December
    assessment. January resets them. A field read mid-year is the income
    *so far*, which is exactly what a policy is entitled to know and exactly
    what it must not mistake for the year's total.

    Kept as components rather than a single total because the tax treatment
    differs: only some of it is eligible for pension income splitting, only
    some counts toward the OAS recovery tax, and dividends and capital gains
    enter taxable income at their own inclusion rates.

    All fields are real dollars, shape ``(n_paths,)``.
    """

    person_id: str
    employment: NDArray[np.float64]
    registered_withdrawals: NDArray[np.float64]
    pension: NDArray[np.float64]
    government_benefits: NDArray[np.float64]
    taxable_investment: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HouseholdIncome:
    """Year-to-date income components for every person in the household."""

    persons: tuple[PersonIncome, ...]


@dataclass(frozen=True, slots=True)
class TaxLedger:
    """What the monthly loop owes, has paid, and will be assessed on.

    This class exists because the timestep is a month and tax is annual. On an
    annual timestep, income and tax coincide; on a monthly one they do not, and
    three separate quantities have to be carried:

    - what has accrued so far this year (``ytd_income``),
    - what has already been handed over (``remitted_ytd``),
    - what was assessed for the year that closed and has not yet been paid
      (``balance_owing``).

    Confusing the first with the third produces a household that pays its tax
    in the year it earned the income, which is a full year early and quietly
    flatters every decumulation path.

    Attributes:
        ytd_income: Income accumulated so far in ``year``, per person. Reset in
            January.
        remitted_ytd: Tax already paid during ``year`` through withholding on
            employment and pension income and through instalments, per person,
            ``(n_persons, n_paths)``. Reduces the balance the assessment
            leaves owing.
        balance_owing: Assessed tax for the *prior* year not yet paid, per
            person, ``(n_persons, n_paths)``. Created by the December
            assessment, discharged in the filing month, zero in between only if
            withholding happened to be exact.
        prior_year_net_income: Net income per person for each recent calendar
            year, keyed by year, ``(n_persons, n_paths)``. The OAS recovery tax
            and GIS are assessed against an earlier year's figure, so the
            figure has to survive past the year that produced it.
    """

    ytd_income: HouseholdIncome
    remitted_ytd: NDArray[np.float64]
    balance_owing: NDArray[np.float64]
    prior_year_net_income: dict[int, NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class HouseholdState:
    """The full state of the household at the boundary between two months.

    Returned by the monthly step and fed straight back into it. A policy
    function receives this and may read all of it — everything in here is
    knowable at that simulated moment. It may not read anything else.

    Attributes:
        year: The calendar year this state is the opening position for.
        month: The month this state is the opening position for, ``1..12``.
            January is 1. Every rule that fires in a particular month reads
            this rather than counting steps.
        household: The people. Constant across the run except for deaths.
        alive: Per-person survival flag, ``(n_persons, n_paths)``. Once false
            it stays false. Resolved monthly, so a death lands in the month it
            happens and the benefits that stop, stop from that month.
        accounts: Account balances by person and account type. Real dollars.
            Also carries the annual quantities fixed in January and drawn down
            over the year — the RRIF minimum still to be taken, the LIF maximum
            still available, contribution room — and the prior year's TFSA
            withdrawals awaiting their January restoration.
        tax: The bridge between the monthly loop and the annual assessment.
        history: Accumulated records for reporting, appended at each year end.
            Append-only.
    """

    year: int
    month: int
    household: Household
    alive: NDArray[np.bool_]
    accounts: dict[str, Any]
    tax: TaxLedger
    history: tuple[dict[str, Any], ...]

"""Household state carried from one simulated year to the next.

Everything here is a frozen dataclass. The annual step returns a new state
rather than mutating the old one, which keeps a path's history inspectable and
makes it impossible for a policy to write into state it should only read.

Shape convention: scalar fields (a person's birth year) are plain Python
values, shared across all paths. Anything that varies by path — every balance,
every realized return — is an array of shape ``(n_paths,)``.

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
        birth_year: Calendar year of birth. Age within a simulated year is
            derived from this, never stored, so the two cannot drift apart.
    """

    person_id: str
    birth_year: int

    def age_in(self, year: int) -> int:
        """Age in whole years at the end of calendar ``year``.

        The convention used everywhere: age at year end, not on the birthday.
        Benefit and RRIF rules that key off age at the *start* of a year must
        say so explicitly at their call site.
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
    """

    beneficiary_id: str
    birth_year: int


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
    """One person's income components for one simulated year.

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
    """Income components for every person in the household, for one year."""

    persons: tuple[PersonIncome, ...]


@dataclass(frozen=True, slots=True)
class HouseholdState:
    """The full state of the household at the boundary between two years.

    Returned by the annual step and fed straight back into it. A policy
    function receives this and may read all of it — everything in here is
    knowable at that simulated moment. It may not read anything else.

    Attributes:
        year: The calendar year this state is the opening position for.
        household: The people. Constant across the run except for deaths.
        alive: Per-person survival flag, ``(n_persons, n_paths)``. Once false
            it stays false.
        accounts: Account balances by person and account type. Real dollars.
        history: Accumulated per-year records for reporting. Append-only.
    """

    year: int
    household: Household
    alive: NDArray[np.bool_]
    accounts: dict[str, Any]
    history: tuple[dict[str, Any], ...]

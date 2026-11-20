# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Refusing a locked-in account the engine cannot carry through a run.

Two error classes, both :class:`~engine.scenario.load.ScenarioError`
subclasses. :func:`check_locked_in` raises on the first failure, in this order:

1. **A couple's jurisdictions** (:class:`LockedInJurisdictionError`), for a
   household of two: one person's LIRA holds a balance and names a
   jurisdiction other than the one the other person states for a LIRA or a
   LIF. Two jurisdictions do not merge, at a first death or at the survivor's
   later conversion to a LIF (``docs/limitations.md`` L47). Read from the
   scenario alone, in both directions, and applied even where the LIRA would
   be converted before any death.
2. **An unfiled jurisdiction** (:class:`LockedInJurisdictionError`), per person
   in household order, LIRA then LIF: a stated jurisdiction, whatever the
   balance, with no parameter file in the year (L3).
3. **An age** (:class:`LockedInAgeError`), in the same pass as 2 for that
   person: a stated LIF jurisdiction, whatever the balance, for someone below
   the jurisdiction's unlocking age on 1 January of ``start_year``
   (``docs/limitations.md`` L62). The stated jurisdiction alone opens the LIF.
   The committed parameter files pin the maximum table's first row to the
   unlocking age
   (``tests/params/test_param_file_structure.py::test_lif_tables_start_at_the_unlocking_age``),
   so for those files the refusal closes the age at which
   :func:`engine.accounts.lif.maximum_withdrawal` raises.

A parameter year that does not match ``scenario.start_year`` is a caller
mistake and raises ``ValueError``, as in
:func:`engine.scenario.lifespan.check_lifespan`.

``prepare_run``, through :func:`engine.mc.prepare.check_scenario`, calls
:func:`check_locked_in` once, on the grid-expanded scenario, right after
:func:`~engine.scenario.lifespan.check_lifespan`.
"""

from __future__ import annotations

from engine.accounts.lira import withdrawals_permitted
from engine.core.timeline import age_at_start_of_year
from engine.params.loader import ParamFileMissingError, ParamYear
from engine.scenario.load import ScenarioError
from engine.scenario.schema import Scenario

__all__ = ["LockedInAgeError", "LockedInJurisdictionError", "check_locked_in"]


class LockedInAgeError(ScenarioError):
    """A LIF stated for someone below the unlocking age on 1 January of the start year.

    Raised by :func:`check_locked_in`. The message names the person, their age,
    the unlocking age and the parameter file's source.
    """


class LockedInJurisdictionError(ScenarioError):
    """A locked-in account whose jurisdiction the run cannot carry.

    Raised by :func:`check_locked_in`: either a couple in which one person's
    LIRA holds a balance and names a jurisdiction other than the one the other
    person states for a LIRA or a LIF, or a jurisdiction with no parameter
    file. The message names the persons, the accounts and the codes.
    """


def check_locked_in(scenario: Scenario, params: ParamYear) -> None:
    """Refuse a locked-in account ``params`` cannot carry for ``scenario``.

    Checks run in the order the module docstring gives and this raises on the
    first failure, never collecting more than one.

    Args:
        scenario: The validated scenario.
        params: The parameter year matching ``scenario.start_year``.

    Raises:
        ValueError: If ``params.year != scenario.start_year``.
        LockedInJurisdictionError: On a couple in which one person's LIRA holds a
            balance and names a jurisdiction other than the one the other person
            states for a LIRA or a LIF, or on a stated jurisdiction with no
            parameter file. For the latter the ``ParamFileMissingError`` is its
            ``__cause__``.
        LockedInAgeError: On a LIF stated for someone below the unlocking age.
    """
    if params.year != scenario.start_year:
        raise ValueError(
            f"check_locked_in: params is for tax year {params.year} but "
            f"scenario.start_year is {scenario.start_year}. Load "
            f"params/{scenario.start_year}/ for this scenario."
        )

    persons = scenario.household.persons

    if len(persons) == 2:
        for i, j in ((0, 1), (1, 0)):
            holder = persons[i]
            other = persons[j]
            lira = holder.accounts.lira
            if not lira.balance > 0:
                continue
            for kind, account in (("lira", other.accounts.lira), ("lif", other.accounts.lif)):
                if account.jurisdiction is not None and account.jurisdiction != lira.jurisdiction:
                    raise LockedInJurisdictionError(
                        f"household.persons: {holder.id!r} holds a LIRA balance "
                        f"({lira.balance}) registered in {lira.jurisdiction!r}, and "
                        f"{other.id!r} states a {kind.upper()} registered in "
                        f"{account.jurisdiction!r}. Two jurisdictions do not merge, at a "
                        f"first death or at the survivor's later conversion to a LIF, so the "
                        f"couple is refused, even where the LIRA would be converted to a LIF "
                        f"before any death "
                        f"(docs/limitations.md L47)."
                    )

    for person in persons:
        for kind, account in (("lira", person.accounts.lira), ("lif", person.accounts.lif)):
            code = account.jurisdiction
            if code is None:
                continue
            try:
                jurisdiction = params.jurisdiction(code)
            except ParamFileMissingError as error:
                raise LockedInJurisdictionError(
                    f"household.persons: {person.id!r} states accounts.{kind}.jurisdiction "
                    f"{code!r}, which has no parameter file for tax year {params.year} in "
                    f"{params.root}. Available: {', '.join(params.names())} "
                    f"(docs/limitations.md L3)."
                ) from error
            if kind == "lif":
                age = age_at_start_of_year(
                    person.birth_year, person.birth_month, scenario.start_year
                )
                if not withdrawals_permitted(age, jurisdiction):
                    unlocking_age = jurisdiction.integer("lif.unlocking_age_years")
                    raise LockedInAgeError(
                        f"household.persons: {person.id!r} states a LIF "
                        f"(accounts.lif.jurisdiction {code!r}) but is {age} on 1 January "
                        f"{scenario.start_year}, below lif.unlocking_age_years "
                        f"({unlocking_age}) in {jurisdiction.source}. A zero balance "
                        f"still counts: the stated jurisdiction alone opens the LIF "
                        f"(docs/limitations.md L62)."
                    )

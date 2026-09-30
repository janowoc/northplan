# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Refusing a CPP, OAS or RRIF conversion age the scenario's parameter year does not allow.

Four refusals, all raised as :class:`StartAgeNotAllowedError`:

1. An elected start age (``cpp_start_age_years`` or ``oas_start_age_years``)
   outside ``[start_age.earliest_months, start_age.latest_months]`` for that
   benefit, for every election written, including one for a person whose
   benefit is already in pay.
2. An election, for a person whose benefit is *not* in pay, below that
   person's age in whole years on 1 January of ``start_year``. An election
   equal to that age is accepted and is first paid at the opening, except
   an OAS election whose age in months is reached in the opening month,
   which is first paid the month after (L54).
3. CPP already ``in_pay_monthly`` for a person younger than CPP's
   ``start_age.earliest_months`` at the run's opening, or OAS already
   ``in_pay_monthly`` for a person not older than OAS's
   ``start_age.earliest_months``: CPP is payable from the month that age is
   reached, OAS from the month after (L54).
4. A ``rrif_conversion.age_years`` election above ``rrif``'s
   ``conversion_age_years``: an RRSP must already be converted to a RRIF by
   then, so the election could never take effect. An election equal to
   ``conversion_age_years`` is accepted, but the statutory conversion that
   year moves the whole RRSP, whatever the election's ``fraction``
   (``docs/limitations.md`` L26). An election below every person's age on
   31 December of ``start_year`` is not refused; it never takes effect.

A parameter year that does not match ``scenario.start_year`` is a caller
mistake rather than a bad scenario, and raises ``ValueError`` instead.

:func:`engine.mc.prepare.prepare_run` calls :func:`check_start_ages` once, on
the grid-expanded scenario, before anything is built.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from engine.core.timeline import MONTHS_PER_YEAR, age_in_months
from engine.params.loader import ParamYear
from engine.scenario.load import ScenarioError
from engine.scenario.schema import Person, PolicySpec, Scenario

__all__ = ["StartAgeNotAllowedError", "check_start_ages"]


class StartAgeNotAllowedError(ScenarioError):
    """A CPP, OAS, or RRIF conversion age the parameter year does not allow.

    Raised by :func:`check_start_ages`. For a CPP or OAS election, it names
    the person, the policy, the benefit, the elected age, and the window or
    the age at the opening it was checked against; for the RRIF conversion
    election, the policy, the elected age, ``conversion_age_years``, and the
    source it was checked against; for CPP or OAS already in pay, the person,
    their age in months at the opening, and the earliest start age it was
    checked against.
    """


def _format_number(number: float) -> str:
    """Render a parameter number without a spurious trailing ``.0``."""
    return str(int(number)) if number.is_integer() else repr(number)


def _age_at_open(person: Person, start_year: int) -> int:
    return age_in_months(person.birth_year, person.birth_month, start_year, 1)


def _check_election(
    *,
    policy: PolicySpec,
    person: Person,
    field_name: str,
    benefit_name: str,
    elections: Mapping[str, int],
    earliest: float,
    latest: float,
    source: Path,
    in_pay: bool,
    start_year: int,
    age_at_open: int,
) -> None:
    """Case 1 then case 2 for one person's election of one benefit, on one policy.

    Case 2 refuses an election below the person's age in whole years at the
    run's opening; an election equal to that age is accepted and is first
    paid at the opening, except an OAS election whose age in months is
    reached in the opening month, which is first paid the month after (L54).
    Does nothing when ``person.id`` has no entry in ``elections`` — the
    schema already guarantees an entry for anyone whose benefit is not in
    pay, and a leftover entry for someone in pay is still checked by case 1,
    never case 2.
    """
    if person.id not in elections:
        return
    years = elections[person.id]
    months = years * MONTHS_PER_YEAR
    window = f"{_format_number(earliest)}-{_format_number(latest)} months"

    if not (earliest <= months <= latest):
        raise StartAgeNotAllowedError(
            f"policies[{policy.name!r}].elections.{field_name}[{person.id!r}]: "
            f"{benefit_name} start age {years} ({months} months) is outside "
            f"the window {window} set by start_age.earliest_months and "
            f"start_age.latest_months in {source}."
        )

    if in_pay:
        return

    age_at_open_years = age_at_open // MONTHS_PER_YEAR
    if years < age_at_open_years:
        raise StartAgeNotAllowedError(
            f"policies[{policy.name!r}].elections.{field_name}[{person.id!r}]: "
            f"{benefit_name} start age {years} ({months} months) is below "
            f"the age of {person.id!r} at the run's opening on 1 January "
            f"{start_year}, {age_at_open_years} years ({age_at_open} months) "
            f"(window {window} from start_age.earliest_months and "
            f"start_age.latest_months in {source}; docs/limitations.md L54)."
        )


def check_start_ages(scenario: Scenario, params: ParamYear) -> None:
    """Refuse a CPP, OAS or RRIF conversion age ``params`` does not allow for ``scenario``.

    Checks run in a fixed order and this raises on the first failure, never
    collecting more than one:

    1. For each person in household order, CPP then OAS already in pay
       (case 3).
    2. For each policy in file order, and each person in household order: CPP
       before OAS, and for each benefit the statutory window (case 1) before
       the past-start check (case 2): an election below the person's age in
       whole years at the opening is refused, and an election equal to that
       age is accepted and is first paid at the opening, except an OAS
       election reached in the opening month, which is first paid the month
       after (L54).
    3. After the whole loop above: for each policy in file order, a
       ``rrif_conversion.age_years`` above ``conversion_age_years`` (case 4).

    Grid values (``scenario.grid``) are not checked; only the elections a
    policy states directly. :func:`engine.mc.prepare.prepare_run` passes the
    grid-expanded scenario, so grid values are checked there.

    Args:
        scenario: The validated scenario.
        params: The parameter year matching ``scenario.start_year``.

    Raises:
        ValueError: If ``params.year != scenario.start_year``.
        StartAgeNotAllowedError: On the first election, or person with CPP or
            OAS in pay, the rules do not allow.
    """
    if params.year != scenario.start_year:
        raise ValueError(
            f"check_start_ages: params is for tax year {params.year} but "
            f"scenario.start_year is {scenario.start_year}. Load "
            f"params/{scenario.start_year}/ for this scenario."
        )

    cpp_earliest = params.cpp.number("start_age.earliest_months")
    cpp_latest = params.cpp.number("start_age.latest_months")
    oas_earliest = params.oas.number("start_age.earliest_months")
    oas_latest = params.oas.number("start_age.latest_months")

    for person in scenario.household.persons:
        age_at_open = _age_at_open(person, scenario.start_year)
        if person.cpp.in_pay_monthly is not None and age_at_open < cpp_earliest:
            raise StartAgeNotAllowedError(
                f"household.persons: {person.id!r} has CPP in pay "
                f"(cpp.in_pay_monthly) but is {age_at_open} months old when "
                f"the run opens on 1 January {scenario.start_year}, younger "
                f"than start_age.earliest_months "
                f"({_format_number(cpp_earliest)}) in {params.cpp.source}; "
                f"CPP is first paid in the month that age is reached."
            )
        if person.oas.in_pay_monthly is not None and age_at_open <= oas_earliest:
            raise StartAgeNotAllowedError(
                f"household.persons: {person.id!r} has OAS in pay "
                f"(oas.in_pay_monthly) but is {age_at_open} months old when "
                f"the run opens on 1 January {scenario.start_year}, not older "
                f"than start_age.earliest_months "
                f"({_format_number(oas_earliest)}) in {params.oas.source}; "
                f"OAS is first paid in the month after that age."
            )

    for policy in scenario.policies:
        for person in scenario.household.persons:
            age_at_open = _age_at_open(person, scenario.start_year)
            _check_election(
                policy=policy,
                person=person,
                field_name="cpp_start_age_years",
                benefit_name="CPP",
                elections=policy.elections.cpp_start_age_years,
                earliest=cpp_earliest,
                latest=cpp_latest,
                source=params.cpp.source,
                in_pay=person.cpp.in_pay_monthly is not None,
                start_year=scenario.start_year,
                age_at_open=age_at_open,
            )
            _check_election(
                policy=policy,
                person=person,
                field_name="oas_start_age_years",
                benefit_name="OAS",
                elections=policy.elections.oas_start_age_years,
                earliest=oas_earliest,
                latest=oas_latest,
                source=params.oas.source,
                in_pay=person.oas.in_pay_monthly is not None,
                start_year=scenario.start_year,
                age_at_open=age_at_open,
            )

    conversion_age_years = params.rrif.number("conversion_age_years")
    for policy in scenario.policies:
        elected = policy.elections.rrif_conversion.age_years
        if elected > conversion_age_years:
            raise StartAgeNotAllowedError(
                f"policies[{policy.name!r}].elections.rrif_conversion.age_years: "
                f"elected age {elected} is above conversion_age_years "
                f"({_format_number(conversion_age_years)}) in {params.rrif.source}; "
                "an RRSP must already be converted to a RRIF by then, so this "
                "election could never take effect."
            )

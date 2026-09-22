# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Refusing a scenario the life table cannot represent.

Two refusals, both raised as :class:`LifespanNotRepresentableError`:

1. A person whose age in whole years on 1 January of ``scenario.start_year``
   already exceeds ``mortality.terminal_age_years``. The model has no
   mortality data past the terminal age, so
   :func:`engine.core.mortality.survival_curve` would answer with the
   degenerate length-1 curve ``[0.0]`` and kill them silently at month zero
   (``docs/limitations.md`` L10) — refusing is better than a one-month run
   nobody asked for. Checked even when another person in the household keeps
   the run alive: a person past the terminal age beside one of ordinary age
   is a data error, not a scenario.
2. Only once every person clears check 1: a household whose longest
   :func:`~engine.core.mortality.months_to_terminal` across its persons is
   below twelve. Month index 11 is December of the start year, and a
   ``SimulationResult`` row exists only for a year whose December close ran
   (issue #19); a run that never reaches one produces an empty first axis
   that every objective in ``engine/optimize/objective.py`` would index into
   or reduce over.

A parameter year that does not match ``scenario.start_year`` is a caller
mistake rather than a bad scenario, and raises ``ValueError`` instead, exactly
like :func:`engine.scenario.start_ages.check_start_ages`.

Whatever opens a run calls :func:`check_lifespan` alongside
:func:`~engine.scenario.start_ages.check_start_ages`.
"""

from __future__ import annotations

from engine.core.mortality import months_to_terminal
from engine.core.timeline import MONTHS_PER_YEAR, age_at_start_of_year
from engine.params.loader import ParamYear
from engine.scenario.load import ScenarioError
from engine.scenario.schema import Scenario

__all__ = ["LifespanNotRepresentableError", "check_lifespan"]


class LifespanNotRepresentableError(ScenarioError):
    """A person, or the household as a whole, outlives what the life table can represent.

    Raised by :func:`check_lifespan`. Check 1's message names the person,
    their age at the opening, the terminal age, and the mortality parameter
    file's source. Check 2's message additionally names the household's
    longest month count and the twelve months it needed to reach a first
    December close.
    """


def check_lifespan(scenario: Scenario, params: ParamYear) -> None:
    """Refuse a scenario the ``mortality`` parameter year cannot represent.

    Checks run in a fixed order and this raises on the first failure, never
    collecting more than one:

    1. For each person in household order: refused if their age in whole
       years on 1 January of ``scenario.start_year`` exceeds
       ``mortality.terminal_age_years``. The more specific diagnosis, so it
       runs first.
    2. Only once every person clears check 1: refused if the household's
       longest :func:`~engine.core.mortality.months_to_terminal` across its
       persons is below twelve, i.e. the run cannot reach its first December
       close.

    Args:
        scenario: The validated scenario.
        params: The parameter year matching ``scenario.start_year``.

    Raises:
        ValueError: If ``params.year != scenario.start_year``.
        LifespanNotRepresentableError: On the first person, or household, the
            table cannot represent.
    """
    if params.year != scenario.start_year:
        raise ValueError(
            f"check_lifespan: params is for tax year {params.year} but "
            f"scenario.start_year is {scenario.start_year}. Load "
            f"params/{scenario.start_year}/ for this scenario."
        )

    mortality = params["mortality"]
    terminal_age = int(mortality.number("terminal_age_years"))

    for person in scenario.household.persons:
        age_at_open = age_at_start_of_year(
            person.birth_year, person.birth_month, scenario.start_year
        )
        if age_at_open > terminal_age:
            raise LifespanNotRepresentableError(
                f"household.persons: {person.id!r} is {age_at_open} years old "
                f"on 1 January {scenario.start_year}, older than "
                f"terminal_age_years ({terminal_age}) in {mortality.source}. "
                "The model has no mortality data past the terminal age, so "
                "this person cannot be simulated."
            )

    longest = max(
        months_to_terminal(
            person.birth_year,
            person.birth_month,
            person.sex,
            scenario.start_year,
            mortality,
        )
        for person in scenario.household.persons
    )
    # Month index 11 is December of the start year, since the run opens on 1
    # January (L4) and December is the twelfth month — so a curve needs at
    # least MONTHS_PER_YEAR months to reach the household's first December
    # close. The two coincide, not two independent quantities.
    if longest < MONTHS_PER_YEAR:
        raise LifespanNotRepresentableError(
            f"household: every person in the household dies, per "
            f"terminal_age_years ({terminal_age}) in {mortality.source}, "
            f"within {longest} month(s) of the run's opening on 1 January "
            f"{scenario.start_year} — short of the {MONTHS_PER_YEAR} months "
            f"needed to reach the household's first December close. This "
            "scenario would produce a SimulationResult with zero year rows."
        )

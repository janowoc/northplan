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
   twelve or fewer. A person's death month index is at most that count less
   one, and being alive at the run's first December close (month index 11)
   needs a death month index above 11, so no one in such a household can be
   alive at that close. The run would still reach the close, since the draws
   are sized to whole years (:func:`engine.core.build.build_draws`), but its
   one row would describe a household already gone; like check 1, that is
   refused as a data error rather than simulated.

A parameter year that does not match ``scenario.start_year`` is a caller
mistake rather than a bad scenario, and raises ``ValueError`` instead, exactly
like :func:`engine.scenario.start_ages.check_start_ages`.

:func:`engine.mc.prepare.prepare_run` calls :func:`check_lifespan` once, on
the grid-expanded scenario, right after
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
    file's source. Check 2's message additionally names the household's longest
    month count and the twelve months someone must outlive to be alive at the
    run's first December close.
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
       persons is twelve or fewer, i.e. no one in it can be alive at the run's
       first December close.

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
    # January (L4). A death drawn for month index k means not alive at the
    # opening of month k, and death_month_index is at most longest - 1, so
    # someone can be alive at the first December close only if longest exceeds
    # MONTHS_PER_YEAR.
    if longest <= MONTHS_PER_YEAR:
        raise LifespanNotRepresentableError(
            f"household: no one in the household can be alive at the run's first "
            f"December close. Per terminal_age_years ({terminal_age}) in "
            f"{mortality.source}, every person dies within {longest} month(s) of the "
            f"run's opening on 1 January {scenario.start_year}, and someone must "
            f"outlive {MONTHS_PER_YEAR} months to be alive at that close. Like a "
            "person past the terminal age, that is refused rather than simulated."
        )

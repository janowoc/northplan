# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Refusing a scenario the 2026 life table cannot represent.

Every age and birth date below is derived from ``params/2026/mortality.yaml``
via ``terminal_age_years``, never typed in as a bare literal. Only small,
meaningless offsets ("one year older") are literals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from engine.core.mortality import months_to_terminal
from engine.core.timeline import MONTHS_PER_YEAR
from engine.params.loader import ParamYear, load_year
from engine.scenario import LifespanNotRepresentableError, Scenario, check_lifespan

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "scenarios" / "example.yaml"


def example_values() -> dict[str, Any]:
    """The example scenario, parsed, as a fresh mutable structure each call."""
    values = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    assert isinstance(values, dict)
    return values


def _person(values: dict[str, Any]) -> dict[str, Any]:
    """The example's one adult, ``a``."""
    return values["household"]["persons"][0]


def _clear_db_pensions(values: dict[str, Any]) -> None:
    """Drop the example's DB pension.

    Dropped as a precaution when a birth date moves, so no pension-date
    validator can refuse the scenario before ``check_lifespan`` runs.
    """
    _person(values)["db_pensions"] = []


def _birth_for_age_at_open(start_year: int, age_months: int) -> tuple[int, int]:
    """A ``(birth_year, birth_month)`` giving exactly ``age_months`` on 1 January of ``start_year``.

    Inverts ``engine.core.timeline.age_in_months`` for a fixed ``start_year``
    and ``month=1``. Exact integer arithmetic: ``years_back`` is the ceiling
    of ``age_months / MONTHS_PER_YEAR``, chosen so the remaining month count
    always lands in ``1..12``.
    """
    years_back = -(-age_months // MONTHS_PER_YEAR)  # ceiling division
    birth_month = 1 - age_months + MONTHS_PER_YEAR * years_back
    assert 1 <= birth_month <= MONTHS_PER_YEAR
    return start_year - years_back, birth_month


def _append_minimal_person(
    values: dict[str, Any], person_id: str, birth_year: int, birth_month: int
) -> None:
    """Append a second, minimal adult at ``(birth_year, birth_month)``.

    Given ``in_pay_monthly`` for both CPP and OAS so that no policy election
    is required for them (``Scenario._check_every_person_has_a_start_age``
    only demands one for a person not already in pay), and no accounts or
    ``db_pensions`` so nothing else about them can refuse the scenario first.
    """
    values["household"]["persons"].append(
        {
            "id": person_id,
            "birth_year": birth_year,
            "birth_month": birth_month,
            "sex": "f",
            "cpp": {"in_pay_monthly": 800.0},
            "oas": {"in_pay_monthly": 500.0},
            "prior_year_net_income": 0,
            "net_income_two_years_prior": 0,
        }
    )


@pytest.fixture(scope="module")
def params() -> ParamYear:
    return load_year(2026)


@pytest.fixture(scope="module")
def start_year() -> int:
    return example_values()["start_year"]


@pytest.fixture(scope="module")
def terminal_age_years(params: ParamYear) -> int:
    return int(params["mortality"].number("terminal_age_years"))


# --- The example scenario itself --------------------------------------------


def test_the_committed_example_is_accepted(params: ParamYear) -> None:
    scenario = Scenario.model_validate(example_values())
    check_lifespan(scenario, params)  # must not raise


# --- Check 1: a person already past the terminal age ------------------------


def test_person_one_year_past_terminal_age_is_refused_by_check_one(
    params: ParamYear, start_year: int, terminal_age_years: int
) -> None:
    age_months = (terminal_age_years + 1) * MONTHS_PER_YEAR
    birth_year, birth_month = _birth_for_age_at_open(start_year, age_months)

    values = example_values()
    _person(values)["birth_year"] = birth_year
    _person(values)["birth_month"] = birth_month
    _clear_db_pensions(values)
    scenario = Scenario.model_validate(values)

    with pytest.raises(LifespanNotRepresentableError) as excinfo:
        check_lifespan(scenario, params)

    message = str(excinfo.value)
    assert "'a'" in message
    assert str(terminal_age_years + 1) in message  # the person's age at the opening
    assert str(terminal_age_years) in message  # terminal_age_years itself
    assert "older than" in message


def test_person_past_terminal_age_beside_ordinary_person_is_refused_by_check_one(
    params: ParamYear, start_year: int, terminal_age_years: int
) -> None:
    """Even when another person keeps the run alive, the old person is refused.

    ``a`` is left at the example's own, ordinary birth date; a second person,
    one year past the terminal age, is added beside them. Check 1 runs over
    every person in household order and names the old one, not ``a``.
    """
    age_months = (terminal_age_years + 1) * MONTHS_PER_YEAR
    old_birth_year, old_birth_month = _birth_for_age_at_open(start_year, age_months)

    values = example_values()
    _append_minimal_person(values, "old", old_birth_year, old_birth_month)
    scenario = Scenario.model_validate(values)

    with pytest.raises(LifespanNotRepresentableError) as excinfo:
        check_lifespan(scenario, params)

    message = str(excinfo.value)
    assert "'old'" in message
    assert str(terminal_age_years + 1) in message
    assert "'a'" not in message


# --- Check 2: no one alive at the household's first December close ----------


def test_person_exactly_terminal_age_january_birthday_is_refused_by_check_two(
    params: ParamYear, start_year: int, terminal_age_years: int
) -> None:
    """The pin between the two checks: exactly the terminal age, January birthday.

    Passes check 1 (age at open equals, does not exceed, terminal_age_years)
    but no one can be alive at the run's first December close, and check 2 is
    what refuses it.
    """
    age_months = terminal_age_years * MONTHS_PER_YEAR
    birth_year, birth_month = _birth_for_age_at_open(start_year, age_months)
    assert birth_month == 1

    values = example_values()
    _person(values)["birth_year"] = birth_year
    _person(values)["birth_month"] = birth_month
    _clear_db_pensions(values)
    scenario = Scenario.model_validate(values)

    with pytest.raises(LifespanNotRepresentableError) as excinfo:
        check_lifespan(scenario, params)

    message = str(excinfo.value)
    # This is check 2's message, not check 1's: no "older than" (check 1's
    # wording), and it names the December close and the 12 months someone
    # must outlive.
    assert "older than" not in message
    assert "December" in message
    assert "12" in message
    assert str(terminal_age_years) in message

    expected_months = months_to_terminal(
        birth_year, birth_month, _person(values)["sex"], start_year, params["mortality"]
    )
    assert f"within {expected_months} month(s)" in message


def test_person_turning_terminal_age_in_march_is_refused_by_check_two(
    params: ParamYear, start_year: int, terminal_age_years: int
) -> None:
    birth_year = start_year - terminal_age_years
    birth_month = 3

    values = example_values()
    _person(values)["birth_year"] = birth_year
    _person(values)["birth_month"] = birth_month
    _clear_db_pensions(values)
    scenario = Scenario.model_validate(values)

    with pytest.raises(LifespanNotRepresentableError) as excinfo:
        check_lifespan(scenario, params)

    message = str(excinfo.value)
    assert "older than" not in message
    assert "December" in message
    assert "12" in message

    expected_months = months_to_terminal(
        birth_year, birth_month, _person(values)["sex"], start_year, params["mortality"]
    )
    assert f"within {expected_months} month(s)" in message


def test_person_turning_terminal_age_in_december_is_accepted(
    params: ParamYear, start_year: int, terminal_age_years: int
) -> None:
    """The boundary check 2 tests, accepted side: a thirteen-month curve.

    A December birthday gives ``months_to_terminal == 13``: a death month index of 12 is
    possible, so someone can be alive at the first December close.
    """
    birth_year = start_year - terminal_age_years
    birth_month = MONTHS_PER_YEAR

    values = example_values()
    _person(values)["birth_year"] = birth_year
    _person(values)["birth_month"] = birth_month
    _clear_db_pensions(values)
    scenario = Scenario.model_validate(values)

    assert (
        months_to_terminal(
            birth_year, birth_month, _person(values)["sex"], start_year, params["mortality"]
        )
        == MONTHS_PER_YEAR + 1
    )
    check_lifespan(scenario, params)  # must not raise


def test_person_turning_terminal_age_in_november_is_refused_by_check_two(
    params: ParamYear, start_year: int, terminal_age_years: int
) -> None:
    """The boundary check 2 tests, refused side: a twelve-month curve.

    A November birthday gives ``months_to_terminal == 12``: the latest death month index is
    11, so no one can be alive at the first December close, and check 2 refuses it. The
    December birthday's thirteen-month curve, in
    ``test_person_turning_terminal_age_in_december_is_accepted``, is the accepted side.
    """
    birth_year = start_year - terminal_age_years
    birth_month = MONTHS_PER_YEAR - 1

    values = example_values()
    _person(values)["birth_year"] = birth_year
    _person(values)["birth_month"] = birth_month
    _clear_db_pensions(values)
    scenario = Scenario.model_validate(values)

    months = months_to_terminal(
        birth_year, birth_month, _person(values)["sex"], start_year, params["mortality"]
    )
    assert months == MONTHS_PER_YEAR

    with pytest.raises(LifespanNotRepresentableError) as excinfo:
        check_lifespan(scenario, params)

    message = str(excinfo.value)
    assert "within 12 month(s)" in message
    assert "older than" not in message


def test_short_lived_person_beside_ordinary_person_is_accepted_by_check_two(
    params: ParamYear, start_year: int, terminal_age_years: int
) -> None:
    """Check 2 takes the household's *longest* curve across persons, not any one person's.

    Person ``a`` is mutated to a March birthday in their terminal year (they
    turn the terminal age in March, a 4-month curve, as in
    ``test_person_turning_terminal_age_in_march_is_refused_by_check_two``).
    With a March birthday the person is still a year below the terminal age
    on 1 January, so check 1 passes and execution reaches check 2. A second,
    ordinary-age person is appended beside them, at the example's own birth
    date for ``a`` before it was mutated. Under ``max`` the household's
    longest curve is the ordinary person's several hundred months and the
    scenario is accepted; under ``min`` it would be 4 and the scenario would
    be wrongly refused.
    """
    values = example_values()
    ordinary_birth_year = _person(values)["birth_year"]
    ordinary_birth_month = _person(values)["birth_month"]
    short_sex = _person(values)["sex"]

    birth_year = start_year - terminal_age_years
    birth_month = 3
    _person(values)["birth_year"] = birth_year
    _person(values)["birth_month"] = birth_month
    _clear_db_pensions(values)
    _append_minimal_person(values, "ordinary", ordinary_birth_year, ordinary_birth_month)
    ordinary_sex = values["household"]["persons"][-1]["sex"]
    scenario = Scenario.model_validate(values)

    short_months = months_to_terminal(
        birth_year, birth_month, short_sex, start_year, params["mortality"]
    )
    ordinary_months = months_to_terminal(
        ordinary_birth_year, ordinary_birth_month, ordinary_sex, start_year, params["mortality"]
    )
    assert short_months < MONTHS_PER_YEAR
    assert ordinary_months > MONTHS_PER_YEAR

    check_lifespan(scenario, params)  # must not raise


def test_ordinary_person_first_short_lived_person_second_is_accepted_by_check_two(
    params: ParamYear, start_year: int, terminal_age_years: int
) -> None:
    """Check 2 must look at every person, not just the last one in household order.

    The mirror of the pair above: ``a`` is left at the example's own, ordinary
    birth date, and a second person, turning the terminal age in March
    (a 4-month curve, as in
    ``test_person_turning_terminal_age_in_march_is_refused_by_check_two``), is
    appended after them. The household's longest curve is still the ordinary
    person's several hundred months, so the correct answer is acceptance —
    but a collector restricted to ``persons[-1:]`` would see only the short
    person's 4-month curve and wrongly refuse. The companion case,
    ``test_short_lived_person_beside_ordinary_person_is_accepted_by_check_two``,
    puts the short-lived person first and the ordinary one last, so a
    ``persons[-1:]`` collector would wrongly see only the ordinary curve and
    happen to still accept; this one closes that gap.
    """
    values = example_values()
    ordinary_birth_year = _person(values)["birth_year"]
    ordinary_birth_month = _person(values)["birth_month"]
    ordinary_sex = _person(values)["sex"]

    short_birth_year = start_year - terminal_age_years
    short_birth_month = 3
    _append_minimal_person(values, "short", short_birth_year, short_birth_month)
    short_sex = values["household"]["persons"][-1]["sex"]
    scenario = Scenario.model_validate(values)

    ordinary_months = months_to_terminal(
        ordinary_birth_year, ordinary_birth_month, ordinary_sex, start_year, params["mortality"]
    )
    short_months = months_to_terminal(
        short_birth_year, short_birth_month, short_sex, start_year, params["mortality"]
    )
    assert short_months < MONTHS_PER_YEAR
    assert ordinary_months > MONTHS_PER_YEAR

    check_lifespan(scenario, params)  # must not raise


# --- Wrong parameter year ----------------------------------------------------


def test_wrong_parameter_year_is_a_value_error(params: ParamYear) -> None:
    scenario = Scenario.model_validate(example_values())
    mismatched = scenario.model_copy(update={"start_year": scenario.start_year + 1})

    with pytest.raises(ValueError, match=f"{scenario.start_year}") as excinfo:
        check_lifespan(mismatched, params)

    message = str(excinfo.value)
    assert str(scenario.start_year) in message
    assert str(scenario.start_year + 1) in message

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Refusing a CPP or OAS start age the 2026 parameter year does not allow.

Every window edge, birth date, and age used below is derived from
``params/2026/{cpp,oas}.yaml`` and ``MONTHS_PER_YEAR``, never typed in as a
bare 60/65/70. Only small, meaningless offsets ("two years older") are
literals.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from engine.core.timeline import MONTHS_PER_YEAR, age_at_end_of_year, age_in_months
from engine.params.loader import ParamYear, load_year
from engine.scenario import Scenario, StartAgeNotAllowedError, check_start_ages

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


def _policy(values: dict[str, Any], name: str = "taxable-first") -> dict[str, Any]:
    for policy in values["policies"]:
        if policy["name"] == name:
            return policy
    raise AssertionError(f"no policy named {name!r}")


def _clear_db_pensions(values: dict[str, Any]) -> None:
    """Drop the example's DB pension, required whenever a birth date changes.

    ``Person._check_bridges_do_not_end_before_they_start`` may otherwise
    refuse the scenario before ``check_start_ages`` ever runs.
    """
    _person(values)["db_pensions"] = []


def _safe_election_years(window: tuple[float, float], age_at_open_months: int) -> int:
    """A whole-year election for ``window`` that passes both case 1 and case 2.

    The smallest election at or above the person's age in whole years at the
    opening that still falls inside ``window``. Used to keep the *other*
    benefit's election out of the way while a test exercises one benefit's
    past-start check under a birth date it does not control for.
    """
    earliest, latest = window
    earliest_years = math.ceil(earliest / MONTHS_PER_YEAR)
    latest_years = math.floor(latest / MONTHS_PER_YEAR)
    years = max(earliest_years, age_at_open_months // MONTHS_PER_YEAR)
    assert years <= latest_years
    return years


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


@pytest.fixture(scope="module")
def params() -> ParamYear:
    return load_year(2026)


@pytest.fixture(scope="module")
def start_year() -> int:
    return example_values()["start_year"]


@pytest.fixture(scope="module")
def cpp_window(params: ParamYear) -> tuple[float, float]:
    return (
        params.cpp.number("start_age.earliest_months"),
        params.cpp.number("start_age.latest_months"),
    )


@pytest.fixture(scope="module")
def oas_window(params: ParamYear) -> tuple[float, float]:
    return (
        params.oas.number("start_age.earliest_months"),
        params.oas.number("start_age.latest_months"),
    )


# --- The example scenario itself --------------------------------------------


def test_the_committed_example_is_accepted(params: ParamYear) -> None:
    scenario = Scenario.model_validate(example_values())
    check_start_ages(scenario, params)  # must not raise


# --- Case 1: outside the statutory window -----------------------------------


@pytest.mark.parametrize(
    ("field", "benefit", "window_fixture", "direction"),
    [
        pytest.param("cpp_start_age_years", "CPP", "cpp_window", "below", id="cpp-below-window"),
        pytest.param("cpp_start_age_years", "CPP", "cpp_window", "above", id="cpp-above-window"),
        pytest.param("oas_start_age_years", "OAS", "oas_window", "below", id="oas-below-window"),
        pytest.param("oas_start_age_years", "OAS", "oas_window", "above", id="oas-above-window"),
    ],
)
def test_election_outside_the_window_is_refused(
    request: pytest.FixtureRequest,
    params: ParamYear,
    field: str,
    benefit: str,
    window_fixture: str,
    direction: str,
) -> None:
    earliest, latest = request.getfixturevalue(window_fixture)
    earliest_years = math.ceil(earliest / MONTHS_PER_YEAR)
    latest_years = math.floor(latest / MONTHS_PER_YEAR)
    years = earliest_years - 1 if direction == "below" else latest_years + 1

    values = example_values()
    _policy(values)["elections"][field]["a"] = years
    scenario = Scenario.model_validate(values)

    with pytest.raises(StartAgeNotAllowedError) as excinfo:
        check_start_ages(scenario, params)

    message = str(excinfo.value)
    assert "outside" in message
    assert "'a'" in message
    assert "'taxable-first'" in message
    assert benefit in message
    assert str(int(earliest)) in message
    assert str(int(latest)) in message


# --- Case 2: an election below the age at opening, in whole years ----------


def test_cpp_start_already_past_is_refused(params: ParamYear, cpp_window, start_year: int) -> None:
    earliest, latest = cpp_window
    earliest_years = math.ceil(earliest / MONTHS_PER_YEAR)
    election_years = earliest_years + 1  # one year above the earliest whole-year age
    age_at_open_years = election_years + 1  # older, so the election is already past
    assert election_years < age_at_open_years  # at least one whole year below

    values = example_values()
    _person(values)["birth_year"] = start_year - age_at_open_years
    _person(values)["birth_month"] = 1
    _clear_db_pensions(values)
    _policy(values)["elections"]["cpp_start_age_years"]["a"] = election_years
    scenario = Scenario.model_validate(values)

    with pytest.raises(StartAgeNotAllowedError) as excinfo:
        check_start_ages(scenario, params)

    message = str(excinfo.value)
    assert "is below the age of" in message
    assert "'a'" in message
    assert "'taxable-first'" in message
    assert "CPP" in message
    assert str(int(earliest)) in message
    assert str(int(latest)) in message
    assert str(age_at_open_years * MONTHS_PER_YEAR) in message


def test_oas_start_already_past_is_refused(params: ParamYear, oas_window, start_year: int) -> None:
    earliest, latest = oas_window
    earliest_years = math.ceil(earliest / MONTHS_PER_YEAR)
    election_years = earliest_years + 1  # one year above the earliest whole-year age
    age_at_open_years = election_years + 1  # older, so the election is already past
    assert election_years < age_at_open_years  # at least one whole year below

    values = example_values()
    _person(values)["birth_year"] = start_year - age_at_open_years
    _person(values)["birth_month"] = 1
    _clear_db_pensions(values)
    # Keep CPP valid for this new birth date so its check does not fire first.
    _policy(values)["elections"]["cpp_start_age_years"]["a"] = age_at_open_years
    _policy(values)["elections"]["oas_start_age_years"]["a"] = election_years
    scenario = Scenario.model_validate(values)

    with pytest.raises(StartAgeNotAllowedError) as excinfo:
        check_start_ages(scenario, params)

    message = str(excinfo.value)
    assert "is below the age of" in message
    assert "'a'" in message
    assert "'taxable-first'" in message
    assert "OAS" in message
    assert str(int(earliest)) in message
    assert str(int(latest)) in message
    assert str(age_at_open_years * MONTHS_PER_YEAR) in message


# --- Case 3: OAS already in pay, too young ----------------------------------


def test_oas_in_pay_below_earliest_start_age_is_refused(
    params: ParamYear, oas_window, start_year: int
) -> None:
    earliest, _latest = oas_window
    age_months = int(earliest) - 1
    birth_year, birth_month = _birth_for_age_at_open(start_year, age_months)

    values = example_values()
    _person(values)["birth_year"] = birth_year
    _person(values)["birth_month"] = birth_month
    _clear_db_pensions(values)
    _person(values)["oas"] = {"in_pay_monthly": 500.0}
    scenario = Scenario.model_validate(values)

    with pytest.raises(StartAgeNotAllowedError) as excinfo:
        check_start_ages(scenario, params)

    message = str(excinfo.value)
    assert "'a'" in message
    assert "OAS" in message
    assert "'taxable-first'" not in message
    assert str(int(earliest)) in message
    assert str(age_months) in message


def test_oas_in_pay_wins_over_an_out_of_window_cpp_election(
    params: ParamYear, cpp_window, oas_window, start_year: int
) -> None:
    """Case 3 runs over every person before any policy is examined, so it wins."""
    earliest, _latest = oas_window
    age_months = int(earliest) - 1
    birth_year, birth_month = _birth_for_age_at_open(start_year, age_months)
    cpp_earliest_years = math.ceil(cpp_window[0] / MONTHS_PER_YEAR)

    values = example_values()
    _person(values)["birth_year"] = birth_year
    _person(values)["birth_month"] = birth_month
    _clear_db_pensions(values)
    _person(values)["oas"] = {"in_pay_monthly": 500.0}
    _policy(values)["elections"]["cpp_start_age_years"]["a"] = cpp_earliest_years - 1
    scenario = Scenario.model_validate(values)

    with pytest.raises(StartAgeNotAllowedError) as excinfo:
        check_start_ages(scenario, params)

    message = str(excinfo.value)
    assert "oas.in_pay_monthly" in message
    assert "'taxable-first'" not in message


# --- Case 1 in a second policy only ------------------------------------------


def test_refusal_in_a_second_policy_only(params: ParamYear, cpp_window) -> None:
    earliest, _latest = cpp_window
    earliest_years = math.ceil(earliest / MONTHS_PER_YEAR)

    values = example_values()
    second = yaml.safe_load(yaml.safe_dump(_policy(values)))
    second["name"] = "taxable-first-2"
    second["elections"]["cpp_start_age_years"]["a"] = earliest_years - 1
    values["policies"].append(second)
    scenario = Scenario.model_validate(values)

    with pytest.raises(StartAgeNotAllowedError) as excinfo:
        check_start_ages(scenario, params)

    message = str(excinfo.value)
    assert "'taxable-first-2'" in message
    assert "'taxable-first'" not in message


# --- Wrong parameter year ----------------------------------------------------


def test_wrong_parameter_year_is_a_value_error(params: ParamYear) -> None:
    scenario = Scenario.model_validate(example_values())
    mismatched = scenario.model_copy(update={"start_year": scenario.start_year + 1})

    with pytest.raises(ValueError, match=f"{scenario.start_year}") as excinfo:
        check_start_ages(mismatched, params)

    message = str(excinfo.value)
    assert str(scenario.start_year) in message
    assert str(scenario.start_year + 1) in message


def test_wrong_parameter_year_wins_over_an_out_of_window_election(
    params: ParamYear, cpp_window
) -> None:
    """The year mismatch is checked before any election, so it is what raises."""
    earliest, _latest = cpp_window
    earliest_years = math.ceil(earliest / MONTHS_PER_YEAR)

    values = example_values()
    _policy(values)["elections"]["cpp_start_age_years"]["a"] = earliest_years - 1
    scenario = Scenario.model_validate(values)
    mismatched = scenario.model_copy(update={"start_year": scenario.start_year + 1})

    with pytest.raises(ValueError) as excinfo:
        check_start_ages(mismatched, params)

    message = str(excinfo.value)
    assert str(scenario.start_year) in message
    assert str(scenario.start_year + 1) in message


# --- Accepted cases -----------------------------------------------------------


def test_cpp_election_equal_to_age_at_opening_is_accepted(
    params: ParamYear, cpp_window, start_year: int
) -> None:
    earliest, _latest = cpp_window
    k = math.ceil(earliest / MONTHS_PER_YEAR) + 1

    values = example_values()
    _person(values)["birth_year"] = start_year - k
    _person(values)["birth_month"] = 1
    _clear_db_pensions(values)
    _policy(values)["elections"]["cpp_start_age_years"]["a"] = k
    scenario = Scenario.model_validate(values)

    check_start_ages(scenario, params)  # must not raise


def test_oas_election_equal_to_age_at_opening_is_accepted(
    params: ParamYear, oas_window, start_year: int
) -> None:
    earliest, _latest = oas_window
    m = math.ceil(earliest / MONTHS_PER_YEAR) + 1

    values = example_values()
    _person(values)["birth_year"] = start_year - m
    _person(values)["birth_month"] = 1
    _clear_db_pensions(values)
    _policy(values)["elections"]["cpp_start_age_years"]["a"] = m
    _policy(values)["elections"]["oas_start_age_years"]["a"] = m
    scenario = Scenario.model_validate(values)

    check_start_ages(scenario, params)  # must not raise


# --- Whole-year comparison: not born in January -----------------------------


def test_cpp_election_at_the_whole_year_floor_is_accepted(
    params: ParamYear, cpp_window, oas_window, start_year: int
) -> None:
    """``years < age_at_open // MONTHS_PER_YEAR`` compares whole years, not months.

    ``a`` is k years and 11 months old at the opening, not born in January, so
    an election of exactly ``k`` is one month short of ``a``'s age in months
    but equal to it in whole years, and is accepted.
    """
    earliest, _latest = cpp_window
    k = math.ceil(earliest / MONTHS_PER_YEAR) + 1
    age_at_open_months = k * MONTHS_PER_YEAR + 11
    birth_year, birth_month = _birth_for_age_at_open(start_year, age_at_open_months)

    values = example_values()
    _person(values)["birth_year"] = birth_year
    _person(values)["birth_month"] = birth_month
    _clear_db_pensions(values)
    _policy(values)["elections"]["cpp_start_age_years"]["a"] = k
    _policy(values)["elections"]["oas_start_age_years"]["a"] = _safe_election_years(
        oas_window, age_at_open_months
    )
    scenario = Scenario.model_validate(values)

    check_start_ages(scenario, params)  # must not raise


def test_cpp_election_one_year_below_the_whole_year_floor_is_refused(
    params: ParamYear, cpp_window, oas_window, start_year: int
) -> None:
    """One year below the election :func:`test_cpp_election_at_the_whole_year_floor_is_accepted` accepts."""
    earliest, latest = cpp_window
    k = math.ceil(earliest / MONTHS_PER_YEAR) + 1
    age_at_open_months = k * MONTHS_PER_YEAR + 11
    birth_year, birth_month = _birth_for_age_at_open(start_year, age_at_open_months)
    election_years = k - 1
    assert math.ceil(earliest / MONTHS_PER_YEAR) <= election_years  # still inside the window

    values = example_values()
    _person(values)["birth_year"] = birth_year
    _person(values)["birth_month"] = birth_month
    _clear_db_pensions(values)
    _policy(values)["elections"]["cpp_start_age_years"]["a"] = election_years
    _policy(values)["elections"]["oas_start_age_years"]["a"] = _safe_election_years(
        oas_window, age_at_open_months
    )
    scenario = Scenario.model_validate(values)

    with pytest.raises(StartAgeNotAllowedError) as excinfo:
        check_start_ages(scenario, params)

    message = str(excinfo.value)
    assert "is below the age of" in message
    assert "'a'" in message
    assert "CPP" in message
    assert str(int(earliest)) in message
    assert str(int(latest)) in message


def test_oas_election_at_the_whole_year_floor_is_accepted(
    params: ParamYear, cpp_window, oas_window, start_year: int
) -> None:
    """``years < age_at_open // MONTHS_PER_YEAR`` compares whole years, not months.

    ``a`` is k years and 11 months old at the opening, not born in January, so
    an election of exactly ``k`` is one month short of ``a``'s age in months
    but equal to it in whole years, and is accepted.
    """
    earliest, _latest = oas_window
    k = math.ceil(earliest / MONTHS_PER_YEAR) + 1
    age_at_open_months = k * MONTHS_PER_YEAR + 11
    birth_year, birth_month = _birth_for_age_at_open(start_year, age_at_open_months)

    values = example_values()
    _person(values)["birth_year"] = birth_year
    _person(values)["birth_month"] = birth_month
    _clear_db_pensions(values)
    _policy(values)["elections"]["cpp_start_age_years"]["a"] = _safe_election_years(
        cpp_window, age_at_open_months
    )
    _policy(values)["elections"]["oas_start_age_years"]["a"] = k
    scenario = Scenario.model_validate(values)

    check_start_ages(scenario, params)  # must not raise


def test_oas_election_one_year_below_the_whole_year_floor_is_refused(
    params: ParamYear, cpp_window, oas_window, start_year: int
) -> None:
    """One year below the election :func:`test_oas_election_at_the_whole_year_floor_is_accepted` accepts."""
    earliest, latest = oas_window
    k = math.ceil(earliest / MONTHS_PER_YEAR) + 1
    age_at_open_months = k * MONTHS_PER_YEAR + 11
    birth_year, birth_month = _birth_for_age_at_open(start_year, age_at_open_months)
    election_years = k - 1
    assert math.ceil(earliest / MONTHS_PER_YEAR) <= election_years  # still inside the window

    values = example_values()
    _person(values)["birth_year"] = birth_year
    _person(values)["birth_month"] = birth_month
    _clear_db_pensions(values)
    _policy(values)["elections"]["cpp_start_age_years"]["a"] = _safe_election_years(
        cpp_window, age_at_open_months
    )
    _policy(values)["elections"]["oas_start_age_years"]["a"] = election_years
    scenario = Scenario.model_validate(values)

    with pytest.raises(StartAgeNotAllowedError) as excinfo:
        check_start_ages(scenario, params)

    message = str(excinfo.value)
    assert "is below the age of" in message
    assert "'a'" in message
    assert "OAS" in message
    assert str(int(earliest)) in message
    assert str(int(latest)) in message


@pytest.mark.parametrize(
    ("field", "window_fixture", "bound"),
    [
        pytest.param("cpp_start_age_years", "cpp_window", "earliest", id="cpp-earliest-edge"),
        pytest.param("cpp_start_age_years", "cpp_window", "latest", id="cpp-latest-edge"),
        pytest.param("oas_start_age_years", "oas_window", "earliest", id="oas-earliest-edge"),
        pytest.param("oas_start_age_years", "oas_window", "latest", id="oas-latest-edge"),
    ],
)
def test_election_at_the_inclusive_window_edge_is_accepted(
    request: pytest.FixtureRequest,
    params: ParamYear,
    field: str,
    window_fixture: str,
    bound: str,
) -> None:
    earliest, latest = request.getfixturevalue(window_fixture)
    edge = earliest if bound == "earliest" else latest
    earliest_years = math.ceil(earliest / MONTHS_PER_YEAR)
    latest_years = math.floor(latest / MONTHS_PER_YEAR)
    years = earliest_years if bound == "earliest" else latest_years
    # The edge must be a whole number of years, or this test would be quietly
    # exercising an interior age instead of the window edge it names.
    assert years * MONTHS_PER_YEAR == edge

    values = example_values()
    person = _person(values)
    age_at_open = age_in_months(
        person["birth_year"], person["birth_month"], values["start_year"], 1
    )
    # This guard keeps a's age at opening at or below the edge, so the
    # election is not below a's whole-year age and case 2 cannot fire; it is
    # stricter than the whole-year rule requires.
    assert years * MONTHS_PER_YEAR >= age_at_open

    _policy(values)["elections"][field]["a"] = years
    scenario = Scenario.model_validate(values)

    check_start_ages(scenario, params)  # must not raise


def test_oas_in_pay_exactly_at_earliest_start_age_is_accepted(
    params: ParamYear, oas_window, start_year: int
) -> None:
    earliest, _latest = oas_window
    age_months = int(earliest)
    birth_year, birth_month = _birth_for_age_at_open(start_year, age_months)

    values = example_values()
    _person(values)["birth_year"] = birth_year
    _person(values)["birth_month"] = birth_month
    _clear_db_pensions(values)
    _person(values)["oas"] = {"in_pay_monthly": 500.0}
    scenario = Scenario.model_validate(values)

    check_start_ages(scenario, params)  # must not raise


def test_cpp_in_pay_skips_the_past_start_check(
    params: ParamYear, cpp_window, start_year: int
) -> None:
    earliest, _latest = cpp_window
    earliest_years = math.ceil(earliest / MONTHS_PER_YEAR)
    age_at_open_years = earliest_years + 5  # older than the window's earliest age
    leftover_election_years = earliest_years + 1  # inside the window, below age at opening

    values = example_values()
    _person(values)["birth_year"] = start_year - age_at_open_years
    _person(values)["birth_month"] = 1
    _clear_db_pensions(values)
    _person(values)["cpp"] = {"in_pay_monthly": 800.0}
    _policy(values)["elections"]["cpp_start_age_years"]["a"] = leftover_election_years
    # oas_start_age_years must stay valid for this new, older birth date.
    _policy(values)["elections"]["oas_start_age_years"]["a"] = age_at_open_years
    scenario = Scenario.model_validate(values)

    check_start_ages(scenario, params)  # must not raise


# --- Case 4: RRIF conversion age above conversion_age_years -----------------


def test_rrif_conversion_above_limit_is_refused(params: ParamYear) -> None:
    limit = int(params.rrif.number("conversion_age_years"))
    elected = limit + 1

    values = example_values()
    _policy(values)["elections"]["rrif_conversion"]["age_years"] = elected
    scenario = Scenario.model_validate(values)

    with pytest.raises(StartAgeNotAllowedError) as excinfo:
        check_start_ages(scenario, params)

    message = str(excinfo.value)
    assert "'taxable-first'" in message
    assert "rrif_conversion.age_years" in message
    assert str(elected) in message
    assert "conversion_age_years" in message
    assert str(params.rrif.source) in message


def test_rrif_conversion_equal_to_limit_is_accepted(params: ParamYear) -> None:
    limit = int(params.rrif.number("conversion_age_years"))

    values = example_values()
    _policy(values)["elections"]["rrif_conversion"]["age_years"] = limit
    scenario = Scenario.model_validate(values)

    check_start_ages(scenario, params)  # must not raise


def test_rrif_conversion_below_everyones_december_age_is_accepted(
    params: ParamYear, start_year: int
) -> None:
    values = example_values()
    persons = values["household"]["persons"]
    december_age = min(
        age_at_end_of_year(person["birth_year"], person["birth_month"], start_year)
        for person in persons
    )
    elected = december_age - 1

    _policy(values)["elections"]["rrif_conversion"]["age_years"] = elected
    scenario = Scenario.model_validate(values)

    check_start_ages(scenario, params)  # must not raise


def test_committed_late_life_couple_is_accepted(params: ParamYear) -> None:
    """Its RRIF election, 71, is accepted because it does not exceed conversion_age_years;
    it is inert because it is below both persons' ages at the opening, so no year-end age
    ever equals it.
    """
    values = yaml.safe_load(
        (REPO_ROOT / "scenarios" / "late_life_couple.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(values, dict)
    scenario = Scenario.model_validate(values)

    check_start_ages(scenario, params)  # must not raise


def test_rrif_step_runs_after_the_whole_cpp_oas_loop(params: ParamYear, cpp_window) -> None:
    """The first policy's RRIF election is above the limit, and would raise if the RRIF
    step ran first; the second policy's CPP election is outside the window, and is what
    actually raises, since the CPP/OAS loop runs over every policy before the RRIF step
    begins.
    """
    limit = int(params.rrif.number("conversion_age_years"))
    earliest, _latest = cpp_window
    earliest_years = math.ceil(earliest / MONTHS_PER_YEAR)

    values = example_values()
    _policy(values)["elections"]["rrif_conversion"]["age_years"] = limit + 1

    second = yaml.safe_load(yaml.safe_dump(_policy(values)))
    second["name"] = "taxable-first-2"
    second["elections"]["cpp_start_age_years"]["a"] = earliest_years - 1
    second["elections"]["rrif_conversion"]["age_years"] = limit
    values["policies"].append(second)
    scenario = Scenario.model_validate(values)

    with pytest.raises(StartAgeNotAllowedError) as excinfo:
        check_start_ages(scenario, params)

    message = str(excinfo.value)
    assert "'taxable-first-2'" in message
    assert "CPP" in message

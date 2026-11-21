# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Refusing a locked-in account the run cannot carry: age, unfiled jurisdiction, couples.

The unlocking age is read from ``params/2026/ab.yaml``, never typed; every birth year is
derived from it. Balances are synthetic. ``bc`` stands for any jurisdiction without a
parameter file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from engine.core.timeline import age_at_start_of_year
from engine.params.loader import ParamFileMissingError, ParamYear, load_year
from engine.scenario import (
    LockedInAgeError,
    LockedInJurisdictionError,
    Scenario,
    ScenarioError,
    check_locked_in,
    load_scenario,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MINIMAL = REPO_ROOT / "scenarios" / "minimal.yaml"
COUPLE = REPO_ROOT / "scenarios" / "late_life_couple.yaml"

START_YEAR = 2026
BALANCE = 10_000.0


@pytest.fixture(scope="module")
def params() -> ParamYear:
    return load_year(START_YEAR)


@pytest.fixture(scope="module")
def unlocking_age(params: ParamYear) -> int:
    return params.jurisdiction("ab").integer("lif.unlocking_age_years")


def _person(
    person_id: str,
    birth_year: int,
    birth_month: int = 1,
    *,
    lira: tuple[float, str | None] | None = None,
    lif: tuple[float, str | None] | None = None,
) -> dict[str, Any]:
    accounts: dict[str, Any] = {}
    if lira is not None:
        accounts["lira"] = {"balance": lira[0], "jurisdiction": lira[1]}
    if lif is not None:
        accounts["lif"] = {"balance": lif[0], "jurisdiction": lif[1]}
    return {
        "id": person_id,
        "birth_year": birth_year,
        "birth_month": birth_month,
        "sex": "f",
        "cpp": {"contributory_history": 0.0},
        "accounts": accounts,
        "prior_year_net_income": 0,
        "net_income_two_years_prior": 0,
    }


def _scenario(*persons: dict[str, Any]) -> Scenario:
    """The minimal scenario with its household replaced by ``persons``."""
    values = yaml.safe_load(MINIMAL.read_text(encoding="utf-8"))
    values["household"]["persons"] = list(persons)
    elections = values["policies"][0]["elections"]
    elections["cpp_start_age_years"] = {p["id"]: 65 for p in persons}
    elections["oas_start_age_years"] = {p["id"]: 65 for p in persons}
    return Scenario.model_validate(values)


def _old_enough(unlocking_age: int) -> int:
    """A birth year giving exactly the unlocking age on 1 January of the start year."""
    return START_YEAR - unlocking_age


def test_a_parameter_year_that_is_not_the_start_year_is_a_value_error(
    params: ParamYear, unlocking_age: int
) -> None:
    scenario = _scenario(_person("a", _old_enough(unlocking_age)))
    mismatched = scenario.model_copy(update={"start_year": START_YEAR + 1})

    with pytest.raises(ValueError, match=str(START_YEAR + 1)) as excinfo:
        check_locked_in(mismatched, params)

    assert str(START_YEAR) in str(excinfo.value)
    assert not isinstance(excinfo.value, ScenarioError)


# --- the age rule ------------------------------------------------------------


def test_a_funded_lif_below_the_unlocking_age_is_refused(
    params: ParamYear, unlocking_age: int
) -> None:
    scenario = _scenario(_person("a", _old_enough(unlocking_age) + 5, lif=(BALANCE, "ab")))

    with pytest.raises(LockedInAgeError) as excinfo:
        check_locked_in(scenario, params)

    message = str(excinfo.value)
    age = unlocking_age - 5
    assert "'a'" in message
    assert f"is {age} on 1 January {START_YEAR}" in message
    assert f"({unlocking_age})" in message
    assert str(params.jurisdiction("ab").source) in message


def test_an_empty_lif_with_a_jurisdiction_below_the_unlocking_age_is_refused(
    params: ParamYear, unlocking_age: int
) -> None:
    scenario = _scenario(_person("a", _old_enough(unlocking_age) + 5, lif=(0.0, "ab")))

    with pytest.raises(LockedInAgeError, match="zero balance still counts"):
        check_locked_in(scenario, params)


def test_a_person_who_turns_the_unlocking_age_later_in_the_start_year_is_refused(
    params: ParamYear, unlocking_age: int
) -> None:
    born_in_november = _person("a", _old_enough(unlocking_age), 11, lif=(BALANCE, "ab"))
    assert age_at_start_of_year(_old_enough(unlocking_age), 11, START_YEAR) == unlocking_age - 1

    with pytest.raises(LockedInAgeError, match=f"is {unlocking_age - 1} on 1 January"):
        check_locked_in(_scenario(born_in_november), params)


def test_the_unlocking_age_on_1_january_is_accepted(params: ParamYear, unlocking_age: int) -> None:
    born_in_january = _person("a", _old_enough(unlocking_age), 1, lif=(BALANCE, "ab"))
    assert age_at_start_of_year(_old_enough(unlocking_age), 1, START_YEAR) == unlocking_age

    check_locked_in(_scenario(born_in_january), params)  # must not raise


def test_a_young_person_with_no_lif_jurisdiction_is_not_age_checked(
    params: ParamYear, unlocking_age: int
) -> None:
    young = _person("a", _old_enough(unlocking_age) + 20)

    check_locked_in(_scenario(young), params)  # must not raise


def test_a_lira_for_a_young_person_is_accepted(params: ParamYear, unlocking_age: int) -> None:
    young = _person("a", _old_enough(unlocking_age) + 20, lira=(BALANCE, "ab"))

    check_locked_in(_scenario(young), params)  # must not raise


# --- the unfiled jurisdiction ------------------------------------------------


def test_a_lira_in_an_unfiled_jurisdiction_is_refused(
    params: ParamYear, unlocking_age: int
) -> None:
    scenario = _scenario(_person("a", _old_enough(unlocking_age), lira=(BALANCE, "bc")))

    with pytest.raises(LockedInJurisdictionError) as excinfo:
        check_locked_in(scenario, params)

    message = str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ParamFileMissingError)
    assert "'a'" in message
    assert "lira" in message
    assert "'bc'" in message
    assert str(params.root) in message
    assert message.endswith("Available: ab (docs/limitations.md L3).")
    assert "L3" in message


def test_an_unfiled_jurisdiction_in_a_year_with_no_jurisdiction_file_lists_none(
    tmp_path: Path, unlocking_age: int
) -> None:
    """The parameter root is synthetic: one empty ``federal.yaml``, no two-letter file."""
    (tmp_path / str(START_YEAR)).mkdir()
    (tmp_path / str(START_YEAR) / "federal.yaml").write_text("", encoding="utf-8")
    synthetic = load_year(START_YEAR, tmp_path)
    scenario = _scenario(_person("a", _old_enough(unlocking_age), lira=(BALANCE, "bc")))

    with pytest.raises(LockedInJurisdictionError) as excinfo:
        check_locked_in(scenario, synthetic)

    assert str(excinfo.value).endswith("Available: none (docs/limitations.md L3).")


def test_the_available_list_holds_every_two_letter_lowercase_code_and_nothing_else(
    tmp_path: Path, unlocking_age: int
) -> None:
    """The parameter root is synthetic. ``AB`` and ``a1`` are two-character stems the pattern
    refuses, ``a`` is a one-letter stem it refuses, and ``federal`` is longer than two
    characters."""
    (tmp_path / str(START_YEAR)).mkdir()
    for stem in ("on", "sk", "AB", "a1", "a", "federal"):
        (tmp_path / str(START_YEAR) / f"{stem}.yaml").write_text("", encoding="utf-8")
    synthetic = load_year(START_YEAR, tmp_path)
    scenario = _scenario(_person("a", _old_enough(unlocking_age), lira=(BALANCE, "bc")))

    with pytest.raises(LockedInJurisdictionError) as excinfo:
        check_locked_in(scenario, synthetic)

    assert str(excinfo.value).endswith("Available: on, sk (docs/limitations.md L3).")


def test_a_lif_in_an_unfiled_jurisdiction_is_refused(params: ParamYear, unlocking_age: int) -> None:
    scenario = _scenario(_person("a", _old_enough(unlocking_age), lif=(BALANCE, "bc")))

    with pytest.raises(LockedInJurisdictionError) as excinfo:
        check_locked_in(scenario, params)

    assert isinstance(excinfo.value.__cause__, ParamFileMissingError)
    assert "accounts.lif" in str(excinfo.value)
    assert "'bc'" in str(excinfo.value)


def test_the_unfiled_jurisdiction_is_refused_before_the_age(
    params: ParamYear, unlocking_age: int
) -> None:
    young = _person("a", _old_enough(unlocking_age) + 5, lif=(0.0, "bc"))

    with pytest.raises(LockedInJurisdictionError):
        check_locked_in(_scenario(young), params)


def test_an_alberta_lira_and_lif_are_accepted(params: ParamYear, unlocking_age: int) -> None:
    check_locked_in(
        _scenario(_person("a", _old_enough(unlocking_age), lira=(BALANCE, "ab"))), params
    )
    check_locked_in(
        _scenario(_person("a", _old_enough(unlocking_age), lif=(BALANCE, "ab"))), params
    )


# --- the couple rule ---------------------------------------------------------


def test_a_funded_lira_against_the_others_lira_in_another_jurisdiction_is_refused(
    params: ParamYear, unlocking_age: int
) -> None:
    year = _old_enough(unlocking_age)
    scenario = _scenario(
        _person("a", year, lira=(BALANCE, "ab")), _person("b", year, lira=(BALANCE, "bc"))
    )

    with pytest.raises(LockedInJurisdictionError) as excinfo:
        check_locked_in(scenario, params)

    message = str(excinfo.value)
    assert "'a'" in message and "'b'" in message
    assert "LIRA" in message
    assert "'ab'" in message and "'bc'" in message
    assert "L47" in message
    assert "before any death" in message
    assert excinfo.value.__cause__ is None


def test_a_funded_lira_against_the_others_empty_lif_in_another_jurisdiction_is_refused(
    params: ParamYear, unlocking_age: int
) -> None:
    year = _old_enough(unlocking_age)
    scenario = _scenario(
        _person("a", year, lira=(BALANCE, "ab")), _person("b", year, lif=(0.0, "bc"))
    )

    with pytest.raises(LockedInJurisdictionError, match="LIF") as excinfo:
        check_locked_in(scenario, params)

    assert "L47" in str(excinfo.value)


def test_the_couple_rule_runs_in_the_second_to_first_direction(
    params: ParamYear, unlocking_age: int
) -> None:
    year = _old_enough(unlocking_age)
    scenario = _scenario(
        _person("a", year, lif=(0.0, "bc")), _person("b", year, lira=(BALANCE, "ab"))
    )

    with pytest.raises(LockedInJurisdictionError, match="'b' holds a LIRA balance") as excinfo:
        check_locked_in(scenario, params)

    assert "L47" in str(excinfo.value)


def test_an_empty_lira_is_not_refused_by_the_couple_rule(
    params: ParamYear, unlocking_age: int
) -> None:
    year = _old_enough(unlocking_age)
    scenario = _scenario(
        _person("a", year, lira=(0.0, "bc")), _person("b", year, lif=(BALANCE, "ab"))
    )

    with pytest.raises(LockedInJurisdictionError) as excinfo:
        check_locked_in(scenario, params)

    # The unfiled-jurisdiction message, which proves the couple rule skipped it.
    assert "no parameter file" in str(excinfo.value)
    assert "L47" not in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ParamFileMissingError)


def test_a_couple_in_one_jurisdiction_is_accepted(params: ParamYear, unlocking_age: int) -> None:
    year = _old_enough(unlocking_age)
    check_locked_in(
        _scenario(
            _person("a", year, lira=(BALANCE, "ab")), _person("b", year, lira=(BALANCE, "ab"))
        ),
        params,
    )
    check_locked_in(
        _scenario(
            _person("a", year, lira=(BALANCE, "ab")), _person("b", year, lif=(BALANCE, "ab"))
        ),
        params,
    )


def test_the_committed_couple_is_accepted(params: ParamYear) -> None:
    check_locked_in(load_scenario(COUPLE), params)  # must not raise


def test_a_household_of_one_is_not_couple_checked(params: ParamYear, unlocking_age: int) -> None:
    check_locked_in(
        _scenario(_person("a", _old_enough(unlocking_age), lira=(BALANCE, "ab"))), params
    )


def test_the_couple_rule_runs_before_the_per_person_rules(
    params: ParamYear, unlocking_age: int
) -> None:
    year = _old_enough(unlocking_age)
    scenario = _scenario(
        _person("a", year, lira=(BALANCE, "ab")), _person("b", year, lira=(BALANCE, "bc"))
    )

    with pytest.raises(LockedInJurisdictionError) as excinfo:
        check_locked_in(scenario, params)

    assert "L47" in str(excinfo.value)
    assert "no parameter file" not in str(excinfo.value)

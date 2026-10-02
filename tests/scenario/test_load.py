# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reading a scenario file: the failures that happen before validation.

Everything here is about the *file* rather than the household: a path that is
not there, bytes that are not YAML, and the one thing a file can express that a
mapping cannot — the same key twice. The validation rules themselves are in
``test_schema.py``. This module also pins the two committed scenarios
themselves: that each loads, passes the run-opening checks, builds, and holds
the mechanisms it exists for.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from engine.core.build import build_initial_state
from engine.core.indexation import real_year
from engine.core.mortality import months_to_terminal
from engine.core.state import DEATH_NOT_DRAWN
from engine.params.loader import ParamYear, YamlConstructionError, load_year
from engine.scenario import (
    InvalidScenarioError,
    MalformedScenarioFileError,
    Scenario,
    ScenarioError,
    ScenarioFileMissingError,
    check_lifespan,
    check_start_ages,
    load_scenario,
)
from engine.scenario.load import DuplicateKeyError
from engine.tax.combined import household_assessment

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The committed single-person example. Every later issue that does not need a
#: second person runs against this file, so a change that stops it loading
#: breaks the whole roadmap below issue 10, not just here.
EXAMPLE = REPO_ROOT / "scenarios" / "example.yaml"

#: The committed two-person fixture: the couple decumulation path, used
#: wherever a later issue needs a second person, a spousal rollover, or a
#: survivor share.
COUPLE = REPO_ROOT / "scenarios" / "late_life_couple.yaml"

#: Small enough that building state for both committed scenarios stays cheap.
N_PATHS = 3


def write(tmp_path: Path, text: str) -> Path:
    """Write ``text`` to a scenario file under ``tmp_path`` and return its path."""
    path = tmp_path / "scenario.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_committed_example_loads() -> None:
    """``load_scenario("scenarios/example.yaml")`` returns a ``Scenario``.

    The reason the file is committed: it is the single-person fixture every
    later issue builds a state, a run, and a search from, when it does not
    need a second person.
    """
    scenario = load_scenario(EXAMPLE)

    assert isinstance(scenario, Scenario)
    assert scenario.name == "example-household"
    assert scenario.start_year == 2026
    assert scenario.household.person_ids == ("a",)
    assert scenario.assumptions.asset_class_names == ("equity", "bonds")


def test_a_missing_file_names_the_path(tmp_path: Path) -> None:
    """A path that is not there stops, and says which path."""
    absent = tmp_path / "not-here.yaml"

    with pytest.raises(ScenarioFileMissingError) as excinfo:
        load_scenario(absent)

    assert str(absent) in str(excinfo.value)


def test_a_directory_is_not_a_scenario(tmp_path: Path) -> None:
    """Pointing at a directory fails as a missing file, not as an OSError.

    ``scenarios/`` and ``scenarios/example.yaml`` are one tab-completion apart.
    """
    with pytest.raises(ScenarioFileMissingError):
        load_scenario(tmp_path)


def test_non_utf8_file_is_malformed(tmp_path: Path) -> None:
    """A hand-edited scenario saved in the wrong encoding names itself, not `UnicodeDecodeError`."""
    path = tmp_path / "scenario.yaml"
    path.write_bytes("# Québec\nx: 1\n".encode("cp1252"))

    with pytest.raises(MalformedScenarioFileError) as excinfo:
        load_scenario(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "UTF-8" in message
    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_invalid_yaml_is_malformed(tmp_path: Path) -> None:
    """Bytes that do not parse stop with the parser's own complaint attached."""
    path = write(tmp_path, "name: example\n  bad: indentation\n")

    with pytest.raises(MalformedScenarioFileError) as excinfo:
        load_scenario(path)

    assert str(path) in str(excinfo.value)


@pytest.mark.parametrize(
    ("text", "cause_type"),
    [
        pytest.param("name: !!bool maybe\n", KeyError, id="bool"),
        pytest.param("name: !!timestamp abc\n", AttributeError, id="timestamp"),
        pytest.param("name: !!int abc\n", ValueError, id="int"),
    ],
)
def test_a_construction_error_is_malformed_not_a_duplicate_key(
    tmp_path: Path, text: str, cause_type: type[Exception]
) -> None:
    """PyYAML's scalar constructors raise a bare exception with no filename attached.

    ``load_scenario`` must still name the file and say the file is not valid
    YAML, rather than letting the bare exception escape or misreporting it as
    ``DuplicateKeyError``. ``parse_yaml`` converts the bare exception to
    :class:`YamlConstructionError` once, so it is that class's ``__cause__``
    that carries the original.
    """
    path = write(tmp_path, text)

    with pytest.raises(MalformedScenarioFileError) as excinfo:
        load_scenario(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "not valid YAML" in message
    assert not isinstance(excinfo.value, DuplicateKeyError)
    assert isinstance(excinfo.value.__cause__, YamlConstructionError)
    assert isinstance(excinfo.value.__cause__.__cause__, cause_type)


@pytest.mark.parametrize(
    ("text", "what"),
    [
        pytest.param("", "an empty file", id="empty"),
        pytest.param("- a\n- b\n", "a list", id="list"),
        pytest.param("42\n", "a scalar", id="scalar"),
    ],
)
def test_a_top_level_that_is_not_a_mapping_is_malformed(
    tmp_path: Path, text: str, what: str
) -> None:
    """A scenario is a mapping of top-level keys.

    An empty file is malformed here, unlike an empty file under ``params/``,
    which is the expected state of a tax year nobody has transcribed yet. There
    is no equivalent half-written scenario: a file with no household describes
    nothing to simulate.
    """
    path = write(tmp_path, text)

    with pytest.raises(MalformedScenarioFileError):
        load_scenario(path)

    assert what  # names the case in the failure output


def test_a_repeated_key_is_rejected_rather_than_silently_overwritten(tmp_path: Path) -> None:
    """The rule ``Scenario.model_validate`` cannot enforce, because a dict cannot hold it.

    ``yaml.safe_load`` keeps the last of a repeated key and says nothing. Here
    the second ``equity`` would replace the first, leaving two asset classes
    whose ordering the correlation matrix still refers to positionally — every
    off-diagonal entry now meaning something the author did not write.
    """
    text = (
        "name: example\n"
        "assumptions:\n"
        "  asset_classes:\n"
        "    equity: {real_mean: 0.05}\n"
        "    bonds: {real_mean: 0.01}\n"
        "    equity: {real_mean: 0.02}\n"
    )
    path = write(tmp_path, text)

    with pytest.raises(DuplicateKeyError) as excinfo:
        load_scenario(path)

    message = str(excinfo.value)
    assert "equity" in message
    assert str(path) in message
    assert "line 6" in message, "The message must point at the repeat, not at the file."


def test_a_repeated_key_at_the_top_level_is_rejected_too(tmp_path: Path) -> None:
    """The check is on every mapping in the file, not only on the ones we thought of."""
    path = write(tmp_path, "name: first\nstart_year: 2026\nname: second\n")

    with pytest.raises(DuplicateKeyError):
        load_scenario(path)


def test_a_duplicate_key_is_a_malformed_file(tmp_path: Path) -> None:
    """The error hierarchy is usable: one ``except`` catches every file problem."""
    path = write(tmp_path, "name: first\nname: second\n")

    with pytest.raises(MalformedScenarioFileError):
        load_scenario(path)

    assert issubclass(DuplicateKeyError, MalformedScenarioFileError)
    assert issubclass(MalformedScenarioFileError, ScenarioError)
    assert issubclass(ScenarioFileMissingError, ScenarioError)
    assert issubclass(InvalidScenarioError, ScenarioError)


def test_a_validation_failure_names_the_file(tmp_path: Path) -> None:
    """``load_scenario`` wraps the validation error so the message says which file.

    A run loads several scenarios. ``household.persons.0.birth_month`` on its
    own sends the reader looking through all of them, and the pydantic error
    has no way to know which one it came from.
    """
    values = EXAMPLE.read_text(encoding="utf-8").replace("birth_month: 3", "birth_month: 13")
    path = write(tmp_path, values)

    with pytest.raises(InvalidScenarioError) as excinfo:
        load_scenario(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "birth_month" in message
    assert isinstance(excinfo.value.__cause__, ValidationError), (
        "The structured pydantic error must stay reachable rather than being flattened away."
    )


def test_model_validate_raises_pydantic_s_own_error(tmp_path: Path) -> None:
    """The in-memory entry point does not wrap.

    A caller holding a mapping — a request body, a fixture, a scenario built in
    code — has no file to be told about and does want the structured error.
    """
    del tmp_path
    with pytest.raises(ValidationError):
        Scenario.model_validate({"name": "nothing else"})


def test_a_loaded_scenario_cannot_be_written_to() -> None:
    """Frozen models and read-only mappings, all the way down.

    One scenario is shared across every path and every policy the optimizer
    evaluates. A write into it anywhere would change the run under paths
    already in flight, and nothing in the output would show it.
    """
    scenario = load_scenario(EXAMPLE)

    with pytest.raises(ValidationError):
        scenario.name = "renamed"

    with pytest.raises(TypeError):
        scenario.assumptions.allocations["default"]["equity"] = 1.0

    with pytest.raises(TypeError):
        scenario.assumptions.asset_classes["equity"] = scenario.assumptions.asset_classes["bonds"]

    with pytest.raises(TypeError):
        scenario.policies[0].elections.cpp_start_age_years["a"] = 70

    assert scenario.assumptions.allocations["default"]["equity"] == 0.6


def test_the_example_path_in_the_success_criterion_is_the_one_that_is_committed() -> None:
    """``scenarios/example.yaml`` exists at the path issue 10 names.

    Written as its own assertion so that renaming or moving the file fails here
    with the reason, rather than as a confusing missing-file error in every
    later issue's fixtures.
    """
    assert EXAMPLE.is_file(), (
        f"{EXAMPLE} is the single-person fixture every later issue that does "
        "not need a second person runs against."
    )


# --- The committed couple ----------------------------------------------------


@pytest.fixture(scope="module")
def params() -> ParamYear:
    return load_year(2026)


def test_the_couple_loads() -> None:
    """``load_scenario("scenarios/late_life_couple.yaml")`` returns a two-person ``Scenario``."""
    scenario = load_scenario(COUPLE)

    assert isinstance(scenario, Scenario)
    assert scenario.name == "late-life-couple"
    assert scenario.start_year == 2026
    assert scenario.household.person_ids == ("a", "b")
    assert scenario.household.beneficiaries == ()


def test_the_couple_passes_the_run_opening_checks(params: ParamYear) -> None:
    """Neither of the two checks that gate a run refuses this household."""
    scenario = load_scenario(COUPLE)

    check_start_ages(scenario, params)  # must not raise
    check_lifespan(scenario, params)  # must not raise


def test_the_couple_exercises_what_it_is_for() -> None:
    """Structural assertions on the loaded ``Scenario``.

    Reads the mechanisms the header says this file is for: two distinct
    birth dates, CPP and OAS already in pay, a funded RRIF and no RRSP for
    each person, a funded LIF registered in Alberta, a DB pension with a
    positive survivor share, and a survivor share below one. No numeric
    literal from the YAML appears below other than 0 and 1; ``"ab"`` is a
    jurisdiction code, not a number.
    """
    scenario = load_scenario(COUPLE)
    persons = scenario.household.persons

    birth_dates = [(person.birth_year, person.birth_month) for person in persons]
    assert len(set(birth_dates)) == len(birth_dates)

    for person in persons:
        assert person.cpp.in_pay_monthly is not None
        assert person.oas.in_pay_monthly is not None
        assert person.accounts.rrsp.balance == 0
        assert person.accounts.rrif.balance > 0
        assert person.accounts.tfsa.balance > 0
        assert 0 < person.accounts.taxable.acb < person.accounts.taxable.balance

    funded_lifs = [person for person in persons if person.accounts.lif.balance > 0]
    assert funded_lifs
    for person in funded_lifs:
        assert person.accounts.lif.jurisdiction == "ab"

    assert any(pension.survivor_share > 0 for person in persons for pension in person.db_pensions)
    assert scenario.spending.survivor_share < 1


def test_the_couple_builds_two_persons() -> None:
    """``build_initial_state`` gives the couple a two-person opening state.

    Checks every field this file exists to make non-trivial: both persons
    alive with no death drawn yet, a RRIF opened the year before the run for
    each, CPP and OAS already in pay with no start-age election, and a LIF
    opened the year before the run for each person who holds one.
    """
    scenario = load_scenario(COUPLE)
    state = build_initial_state(scenario, N_PATHS)

    assert len(state.persons) == 2
    for person in state.persons:
        assert person.alive.all()
        assert (person.death_month_index == DEATH_NOT_DRAWN).all()
        assert person.rrif.opened_year == scenario.start_year - 1
        assert person.cpp.start_age_months is None
        assert person.oas.start_age_months is None
        assert person.cpp.in_pay_monthly is not None
        assert person.oas.in_pay_monthly is not None
        if person.lif.balance[0] > 0:
            assert person.lif.opened_year == scenario.start_year - 1


def test_the_couple_s_run_is_materially_shorter_than_the_example_s(params: ParamYear) -> None:
    """The couple's longest survival curve is much shorter than the example's.

    Each scenario's household month count is the maximum, across its
    persons, of ``months_to_terminal``; the couple opens with both persons
    already old, the example with one middle-aged person. Neither month
    count is hard-coded: both derive from ``terminal_age_years``, a
    parameter, via the same public function the builder uses to size a run.
    """
    example = load_scenario(EXAMPLE)
    couple = load_scenario(COUPLE)

    def household_months(scenario: Scenario) -> int:
        return max(
            months_to_terminal(
                person.birth_year,
                person.birth_month,
                person.sex,
                scenario.start_year,
                params["mortality"],
            )
            for person in scenario.household.persons
        )

    example_months = household_months(example)
    couple_months = household_months(couple)

    assert couple_months * 3 < example_months * 2


def test_the_couple_reaches_the_two_person_branch_of_household_assessment(
    params: ParamYear,
) -> None:
    """``household_assessment`` returns one ``Assessment`` per person.

    A one-person household returns one ``Assessment``; the couple returning
    two is what proves its two-person branch — pension income splitting —
    ran at all.
    """
    scenario = load_scenario(COUPLE)
    state = build_initial_state(scenario, N_PATHS)
    real_params = real_year(params, scenario.assumptions.inflation)

    assessments = household_assessment(state, real_params)

    assert len(assessments) == 2

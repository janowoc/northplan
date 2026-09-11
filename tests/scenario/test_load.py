# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reading a scenario file: the failures that happen before validation.

Everything here is about the *file* rather than the household: a path that is
not there, bytes that are not YAML, and the one thing a file can express that a
mapping cannot — the same key twice. The validation rules themselves are in
``test_schema.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from engine.scenario import (
    InvalidScenarioError,
    MalformedScenarioFileError,
    Scenario,
    ScenarioError,
    ScenarioFileMissingError,
    load_scenario,
)
from engine.scenario.load import DuplicateKeyError

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The committed example. Every later issue runs against this file, so a change
#: that stops it loading breaks the whole roadmap below issue 10, not just here.
EXAMPLE = REPO_ROOT / "scenarios" / "example.yaml"


def write(tmp_path: Path, text: str) -> Path:
    """Write ``text`` to a scenario file under ``tmp_path`` and return its path."""
    path = tmp_path / "scenario.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_committed_example_loads() -> None:
    """``load_scenario("scenarios/example.yaml")`` returns a ``Scenario``.

    The first success criterion of issue 10, and the reason the file is
    committed: it is the fixture every later issue builds a state, a run, and a
    search from.
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
    assert EXAMPLE.is_file(), f"{EXAMPLE} is the fixture every later issue runs against."

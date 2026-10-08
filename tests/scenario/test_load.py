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

import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from engine.core.build import build_initial_state
from engine.core.indexation import real_year
from engine.core.mortality import months_to_terminal
from engine.core.state import DEATH_NOT_DRAWN
from engine.params.loader import ParamYear, YamlConstructionError, load_year, parse_yaml
from engine.policy.build import expand_grid
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
from engine.scenario.load import (
    MAX_DOCUMENT_CHARACTERS,
    MAX_DOCUMENT_VALUES,
    MAX_REPORTED_ERRORS,
    DuplicateKeyError,
    _expanded_size,
    _text_length,
    scenario_from_text,
    validation_message,
)
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


# =============================================================================
# scenario_from_text: text already in memory
# =============================================================================


def example_text() -> str:
    return EXAMPLE.read_text(encoding="utf-8")


def example_json() -> str:
    return json.dumps(parse_yaml(example_text()))


def test_yaml_text_gives_the_scenario_the_file_gives() -> None:
    assert scenario_from_text(example_text(), "x", syntax="yaml") == load_scenario(EXAMPLE)


def test_json_text_gives_the_scenario_the_yaml_gives() -> None:
    from_json = scenario_from_text(example_json(), "request body", syntax="json")
    assert from_json == scenario_from_text(example_text(), "x", syntax="yaml")


def test_a_repeated_top_level_json_key_is_rejected() -> None:
    with pytest.raises(DuplicateKeyError, match="key 'seed'") as caught:
        scenario_from_text('{"seed": 1, "seed": 2}', "request body", syntax="json")
    assert str(caught.value).startswith("request body: key ")


def test_a_repeated_nested_json_key_is_rejected() -> None:
    text = '{"assumptions": {"inflation": 0.02, "inflation": 0.03}}'
    with pytest.raises(DuplicateKeyError, match="key 'inflation'"):
        scenario_from_text(text, "request body", syntax="json")


def test_a_json_syntax_error_is_malformed_and_not_a_duplicate_key() -> None:
    with pytest.raises(MalformedScenarioFileError, match="is not valid JSON") as caught:
        scenario_from_text('{"seed": ', "request body", syntax="json")
    assert not isinstance(caught.value, DuplicateKeyError)


def test_a_json_integer_over_the_digit_limit_is_malformed() -> None:
    with pytest.raises(MalformedScenarioFileError, match="is not valid JSON"):
        scenario_from_text("1" * 5000, "request body", syntax="json")


def test_json_nested_too_deep_is_malformed() -> None:
    text = "[" * 100000 + "]" * 100000
    with pytest.raises(MalformedScenarioFileError, match="is not valid JSON"):
        scenario_from_text(text, "request body", syntax="json")


def test_a_json_top_level_array_is_not_a_mapping() -> None:
    with pytest.raises(MalformedScenarioFileError, match=r"this parses to list\."):
        scenario_from_text("[1, 2]", "request body", syntax="json")


def test_a_json_object_that_fails_validation_names_the_source() -> None:
    values = parse_yaml(example_text())
    values["n_paths"] = 0
    with pytest.raises(InvalidScenarioError) as caught:
        scenario_from_text(json.dumps(values), "request body", syntax="json")
    assert str(caught.value).startswith("request body is not a valid scenario:")


def test_a_repeated_yaml_key_starts_with_the_source() -> None:
    with pytest.raises(DuplicateKeyError) as caught:
        scenario_from_text("seed: 1\nseed: 2\n", "request body", syntax="yaml")
    assert str(caught.value).startswith("request body: ")


def test_a_yaml_syntax_error_is_malformed() -> None:
    with pytest.raises(MalformedScenarioFileError, match="is not valid YAML"):
        scenario_from_text("a: [1, 2\n", "request body", syntax="yaml")


def test_a_yaml_top_level_list_is_not_a_mapping() -> None:
    with pytest.raises(MalformedScenarioFileError, match=r"this parses to list\."):
        scenario_from_text("- 1\n- 2\n", "request body", syntax="yaml")


def test_an_unknown_syntax_is_a_value_error_naming_it() -> None:
    with pytest.raises(ValueError, match="'toml'"):
        scenario_from_text(example_text(), "x", syntax="toml")  # type: ignore[arg-type]


def test_json_goes_through_the_json_parser_and_not_the_yaml_one() -> None:
    # YAML 1.1 reads 4.2e1 as the string "4.2e1", which an int field refuses; JSON
    # reads it as the number 42.
    text = example_json()
    assert '"seed": 42' in text
    exponent = text.replace('"seed": 42', '"seed": 4.2e1')
    assert exponent != text
    assert scenario_from_text(exponent, "request body", syntax="json").seed == 42
    with pytest.raises(InvalidScenarioError):
        scenario_from_text(exponent, "request body", syntax="yaml")


def alias_bomb(levels: int = 7) -> str:
    """YAML whose first level is a ten-item list and each next one ten aliases of the last."""
    lines = ["l0: &l0 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"]
    for level in range(1, levels):
        lines.append(f"l{level}: &l{level} [{', '.join([f'*l{level - 1}'] * 10)}]")
    lines.append(f"name: *l{levels - 1}")
    return "\n".join(lines) + "\n"


CAP_REFUSAL = "more than 100,000 values"
CYCLE_REFUSAL = "refers to itself through a YAML alias"


def test_a_yaml_alias_bomb_is_refused_by_the_cap_quickly() -> None:
    started = time.perf_counter()
    with pytest.raises(MalformedScenarioFileError, match=CAP_REFUSAL) as caught:
        scenario_from_text(alias_bomb(), "request body", syntax="yaml")
    assert time.perf_counter() - started < 1.0
    assert str(caught.value).startswith("request body: ")


def test_a_document_of_exactly_the_cap_is_not_refused_by_the_cap() -> None:
    # Root object 1, the list 1, and the leaves.
    leaves = MAX_DOCUMENT_VALUES - 2
    text = json.dumps({"a": [0] * leaves})
    with pytest.raises(InvalidScenarioError):
        scenario_from_text(text, "request body", syntax="json")


def test_a_document_one_value_over_the_cap_is_refused_by_the_cap() -> None:
    text = json.dumps({"a": [0] * (MAX_DOCUMENT_VALUES - 1)})
    with pytest.raises(MalformedScenarioFileError, match=CAP_REFUSAL):
        scenario_from_text(text, "request body", syntax="json")


def test_a_self_referencing_yaml_list_is_refused_as_a_cycle() -> None:
    with pytest.raises(MalformedScenarioFileError, match=CYCLE_REFUSAL):
        scenario_from_text("a: &a [1, *a]\n", "request body", syntax="yaml")


def test_a_self_referencing_yaml_mapping_is_refused_as_a_cycle() -> None:
    with pytest.raises(MalformedScenarioFileError, match=CYCLE_REFUSAL):
        scenario_from_text("a: &a {b: *a}\n", "request body", syntax="yaml")


def test_values_held_only_in_omap_tuples_count_toward_the_cap() -> None:
    big = ", ".join(["0"] * 1000)
    text = "big: &big [" + big + "]\n"
    text += "pairs: !!omap\n" + "".join(f"  - k{i}: *big\n" for i in range(150))
    # 150 * 1000 leaves live only under the tuples !!omap builds; the document as
    # written holds about 1,150 values, so only counting tuples refuses it.
    with pytest.raises(MalformedScenarioFileError, match=CAP_REFUSAL):
        scenario_from_text(text, "request body", syntax="yaml")


def test_a_shared_but_small_alias_is_not_refused_by_the_cap() -> None:
    with pytest.raises(InvalidScenarioError):
        scenario_from_text("x: &x [1, 2]\ny: *x\nz: *x\n", "request body", syntax="yaml")


def test_load_scenario_refuses_a_bomb_the_same_way(tmp_path: Path) -> None:
    path = tmp_path / "bomb.yaml"
    path.write_text(alias_bomb(), encoding="utf-8")
    with pytest.raises(MalformedScenarioFileError, match=CAP_REFUSAL):
        load_scenario(path)


class CountingList(list[int]):
    """A list that counts how many times it is iterated."""

    iterations = 0

    def __iter__(self) -> Iterator[int]:
        type(self).iterations += 1
        return super().__iter__()


def _naive_size(node: object) -> int:
    if isinstance(node, dict):
        return 1 + sum(_naive_size(child) for child in node.values())
    if isinstance(node, (list, tuple, set, frozenset)):
        return 1 + sum(_naive_size(child) for child in node)
    return 1


def _nested_through(shared: object, depth: int) -> list[object]:
    nested: list[object] = [shared]
    for _ in range(depth):
        nested = [shared, nested]
    return nested


def test_the_children_of_each_container_are_copied_once() -> None:
    CountingList.iterations = 0
    big = CountingList(range(1000))
    value = {"big": big, "x": _nested_through(big, 300)}
    _expanded_size(value, 10**9)
    assert CountingList.iterations == 1


def test_the_expanded_count_equals_a_naive_count() -> None:
    big = list(range(1000))
    value = {"big": big, "x": _nested_through(big, 50)}
    assert _expanded_size(value, 10**9) == _naive_size(value)
    aliased = parse_yaml("x: &x [1, 2]\ny: *x\nz: *x\n")
    assert _expanded_size(aliased, 10**9) == _naive_size(aliased)


# =============================================================================
# The character budget
# =============================================================================

CHARACTER_REFUSAL = "more than 1,000,000 characters of text"


def test_scenario_from_text_refuses_aliased_long_strings_quickly() -> None:
    text = "long: &s " + "x" * 10_000 + "\nmany:\n" + "  - *s\n" * 10_000
    started = time.perf_counter()
    with pytest.raises(MalformedScenarioFileError, match=CHARACTER_REFUSAL) as caught:
        scenario_from_text(text, "request body", syntax="yaml")
    assert time.perf_counter() - started < 1.0
    assert "1,000,000" in str(caught.value)
    assert str(caught.value).startswith("request body: ")


def test_scenario_from_text_counts_mapping_keys_as_text() -> None:
    key = "k" * 200
    text = f"shared: &m {{{key}: 1}}\nmany:\n" + "  - *m\n" * 10_000
    # The strings alone: the one key is not a string value, and nothing else is long.
    values = parse_yaml(text)
    assert _expanded_size(values, 10**9, _text_length) > MAX_DOCUMENT_CHARACTERS
    assert _expanded_size(values, 10**9) < MAX_DOCUMENT_VALUES
    assert sum(len(v) for v in values["many"][0].values() if isinstance(v, str)) == 0

    with pytest.raises(MalformedScenarioFileError, match=CHARACTER_REFUSAL):
        scenario_from_text(text, "request body", syntax="yaml")


def test_scenario_from_text_character_budget_applies_to_json() -> None:
    text = json.dumps({"name": "x" * (MAX_DOCUMENT_CHARACTERS + 1)})
    with pytest.raises(MalformedScenarioFileError, match=CHARACTER_REFUSAL):
        scenario_from_text(text, "request body", syntax="json")


def test_text_length_weights() -> None:
    def weight(value: object) -> int:
        return _expanded_size(value, 10**9, _text_length)

    shared = "abcd"
    assert weight("abc") == 3
    assert weight(b"abcde") == 5
    assert weight(7) == 0
    assert weight({"ab": 1, 1: 2, "cde": 3}) == 5
    assert weight({"ab": "xyz"}) == 5
    assert weight([shared, shared, shared]) == 12
    assert weight({"k": [shared, shared]}) == 9


def test_omap_repeat_in_a_scenario_is_a_duplicate_key_error() -> None:
    with pytest.raises(DuplicateKeyError, match="ordered map"):
        scenario_from_text("x: !!omap [{a: 1}, {a: 2}]\n", "request body", syntax="yaml")


@pytest.mark.parametrize(
    "path", sorted((REPO_ROOT / "scenarios").glob("*.yaml")), ids=lambda path: path.name
)
def test_committed_scenarios_are_far_under_the_character_budget(path: Path) -> None:
    values = parse_yaml(path.read_text(encoding="utf-8"))
    assert _expanded_size(values, 10**9, _text_length) < MAX_DOCUMENT_CHARACTERS / 100


# =============================================================================
# The cap on a validation message
# =============================================================================


def _invalid(policies: object) -> ValidationError:
    values = parse_yaml((REPO_ROOT / "scenarios" / "example.yaml").read_text(encoding="utf-8"))
    values["policies"] = policies
    with pytest.raises(ValidationError) as caught:
        Scenario.model_validate(values)
    return caught.value


@pytest.mark.parametrize("count", [2, MAX_REPORTED_ERRORS])
def test_validation_message_is_unchanged_up_to_the_limit(count: int) -> None:
    exc = _invalid(["x"] * count)
    assert exc.error_count() == count
    assert validation_message(exc) == str(exc)


def test_validation_message_caps_at_the_limit() -> None:
    exc = _invalid(["x"] * (MAX_REPORTED_ERRORS + 1))
    assert exc.error_count() == MAX_REPORTED_ERRORS + 1

    message = validation_message(exc)

    assert message.endswith("\n... and 1 more error, not shown.")
    shown = message.split("\n", 1)[1].removesuffix("\n... and 1 more error, not shown.")
    full = str(exc).split("\n", 1)[1]
    assert full.startswith(shown)
    assert shown.count("policies.") == MAX_REPORTED_ERRORS
    assert f"policies.{MAX_REPORTED_ERRORS}" not in shown
    assert f"policies.{MAX_REPORTED_ERRORS - 1}" in shown


def test_scenario_from_text_caps_a_huge_validation_message() -> None:
    values = parse_yaml((REPO_ROOT / "scenarios" / "example.yaml").read_text(encoding="utf-8"))
    values["policies"] = ["x"] * 99_000
    text = json.dumps(values)

    started = time.perf_counter()
    with pytest.raises(InvalidScenarioError) as caught:
        scenario_from_text(text, "request body", syntax="json")
    elapsed = time.perf_counter() - started

    assert len(str(caught.value)) < 10_000
    assert "98,980 more" in str(caught.value)
    assert elapsed < 2.0


def test_validation_message_keeps_a_validator_error() -> None:
    exc = _invalid([])
    assert exc.error_count() == 1
    assert validation_message(exc) == str(exc)
    assert "policies: none given" in validation_message(exc)


def test_validation_message_rebuilds_a_validator_error() -> None:
    """Thirty value_error entries with ctx, from the grid's re-validation in expand_grid."""
    scenario = load_scenario(REPO_ROOT / "scenarios" / "example.yaml")
    grid = {"contribution.weights.rrsp": tuple(-float(i) for i in range(1, 31))}
    with pytest.raises(ValidationError) as caught:
        expand_grid(scenario.model_copy(update={"grid": grid}))
    exc = caught.value
    assert exc.error_count() == 30
    assert all(e["type"] == "value_error" and "ctx" in e for e in exc.errors())

    message = validation_message(exc)

    assert message.endswith("\n... and 10 more errors, not shown.")
    shown = message.split("\n", 1)[1].removesuffix("\n... and 10 more errors, not shown.")
    assert str(exc).split("\n", 1)[1].startswith(shown)
    assert shown.count("[type=value_error") == MAX_REPORTED_ERRORS

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The scenario editor's logic: the mapping, the emitted YAML, the checks, the schema, the
reference-aware edits and the grid paths."""

from __future__ import annotations

import base64
import copy
import json
import re
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pydantic
import pytest
import yaml

import report.editor as editor
from engine.core.indexation import RoutedParameterError
from engine.mc.prepare import MAX_PATHS
from engine.params.loader import DEFAULT_PARAMS_ROOT, load_year
from engine.scenario.load import MAX_DOCUMENT_VALUES, scenario_from_text
from engine.scenario.schema import (
    CONTRIBUTION_KINDS,
    DEFAULT_ALLOCATION,
    INVESTABLE_KINDS,
    WITHDRAWAL_KINDS,
    AssetClass,
    CppEntitlement,
    Person,
    Scenario,
    resolve_policy_path,
)
from report.editor import (
    FLATTENED,
    EditRefusedError,
    add_asset_class,
    check,
    emit,
    grid_paths,
    move_asset_class,
    parse,
    references,
    remove,
    rename,
    schema,
)
from report.run import simulate_document

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "scenarios" / "example.yaml"
COUPLE = REPO_ROOT / "scenarios" / "late_life_couple.yaml"
MINIMAL = REPO_ROOT / "scenarios" / "minimal.yaml"
ROOT = DEFAULT_PARAMS_ROOT

EXAMPLE_TEXT = EXAMPLE.read_text(encoding="utf-8")
COUPLE_TEXT = COUPLE.read_text(encoding="utf-8")
MINIMAL_TEXT = MINIMAL.read_text(encoding="utf-8")
GRID_LINE = "elections.cpp_start_age_years.a: [60, 65, 70]"

PERSON_A_REFERENCES = [
    ["policies", 0, "elections", "cpp_start_age_years", "a"],
    ["policies", 0, "elections", "oas_start_age_years", "a"],
    ["household", "beneficiaries", 0, "resp", "subscriber"],
    ["grid", "elections.cpp_start_age_years.a"],
]


def strict(value: Any) -> str:
    """JSON text of ``value``: it tells ``1`` from ``1.0``, ``-0.0`` from ``0.0``, and keeps order."""
    return json.dumps(value)


def changed(text: str, old: str, new: str) -> str:
    """``text`` with the first ``old`` replaced; fails when it was not there."""
    assert old in text
    return text.replace(old, new, 1)


def binary(text: str) -> str:
    """``text`` as a ``!!binary`` scalar, which YAML reads as bytes."""
    return "!!binary " + base64.b64encode(text.encode()).decode()


def mapping_of(text: str) -> Any:
    result = parse(text, params_root=ROOT)
    assert result["errors"] == []
    return result["mapping"]


def example() -> Any:
    return mapping_of(EXAMPLE_TEXT)


def couple() -> Any:
    return mapping_of(COUPLE_TEXT)


def minimal() -> Any:
    return mapping_of(MINIMAL_TEXT)


def check_errors(mapping: Any) -> list[dict[str, Any]]:
    return check(mapping, params_root=ROOT)["errors"]


# --- round trip -----------------------------------------------------------------------

ROUND_TRIP_TEXTS = [
    pytest.param(path.read_text(encoding="utf-8"), id=path.name)
    for path in sorted((REPO_ROOT / "scenarios").glob("*.yaml"))
] + [
    pytest.param(changed(EXAMPLE_TEXT, "vol: 0.16", "vol: 16e-2"), id="exponent-float"),
    pytest.param(changed(EXAMPLE_TEXT, "household:", "household: &h"), id="anchor"),
]


@pytest.mark.parametrize("text", ROUND_TRIP_TEXTS)
def test_round_trip_keeps_the_scenario(text: str) -> None:
    first = parse(text, params_root=ROOT)
    assert first["errors"] == []
    m1 = first["mapping"]
    t1 = emit(m1)
    original = scenario_from_text(text, "original", syntax="yaml")
    again = scenario_from_text(t1, "emitted", syntax="yaml")
    assert again == original
    assert again.assumptions.asset_class_names == original.assumptions.asset_class_names
    m2 = parse(t1, params_root=ROOT)["mapping"]
    assert strict(m2) == strict(m1)
    assert emit(m2) == t1
    assert check_errors(m1) == []


def test_round_trip_keeps_synthetic_floats() -> None:
    m = example()
    m["assumptions"]["inflation"] = 0.1 + 0.2
    first_class = next(iter(m["assumptions"]["asset_classes"].values()))
    first_class["real_mean"] = -0.0
    first_class["interest_yield"] = 1e-9
    m["household"]["persons"][0]["accounts"]["tfsa"]["balance"] = 1e20
    m["household"]["persons"][0]["accounts"]["taxable"]["acb"] = 1e16
    text = emit(m)
    result = parse(text, params_root=ROOT)
    assert result["errors"] == []
    assert strict(result["mapping"]) == strict(m)
    assert emit(result["mapping"]) == text


# --- parse ----------------------------------------------------------------------------


def test_parse_valid_text_comes_in_schema_order() -> None:
    result = parse(EXAMPLE_TEXT, params_root=ROOT)
    m = result["mapping"]
    assert list(m) == [key for key in Scenario.model_fields if key in m]
    person = m["household"]["persons"][0]
    assert list(person) == [key for key in Person.model_fields if key in person]
    assert result["errors"] == []
    assert result["notices"] == []


def test_parse_keeps_an_int_given_for_a_float_field() -> None:
    annual = example()["household"]["persons"][0]["employment"][0]["annual"]
    assert annual == 95000
    assert type(annual) is int


def test_parse_reads_an_exponent_float_as_a_float() -> None:
    m = mapping_of(changed(EXAMPLE_TEXT, "vol: 0.16", "vol: 16e-2"))
    vol = m["assumptions"]["asset_classes"]["equity"]["vol"]
    assert vol == 0.16
    assert type(vol) is float
    assert strict(m) == strict(example())


def test_parse_normalises_an_integral_float() -> None:
    seed = mapping_of(changed(EXAMPLE_TEXT, "seed: 42 ", "seed: 42.0 "))["seed"]
    assert seed == 42
    assert type(seed) is int


def test_parse_refuses_a_number_given_for_the_flag() -> None:
    text = changed(EXAMPLE_TEXT, "fill_pension_credit: true", "fill_pension_credit: 1")
    result = parse(text, params_root=ROOT)
    assert result["mapping"] is None
    assert result["errors"] == [
        {
            "loc": ["policies", 0, "withdrawal", "fill_pension_credit"],
            "message": "Input should be a valid boolean",
        }
    ]


def test_parse_refuses_a_flag_given_for_a_number() -> None:
    result = parse(changed(EXAMPLE_TEXT, "seed: 42 ", "seed: true "), params_root=ROOT)
    assert result["mapping"] is None
    assert result["errors"] == [
        {
            "loc": ["seed"],
            "message": "Value error, true and false are flags, not numbers; YAML also reads "
            "yes, no, on and off as flags. Write the number.",
        }
    ]


def test_parse_refuses_a_set() -> None:
    text = changed(
        EXAMPLE_TEXT,
        "spill_order: [tfsa, rrsp, taxable]",
        "spill_order: !!set {tfsa, rrsp, taxable}",
    )
    result = parse(text, params_root=ROOT)
    assert result["mapping"] is None
    assert result["notices"] == []
    assert result["errors"] == [
        {
            "loc": [],
            "message": "the text: the value at policies.0.contribution.spill_order is a set "
            "(!!set), which does not keep the order it was written in. Write it as a list.",
        }
    ]


def test_parse_turns_binary_into_a_string() -> None:
    m = mapping_of(changed(EXAMPLE_TEXT, "name: example-household", "name: !!binary ZXhhbXBsZQ=="))
    assert m["name"] == "example"
    assert type(m["name"]) is str


def test_parse_reports_text_that_is_not_yaml() -> None:
    result = parse("a: [\n", params_root=ROOT)
    assert result["mapping"] is None
    assert result["notices"] == []
    (error,) = result["errors"]
    assert error["loc"] == []
    assert error["message"].startswith("the text is not valid YAML")


def test_parse_reports_a_repeated_key() -> None:
    result = parse("a: 1\na: 2\n", params_root=ROOT)
    assert result["mapping"] is None
    (error,) = result["errors"]
    assert error["loc"] == []
    assert error["message"].startswith("the text: ")


def test_parse_reports_a_top_level_list() -> None:
    result = parse("- 1\n", params_root=ROOT)
    assert result["mapping"] is None
    (error,) = result["errors"]
    assert error["loc"] == []
    assert "list" in error["message"]


def test_parse_reports_a_schema_error_with_its_location() -> None:
    result = parse(changed(EXAMPLE_TEXT, "birth_month: 3", "birth_month: 13"), params_root=ROOT)
    assert result["mapping"] is None
    (error,) = result["errors"]
    assert error["loc"] == ["household", "persons", 0, "birth_month"]


def test_parse_refuses_a_start_age_the_parameter_year_refuses() -> None:
    text = changed(EXAMPLE_TEXT, GRID_LINE, "elections.cpp_start_age_years.a: [50]")
    scenario_from_text(text, "the text", syntax="yaml")
    result = parse(text, params_root=ROOT)
    assert result["mapping"] is None
    (error,) = result["errors"]
    assert error["loc"] == []


def test_parse_refuses_a_grid_over_an_integer_too_large_for_a_float() -> None:
    text = changed(
        EXAMPLE_TEXT, "cpp_start_age_years: {a: 65}", f"cpp_start_age_years: {{a: {10**400}}}"
    )
    result = parse(text, params_root=ROOT)
    assert result["mapping"] is None
    (error,) = result["errors"]
    assert (
        "elections.cpp_start_age_years.a is an integer too large to convert to a float."
        in error["message"]
    )
    assert "0" * 100 not in error["message"]


NON_TEXT_KEY_TAIL = (
    " is not text. Write it as plain text, without a tag such as !!binary, and quote it if "
    "YAML would read it as something else, as it reads 2 or true."
)

NON_TEXT_KEYS = [
    pytest.param(
        changed(EXAMPLE_TEXT, "    default:", "    !!binary ZGVmYXVsdA==:"),
        "assumptions.allocations",
        "default",
        id="allocations",
    ),
    pytest.param(
        changed(
            EXAMPLE_TEXT,
            "  " + GRID_LINE,
            "  !!binary ZWxlY3Rpb25zLmNwcF9zdGFydF9hZ2VfeWVhcnMuYQ==: [60, 65, 70]",
        ),
        "grid",
        GRID_LINE.split(":")[0],
        id="grid",
    ),
    pytest.param(
        changed(EXAMPLE_TEXT, "{equity: 0.6", "{!!binary ZXF1aXR5: 0.6"),
        "assumptions.allocations.default",
        "equity",
        id="default-allocation",
    ),
]


@pytest.mark.parametrize(("text", "place", "key"), NON_TEXT_KEYS)
def test_parse_refuses_a_key_that_is_not_text(text: str, place: str, key: str) -> None:
    result = parse(text, params_root=ROOT)
    assert result["mapping"] is None
    assert result["notices"] == []
    assert result["errors"] == [
        {
            "loc": [],
            "message": f"the text: key {key.encode()!r} in the mapping at {place}"
            + NON_TEXT_KEY_TAIL,
        }
    ]


def test_parse_reports_only_the_first_non_text_key() -> None:
    m = example()
    first = m["policies"][0]
    m["policies"] = [{**copy.deepcopy(first), "name": f"p{i}"} for i in range(7)]
    text = emit(m)
    pattern = re.compile(r"^      (rrsp|tfsa|taxable|resp): ", re.M)
    text, count = pattern.subn(lambda hit: f"      {binary(hit.group(1))}: ", text)
    assert count == 28
    result = parse(text, params_root=ROOT)
    assert result["mapping"] is None
    assert result["notices"] == []
    assert result["errors"] == [
        {
            "loc": [],
            "message": "the text: key b'rrsp' in the mapping at policies.0.contribution.weights"
            + NON_TEXT_KEY_TAIL,
        }
    ]


def test_parse_reports_a_nested_non_text_key_once() -> None:
    text = changed(
        EXAMPLE_TEXT,
        "    default: {equity: 0.6, bonds: 0.4}",
        "    " + binary("default") + ": {" + binary("equity") + ": 0.6, bonds: 0.4}",
    )
    result = parse(text, params_root=ROOT)
    assert result["mapping"] is None
    assert result["errors"] == [
        {
            "loc": [],
            "message": "the text: key b'default' in the mapping at assumptions.allocations"
            + NON_TEXT_KEY_TAIL,
        }
    ]


@pytest.fixture
def default_int_digit_limit() -> Iterator[None]:
    """Python's default limit on the digits of an int written in decimal, whatever the
    environment set."""
    limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(sys.int_info.default_max_str_digits)
    yield
    sys.set_int_max_str_digits(limit)


@pytest.mark.usefixtures("default_int_digit_limit")
def test_parse_refuses_an_integer_key_too_long_for_decimal() -> None:
    line = "    default: {equity: 0.6, bonds: 0.4}\n"
    text = changed(
        EXAMPLE_TEXT, line, line + "    ? 0x" + "f" * 5000 + "\n    : {equity: 0.6, bonds: 0.4}\n"
    )
    result = parse(text, params_root=ROOT)
    assert result["mapping"] is None
    assert result["errors"] == [
        {
            "loc": [],
            "message": "the text: key 0x"
            + "f" * 38
            + "… in the mapping at assumptions.allocations"
            + NON_TEXT_KEY_TAIL,
        }
    ]


def test_parse_refuses_a_binary_twin_of_a_text_key() -> None:
    line = "    default: {equity: 0.6, bonds: 0.4}\n"
    text = changed(
        EXAMPLE_TEXT, line, line + "    " + binary("default") + ": {equity: 0.6, bonds: 0.4}\n"
    )
    result = parse(text, params_root=ROOT)
    assert result["mapping"] is None
    assert result["errors"] == [
        {
            "loc": [],
            "message": "the text: key b'default' in the mapping at assumptions.allocations"
            + NON_TEXT_KEY_TAIL,
        }
    ]


@pytest.mark.parametrize("spelling", ['"<<"', "'<<'", "!!str <<"])
def test_parse_does_not_take_a_quoted_or_tagged_merge_key_for_a_merge(spelling: str) -> None:
    text = EXAMPLE_TEXT.replace("equity", spelling)
    assert text != EXAMPLE_TEXT
    result = parse(text, params_root=ROOT)
    assert result["errors"] == []
    assert result["notices"] == []
    assert list(result["mapping"]["assumptions"]["asset_classes"]) == ["<<", "bonds"]


def test_parse_notices_anchors_aliases_and_merge_keys() -> None:
    anchor = changed(EXAMPLE_TEXT, "household:", "household: &h")
    merged = changed(
        EXAMPLE_TEXT,
        "    withdrawal:\n",
        "    withdrawal:\n      <<: {fill_pension_credit: true}\n",
    )
    assert parse(anchor, params_root=ROOT)["notices"] == [FLATTENED]
    assert parse(merged, params_root=ROOT)["notices"] == [FLATTENED]
    assert parse(EXAMPLE_TEXT, params_root=ROOT)["notices"] == []


@pytest.mark.parametrize(
    "line",
    [
        "      !!merge <<: {fill_pension_credit: true}\n",
        "      !<tag:yaml.org,2002:merge> zz: {fill_pension_credit: true}\n",
    ],
)
def test_parse_notices_a_merge_key_written_with_a_tag(line: str) -> None:
    text = changed(EXAMPLE_TEXT, "    withdrawal:\n", "    withdrawal:\n" + line)
    result = parse(text, params_root=ROOT)
    assert result["errors"] == []
    assert result["notices"] == [FLATTENED]


# --- emit -----------------------------------------------------------------------------


def test_emit_writes_models_in_schema_order_and_unknown_keys_last() -> None:
    m = example()
    shuffled = {"zzz": 1, **dict(reversed(list(m.items())))}
    written = yaml.safe_load(emit(shuffled))
    assert list(written) == [key for key in Scenario.model_fields if key in m] + ["zzz"]
    assert check_errors(shuffled) == [{"loc": ["zzz"], "message": "Extra inputs are not permitted"}]


def test_emit_keeps_the_order_of_a_mapping_typed_field() -> None:
    m = example()
    assumptions = m["assumptions"]
    assumptions["asset_classes"] = dict(reversed(list(assumptions["asset_classes"].items())))
    assert assumptions["correlation"] == [[1.0, 0.1], [0.1, 1.0]]
    written = yaml.safe_load(emit(m))
    assert list(written["assumptions"]["asset_classes"]) == ["bonds", "equity"]
    assert check_errors(m) == []


def test_emit_writes_no_alias_for_a_shared_object() -> None:
    m = example()
    allocations = m["assumptions"]["allocations"]
    allocations["resp"] = allocations["default"]
    text = emit(m)
    assert "&" not in text
    assert "*" not in text
    assert set(yaml.safe_load(text)["assumptions"]["allocations"]) == {"default", "resp"}


def test_emit_writes_unicode_as_is() -> None:
    m = example()
    m["name"] = "Zoë"
    assert "name: Zoë\n" in emit(m)


def test_emit_keeps_a_next_line_character() -> None:
    m = example()
    m["name"] = "x\x85y"
    renamed = rename(m, "asset-class", "equity", "eq\x85uity")
    result = check(renamed, params_root=ROOT)
    assert result["errors"] == []
    assert strict(parse(result["yaml"], params_root=ROOT)["mapping"]) == strict(renamed)


def test_emit_round_trips_every_character() -> None:
    chars = [chr(c) for c in range(0x3000) if not 0xD800 <= c <= 0xDFFF] + [
        "\ufeff",
        "\uffff",
        "\U0010ffff",
    ]
    document = {"k" + c + "z": ["v" + c, c] for c in chars}
    assert yaml.safe_load(emit(document)) == document


# --- check ----------------------------------------------------------------------------


def test_check_of_a_valid_mapping_returns_its_text_and_no_errors() -> None:
    m = example()
    result = check(m, params_root=ROOT)
    assert result["yaml"] == emit(m)
    assert result["errors"] == []


@pytest.mark.parametrize(
    ("count", "last"),
    [
        (20, None),
        (21, "... and 1 more error, not shown."),
        (30, "... and 10 more errors, not shown."),
    ],
)
def test_check_caps_the_errors_it_reports(count: int, last: str | None) -> None:
    m = example()
    m["policies"] = ["x"] * count
    errors = check_errors(m)
    if last is None:
        assert len(errors) == count
        assert all(error["loc"] != [] for error in errors)
    else:
        assert len(errors) == 21
        assert errors[-1] == {"loc": [], "message": last}
        assert all(error["loc"] != [] for error in errors[:-1])


def test_check_refuses_a_document_over_the_size_cap() -> None:
    m = example()
    m["name"] = [0] * (MAX_DOCUMENT_VALUES + 1)
    (error,) = check_errors(m)
    assert error["loc"] == []
    assert error["message"].startswith("the form: ")


def test_check_refuses_a_top_level_that_is_not_a_mapping() -> None:
    (error,) = check_errors([1, 2])
    assert error["loc"] == []


def test_check_refuses_a_run_that_is_too_large() -> None:
    m = example()
    m["n_paths"] = MAX_PATHS + 1
    (error,) = check_errors(m)
    assert error["loc"] == []
    assert error["message"].startswith("the run asks for")


def test_check_locates_a_grid_value_refused_on_expansion() -> None:
    m = example()
    m["grid"] = {"elections.rrif_conversion.fraction": [-1.0]}
    (error,) = check_errors(m)
    assert error["loc"] == ["grid"]
    assert "rrif_conversion.fraction" in error["message"]


def test_check_refuses_a_start_age_the_parameter_year_refuses() -> None:
    m = example()
    m["grid"] = {"elections.cpp_start_age_years.a": [50]}
    (error,) = check_errors(m)
    assert error["loc"] == []
    assert "CPP start age 50" in error["message"]


def test_check_refuses_a_lifespan_the_life_table_cannot_represent() -> None:
    m = couple()
    terminal = load_year(m["start_year"])["mortality"].number("terminal_age_years")
    m["household"]["persons"][0]["birth_year"] = m["start_year"] - int(terminal) - 2
    (error,) = check_errors(m)
    assert error["loc"] == []
    assert "terminal_age_years" in error["message"]


def test_check_refuses_a_ceiling_bracket_past_the_last_edge() -> None:
    m = example()
    edges = len(load_year(m["start_year"]).federal.sequence("brackets.edges_annual"))
    m["policies"][0]["withdrawal"]["taxable_ceiling_bracket"] = edges
    (error,) = check_errors(m)
    assert error["loc"] == []
    assert "no federal bracket edge has this index" in error["message"]


def test_check_refuses_a_start_year_without_parameters() -> None:
    m = example()
    m["start_year"] = 2040
    (error,) = check_errors(m)
    assert error["loc"] == []
    assert error["message"].startswith("No parameters for tax year 2040")


def test_check_validates_the_text_it_emits(monkeypatch: pytest.MonkeyPatch) -> None:
    text = changed(emit(example()), "start_year: 2026", "start_year: 2040")
    monkeypatch.setattr(editor, "emit", lambda _mapping: text)
    result = check(example(), params_root=ROOT)
    assert result["yaml"] == text
    (error,) = result["errors"]
    assert error["loc"] == []
    assert error["message"].startswith("No parameters for tax year 2040")


def _set_at(mapping: Any, path: tuple[str | int, ...], value: Any) -> None:
    for step in path[:-1]:
        mapping = mapping[step]
    mapping[path[-1]] = value


def _value_at(mapping: Any, path: tuple[str | int, ...]) -> Any:
    for step in path:
        mapping = mapping[step]
    return mapping


@pytest.mark.parametrize(
    ("path", "sent", "number"),
    [
        pytest.param(("seed",), "1_000", 1000, id="seed"),
        pytest.param(("assumptions", "inflation"), "0.0_2", 0.02, id="inflation"),
        pytest.param(("household", "persons", 0, "birth_year"), "1_9_7_0", 1970, id="birth-year"),
        pytest.param(("assumptions", "inflation"), "2e-2", 0.02, id="exponent"),
    ],
)
def test_check_writes_number_text_as_the_number_it_validated_as(
    path: tuple[str | int, ...], sent: str, number: float
) -> None:
    m = minimal()
    _set_at(m, path, sent)
    assert isinstance(_value_at(yaml.safe_load(emit(m)), path), str)
    result = check(m, params_root=ROOT)
    assert result["errors"] == []
    expected = minimal()
    _set_at(expected, path, number)
    assert result["yaml"] == emit(expected)
    assert emit(mapping_of(result["yaml"])) == result["yaml"]


def _number_leaves(value: Any, path: tuple[str | int, ...] = ()) -> Iterator[tuple[str | int, ...]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _number_leaves(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _number_leaves(item, (*path, index))
    elif type(value) in (int, float):
        yield path


@pytest.mark.parametrize(
    "path", sorted((REPO_ROOT / "scenarios").glob("*.yaml")), ids=lambda p: p.name
)
def test_check_writes_what_parse_opens_from_the_text_it_emits(path: Path) -> None:
    base = mapping_of(path.read_text(encoding="utf-8"))
    leaves = list(_number_leaves(base))
    assert len(leaves) >= 20
    assert {type(_value_at(base, leaf)) for leaf in leaves} == {int, float}
    for leaf in leaves:
        sent = copy.deepcopy(base)
        _set_at(sent, leaf, f" {_value_at(base, leaf)!r} ")
        result = check(sent, params_root=ROOT)
        assert result["errors"] == [], leaf
        assert result["yaml"] == emit(parse(emit(sent), params_root=ROOT)["mapping"]), leaf
        assert result["yaml"] != emit(sent), leaf


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(
            lambda m: _set_at(m, ("household", "persons", 0, "birth_month"), 13), id="schema-error"
        ),
        pytest.param(lambda m: _set_at(m, ("n_paths",), MAX_PATHS + 1), id="run-refusal"),
    ],
)
def test_check_with_errors_returns_the_text_emit_writes(change: Callable[[Any], None]) -> None:
    m = minimal()
    _set_at(m, ("seed",), "1_000")
    change(m)
    result = check(m, params_root=ROOT)
    assert result["errors"] != []
    assert result["yaml"] == emit(m)
    assert "seed: '1_000'\n" in result["yaml"]


def test_check_validates_the_retyped_text_it_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    real = editor.emit
    calls: list[int] = []

    def counting(mapping: Any) -> str:
        calls.append(1)
        text = real(mapping)
        if len(calls) == 2:
            return changed(text, "start_year: 2026", "start_year: 2040")
        return text

    monkeypatch.setattr(editor, "emit", counting)
    m = minimal()
    _set_at(m, ("seed",), "1_000")
    result = check(m, params_root=ROOT)
    assert len(calls) == 2
    assert "start_year: 2040\n" in result["yaml"]
    assert "seed: 1000\n" in result["yaml"]
    (error,) = result["errors"]
    assert error["loc"] == []
    assert error["message"].startswith("No parameters for tax year 2040")


def _accented() -> Any:
    m = minimal()
    m["name"] = "Élodie et François"
    return m


def _size_error(n: str, limit: str) -> dict[str, Any]:
    return {
        "loc": [],
        "message": (
            f"the form's YAML is {n} bytes, more than the {limit} "
            "a run's request may hold; a scenario is a few kilobytes."
        ),
    }


def test_check_passes_a_text_of_exactly_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _accented()
    size = len(check(m, params_root=ROOT)["yaml"].encode("utf-8"))
    monkeypatch.setattr(editor, "MAX_BODY_BYTES", size)
    assert check(m, params_root=ROOT)["errors"] == []


def test_check_refuses_a_text_one_byte_over_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _accented()
    text = check(m, params_root=ROOT)["yaml"]
    size = len(text.encode("utf-8"))
    assert size == 934
    monkeypatch.setattr(editor, "MAX_BODY_BYTES", size - 1)
    assert len(text) <= size - 1
    result = check(m, params_root=ROOT)
    assert result["yaml"] == text
    assert result["errors"] == [_size_error("934", "933")]


def test_check_measures_the_text_it_returns_not_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m = minimal()
    _set_at(m, ("spending", "schedule", 0, "annual"), "1e15")
    first = len(emit(m).encode("utf-8"))
    returned = check(m, params_root=ROOT)["yaml"]
    assert first < len(returned.encode("utf-8"))
    monkeypatch.setattr(editor, "MAX_BODY_BYTES", first)
    result = check(m, params_root=ROOT)
    assert result["yaml"] == returned
    assert result["errors"] == [
        _size_error(f"{len(returned.encode('utf-8')):,}", f"{first:,}"),
    ]


def test_check_adds_no_size_error_to_a_first_text_with_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m = minimal()
    _set_at(m, ("household", "persons", 0, "birth_month"), 13)
    expected = check(m, params_root=ROOT)
    assert expected["errors"] != []
    monkeypatch.setattr(editor, "MAX_BODY_BYTES", 1)
    result = check(m, params_root=ROOT)
    assert result == expected
    assert all("bytes" not in error["message"] for error in result["errors"])


def test_check_adds_no_size_error_to_a_retyped_text_with_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = editor.emit
    calls: list[int] = []

    def counting(mapping: Any) -> str:
        calls.append(1)
        text = real(mapping)
        if len(calls) == 2:
            return changed(text, "start_year: 2026", "start_year: 2040")
        return text

    monkeypatch.setattr(editor, "emit", counting)
    monkeypatch.setattr(editor, "MAX_BODY_BYTES", 1)
    m = minimal()
    _set_at(m, ("seed",), "1_000")
    result = check(m, params_root=ROOT)
    assert len(calls) == 2
    (error,) = result["errors"]
    assert error["loc"] == []
    assert error["message"].startswith("No parameters for tax year 2040")


@pytest.mark.parametrize(
    "path", sorted((REPO_ROOT / "scenarios").glob("*.yaml")), ids=lambda p: p.name
)
def test_check_differs_from_an_unbounded_check_only_by_the_size_error(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = mapping_of(path.read_text(encoding="utf-8"))
    real = check(m, params_root=ROOT)
    assert real["errors"] == []
    monkeypatch.setattr(editor, "MAX_BODY_BYTES", 10**12)
    unbounded = check(m, params_root=ROOT)
    assert unbounded == real
    size = len(real["yaml"].encode("utf-8"))
    monkeypatch.setattr(editor, "MAX_BODY_BYTES", size - 1)
    refused = check(m, params_root=ROOT)
    assert refused["yaml"] == real["yaml"]
    assert refused["errors"] == [_size_error(f"{size:,}", f"{size - 1:,}")]


def test_check_passes_a_routed_parameter_error_through(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise RoutedParameterError("x")

    monkeypatch.setattr(editor, "check_scenario", refuse)
    with pytest.raises(RoutedParameterError):
        check(example(), params_root=ROOT)


@pytest.fixture
def empty_cache() -> Iterator[None]:
    editor._load_year.cache_clear()
    try:
        yield
    finally:
        editor._load_year.cache_clear()


@pytest.mark.usefixtures("empty_cache")
def test_check_loads_each_parameter_year_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real = editor.load_year

    def counting(year: int, root: Path) -> Any:
        calls.append(year)
        return real(year, root=root)

    monkeypatch.setattr(editor, "load_year", counting)
    m = example()
    check(m, params_root=ROOT)
    check(m, params_root=ROOT)
    assert len(calls) == 1


# --- schema ---------------------------------------------------------------------------


def _without_descriptions(document: dict[str, Any]) -> dict[str, Any]:
    document.pop("description", None)
    for definition in document["$defs"].values():
        definition.pop("description", None)
    return document


def _node(document: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = document
    for segment in path:
        node = node[segment]
    return node


def test_schema_drops_class_descriptions_and_adds_only_the_table() -> None:
    document = schema()
    assert "description" not in document
    assert all("description" not in definition for definition in document["$defs"].values())
    for path, keywords in editor._KEYWORDS:
        node = _node(document, path)
        for key in keywords:
            del node[key]
    assert document == _without_descriptions(Scenario.model_json_schema())


def test_schema_carries_every_keyword_at_its_node() -> None:
    document = schema()
    for path, keywords in editor._KEYWORDS:
        node = _node(document, path)
        for key, value in keywords.items():
            assert node[key] == value
    allocations = _node(document, ("$defs", "Assumptions", "properties", "allocations"))
    assert allocations["propertyNames"]["enum"] == sorted(INVESTABLE_KINDS | {DEFAULT_ALLOCATION})
    weights = _node(document, ("$defs", "ContributionRule", "properties", "weights"))
    assert weights["propertyNames"]["enum"] == sorted(CONTRIBUTION_KINDS)
    spill = _node(document, ("$defs", "ContributionRule", "properties", "spill_order", "items"))
    assert spill["enum"] == sorted(CONTRIBUTION_KINDS)
    order = _node(document, ("$defs", "WithdrawalRule", "properties", "order", "items"))
    assert order["enum"] == sorted(WITHDRAWAL_KINDS)


def test_schema_exactly_one_list_is_true() -> None:
    fields = schema()["$defs"]["CppEntitlement"]["x-northplan-exactly-one"]
    assert sorted(fields) == sorted(CppEntitlement.model_fields)
    values = {"contributory_history": 0.5, "in_pay_monthly": 1000.0}
    for field in fields:
        CppEntitlement.model_validate({field: values[field]})
    with pytest.raises(pydantic.ValidationError):
        CppEntitlement.model_validate(values)
    with pytest.raises(pydantic.ValidationError):
        CppEntitlement.model_validate({})


def _kinds_named(node: Any, found: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("x-northplan-defines", "x-northplan-keys", "x-northplan-ref"):
                found.add(value)
            _kinds_named(value, found)
    elif isinstance(node, list):
        for item in node:
            _kinds_named(item, found)


def test_schema_kinds_are_kinds_references_accepts() -> None:
    found: set[str] = set()
    _kinds_named(schema(), found)
    assert found == {"person", "asset-class"}
    for kind in found:
        references(example(), kind, "a")  # type: ignore[arg-type]


def test_schema_is_json_and_fresh_each_call() -> None:
    first = schema()
    json.dumps(first)
    first["$defs"]["Person"]["properties"]["id"]["x-northplan-defines"] = "changed"
    first["properties"].clear()
    second = schema()
    assert second["$defs"]["Person"]["properties"]["id"]["x-northplan-defines"] == "person"
    assert second["properties"]


# --- references -----------------------------------------------------------------------


def test_references_lists_every_place_a_person_is_named() -> None:
    assert references(example(), "person", "a") == PERSON_A_REFERENCES


def test_references_counts_a_zero_weight_class() -> None:
    m = example()
    m["assumptions"]["allocations"]["rrsp"] = {"equity": 1.0, "bonds": 0.0}
    assert ["assumptions", "allocations", "rrsp", "bonds"] in references(m, "asset-class", "bonds")


@pytest.mark.parametrize(
    "malformed",
    [
        {},
        {"policies": 3},
        {"household": {"beneficiaries": [1]}},
        {"grid": {"a.b": [1]}, "policies": [None]},
        {"assumptions": {"allocations": {"x": 1}}},
    ],
)
def test_references_reads_a_malformed_mapping_without_raising(malformed: Any) -> None:
    assert references(malformed, "person", "a") == []
    assert references(malformed, "asset-class", "a") == []


def test_an_unknown_kind_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="kind must be 'person' or 'asset-class', not 'pet'"):
        references(example(), "pet", "a")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind must be"):
        rename(example(), "pet", "a", "b")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind must be"):
        remove(example(), "pet", "a")  # type: ignore[arg-type]


# --- person edits ---------------------------------------------------------------------


def test_renaming_a_person_carries_every_reference() -> None:
    renamed = rename(example(), "person", "a", "z")
    assert check_errors(renamed) == []
    expected = [
        [
            "z"
            if part == "a"
            else part[:-1] + "z"
            if isinstance(part, str) and part.endswith(".a")
            else part
            for part in loc
        ]
        for loc in PERSON_A_REFERENCES
    ]
    assert references(renamed, "person", "z") == expected
    assert references(renamed, "person", "a") == []
    assert renamed["household"]["beneficiaries"][0]["resp"]["subscriber"] == "z"


def test_renaming_a_person_keeps_the_order_of_each_election_map() -> None:
    m = example()
    elections = m["policies"][0]["elections"]
    elections["cpp_start_age_years"] = {"p": 60, "a": 65, "q": 70}
    elections["oas_start_age_years"] = {"a": 65, "q": 70}
    m["grid"] = {
        "x.y": [1],
        "elections.cpp_start_age_years.a": [60],
        "elections.oas_start_age_years.a": [60],
    }
    renamed = rename(m, "person", "a", "z")
    after = renamed["policies"][0]["elections"]
    assert list(after["cpp_start_age_years"]) == ["p", "z", "q"]
    assert list(after["oas_start_age_years"]) == ["z", "q"]
    assert list(renamed["grid"]) == [
        "x.y",
        "elections.cpp_start_age_years.z",
        "elections.oas_start_age_years.z",
    ]


def test_renaming_a_person_is_invertible_and_leaves_its_argument_alone() -> None:
    m = example()
    before = copy.deepcopy(m)
    there = rename(m, "person", "a", "z")
    back = rename(there, "person", "z", "a")
    assert strict(back) == strict(m)
    assert strict(m) == strict(before)
    assert strict(there) != strict(m)


def test_renaming_a_person_to_itself_returns_a_copy() -> None:
    m = example()
    same = rename(m, "person", "a", "a")
    assert strict(same) == strict(m)
    assert same is not m


def test_renaming_a_missing_person_is_refused() -> None:
    with pytest.raises(EditRefusedError) as raised:
        rename(example(), "person", "nobody", "z")
    assert str(raised.value) == "household.persons: no person has id 'nobody'."
    assert raised.value.references == []


def test_renaming_an_ambiguous_person_is_refused() -> None:
    m = example()
    m["household"]["persons"].append(copy.deepcopy(m["household"]["persons"][0]))
    with pytest.raises(EditRefusedError) as raised:
        rename(m, "person", "a", "z")
    assert str(raised.value) == "household.persons: more than one person has id 'a'."


def test_renaming_a_person_to_a_beneficiarys_id_is_refused() -> None:
    with pytest.raises(EditRefusedError) as raised:
        rename(example(), "person", "a", "child1")
    assert str(raised.value) == "household: id 'child1' is already in use."


def test_renaming_a_person_to_another_persons_id_is_refused() -> None:
    with pytest.raises(EditRefusedError) as raised:
        rename(couple(), "person", "a", "b")
    assert str(raised.value) == "household: id 'b' is already in use."
    assert raised.value.references == []


def test_an_edit_refusal_is_not_a_value_error() -> None:
    assert not issubclass(EditRefusedError, ValueError)


def test_renaming_a_person_to_a_name_already_referenced_is_refused() -> None:
    m = example()
    m["policies"][0]["elections"]["cpp_start_age_years"]["q"] = 65
    with pytest.raises(EditRefusedError) as raised:
        rename(m, "person", "a", "q")
    assert str(raised.value) == (
        "person 'q' is already named in 1 place; renaming 'a' to it would merge those with its own."
    )
    assert raised.value.references == [["policies", 0, "elections", "cpp_start_age_years", "q"]]


def test_removing_a_referenced_person_is_refused_with_its_references() -> None:
    with pytest.raises(EditRefusedError) as raised:
        remove(example(), "person", "a")
    assert str(raised.value) == "person 'a' is still named in 4 places; remove those first."
    assert raised.value.references == PERSON_A_REFERENCES


def test_removing_an_unreferenced_person() -> None:
    m = couple()
    assert [p["id"] for p in m["household"]["persons"]] == ["a", "b"]
    assert references(m, "person", "b") == []
    after = remove(m, "person", "b")
    assert [p["id"] for p in after["household"]["persons"]] == ["a"]
    assert [p["id"] for p in m["household"]["persons"]] == ["a", "b"]


def test_removing_an_ambiguous_person_is_refused() -> None:
    m = example()
    m["household"]["persons"].append(copy.deepcopy(m["household"]["persons"][0]))
    with pytest.raises(EditRefusedError) as raised:
        remove(m, "person", "a")
    assert str(raised.value) == "household.persons: more than one person has id 'a'."


def test_removing_a_missing_person_is_refused() -> None:
    with pytest.raises(EditRefusedError, match="no person has id 'nobody'"):
        remove(example(), "person", "nobody")


# --- asset-class edits ----------------------------------------------------------------


def test_renaming_an_asset_class_keeps_every_position() -> None:
    m = example()
    m["assumptions"]["allocations"]["rrsp"] = {"bonds": 0.5, "equity": 0.5}
    before = copy.deepcopy(m)
    renamed = rename(m, "asset-class", "equity", "stocks")
    assumptions = renamed["assumptions"]
    assert list(assumptions["asset_classes"]) == ["stocks", "bonds"]
    assert list(assumptions["allocations"]["default"]) == ["stocks", "bonds"]
    assert list(assumptions["allocations"]["rrsp"]) == ["bonds", "stocks"]
    assert assumptions["correlation"] == m["assumptions"]["correlation"]
    assert strict(m) == strict(before)
    assert check_errors(renamed) == []


def test_renaming_an_asset_class_to_itself_returns_a_copy() -> None:
    m = example()
    same = rename(m, "asset-class", "equity", "equity")
    assert strict(same) == strict(m)
    assert same is not m


def test_renaming_an_asset_class_is_refused_when_it_would_collide() -> None:
    m = example()
    with pytest.raises(EditRefusedError) as raised:
        rename(m, "asset-class", "nothing", "stocks")
    assert str(raised.value) == "assumptions.asset_classes: no class named 'nothing'."
    with pytest.raises(EditRefusedError) as raised:
        rename(m, "asset-class", "equity", "bonds")
    assert str(raised.value) == "assumptions.asset_classes: a class named 'bonds' already exists."
    m["assumptions"]["allocations"]["default"]["q"] = 0.0
    with pytest.raises(EditRefusedError) as raised:
        rename(m, "asset-class", "equity", "q")
    assert str(raised.value) == (
        "asset class 'q' is already named in 1 place; renaming 'equity' to it would merge "
        "those with its own."
    )
    assert raised.value.references == [["assumptions", "allocations", "default", "q"]]


def test_asset_class_edits_refuse_classes_that_are_not_a_mapping() -> None:
    m = example()
    m["assumptions"]["asset_classes"] = []
    message = "assumptions.asset_classes is not a mapping of classes."
    for edit in (
        lambda: rename(m, "asset-class", "equity", "stocks"),
        lambda: remove(m, "asset-class", "equity"),
        lambda: add_asset_class(m, "gold"),
        lambda: move_asset_class(m, "equity", 0),
    ):
        with pytest.raises(EditRefusedError) as raised:
            edit()
        assert str(raised.value) == message


def test_adding_an_asset_class() -> None:
    m = example()
    before = copy.deepcopy(m)
    added = add_asset_class(m, "gold")
    assumptions = added["assumptions"]
    assert list(assumptions["asset_classes"]) == ["equity", "bonds", "gold"]
    assert assumptions["asset_classes"]["gold"] == {field: 0.0 for field in AssetClass.model_fields}
    assert assumptions["correlation"] == [[1.0, 0.1, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert check_errors(added) == []
    assert strict(m) == strict(before)


def test_adding_an_asset_class_is_refused_when_it_cannot_be_added() -> None:
    m = example()
    with pytest.raises(EditRefusedError) as raised:
        add_asset_class(m, "equity")
    assert str(raised.value) == "assumptions.asset_classes: a class named 'equity' already exists."
    referenced = copy.deepcopy(m)
    referenced["assumptions"]["allocations"]["default"]["gold"] = 0.0
    with pytest.raises(EditRefusedError) as raised:
        add_asset_class(referenced, "gold")
    assert str(raised.value) == (
        "asset class 'gold' is already named in 1 place; remove those first."
    )
    assert raised.value.references == [["assumptions", "allocations", "default", "gold"]]
    ragged = copy.deepcopy(m)
    del ragged["assumptions"]["correlation"][1]
    with pytest.raises(EditRefusedError) as raised:
        add_asset_class(ragged, "gold")
    assert str(raised.value) == (
        "assumptions.correlation is not a 2x2 matrix for the 2 asset classes; correct it first."
    )


def _three_classes() -> Any:
    m = add_asset_class(example(), "gold")
    m["assumptions"]["correlation"] = [[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]]
    return m


def test_moving_an_asset_class_permutes_the_correlation_with_it() -> None:
    m = _three_classes()
    before = copy.deepcopy(m)
    moved = move_asset_class(m, "gold", 0)
    old_names = list(m["assumptions"]["asset_classes"])
    new_names = list(moved["assumptions"]["asset_classes"])
    assert new_names == ["gold", "equity", "bonds"]
    old = m["assumptions"]["correlation"]
    new = moved["assumptions"]["correlation"]
    assert old != new
    for first in new_names:
        for second in new_names:
            assert (
                new[new_names.index(first)][new_names.index(second)]
                == old[old_names.index(first)][old_names.index(second)]
            )
    assert check_errors(moved) == []
    assert strict(m) == strict(before)


def test_moving_an_asset_class_to_its_own_index_returns_a_copy() -> None:
    m = _three_classes()
    same = move_asset_class(m, "equity", 0)
    assert strict(same) == strict(m)
    assert same is not m


@pytest.mark.parametrize(
    ("name", "index", "message"),
    [
        ("nothing", 0, "assumptions.asset_classes: no class named 'nothing'."),
        ("gold", -1, "index must be an integer from 0 to 2, not -1."),
        ("gold", 3, "index must be an integer from 0 to 2, not 3."),
        ("gold", True, "index must be an integer from 0 to 2, not True."),
    ],
)
def test_moving_an_asset_class_is_refused_for_a_bad_name_or_index(
    name: str, index: Any, message: str
) -> None:
    with pytest.raises(EditRefusedError) as raised:
        move_asset_class(_three_classes(), name, index)
    assert str(raised.value) == message


def test_moving_an_asset_class_is_refused_for_a_correlation_that_is_not_square() -> None:
    m = _three_classes()
    del m["assumptions"]["correlation"][2]
    with pytest.raises(EditRefusedError) as raised:
        move_asset_class(m, "gold", 0)
    assert str(raised.value) == (
        "assumptions.correlation is not a 3x3 matrix for the 3 asset classes; correct it first."
    )


def test_removing_an_asset_class_undoes_adding_it() -> None:
    m = example()
    assert strict(remove(add_asset_class(m, "gold"), "asset-class", "gold")) == strict(m)


def test_removing_a_referenced_asset_class_is_refused() -> None:
    with pytest.raises(EditRefusedError) as raised:
        remove(example(), "asset-class", "equity")
    assert (
        str(raised.value) == "asset class 'equity' is still named in 2 places; remove those first."
    )
    assert raised.value.references == [
        ["assumptions", "allocations", "default", "equity"],
        ["assumptions", "allocations", "resp", "equity"],
    ]


def test_removing_an_asset_class_is_refused_for_a_missing_name_or_a_ragged_correlation() -> None:
    with pytest.raises(EditRefusedError) as raised:
        remove(example(), "asset-class", "nothing")
    assert str(raised.value) == "assumptions.asset_classes: no class named 'nothing'."
    m = add_asset_class(example(), "gold")
    del m["assumptions"]["correlation"][0]
    with pytest.raises(EditRefusedError) as raised:
        remove(m, "asset-class", "gold")
    assert str(raised.value) == (
        "assumptions.correlation is not a 3x3 matrix for the 3 asset classes; correct it first."
    )


# --- grid paths -----------------------------------------------------------------------


def _resolves_everywhere(text: str, paths: list[str]) -> None:
    for policy in scenario_from_text(text, "the text", syntax="yaml").policies:
        for path in paths:
            resolve_policy_path(policy, path)


def test_grid_paths_of_the_example() -> None:
    paths = grid_paths(example())
    assert paths == [
        "contribution.weights.rrsp",
        "contribution.weights.tfsa",
        "contribution.weights.taxable",
        "contribution.weights.resp",
        "elections.cpp_start_age_years.a",
        "elections.oas_start_age_years.a",
        "elections.rrif_conversion.age_years",
        "elections.rrif_conversion.fraction",
    ]
    _resolves_everywhere(EXAMPLE_TEXT, paths)


def test_grid_paths_skip_a_person_id_with_a_dot() -> None:
    m = example()
    del m["grid"]
    renamed = rename(m, "person", "a", "a.b")
    result = check(renamed, params_root=ROOT)
    assert result["errors"] == []
    paths = grid_paths(renamed)
    assert paths == [
        path
        for path in grid_paths(example())
        if path
        not in (
            "elections.cpp_start_age_years.a",
            "elections.oas_start_age_years.a",
        )
    ]
    _resolves_everywhere(result["yaml"], paths)


def test_grid_paths_of_the_couple() -> None:
    paths = grid_paths(couple())
    assert paths == [
        "contribution.weights.tfsa",
        "withdrawal.taxable_ceiling_bracket",
        "elections.rrif_conversion.age_years",
        "elections.rrif_conversion.fraction",
    ]
    _resolves_everywhere(COUPLE_TEXT, paths)


def test_grid_paths_are_those_every_policy_has() -> None:
    m = couple()
    other = copy.deepcopy(m["policies"][0])
    other["name"] = "other"
    other["withdrawal"]["taxable_ceiling_bracket"] = None
    m["policies"].append(other)
    assert grid_paths(m) == [
        "contribution.weights.tfsa",
        "elections.rrif_conversion.age_years",
        "elections.rrif_conversion.fraction",
    ]


def _without_huge_rrif_age() -> list[str]:
    paths = grid_paths(example())
    assert "elections.rrif_conversion.age_years" in paths
    return [p for p in paths if p != "elections.rrif_conversion.age_years"]


def test_grid_paths_leave_out_an_integer_too_large_for_a_float() -> None:
    m = example()
    m["policies"][0]["elections"]["rrif_conversion"]["age_years"] = 10**400
    assert grid_paths(m) == _without_huge_rrif_age()


def test_grid_paths_leave_out_an_integer_too_large_for_a_float_on_a_later_policy() -> None:
    m = example()
    other = copy.deepcopy(m["policies"][0])
    other["name"] = "other"
    other["elections"]["rrif_conversion"]["age_years"] = 10**400
    m["policies"].append(other)
    assert grid_paths(m) == _without_huge_rrif_age()


def _without_policies() -> Any:
    m = example()
    del m["policies"]
    return m


def _with_policies(value: Any) -> Any:
    m = example()
    m["policies"] = value
    return m


def _with_invalid_policy() -> Any:
    m = example()
    m["policies"].append({"name": "x"})
    return m


@pytest.mark.parametrize(
    "build",
    [
        _without_policies,
        lambda: _with_policies("x"),
        lambda: _with_policies([]),
        _with_invalid_policy,
    ],
    ids=["absent", "not-a-list", "empty", "invalid-entry"],
)
def test_grid_paths_are_empty_without_valid_policies(build: Callable[[], Any]) -> None:
    assert grid_paths(build()) == []


# --- the minimal scenario -------------------------------------------------------------


def test_the_minimal_scenario_runs() -> None:
    document = simulate_document(
        MINIMAL.read_text(encoding="utf-8"),
        "yaml",
        policy=None,
        paths=None,
        deterministic=True,
        nominal=False,
        params_root=ROOT,
    )
    assert isinstance(document, dict)

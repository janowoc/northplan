# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The editor's calls as JSON text: each answered like the function it names, and every
refusal of the arguments, with its status and exact detail."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from engine.core.indexation import RoutedParameterError
from engine.params.loader import DEFAULT_PARAMS_ROOT
from report import editor
from report.editor_calls import CALLS, answer

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = (REPO_ROOT / "scenarios" / "example.yaml").read_text(encoding="utf-8")
MAPPING = editor.parse(EXAMPLE, params_root=DEFAULT_PARAMS_ROOT)["mapping"]
WITH_GOLD = editor.add_asset_class(MAPPING, "gold")
KINDS_MESSAGE = "kind must be 'person' or 'asset-class', not {!r}."


def call(name: str, arguments_text: str) -> tuple[int, str]:
    return answer(name, arguments_text, params_root=DEFAULT_PARAMS_ROOT)


def detail_of(status: int, body: str, expected_status: int) -> str:
    assert status == expected_status, body
    value = json.loads(body)
    assert list(value) == ["detail"]
    return value["detail"]


def refused(name: str, arguments_text: str) -> str:
    status, body = call(name, arguments_text)
    return detail_of(status, body, 422)


NINE: list[tuple[str, dict[str, Any], Callable[[], Any]]] = [
    ("parse", {"text": EXAMPLE}, lambda: editor.parse(EXAMPLE, params_root=DEFAULT_PARAMS_ROOT)),
    (
        "check",
        {"mapping": MAPPING},
        lambda: editor.check(MAPPING, params_root=DEFAULT_PARAMS_ROOT),
    ),
    ("schema", {}, editor.schema),
    (
        "references",
        {"mapping": MAPPING, "kind": "person", "name": "a"},
        lambda: editor.references(MAPPING, "person", "a"),
    ),
    (
        "rename",
        {"mapping": MAPPING, "kind": "person", "old": "a", "new": "b"},
        lambda: editor.rename(MAPPING, "person", "a", "b"),
    ),
    (
        "remove",
        {"mapping": WITH_GOLD, "kind": "asset-class", "name": "gold"},
        lambda: editor.remove(WITH_GOLD, "asset-class", "gold"),
    ),
    (
        "add-asset-class",
        {"mapping": MAPPING, "name": "gold"},
        lambda: editor.add_asset_class(MAPPING, "gold"),
    ),
    (
        "move-asset-class",
        {"mapping": MAPPING, "name": "bonds", "index": 0},
        lambda: editor.move_asset_class(MAPPING, "bonds", 0),
    ),
    ("grid-paths", {"mapping": MAPPING}, lambda: editor.grid_paths(MAPPING)),
]


@pytest.mark.parametrize(("name", "arguments", "expected"), NINE, ids=[item[0] for item in NINE])
def test_each_call_answers_200_with_its_functions_result(
    name: str, arguments: dict[str, Any], expected: Callable[[], Any]
) -> None:
    assert call(name, json.dumps(arguments)) == (200, json.dumps(expected()))


def test_calls_names_exactly_the_nine_calls_with_their_arguments() -> None:
    assert CALLS == {
        "parse": ("text",),
        "check": ("mapping",),
        "schema": (),
        "references": ("mapping", "kind", "name"),
        "rename": ("mapping", "kind", "old", "new"),
        "remove": ("mapping", "kind", "name"),
        "add-asset-class": ("mapping", "name"),
        "move-asset-class": ("mapping", "name", "index"),
        "grid-paths": ("mapping",),
    }
    assert [item[0] for item in NINE] == list(CALLS)


def test_an_unknown_call_is_a_404_before_the_arguments_are_read() -> None:
    status, body = call("nope", "garbage")
    assert status == 404
    assert body == json.dumps(
        {"detail": f"no editor call named 'nope'; the calls are {sorted(CALLS)!r}."}
    )


@pytest.mark.parametrize(
    ("text", "start"),
    [
        (
            "not json",
            "the arguments are not valid JSON: Expecting value: line 1 column 1 (char 0).",
        ),
        ("", "the arguments are not valid JSON: Expecting value: line 1 column 1 (char 0)."),
        (
            '{"text": 1' + "0" * 5000 + "}",
            "the arguments are not valid JSON: Exceeds the limit (4300 digits)",
        ),
    ],
    ids=["not-json", "empty", "integer-too-long"],
)
def test_arguments_that_are_not_json_are_a_422(text: str, start: str) -> None:
    detail = refused("parse", text)
    assert detail.startswith(start)
    if "Expecting" in start:
        assert detail == start


@pytest.mark.parametrize("name", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_the_infinities_are_refused(name: str) -> None:
    assert refused("check", '{"mapping": ' + name + "}") == (
        f"the arguments are not valid JSON: {name} is not a number JSON allows."
    )


@pytest.mark.parametrize("text", ["1e999", "-1e999"])
def test_a_float_too_large_is_refused(text: str) -> None:
    assert refused("check", '{"mapping": ' + text + "}") == (
        f"the arguments are not valid JSON: {text} is too large for a float."
    )


@pytest.mark.parametrize(
    ("name", "arguments", "key"),
    [
        ("parse", '{"text": "a", "text": "b"}', "text"),
        ("check", '{"mapping": {"a": {"b": 1, "b": 2}}}', "b"),
    ],
    ids=["top", "inside-the-mapping"],
)
def test_a_key_given_twice_is_refused(name: str, arguments: str, key: str) -> None:
    assert refused(name, arguments) == (
        f"the arguments are not valid JSON: the key {key!r} is given twice in one object."
    )


def test_arguments_nested_too_deeply_to_read_are_a_422() -> None:
    text = '{"mapping": ' + "[" * 100_000 + "]" * 100_000 + "}"
    assert refused("check", text) == "the arguments are nested too deeply to read."


@pytest.mark.parametrize(
    ("text", "ending"),
    [("[]", "not list."), ('"x"', "not str."), ("1", "not int."), ("null", "not NoneType.")],
)
def test_arguments_that_are_not_an_object_are_a_422(text: str, ending: str) -> None:
    assert refused("schema", text) == f"the arguments must be a JSON object, {ending}"


def test_an_unknown_argument_is_a_422() -> None:
    assert refused("schema", '{"x": 1, "a": 2}') == "unknown argument 'a'; schema takes []."


def test_an_unknown_argument_is_reported_before_a_missing_one() -> None:
    assert refused("rename", '{"kind": 1, "old": 2, "zz": 3}') == (
        "unknown argument 'zz'; rename takes ['mapping', 'kind', 'old', 'new']."
    )


def test_a_missing_argument_is_the_first_in_the_calls_order() -> None:
    assert refused("rename", '{"mapping": null, "kind": "person"}') == (
        "rename needs the argument 'old'."
    )


@pytest.mark.parametrize("value", ["people", None, 1])
def test_a_kind_that_is_neither_is_a_422(value: object) -> None:
    arguments = json.dumps({"mapping": MAPPING, "kind": value, "name": "a"})
    assert refused("references", arguments) == KINDS_MESSAGE.format(value)


@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_an_index_that_is_not_an_integer_is_a_422(value: object) -> None:
    arguments = json.dumps({"mapping": MAPPING, "name": "bonds", "index": value})
    assert refused("move-asset-class", arguments) == f"index must be an integer, not {value!r}."


STRING_ARGUMENTS = [
    ("parse", "text", {}),
    ("references", "name", {"mapping": MAPPING, "kind": "person"}),
    ("rename", "old", {"mapping": MAPPING, "kind": "person", "new": "b"}),
    ("rename", "new", {"mapping": MAPPING, "kind": "person", "old": "a"}),
    ("remove", "name", {"mapping": MAPPING, "kind": "person"}),
    ("add-asset-class", "name", {"mapping": MAPPING}),
    ("move-asset-class", "name", {"mapping": MAPPING, "index": 0}),
]


@pytest.mark.parametrize(
    ("name", "argument", "others"),
    STRING_ARGUMENTS,
    ids=[f"{item[0]}-{item[1]}" for item in STRING_ARGUMENTS],
)
def test_a_name_that_is_not_a_string_is_a_422(
    name: str, argument: str, others: dict[str, Any]
) -> None:
    arguments = json.dumps({**others, argument: 5})
    assert refused(name, arguments) == f"{argument} must be a string, not 5."


def test_types_are_checked_in_the_calls_order() -> None:
    arguments = json.dumps({"mapping": MAPPING, "kind": "nope", "old": 2, "new": "b"})
    assert refused("rename", arguments) == KINDS_MESSAGE.format("nope")


def test_types_are_checked_in_the_calls_order_not_sorted_order() -> None:
    arguments = json.dumps({"mapping": MAPPING, "kind": "person", "old": 2, "new": 3})
    assert refused("rename", arguments) == "old must be a string, not 2."


@pytest.mark.parametrize("value", [None, [], "x", 5], ids=["null", "list", "string", "int"])
def test_a_mapping_of_any_json_type_is_passed_to_the_call(value: object) -> None:
    expected = editor.check(value, params_root=DEFAULT_PARAMS_ROOT)
    assert call("check", json.dumps({"mapping": value})) == (200, json.dumps(expected))


def test_a_refused_edit_is_a_409_with_its_references() -> None:
    with pytest.raises(editor.EditRefusedError) as raised:
        editor.remove(MAPPING, "person", "a")
    status, body = call("remove", json.dumps({"mapping": MAPPING, "kind": "person", "name": "a"}))
    assert status == 409
    assert json.loads(body) == {
        "detail": str(raised.value),
        "references": json.loads(json.dumps(raised.value.references)),
    }
    assert json.loads(body)["detail"] == (
        "person 'a' is still named in 4 places; remove those first."
    )
    assert json.loads(body)["references"]


def test_a_refused_edit_with_no_places_has_empty_references() -> None:
    status, body = call(
        "rename", json.dumps({"mapping": MAPPING, "kind": "person", "old": "zz", "new": "b"})
    )
    assert status == 409
    assert json.loads(body)["references"] == []


def test_a_mapping_nested_too_deeply_to_work_on_is_a_422() -> None:
    nested = "[" * 400 + "]" * 400
    assert refused("check", '{"mapping": ' + nested + "}") == (
        "the mapping is nested too deeply to work on."
    )


def test_another_exception_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise RoutedParameterError("boom")

    monkeypatch.setattr(editor, "check", boom)
    with pytest.raises(RoutedParameterError, match="boom"):
        call("check", '{"mapping": null}')


def test_a_nan_in_a_result_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(editor, "grid_paths", lambda _mapping: [float("nan")])
    with pytest.raises(ValueError, match="not JSON compliant"):
        call("grid-paths", '{"mapping": null}')


def test_the_body_is_ascii() -> None:
    arguments = json.dumps({"mapping": MAPPING, "kind": "person", "old": "a", "new": "\ud800"})
    status, body = call("rename", arguments)
    assert status == 200
    assert body.isascii()
    assert "\\ud800" in body
    elections = json.loads(body)["policies"][0]["elections"]
    assert "\ud800" in elections["cpp_start_age_years"]

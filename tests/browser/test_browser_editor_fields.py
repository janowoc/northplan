# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The form's fields on the API page: every value of the committed scenarios is shown, typed
numbers reach the YAML exactly, the engine's errors appear at the field or section they
concern (or only in the list), and the optional, nullable, choice and list controls edit the
mapping as the schema says.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from form_helpers import (
    COUPLE,
    EXAMPLE,
    MINIMAL,
    back_to_yaml,
    button,
    control,
    field,
    field_errors,
    fill,
    mapping_of,
    native_errors,
    open_text,
    own,
    shown_errors,
    tick,
    wait_clean,
    wait_errors,
    wait_settled,
)
from offline import Offline
from playwright.sync_api import Page, expect

from engine.params.loader import DEFAULT_PARAMS_ROOT
from report import editor

SCHEMA = editor.schema()


def minimal() -> Any:
    return copy.deepcopy(mapping_of(MINIMAL))


def native_checked(mapping: Any) -> list[dict[str, Any]]:
    return editor.check(mapping, params_root=DEFAULT_PARAMS_ROOT)["errors"]


# --- the walk: the classification of the form's design, applied to a native mapping ---------


def resolve(node: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in node:
        node = SCHEMA["$defs"][node["$ref"].rsplit("/", 1)[1]]
    return node


def unwrap(node: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    options = node.get("anyOf")
    if options and len(options) == 2 and {"type": "null"} in options:
        other = next(o for o in options if o != {"type": "null"})
        return other, True
    return node, False


def kind_of(node: dict[str, Any]) -> str:
    node, _ = unwrap(node)
    res = resolve(node)

    def get(key: str) -> Any:
        return res[key] if key in res else node.get(key)

    extra = get("additionalProperties")
    extra = resolve(extra) if isinstance(extra, dict) else None
    if get("x-northplan-widget") in ("correlation", "grid"):
        return get("x-northplan-widget")
    if get("x-northplan-defines") == "asset-class":
        return "asset-classes"
    if "properties" in res:
        return "model"
    if (
        get("type") == "object"
        and "propertyNames" in res
        and extra is not None
        and "additionalProperties" in extra
    ):
        return "map-of-maps"
    if get("type") == "object" and extra is not None and extra.get("type") in ("number", "integer"):
        return "number-map"
    if get("type") == "array" and "properties" in resolve(get("items")):
        return "list"
    if get("type") == "array" and get("uniqueItems") and "enum" in resolve(get("items")):
        return "pick"
    if get("type") == "string" and "enum" in res:
        return "select"
    if get("type") == "string" and get("x-northplan-ref") == "person":
        return "person-select"
    if get("type") == "string" and get("x-northplan-defines") == "person":
        return "person-id"
    if get("type") == "string":
        return "text"
    if get("type") in ("number", "integer"):
        return "number"
    if get("type") == "boolean":
        return "checkbox"
    return "raw"


def shown_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value)


def walk_value(
    page: Page, node: dict[str, Any], value: Any, path: list[Any], seen: list[str]
) -> None:
    inner, _ = unwrap(node)
    kind = kind_of(node)
    res = resolve(inner)
    if kind == "model":
        walk_model(page, res, value, path, seen)
    elif kind == "list":
        item = resolve(res["items"])
        for i, element in enumerate(value):
            walk_model(page, item, element, [*path, i], seen)
    elif kind == "asset-classes":
        item = resolve(res["additionalProperties"])
        for name, cls in value.items():
            assert control_of(page, "editor-name", [*path, name]).input_value() == name
            seen.append("name")
            walk_model(page, item, cls, [*path, name], seen)
    elif kind == "number-map":
        for key, number in value.items():
            expect(control(page, *path, key)).to_have_value(json.dumps(number))
            seen.append("number")
    elif kind == "map-of-maps":
        for account, weights in value.items():
            for key, number in weights.items():
                expect(control(page, *path, account, key)).to_have_value(json.dumps(number))
                seen.append("number")
    elif kind == "grid":
        for key, values in value.items():
            expect(control(page, *path, key)).to_have_value(
                ", ".join(json.dumps(v) for v in values)
            )
            seen.append("grid")
    elif kind == "pick":
        for i, item_value in enumerate(value):
            span = field(page, *path, i).locator(":scope > span.editor-value")
            expect(span).to_have_text(item_value)
            seen.append("pick")
    elif kind == "correlation":
        for i, row in enumerate(value):
            for j, cell in enumerate(row):
                expect(control(page, *path, i, j)).to_have_value(json.dumps(cell))
                seen.append("cell")
    elif kind == "checkbox":
        assert control(page, *path).is_checked() is value
        seen.append("checkbox")
    else:
        expect(control(page, *path)).to_have_value(shown_text(value))
        seen.append("scalar")


def control_of(page: Page, cls: str, path: list[Any]) -> Any:
    return own(page, cls, *path)


def walk_model(
    page: Page, res: dict[str, Any], value: dict[str, Any], path: list[Any], seen: list[str]
) -> None:
    required = res.get("required", [])
    choice = "x-northplan-exactly-one" in res
    for key, node in res["properties"].items():
        _, nullable = unwrap(node)
        child = [*path, key]
        if not choice:
            if key not in required:
                present = own(page, "editor-present", *child)
                assert present.is_checked() is (key in value)
                seen.append("present")
            if key in required and nullable and value.get(key, 0) is None:
                assert own(page, "editor-null", *child).is_checked()
                assert control(page, *child).is_disabled()
                seen.append("null")
                continue
        if key in value:
            if value[key] is None:
                expect(control(page, *child)).to_have_value("")
            else:
                walk_value(page, node, value[key], child, seen)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(EXAMPLE, id="example"),
        pytest.param(COUPLE, id="couple"),
        pytest.param(MINIMAL, id="minimal"),
    ],
)
def test_every_value_of_each_committed_scenario_is_shown(
    api_form_page: tuple[Page, Offline], text: str
) -> None:
    page, _ = api_form_page
    open_text(page, text)
    wait_clean(page)
    seen: list[str] = []
    walk_model(page, SCHEMA, mapping_of(text), [], seen)
    assert {"scalar", "number", "present", "name", "cell"} <= set(seen)
    if text == MINIMAL:
        assert "null" in seen
    if text == EXAMPLE:
        assert {"grid", "checkbox", "pick"} <= set(seen)


def test_typed_numbers_reach_the_yaml_exactly(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    expected = minimal()
    expected["seed"] = 18446744073709551617
    expected["spending"]["survivor_share"] = 0.50
    expected["spending"]["schedule"][0]["annual"] = 95000
    expected["assumptions"]["inflation"] = 0.02
    expected["assumptions"]["asset_classes"]["cash"]["vol"] = 1e-9
    expected["household"]["persons"][0]["prior_year_net_income"] = 7
    assert native_checked(expected) == []
    open_text(page, MINIMAL)
    fill(page, "18446744073709551617", "seed")
    fill(page, "0.50", "spending", "survivor_share")
    fill(page, " +95000 ", "spending", "schedule", 0, "annual")
    fill(page, ".02", "assumptions", "inflation")
    fill(page, "1e-9", "assumptions", "asset_classes", "cash", "vol")
    fill(page, "007", "household", "persons", 0, "prior_year_net_income")
    wait_clean(page)
    assert back_to_yaml(page) == editor.emit(expected)


def test_a_bad_value_shows_the_engines_error_at_its_field(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    edited = minimal()
    edited["household"]["persons"][0]["birth_year"] = "abc"
    errors = native_checked(edited)
    assert len(errors) == 1
    open_text(page, MINIMAL)
    path = ("household", "persons", 0, "birth_year")
    fill(page, "abc", *path)
    wait_errors(page)
    assert field_errors(page, *path) == [errors[0]["message"]]
    assert shown_errors(page) == native_errors(edited)
    expect(button(page, "Back to YAML")).to_be_disabled()
    expect(button(page, "Download YAML")).to_be_disabled()
    expect(page.locator("#run")).to_be_disabled()
    fill(page, "1970", *path)
    wait_clean(page)
    expect(button(page, "Back to YAML")).to_be_enabled()
    expect(button(page, "Download YAML")).to_be_enabled()
    expect(page.locator("#run")).to_be_enabled()


def test_a_section_error_shows_at_its_section(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    edited = minimal()
    edited["assumptions"]["allocations"]["default"]["cash"] = 0.5
    errors = [e for e in native_checked(edited) if e["loc"] == ["assumptions"]]
    assert len(errors) == 1
    open_text(page, MINIMAL)
    fill(page, "0.5", "assumptions", "allocations", "default", "cash")
    wait_errors(page)
    assert field_errors(page, "assumptions") == [errors[0]["message"]]
    assert shown_errors(page) == native_errors(edited)


def test_a_scenario_level_error_shows_only_in_the_list(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    edited = minimal()
    del edited["policies"][0]["elections"]["cpp_start_age_years"]["a"]
    assert native_errors(edited)
    open_text(page, MINIMAL)
    fill(page, "", "policies", 0, "elections", "cpp_start_age_years", "a")
    wait_errors(page)
    assert shown_errors(page) == native_errors(edited)
    expect(page.locator("#form .editor-field-errors li")).to_have_count(0)


def test_an_optional_field_is_added_filled_and_removed(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    edited = minimal()
    open_text(page, MINIMAL)
    tick(page, "editor-present", "risk_aversion")
    wait_errors(page)
    assert control(page, "risk_aversion").input_value() == ""
    edited["risk_aversion"] = ""
    assert shown_errors(page) == native_errors(edited)
    fill(page, "2", "risk_aversion")
    wait_clean(page)
    tick(page, "editor-present", "risk_aversion")
    wait_clean(page)
    expect(control(page, "risk_aversion")).to_have_count(0)
    tick(page, "editor-present", "risk_aversion")
    fill(page, "2", "risk_aversion")
    wait_clean(page)
    edited["risk_aversion"] = 2
    assert back_to_yaml(page) == editor.emit(edited)


def test_an_optional_section_starts_from_its_skeleton_and_default(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    open_text(page, MINIMAL)
    person = ("household", "persons", 0)
    tick(page, "editor-present", *person, "accounts")
    tick(page, "editor-present", *person, "accounts", "tfsa")
    tick(page, "editor-present", *person, "accounts", "tfsa", "balance")
    tick(page, "editor-present", *person, "employment")
    wait_clean(page)
    expected = minimal()
    expected["household"]["persons"][0]["accounts"] = {"tfsa": {"balance": 0.0}}
    expected["household"]["persons"][0]["employment"] = []
    text = back_to_yaml(page)
    assert text == editor.emit(expected)
    assert "balance: 0.0" in text


def test_the_none_box_writes_null(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    path = ("policies", 0, "withdrawal", "taxable_ceiling_bracket")
    open_text(page, MINIMAL)
    assert own(page, "editor-null", *path).is_checked()
    assert control(page, *path).is_disabled()
    tick(page, "editor-null", *path)
    wait_errors(page)
    edited = minimal()
    edited["policies"][0]["withdrawal"]["taxable_ceiling_bracket"] = ""
    assert shown_errors(page) == native_errors(edited)
    fill(page, "0", *path)
    wait_clean(page)
    tick(page, "editor-null", *path)
    wait_clean(page)
    assert back_to_yaml(page) == editor.emit(minimal())


def test_the_cpp_choice_keeps_exactly_one(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    cpp = ("household", "persons", 0, "cpp")
    original = mapping_of(COUPLE)
    assert original["household"]["persons"][0]["cpp"] == {"in_pay_monthly": 1150}
    open_text(page, COUPLE)
    field(page, *cpp).locator(":scope > .editor-choice input[value=contributory_history]").click()
    wait_errors(page)
    edited = copy.deepcopy(original)
    edited["household"]["persons"][0]["cpp"] = {"contributory_history": ""}
    assert shown_errors(page) == native_errors(edited)
    fill(page, "0.5", *cpp, "contributory_history")
    wait_settled(page)
    edited["household"]["persons"][0]["cpp"] = {"contributory_history": 0.5}
    assert shown_errors(page) == native_errors(edited)
    field(page, *cpp).locator(":scope > .editor-choice input[value=in_pay_monthly]").click()
    fill(page, "1150", *cpp, "in_pay_monthly")
    wait_clean(page)
    assert back_to_yaml(page) == editor.emit(original)


def test_a_list_item_is_added_and_removed(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    open_text(page, MINIMAL)
    own(page, "editor-add", "spending", "schedule").click()
    wait_errors(page)
    edited = minimal()
    edited["spending"]["schedule"].append({})
    assert shown_errors(page) == native_errors(edited)
    fill(page, "2030", "spending", "schedule", 1, "from_year")
    fill(page, "1000", "spending", "schedule", 1, "annual")
    wait_clean(page)
    tick(page, "editor-remove", "spending", "schedule", 1)
    wait_clean(page)
    assert back_to_yaml(page) == editor.emit(minimal())


def test_a_new_person_starts_from_the_required_sections(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    open_text(page, MINIMAL)
    own(page, "editor-add", "household", "persons").click()
    wait_errors(page)
    edited = minimal()
    edited["household"]["persons"].append({"cpp": {}})
    assert shown_errors(page) == native_errors(edited)
    choice = own(page, "editor-choice", "household", "persons", 1, "cpp")
    expect(choice).to_have_count(1)
    expect(choice.locator("input[type=radio]")).to_have_count(2)
    expect(choice.locator("input[type=radio]:checked")).to_have_count(0)
    at_person = [
        e["message"] for e in native_checked(edited) if e["loc"] == ["household", "persons", 1]
    ]
    assert len(at_person) == 1
    assert field_errors(page, "household", "persons", 1) == at_person


def test_a_policy_starts_from_its_skeleton(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    open_text(page, MINIMAL)
    own(page, "editor-add", "policies").click()
    wait_errors(page)
    edited = minimal()
    edited["policies"].append(
        {
            "contribution": {"weights": {}, "spill_order": []},
            "withdrawal": {"order": []},
            "elections": {
                "cpp_start_age_years": {},
                "oas_start_age_years": {},
                "rrif_conversion": {},
            },
        }
    )
    assert shown_errors(page) == native_errors(edited)


def test_a_new_policy_can_be_filled_until_it_is_valid(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    open_text(page, MINIMAL)
    own(page, "editor-add", "policies").click()
    policy = ("policies", 1)
    fill(page, "second", *policy, "name")
    fill(page, "1.0", *policy, "contribution", "weights", "tfsa")
    order = (*policy, "withdrawal", "order")
    own(page, "editor-add-choice", *order).select_option("tfsa")
    own(page, "editor-add", *order).click()
    tick(page, "editor-null", *policy, "withdrawal", "taxable_ceiling_bracket")
    credit = control(page, *policy, "withdrawal", "fill_pension_credit")
    credit.click()
    credit.click()
    fill(page, "65", *policy, "elections", "cpp_start_age_years", "a")
    fill(page, "65", *policy, "elections", "oas_start_age_years", "a")
    fill(page, "65", *policy, "elections", "rrif_conversion", "age_years")
    fill(page, "0.0", *policy, "elections", "rrif_conversion", "fraction")
    wait_clean(page)
    expected = minimal()
    expected["policies"].append({**copy.deepcopy(expected["policies"][0]), "name": "second"})
    assert back_to_yaml(page) == editor.emit(expected)


def test_an_absent_required_checkbox_is_neither_true_nor_false(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    open_text(page, MINIMAL)
    own(page, "editor-add", "policies").click()
    credit = control(page, "policies", 1, "withdrawal", "fill_pension_credit")
    assert credit.evaluate("(box) => box.indeterminate") is True
    own_credit = control(page, "policies", 0, "withdrawal", "fill_pension_credit")
    assert own_credit.evaluate("(box) => box.indeterminate") is False
    credit.click()
    assert credit.evaluate("(box) => box.indeterminate") is False
    assert credit.is_checked()
    wait_settled(page)
    edited = minimal()
    edited["policies"].append(
        {
            "contribution": {"weights": {}, "spill_order": []},
            "withdrawal": {"order": [], "fill_pension_credit": True},
            "elections": {
                "cpp_start_age_years": {},
                "oas_start_age_years": {},
                "rrif_conversion": {},
            },
        }
    )
    assert shown_errors(page) == native_errors(edited)


def test_a_number_too_large_for_a_float_is_an_error_at_its_field(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    open_text(page, MINIMAL)
    fill(page, "1e400", "assumptions", "inflation")
    wait_errors(page)
    edited = minimal()
    edited["assumptions"]["inflation"] = "1e400"
    found = [
        e["message"] for e in native_checked(edited) if e["loc"] == ["assumptions", "inflation"]
    ]
    assert found
    assert field_errors(page, "assumptions", "inflation") == found
    assert shown_errors(page) == native_errors(edited)
    expect(page.locator(".editor-problem")).to_be_hidden()

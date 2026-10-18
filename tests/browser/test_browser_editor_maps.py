# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The form's maps, lists and grid on the API page: a map's rows are the keys present and then
the keys that could be, a cleared row deletes its key and a filled one adds it at the end,
allocations add and remove an account, a pick list adds, moves and removes, the RESP subscriber
is chosen from the persons, the grid offers the paths that resolve and splits its values on
commas, a grid drawn when the form opens gets its paths from the grid-paths reply, a select adds a
current value its choices lack as an option and keeps it selected, the correlation is headed by
the class names, and Down moves a pick-list item and an asset class.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from form_helpers import (
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
    open_example,
    open_new,
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
PATH = "elections.cpp_start_age_years.a"
GRID_WEIGHTS = ("policies", 0, "contribution", "weights")
ELECTIONS = ("policies", 0, "elections", "cpp_start_age_years")


def minimal() -> Any:
    return copy.deepcopy(mapping_of(MINIMAL))


def rows(page: Page, *path: str | int) -> list[str]:
    return (
        field(page, *path)
        .locator(":scope > .editor-field")
        .evaluate_all("(elements) => elements.map((e) => e.getAttribute('data-loc'))")
    )


def expected_rows(
    path: tuple[str | int, ...], present: list[str], possible: list[str]
) -> list[str]:
    keys = [*present, *(key for key in possible if key not in present)]
    return [json.dumps([*path, key], separators=(",", ":")) for key in keys]


def test_map_rows_list_the_map_then_the_rest(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    contribution = SCHEMA["$defs"]["ContributionRule"]["properties"]["weights"]
    possible = contribution["propertyNames"]["enum"]
    example = mapping_of(EXAMPLE)
    open_example(page)
    present = list(example["policies"][0]["contribution"]["weights"])
    assert present != possible
    assert rows(page, *GRID_WEIGHTS) == expected_rows(GRID_WEIGHTS, present, possible)
    people = [p["id"] for p in example["household"]["persons"]]
    assert rows(page, *ELECTIONS) == expected_rows(ELECTIONS, ["a"], people)
    button(page, "Discard form").click()
    open_text(page, MINIMAL)
    present = list(mapping_of(MINIMAL)["policies"][0]["contribution"]["weights"])
    expected = expected_rows(GRID_WEIGHTS, present, possible)
    assert (
        len(expected) == len(possible)
        and expected[0] != expected_rows(GRID_WEIGHTS, [], possible)[0]
    )
    assert rows(page, *GRID_WEIGHTS) == expected
    fill(page, "", *ELECTIONS, "a")
    wait_errors(page)
    assert rows(page, *ELECTIONS) == expected_rows(ELECTIONS, [], ["a"])
    expect(control(page, *ELECTIONS, "a")).to_have_value("")


def test_a_cleared_row_deletes_its_key_and_a_filled_one_adds_it(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    cleared = copy.deepcopy(mapping_of(EXAMPLE))
    del cleared["policies"][0]["contribution"]["weights"]["rrsp"]
    refilled = copy.deepcopy(cleared)
    refilled["policies"][0]["contribution"]["weights"]["rrsp"] = 0.5
    assert editor.check(refilled, params_root=DEFAULT_PARAMS_ROOT)["errors"] == []
    assert list(refilled["policies"][0]["contribution"]["weights"]) != list(
        mapping_of(EXAMPLE)["policies"][0]["contribution"]["weights"]
    )
    open_example(page)
    fill(page, "", *GRID_WEIGHTS, "rrsp")
    wait_errors(page)
    assert shown_errors(page) == native_errors(cleared)
    fill(page, "0.5", *GRID_WEIGHTS, "rrsp")
    wait_clean(page)
    assert back_to_yaml(page) == editor.emit(refilled)


def test_allocations_add_and_remove_an_account(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    open_text(page, MINIMAL)
    allocations = ("assumptions", "allocations")
    own(page, "editor-add-choice", *allocations).select_option("rrsp")
    own(page, "editor-add", *allocations).click()
    wait_errors(page)
    edited = minimal()
    edited["assumptions"]["allocations"]["rrsp"] = {}
    assert shown_errors(page) == native_errors(edited)
    tick(page, "editor-remove", *allocations, "rrsp")
    wait_clean(page)


def test_a_pick_list_adds_moves_and_removes(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    order = ("policies", 0, "withdrawal", "order")
    open_text(page, MINIMAL)
    edited = minimal()

    def step(expected: list[str]) -> None:
        wait_settled(page)
        edited["policies"][0]["withdrawal"]["order"] = expected
        assert shown_errors(page) == native_errors(edited) == []
        spans = field(page, *order).locator(":scope > .editor-field > span.editor-value")
        expect(spans).to_have_text(expected)

    own(page, "editor-add-choice", *order).select_option("rrsp")
    own(page, "editor-add", *order).click()
    step(["tfsa", "rrsp"])
    tick(page, "editor-up", *order, 1)
    step(["rrsp", "tfsa"])
    tick(page, "editor-remove", *order, 1)
    step(["rrsp"])
    assert back_to_yaml(page) == editor.emit(edited)


def test_the_subscriber_is_chosen_from_the_persons(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    open_example(page)
    subscriber = control(page, "household", "beneficiaries", 0, "resp", "subscriber")
    assert subscriber.evaluate("(s) => Array.from(s.options, (o) => o.value)") == ["", "a"]
    expect(subscriber).to_have_value("a")


def test_the_grid_offers_the_paths_that_resolve(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    open_text(page, MINIMAL)
    tick(page, "editor-present", "grid")
    wait_clean(page)
    paths = editor.grid_paths(mapping_of(MINIMAL))
    assert PATH in paths
    options = own(page, "editor-add-choice", "grid").locator("option")
    expect(options).to_have_text(paths)
    own(page, "editor-add-choice", "grid").select_option(PATH)
    own(page, "editor-add", "grid").click()
    wait_errors(page)
    edited = minimal()
    edited["grid"] = {PATH: []}
    assert shown_errors(page) == native_errors(edited)
    expect(options).to_have_text([path for path in paths if path != PATH])
    fill(page, "60, 65, 70", "grid", PATH)
    wait_clean(page)
    edited["grid"] = {PATH: [60, 65, 70]}
    assert back_to_yaml(page) == editor.emit(edited)


def test_grid_values_are_split_on_commas(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    open_text(page, MINIMAL)
    tick(page, "editor-present", "grid")
    own(page, "editor-add-choice", "grid").select_option(PATH)
    own(page, "editor-add", "grid").click()
    wait_errors(page)
    fill(page, "60, x, ", "grid", PATH)
    wait_errors(page)
    edited = minimal()
    edited["grid"] = {PATH: [60, "x", ""]}
    found = editor.check(edited, params_root=DEFAULT_PARAMS_ROOT)["errors"]
    assert shown_errors(page) == native_errors(edited)
    deeper = [e["message"] for e in found if e["loc"][:2] == ["grid", PATH]]
    assert deeper
    assert field_errors(page, "grid", PATH) == deeper
    fill(page, "  ", "grid", PATH)
    wait_settled(page)
    edited["grid"] = {PATH: []}
    assert shown_errors(page) == native_errors(edited)


def test_a_drawn_grid_gets_its_paths_from_the_reply(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    example = mapping_of(EXAMPLE)
    expected = [p for p in editor.grid_paths(example) if p not in example["grid"]]
    assert expected
    open_example(page)
    options = own(page, "editor-add-choice", "grid").locator("option")
    expect(options).to_have_text(expected)
    assert options.evaluate_all("(o) => o.map((e) => e.value)") == expected


def test_a_value_a_select_does_not_list_is_kept_as_an_option(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    changed = editor.schema()
    changed["$defs"]["Person"]["properties"]["sex"]["enum"] = ["m"]
    page.route(
        "**/api/editor/schema",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(changed)
        ),
    )
    open_new(page)
    sex = control(page, "household", "persons", 0, "sex")
    assert sex.evaluate("(s) => Array.from(s.options, (o) => o.value)") == ["", "m", "f"]
    expect(sex).to_have_value("f")


def test_the_correlation_is_headed_by_the_class_names(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    open_example(page)
    matrix = own(page, "editor-matrix", "assumptions", "correlation")
    assert matrix.locator("tr:first-child > th").all_inner_texts() == ["", "equity", "bonds"]
    assert matrix.locator("tr:not(:first-child) > th").all_inner_texts() == ["equity", "bonds"]


def test_down_moves_an_item_and_a_class(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    order = ("policies", 0, "withdrawal", "order")
    open_text(page, MINIMAL)
    own(page, "editor-add-choice", *order).select_option("rrsp")
    own(page, "editor-add", *order).click()
    tick(page, "editor-down", *order, 0)
    wait_settled(page)
    edited = minimal()
    edited["policies"][0]["withdrawal"]["order"] = ["rrsp", "tfsa"]
    assert shown_errors(page) == native_errors(edited)
    spans = field(page, *order).locator(":scope > .editor-field > span.editor-value")
    expect(spans).to_have_text(["rrsp", "tfsa"])
    button(page, "Discard form").click()
    open_example(page)
    tick(page, "editor-down", "assumptions", "asset_classes", "equity")
    wait_clean(page)
    example = mapping_of(EXAMPLE)
    assert back_to_yaml(page) == editor.emit(editor.move_asset_class(example, "equity", 1))

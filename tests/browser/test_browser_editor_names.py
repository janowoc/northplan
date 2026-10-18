# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Renaming and removing a person or an asset class, and adding and moving a class, on the API
page. These edits go to Python, except that the form sets a person's id, and removes a person,
by itself when the person's current id is held by no person or by more than one. Results are
compared with native oracles, refusals and their references included. It also covers a class's
Up and Down buttons stopping at the ends, the correlation's mirror, and the form being disabled
while a Python edit is out. The page commits an id or a class name on ``change``, so a test
fills the input and presses Tab.
"""

from __future__ import annotations

import copy
from typing import Any

from form_helpers import (
    COUPLE,
    EXAMPLE,
    MINIMAL,
    back_to_yaml,
    button,
    control,
    fill,
    mapping_of,
    native_errors,
    open_example,
    open_text,
    own,
    problem,
    shown_errors,
    state,
    tick,
    wait_clean,
    wait_errors,
)
from offline import Offline
from playwright.sync_api import Page, Route, expect
from test_browser_api_page import wait_until_held

from report import editor

CLASSES = ("assumptions", "asset_classes")
PERSONS = ("household", "persons")


def commit(page: Page, value: str, *path: str | int) -> None:
    box = control(page, *path)
    box.fill(value)
    box.press("Tab")


def commit_name(page: Page, value: str, *path: str | int) -> None:
    box = own(page, "editor-name", *path)
    box.fill(value)
    box.press("Tab")


def refusal(call: Any, *arguments: Any) -> editor.EditRefusedError:
    try:
        call(*arguments)
    except editor.EditRefusedError as error:
        return error
    raise AssertionError("the edit was not refused")


def dotted(references: list[list[Any]]) -> list[str]:
    return [".".join(str(step) for step in reference) for reference in references]


def new_person(identifier: str | None = None) -> dict[str, Any]:
    person: dict[str, Any] = {"cpp": {}}
    if identifier is not None:
        person["id"] = identifier
    return person


def test_renaming_a_person_carries_to_every_reference(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    example = mapping_of(EXAMPLE)
    open_example(page)
    commit(page, "z", *PERSONS, 0, "id")
    wait_clean(page)
    assert back_to_yaml(page) == editor.emit(editor.rename(example, "person", "a", "z"))


def test_a_rename_onto_a_used_id_is_refused_and_put_back(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    couple = mapping_of(COUPLE)
    error = refusal(editor.rename, couple, "person", "a", "b")
    assert str(error) == "household: id 'b' is already in use."
    open_text(page, COUPLE)
    wait_clean(page)
    commit(page, "b", *PERSONS, 0, "id")
    expect(problem(page)).to_have_text(str(error))
    expect(page.locator(".editor-problem li")).to_have_count(0)
    expect(control(page, *PERSONS, 0, "id")).to_have_value("a")
    wait_clean(page)
    expect(button(page, "Back to YAML")).to_be_enabled()


def test_an_id_no_single_person_holds_is_set_directly(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    open_text(page, MINIMAL)
    own(page, "editor-add", *PERSONS).click()
    commit(page, "c", *PERSONS, 1, "id")
    wait_errors(page)
    edited = copy.deepcopy(mapping_of(MINIMAL))
    edited["household"]["persons"].append(new_person("c"))
    assert shown_errors(page) == native_errors(edited)
    commit(page, "a", *PERSONS, 1, "id")
    error = refusal(editor.rename, edited, "person", "c", "a")
    expect(problem(page)).to_have_text(str(error))
    expect(control(page, *PERSONS, 1, "id")).to_have_value("c")
    own(page, "editor-add", *PERSONS).click()
    commit(page, "a", *PERSONS, 2, "id")
    commit(page, "d", *PERSONS, 2, "id")
    wait_errors(page)
    edited["household"]["persons"].append(new_person("d"))
    assert shown_errors(page) == native_errors(edited)
    expect(control(page, *PERSONS, 0, "id")).to_have_value("a")
    expect(control(page, "policies", 0, "elections", "cpp_start_age_years", "a")).to_have_value(
        "65"
    )


def test_removing_a_referenced_person_is_refused_with_its_references(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    error = refusal(editor.remove, mapping_of(EXAMPLE), "person", "a")
    assert "grid.elections.cpp_start_age_years.a" in dotted(error.references)
    open_example(page)
    tick(page, "editor-remove", *PERSONS, 0)
    expect(problem(page)).to_have_text(str(error))
    expect(page.locator(".editor-problem li")).to_have_text(dotted(error.references))
    wait_clean(page)


def test_removing_a_person_nobody_names(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    removes: list[str] = []

    def record(route: Route) -> None:
        removes.append(route.request.url)
        route.fallback()

    page.route("**/api/editor/remove", record)
    open_text(page, MINIMAL)
    own(page, "editor-add", *PERSONS).click()
    commit(page, "c", *PERSONS, 1, "id")
    wait_errors(page)
    tick(page, "editor-remove", *PERSONS, 1)
    wait_clean(page)
    assert len(removes) == 1
    expected = editor.emit(mapping_of(MINIMAL))
    assert back_to_yaml(page) == expected
    button(page, "Edit in form").click()
    wait_clean(page)
    own(page, "editor-add", *PERSONS).click()
    wait_errors(page)
    tick(page, "editor-remove", *PERSONS, 1)
    wait_clean(page)
    assert back_to_yaml(page) == expected
    assert len(removes) == 1


def test_asset_classes_are_added_renamed_moved_and_removed(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    example = mapping_of(EXAMPLE)
    open_example(page)
    own(page, "editor-new-name", *CLASSES).fill("gold")
    own(page, "editor-add", *CLASSES).click()
    expect(own(page, "editor-name", *CLASSES, "gold")).to_have_count(1)
    wait_clean(page)
    commit_name(page, "silver", *CLASSES, "gold")
    expect(own(page, "editor-name", *CLASSES, "silver")).to_have_count(1)
    wait_clean(page)
    own(page, "editor-up", *CLASSES, "silver").click()
    expect(own(page, "editor-up", *CLASSES, "silver")).to_be_enabled()
    wait_clean(page)
    expected = editor.move_asset_class(
        editor.rename(editor.add_asset_class(example, "gold"), "asset-class", "gold", "silver"),
        "silver",
        1,
    )
    assert back_to_yaml(page) == editor.emit(expected)
    button(page, "Edit in form").click()
    wait_clean(page)
    tick(page, "editor-remove", *CLASSES, "silver")
    wait_clean(page)
    assert back_to_yaml(page) == editor.emit(example)


def test_removing_a_class_an_allocation_names_is_refused(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    error = refusal(editor.remove, mapping_of(EXAMPLE), "asset-class", "equity")
    assert dotted(error.references) == [
        "assumptions.allocations.default.equity",
        "assumptions.allocations.resp.equity",
    ]
    open_example(page)
    tick(page, "editor-remove", *CLASSES, "equity")
    expect(problem(page)).to_have_text(str(error))
    expect(page.locator(".editor-problem li")).to_have_text(dotted(error.references))


def test_the_up_and_down_buttons_stop_at_the_ends(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    open_example(page)
    expect(own(page, "editor-up", *CLASSES, "equity")).to_be_disabled()
    expect(own(page, "editor-down", *CLASSES, "equity")).to_be_enabled()
    expect(own(page, "editor-up", *CLASSES, "bonds")).to_be_enabled()
    expect(own(page, "editor-down", *CLASSES, "bonds")).to_be_disabled()


def test_a_refused_class_name_is_put_back(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    error = refusal(editor.rename, mapping_of(EXAMPLE), "asset-class", "bonds", "equity")
    open_example(page)
    commit_name(page, "equity", *CLASSES, "bonds")
    expect(problem(page)).to_have_text(str(error))
    expect(own(page, "editor-name", *CLASSES, "bonds")).to_have_value("bonds")


def test_correlation_edits_write_their_mirror(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    correlation = ("assumptions", "correlation")
    open_example(page)
    fill(page, "0.25", *correlation, 0, 1)
    expect(control(page, *correlation, 1, 0)).to_have_value("0.25")
    expect(control(page, *correlation, 0, 0)).to_have_attribute("readonly", "")
    wait_clean(page)
    expected = copy.deepcopy(mapping_of(EXAMPLE))
    expected["assumptions"]["correlation"] = [[1.0, 0.25], [0.25, 1.0]]
    assert back_to_yaml(page) == editor.emit(expected)


def test_a_python_edit_disables_the_form_until_it_is_answered(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    open_example(page)
    held: list[Route] = []
    page.route("**/api/editor/rename", lambda route: held.append(route))
    commit(page, "z", *PERSONS, 0, "id")
    wait_until_held(page, held)
    expect(page.locator("#form")).to_have_attribute("disabled", "disabled")
    expect(control(page, *PERSONS, 0, "id")).to_be_disabled()
    expect(state(page)).to_have_text("Working…")
    held[0].fallback()
    expect(page.locator("#form")).not_to_have_attribute("disabled", "disabled")
    expect(control(page, *PERSONS, 0, "id")).to_be_enabled()
    wait_clean(page)

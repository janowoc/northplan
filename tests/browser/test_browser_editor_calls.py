# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""How the form treats the editor calls themselves, on the API page: a failed check, parse,
schema or rename call is shown with its status and detail, a failed rename frees the form again,
a failed file open or check call clears the errors listed before it, the state reads
"Checking…" until a check answers, a check that finishes after a newer edit is dropped, and a
schema node the form does not know is shown as text without a control.
"""

from __future__ import annotations

import json

import pytest
from form_helpers import (
    ERRORS_TEXT,
    MINIMAL,
    back_to_yaml,
    button,
    control,
    field_errors,
    fill,
    open_example,
    open_new,
    open_text,
    own,
    problem,
    shown_errors,
    state,
    textarea,
    wait_clean,
    wait_errors,
)
from offline import Offline
from playwright.sync_api import Page, Route, expect
from test_browser_api_page import wait_until_held

from report import editor


@pytest.mark.parametrize(
    ("status", "body", "shown"),
    [
        pytest.param(500, "boom", "The check call failed (500): boom", id="text"),
        pytest.param(503, '{"detail": "bad"}', "The check call failed (503): bad", id="detail"),
    ],
)
def test_a_failed_check_is_shown(
    api_form_page: tuple[Page, Offline], status: int, body: str, shown: str
) -> None:
    page, _ = api_form_page
    page.route("**/api/editor/check", lambda route: route.fulfill(status=status, body=body))
    button(page, "New scenario").click()
    wait_errors(page)
    expect(problem(page)).to_have_text(shown)
    expect(button(page, "Back to YAML")).to_be_disabled()


def test_a_stale_check_is_dropped(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    held: list[Route] = []
    holding = {"on": False}

    def handler(route: Route) -> None:
        if holding["on"] and not held:
            held.append(route)
        else:
            route.fallback()

    page.route("**/api/editor/check", handler)
    open_new(page)
    wait_clean(page)
    holding["on"] = True
    fill(page, "abc", "seed")
    wait_until_held(page, held)
    fill(page, "6", "seed")
    wait_clean(page)
    page.evaluate(
        """() => {
            window.__states = [];
            new MutationObserver(() => {
                window.__states.push(document.querySelector('.editor-state').textContent);
            }).observe(document.querySelector('.editor-state'), {
                characterData: true, childList: true, subtree: true,
            });
        }"""
    )
    with page.expect_response("**/api/editor/check") as info:
        held[0].fallback()
    info.value.finished()
    fill(page, "8", "seed")
    wait_clean(page)
    assert ERRORS_TEXT not in page.evaluate("window.__states")
    expect(page.locator("#form .editor-field-errors li")).to_have_count(0)
    assert "seed: 8\n" in back_to_yaml(page)


def test_a_shape_the_form_does_not_know_is_shown_as_text(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    changed = editor.schema()
    changed["properties"]["name"] = {"type": "thing"}
    page.route(
        "**/api/editor/schema",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(changed)
        ),
    )
    open_new(page)
    expect(own(page, "editor-raw", "name")).to_have_text('"new-scenario"')
    expect(control(page, "name")).to_have_count(0)


def test_a_refused_parse_call_is_shown(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    page.route(
        "**/api/editor/parse",
        lambda route: route.fulfill(status=422, body='{"detail": "bad arguments"}'),
    )
    textarea(page).fill(MINIMAL)
    button(page, "Edit in form").click()
    expect(problem(page)).to_have_text("The parse call failed (422): bad arguments")
    expect(page.locator("#form")).to_be_hidden()
    expect(textarea(page)).to_be_visible()


def test_a_failed_open_clears_the_last_errors(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    textarea(page).fill("- 1\n")
    button(page, "Edit in form").click()
    expect(page.locator(".editor-errors li").first).to_be_visible()
    assert shown_errors(page)
    page.route(
        "**/api/minimal",
        lambda route: route.fulfill(status=404, body='{"detail": "no minimal here"}'),
    )
    button(page, "New scenario").click()
    expect(problem(page)).to_have_text("Could not open the file: /api/minimal: 404 no minimal here")
    expect(page.locator(".editor-errors li")).to_have_count(0)
    assert shown_errors(page) == []


def test_a_failed_check_clears_the_last_errors(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    path = ("household", "persons", 0, "birth_year")
    open_text(page, MINIMAL)
    fill(page, "abc", *path)
    wait_errors(page)
    assert len(shown_errors(page)) == 1
    assert len(field_errors(page, *path)) == 1
    page.route("**/api/editor/check", lambda route: route.fulfill(status=500, body="boom"))
    fill(page, "abd", *path)
    expect(problem(page)).to_have_text("The check call failed (500): boom")
    assert shown_errors(page) == []
    expect(page.locator("#form .editor-field-errors li")).to_have_count(0)


def test_a_failed_python_edit_is_shown_and_frees_the_form(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    page.route("**/api/editor/rename", lambda route: route.fulfill(status=500, body="boom"))
    open_example(page)
    box = control(page, "household", "persons", 0, "id")
    box.fill("z")
    box.press("Tab")
    expect(problem(page)).to_have_text("The rename call failed (500): boom")
    expect(control(page, "household", "persons", 0, "id")).to_have_value("a")
    expect(page.locator("#form")).not_to_have_attribute("disabled", "disabled")
    expect(control(page, "household", "persons", 0, "id")).to_be_enabled()
    expect(state(page)).to_have_text("No errors.")


def test_a_failed_schema_call_is_shown(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    page.route("**/api/editor/schema", lambda route: route.fulfill(status=500, body="boom"))
    button(page, "Edit in form").click()
    expect(problem(page)).to_have_text("The schema call failed (500): boom")
    expect(page.locator("#form")).to_be_hidden()
    expect(button(page, "Back to YAML")).to_be_hidden()


def test_the_state_reads_checking_until_the_check_answers(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    held: list[Route] = []
    holding = {"on": False}

    def handler(route: Route) -> None:
        if holding["on"] and not held:
            held.append(route)
        else:
            route.fallback()

    page.route("**/api/editor/check", handler)
    open_new(page)
    wait_clean(page)
    holding["on"] = True
    fill(page, "9", "seed")
    wait_until_held(page, held)
    expect(state(page)).to_have_text("Checking…")
    expect(page.locator("#run")).to_be_disabled()
    held[0].fallback()
    wait_clean(page)

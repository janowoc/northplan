# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What the form does when drawing itself throws, on the API page. A form that cannot be drawn when
it is opened is not opened, and the problem is shown above the YAML box. A form that cannot be drawn
again after an edit, whether the control re-drew it or an edit call's reply did, stays open with the
problem shown and is marked as having errors, so Run, Back to YAML and Download YAML are off and
Discard form is the way out; the list of errors above the form and the references a refusal lists
stay as they were, and the errors beside each field come back when the form is next drawn; a check
already on its way, or one scheduled by an earlier keystroke, is dropped and does not clear that
mark. No failure reaches the page as an uncaught error.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from form_helpers import (
    COUPLE,
    ERRORS_TEXT,
    MINIMAL,
    button,
    control,
    field,
    field_errors,
    fill,
    mapping_of,
    open_text,
    own,
    problem,
    settle_frames,
    shown_errors,
    state,
    textarea,
    wait_clean,
    wait_errors,
)
from offline import Offline
from playwright.sync_api import Page, Request, Route, expect
from test_browser_api_page import wait_until_held

from report import editor

SCHEMA_FAILURE = "The form could not be drawn: TypeError: node.get is not a function"
OVERRIDE_FAILURE = "The form could not be drawn: Error: drawing failed"
OAS = ("household", "persons", 0, "oas")


def undrawable_schema(page: Page) -> None:
    """Serve a schema with one property the form cannot draw, which throws partway through the
    OAS entitlement."""
    changed = editor.schema()
    changed["$defs"]["OasEntitlement"]["properties"]["probe"] = 5
    page.route(
        "**/api/editor/schema",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(changed)
        ),
    )


def make_drawing_throw(page: Page) -> None:
    page.evaluate(
        "() => { document.querySelector('#form').replaceChildren = "
        "() => { throw new Error('drawing failed'); }; }"
    )


def watch_errors(page: Page) -> list[str]:
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    return errors


def watch_calls(page: Page) -> list[str]:
    """The names of the editor calls made from now on."""
    calls: list[str] = []

    def record(request: Request) -> None:
        if request.method == "POST" and "/api/editor/" in request.url:
            calls.append(request.url.rsplit("/", 1)[1])

    page.on("request", record)
    return calls


def stubbed_rename(page: Page, answer: Callable[[Route], None]) -> None:
    page.route("**/api/editor/rename", answer)


def rename_person(page: Page) -> None:
    box = control(page, "household", "persons", 0, "id")
    box.fill("z")
    box.press("Tab")


def expect_marked_with_errors(page: Page) -> None:
    wait_errors(page)
    expect(page.locator("#form")).to_be_visible()
    expect(button(page, "Back to YAML")).to_be_disabled()
    expect(button(page, "Download YAML")).to_be_disabled()
    expect(page.locator("#run")).to_be_disabled()


def test_a_form_that_cannot_be_drawn_is_not_opened(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    undrawable_schema(page)
    textarea(page).fill(COUPLE)
    calls = watch_calls(page)
    button(page, "Edit in form").click()
    expect(problem(page)).to_have_text(SCHEMA_FAILURE)
    settle_frames(page)
    expect(page.locator("#form")).to_be_hidden()
    expect(textarea(page)).to_be_visible()
    expect(button(page, "Back to YAML")).to_be_hidden()
    expect(state(page)).to_be_hidden()
    expect(button(page, "Edit in form")).to_be_enabled()
    page.wait_for_timeout(1000)
    assert "check" not in calls
    assert "grid-paths" not in calls
    open_text(page, MINIMAL)
    wait_clean(page)
    assert errors == []


def test_a_failed_redraw_after_a_control_keeps_the_form(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    undrawable_schema(page)
    open_text(page, MINIMAL)
    wait_clean(page)
    calls = watch_calls(page)
    own(page, "editor-present", *OAS).click()
    expect(problem(page)).to_have_text(SCHEMA_FAILURE)
    expect_marked_with_errors(page)
    expect(field(page, *OAS)).to_be_visible()
    expect(button(page, "Discard form")).to_be_enabled()
    page.wait_for_timeout(1000)
    assert "check" not in calls
    expect(state(page)).to_have_text(ERRORS_TEXT)
    button(page, "Discard form").click()
    expect(page.locator("#form")).to_be_hidden()
    expect(problem(page)).to_be_hidden()
    assert errors == []


def test_a_failed_redraw_after_an_edit_call_keeps_the_form(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    undrawable_schema(page)
    open_text(page, MINIMAL)
    wait_clean(page)
    renamed = editor.rename(mapping_of(MINIMAL), "person", "a", "z")
    renamed["household"]["persons"][0]["oas"] = {}
    stubbed_rename(
        page,
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(renamed)
        ),
    )
    calls = watch_calls(page)
    rename_person(page)
    expect(problem(page)).to_have_text(SCHEMA_FAILURE)
    expect_marked_with_errors(page)
    expect(page.locator("#form")).not_to_have_attribute("disabled", "disabled")
    page.wait_for_timeout(1000)
    assert "check" not in calls
    expect(state(page)).to_have_text(ERRORS_TEXT)
    assert errors == []


def test_a_failed_redraw_after_a_refused_edit_keeps_the_form(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    open_text(page, MINIMAL)
    wait_clean(page)
    stubbed_rename(
        page,
        lambda route: route.fulfill(
            status=409,
            content_type="application/json",
            body=json.dumps(
                {"detail": "refused", "references": [["household", "persons", 0, "id"]]}
            ),
        ),
    )
    make_drawing_throw(page)
    rename_person(page)
    expect(problem(page)).to_have_text(OVERRIDE_FAILURE)
    expect_marked_with_errors(page)
    expect(page.locator("#form")).not_to_have_attribute("disabled", "disabled")
    assert page.locator(".editor-problem li").all_inner_texts() == ["household.persons.0.id"]
    assert errors == []


def test_a_failed_redraw_after_a_failed_edit_keeps_the_form(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    open_text(page, MINIMAL)
    wait_clean(page)
    stubbed_rename(page, lambda route: route.fulfill(status=500, body="boom"))
    make_drawing_throw(page)
    rename_person(page)
    expect(problem(page)).to_have_text(OVERRIDE_FAILURE)
    expect_marked_with_errors(page)
    expect(page.locator("#form")).not_to_have_attribute("disabled", "disabled")
    assert errors == []


def test_a_failed_redraw_keeps_the_errors_listed(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    undrawable_schema(page)
    held: list[Route] = []
    holding = {"on": False}

    def handler(route: Route) -> None:
        if holding["on"] and not held:
            held.append(route)
        else:
            route.fallback()

    page.route("**/api/editor/check", handler)
    open_text(page, MINIMAL)
    wait_clean(page)
    fill(page, "abc", "seed")
    wait_errors(page)
    before = shown_errors(page)
    assert before
    seed_before = field_errors(page, "seed")
    assert seed_before
    holding["on"] = True
    own(page, "editor-present", *OAS).click()
    expect(problem(page)).to_have_text(SCHEMA_FAILURE)
    wait_errors(page)
    assert shown_errors(page) == before
    own(page, "editor-present", *OAS).click()
    wait_until_held(page, held)
    assert field_errors(page, "seed") == seed_before
    with page.expect_response("**/api/editor/check") as info:
        held[0].fallback()
    info.value.finished()
    assert errors == []


def test_a_check_in_flight_does_not_clear_a_failed_redraw(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    undrawable_schema(page)
    held: list[Route] = []
    holding = {"on": False}

    def handler(route: Route) -> None:
        if holding["on"] and not held:
            held.append(route)
        else:
            route.fallback()

    page.route("**/api/editor/check", handler)
    open_text(page, MINIMAL)
    wait_clean(page)
    holding["on"] = True
    fill(page, "6", "seed")
    wait_until_held(page, held)
    own(page, "editor-present", *OAS).click()
    expect(problem(page)).to_have_text(SCHEMA_FAILURE)
    wait_errors(page)
    with page.expect_response("**/api/editor/check") as info:
        held[0].fallback()
    info.value.finished()
    settle_frames(page)
    expect(state(page)).to_have_text(ERRORS_TEXT)
    expect(problem(page)).to_have_text(SCHEMA_FAILURE)
    expect(button(page, "Back to YAML")).to_be_disabled()
    assert errors == []


def test_a_scheduled_check_does_not_clear_a_failed_redraw(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    undrawable_schema(page)
    open_text(page, MINIMAL)
    wait_clean(page)
    calls = watch_calls(page)
    present = '[data-loc=\'["household","persons",0,"oas"]\'] > .editor-present'
    seed = "[data-loc='[\"seed\"]'] > .editor-value"
    page.evaluate(
        """([seed, present]) => {
            const box = document.querySelector(seed);
            box.value = '6';
            box.dispatchEvent(new Event('input', { bubbles: true }));
            document.querySelector(present).click();
        }""",
        [seed, present],
    )
    page.wait_for_timeout(1000)
    assert "check" not in calls
    expect(problem(page)).to_have_text(SCHEMA_FAILURE)
    expect(state(page)).to_have_text(ERRORS_TEXT)
    expect(button(page, "Back to YAML")).to_be_disabled()
    assert errors == []

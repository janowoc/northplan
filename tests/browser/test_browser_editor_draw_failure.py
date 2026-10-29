# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What the form does when drawing itself, or removing what was drawn, throws, on the API page. A
form that cannot be drawn when it is opened is not opened, the problem is shown above the YAML box,
and the schema is dropped, so it is asked for again before the next scenario is parsed. A form that
cannot be drawn again after an edit, whether the control re-drew it or an edit call's reply did,
stays open with the problem shown and is marked as having errors, so Run, Back to YAML and Download
YAML are off and Discard form is the way out; the list of errors above the form and the references a
refusal lists stay as they were, and the errors beside each field come back when the form is next
drawn; a check already on its way, or one scheduled by an earlier keystroke, is dropped and does not
clear that mark. A form whose drawing cannot be removed from the page when it closes is still
closed: Discard form and Back to YAML then show that the form could not be cleared, and a failed
open shows why the form could not be drawn. No failure reaches the page as an uncaught error. A
failed redraw after an edit keeps the schema.
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
CLEAR_FAILURE = "The form could not be cleared: Error: drawing failed"
OAS = ("household", "persons", 0, "oas")


def undrawable_schema(page: Page, *, times: int | None = None) -> None:
    """Serve a schema with one property the form cannot draw, which throws partway through the
    OAS entitlement."""
    changed = editor.schema()
    changed["$defs"]["OasEntitlement"]["properties"]["probe"] = 5
    page.route(
        "**/api/editor/schema",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(changed)
        ),
        times=times,
    )


def make_drawing_throw(page: Page) -> None:
    page.evaluate(
        "() => { document.querySelector('#form').replaceChildren = "
        "() => { throw new Error('drawing failed'); }; }"
    )


def undo_drawing_throw(page: Page) -> None:
    page.evaluate("() => { delete document.querySelector('#form').replaceChildren; }")


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


def fail_an_open(page: Page) -> None:
    textarea(page).fill(COUPLE)
    button(page, "Edit in form").click()
    expect(problem(page)).to_have_text(SCHEMA_FAILURE)


def test_a_failed_open_asks_for_the_schema_again(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    undrawable_schema(page, times=1)
    calls = watch_calls(page)
    fail_an_open(page)
    assert calls[:2] == ["schema", "parse"]
    calls.clear()
    open_text(page, COUPLE)
    wait_clean(page)
    assert calls[:2] == ["schema", "parse"]
    expect(page.locator("#form")).to_be_visible()
    assert errors == []


def test_a_failed_open_drops_a_schema_an_earlier_open_fetched(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    undrawable_schema(page, times=1)
    open_text(page, MINIMAL)
    wait_clean(page)
    button(page, "Discard form").click()
    expect(page.locator("#form")).to_be_hidden()
    calls = watch_calls(page)
    fail_an_open(page)
    assert "schema" not in calls
    calls.clear()
    open_text(page, COUPLE)
    wait_clean(page)
    assert calls[:2] == ["schema", "parse"]
    assert errors == []


REFUSED = (
    409,
    {"detail": "refused", "references": [["household", "persons", 0, "id"]]},
)


def fail_a_redraw(page: Page) -> None:
    undrawable_schema(page, times=1)
    open_text(page, MINIMAL)
    wait_clean(page)
    own(page, "editor-present", *OAS).click()
    expect(problem(page)).to_have_text(SCHEMA_FAILURE)


def reply_with(page: Page, replies: list[tuple[int, object]]) -> None:
    pending = list(replies)

    def answer(route: Route) -> None:
        status, body = pending.pop(0)
        route.fulfill(status=status, content_type="application/json", body=json.dumps(body))

    stubbed_rename(page, answer)


def reopen_without_the_schema(page: Page, errors: list[str]) -> None:
    button(page, "Discard form").click()
    expect(page.locator("#form")).to_be_hidden()
    calls = watch_calls(page)
    open_text(page, MINIMAL)
    wait_clean(page)
    assert "schema" not in calls
    assert calls[0] == "parse"
    assert errors == []


def test_a_failed_redraw_after_a_control_keeps_the_schema(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    fail_a_redraw(page)
    reopen_without_the_schema(page, errors)


def test_a_failed_redraw_after_an_edit_call_keeps_the_schema(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    undrawable_schema(page, times=1)
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
    rename_person(page)
    expect(problem(page)).to_have_text(SCHEMA_FAILURE)
    reopen_without_the_schema(page, errors)


def test_a_failed_redraw_after_a_refused_edit_keeps_the_schema(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    fail_a_redraw(page)
    reply_with(page, [REFUSED])
    rename_person(page)
    expect(page.locator(".editor-problem li")).to_have_text(["household.persons.0.id"])
    expect(problem(page)).to_have_text(SCHEMA_FAILURE)
    reopen_without_the_schema(page, errors)


def test_a_failed_redraw_after_a_failed_edit_keeps_the_schema(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    fail_a_redraw(page)
    reply_with(page, [REFUSED, (500, "boom")])
    rename_person(page)
    expect(page.locator(".editor-problem li")).to_have_text(["household.persons.0.id"])
    rename_person(page)
    expect(page.locator(".editor-problem li")).to_have_count(0)
    expect(problem(page)).to_have_text(SCHEMA_FAILURE)
    reopen_without_the_schema(page, errors)


def test_a_failed_redraw_after_an_unreadable_refusal_keeps_the_schema(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    fail_a_redraw(page)
    reply_with(page, [REFUSED, (409, {"detail": "refused", "references": [5]})])
    rename_person(page)
    expect(page.locator(".editor-problem li")).to_have_text(["household.persons.0.id"])
    rename_person(page)
    expect(page.locator(".editor-problem li")).to_have_count(0)
    expect(problem(page)).to_have_text(SCHEMA_FAILURE)
    reopen_without_the_schema(page, errors)


def test_a_failed_open_that_cannot_be_cleared_is_closed_and_drops_the_schema(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    make_drawing_throw(page)
    calls = watch_calls(page)
    textarea(page).fill(MINIMAL)
    button(page, "Edit in form").click()
    expect(problem(page)).to_have_text(OVERRIDE_FAILURE)
    settle_frames(page)
    expect(page.locator("#form")).to_be_hidden()
    expect(textarea(page)).to_be_visible()
    expect(button(page, "Back to YAML")).to_be_hidden()
    expect(state(page)).to_be_hidden()
    expect(button(page, "Edit in form")).to_be_enabled()
    assert calls[:2] == ["schema", "parse"]
    undo_drawing_throw(page)
    calls.clear()
    open_text(page, MINIMAL)
    wait_clean(page)
    assert calls[:2] == ["schema", "parse"]
    assert errors == []


def test_discarding_a_form_that_cannot_be_cleared_closes_it(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    text = MINIMAL.replace("survivor_share: 1.0", "survivor_share: &share 1.0")
    assert text != MINIMAL
    open_text(page, text)
    wait_clean(page)
    expect(page.locator(".editor-notices li")).to_have_text([editor.FLATTENED])
    fill(page, "abc", "seed")
    wait_errors(page)
    assert shown_errors(page)
    make_drawing_throw(page)
    button(page, "Discard form").click()
    expect(problem(page)).to_have_text(CLEAR_FAILURE)
    settle_frames(page)
    expect(page.locator("#form")).to_be_hidden()
    expect(page.locator(".editor-errors li")).to_have_count(0)
    expect(page.locator(".editor-notices li")).to_have_count(0)
    expect(textarea(page)).to_be_visible()
    expect(textarea(page)).to_have_value(text)
    expect(button(page, "Back to YAML")).to_be_hidden()
    expect(button(page, "Edit in form")).to_be_enabled()
    undo_drawing_throw(page)
    open_text(page, MINIMAL)
    wait_clean(page)
    expect(problem(page)).to_be_hidden()
    assert errors == []


def test_back_to_yaml_on_a_form_that_cannot_be_cleared_closes_it(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
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
    fill(page, "6", "seed")
    wait_clean(page)
    reply_with(page, [REFUSED])
    rename_person(page)
    expect(page.locator(".editor-problem li")).to_have_text(["household.persons.0.id"])
    expect(button(page, "Back to YAML")).to_be_enabled()
    make_drawing_throw(page)
    button(page, "Back to YAML").click()
    expect(problem(page)).to_have_text(CLEAR_FAILURE)
    settle_frames(page)
    expect(page.locator("#form")).to_be_hidden()
    expect(page.locator(".editor-problem li")).to_have_count(0)
    expect(textarea(page)).to_be_visible()
    expect(button(page, "Back to YAML")).to_be_hidden()
    value = textarea(page).input_value()
    assert "seed: 6" in value.splitlines()
    assert value != MINIMAL
    undo_drawing_throw(page)
    holding["on"] = True
    button(page, "Edit in form").click()
    wait_until_held(page, held)
    expect(button(page, "Back to YAML")).to_be_visible()
    expect(button(page, "Back to YAML")).to_be_disabled()
    expect(button(page, "Download YAML")).to_be_disabled()
    expect(page.locator("#run")).to_be_disabled()
    with page.expect_response("**/api/editor/check") as info:
        held[0].fallback()
    info.value.finished()
    wait_clean(page)
    expect(button(page, "Back to YAML")).to_be_enabled()
    expect(button(page, "Download YAML")).to_be_enabled()
    assert errors == []

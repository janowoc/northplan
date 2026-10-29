# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What the form does with a reply it cannot read, on the API page. A schema, parse, check or edit
reply of a shape the form checks for and does not find, or one whose list of errors or references
cannot be read, is shown as a problem and is never half-applied; the types of the entries inside a
list of errors, notices or references are not checked. A check reply that is not an object, or whose
``errors`` is not a list, or that has no errors and no YAML text, is a failed call; a check reply
whose entries cannot be read is shown as the error that reading raised; both leave the form marked
as having errors. An edit reply that is not an object is a failed call and the form keeps its
mapping and does not check it. A refusal is read only when its detail is text and its references are
a list; any other refusal is a failed call, and a failed call or an unreadable reply clears the
references an earlier refusal listed. A parse reply with no mapping and errors that are not a
non-empty list, or with a mapping and notices that are not a list, leaves the form closed. A failed
reply whose body is JSON other than a string or an object with a text detail is shown as that JSON,
with its number text kept. No failure reaches the page as an uncaught error.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest
from form_helpers import (
    MINIMAL,
    button,
    control,
    fill,
    open_example,
    open_text,
    problem,
    settle_frames,
    state,
    textarea,
    wait_clean,
    wait_errors,
)
from offline import Offline
from playwright.sync_api import Page, Route, expect
from test_browser_api_page import wait_until_held
from test_browser_editor_draw_failure import (
    rename_person,
    stubbed_rename,
    watch_calls,
    watch_errors,
)

REFUSAL = '{"detail": "refused", "references": [["household", "persons", 0, "id"]]}'


def answer(status: int, body: str) -> Callable[[Route], None]:
    """A route handler that answers with ``body`` as JSON, or as plain text when it is not JSON."""
    try:
        json.loads(body)
        content_type = "application/json"
    except ValueError:
        content_type = "text/plain"
    return lambda route: route.fulfill(status=status, content_type=content_type, body=body)


def expect_errors_state(page: Page) -> None:
    wait_errors(page)
    expect(button(page, "Back to YAML")).to_be_disabled()


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("not json", "The check call failed (200): not json"),
        ("[1, 2]", "The check call failed (200): [1,2]"),
    ],
)
def test_a_check_reply_that_is_not_an_object_is_shown(
    api_form_page: tuple[Page, Offline], body: str, expected: str
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    page.route("**/api/editor/check", answer(200, body))
    open_example(page)
    expect(problem(page)).to_have_text(expected)
    expect_errors_state(page)
    expect(control(page, "name")).to_be_visible()
    assert errors == []


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            '{"detail": ["a", 1.50, {"k": null}]}',
            'The check call failed (422): {"detail":["a",1.50,{"k":null}]}',
        ),
        ("null", "The check call failed (422): null"),
    ],
)
def test_a_failed_reply_without_a_string_detail_is_shown_as_its_json(
    api_form_page: tuple[Page, Offline], body: str, expected: str
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    page.route("**/api/editor/check", answer(422, body))
    open_example(page)
    expect(problem(page)).to_have_text(expected)
    expect_errors_state(page)
    assert errors == []


def test_a_check_result_the_form_cannot_read_is_shown(
    api_form_page: tuple[Page, Offline],
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    path = ("household", "persons", 0, "birth_year")
    open_text(page, MINIMAL)
    wait_clean(page)
    fill(page, "abc", *path)
    wait_errors(page)
    assert page.locator(".editor-errors li").count() > 0
    assert page.locator("#form .editor-field-errors li").count() > 0
    page.route("**/api/editor/check", answer(200, '{"yaml": null, "errors": ["x"]}'))
    fill(page, "abd", *path)
    expect(problem(page)).to_have_text("TypeError: entry.get is not a function")
    expect_errors_state(page)
    expect(page.locator(".editor-errors li")).to_have_count(0)
    expect(page.locator("#form .editor-field-errors li")).to_have_count(0)
    assert errors == []


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            '{"yaml": 5, "errors": []}',
            'The check call failed (200): {"yaml":5,"errors":[]}',
        ),
        (
            '{"yaml": "x", "errors": 5}',
            'The check call failed (200): {"yaml":"x","errors":5}',
        ),
    ],
)
def test_a_check_result_with_the_wrong_shape_is_a_failed_call(
    api_form_page: tuple[Page, Offline], body: str, expected: str
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    page.route("**/api/editor/check", answer(200, body))
    open_example(page)
    expect(problem(page)).to_have_text(expected)
    expect_errors_state(page)
    assert errors == []


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("[1, 2]", "The rename call failed (200): [1,2]"),
        ("not json", "The rename call failed (200): not json"),
    ],
)
def test_an_edit_reply_that_is_not_an_object_is_a_failed_call(
    api_form_page: tuple[Page, Offline], body: str, expected: str
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    open_text(page, MINIMAL)
    wait_clean(page)
    stubbed_rename(page, answer(200, body))
    calls = watch_calls(page)
    rename_person(page)
    expect(problem(page)).to_have_text(expected)
    expect(control(page, "household", "persons", 0, "id")).to_have_value("a")
    expect(page.locator("#form")).not_to_have_attribute("disabled", "disabled")
    page.wait_for_timeout(1000)
    assert "check" not in calls
    expect(state(page)).to_have_text("No errors.")
    assert errors == []


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            '{"detail": ["x"], "references": []}',
            'The rename call failed (409): {"detail":["x"],"references":[]}',
        ),
        (
            '{"references": [["a"]]}',
            'The rename call failed (409): {"references":[["a"]]}',
        ),
        (
            '{"detail": "refused", "references": "abc"}',
            "The rename call failed (409): refused",
        ),
        ('{"detail": "refused"}', "The rename call failed (409): refused"),
    ],
)
def test_a_refusal_with_the_wrong_shape_is_a_failed_call(
    api_form_page: tuple[Page, Offline], body: str, expected: str
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    open_text(page, MINIMAL)
    wait_clean(page)
    stubbed_rename(page, answer(409, body))
    rename_person(page)
    expect(problem(page)).to_have_text(expected)
    assert page.locator(".editor-problem li").count() == 0
    expect(control(page, "household", "persons", 0, "id")).to_have_value("a")
    assert errors == []


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (500, "boom", "The rename call failed (500): boom"),
        (
            409,
            '{"detail": "refused", "references": [5]}',
            "TypeError: loc.map is not a function",
        ),
    ],
)
def test_a_new_edit_problem_replaces_the_last_references(
    api_form_page: tuple[Page, Offline], status: int, body: str, expected: str
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    open_text(page, MINIMAL)
    wait_clean(page)
    replies = [answer(409, REFUSAL), answer(status, body)]

    def handler(route: Route) -> None:
        replies.pop(0)(route)

    stubbed_rename(page, handler)
    rename_person(page)
    expect(problem(page)).to_have_text("refused")
    assert page.locator(".editor-problem li").all_inner_texts() == ["household.persons.0.id"]
    rename_person(page)
    expect(problem(page)).to_have_text(expected)
    assert page.locator(".editor-problem li").count() == 0
    expect(control(page, "household", "persons", 0, "id")).to_have_value("a")
    expect(page.locator("#form")).not_to_have_attribute("disabled", "disabled")
    assert errors == []


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (500, "boom", "The check call failed (500): boom"),
        (200, '{"yaml": null, "errors": ["x"]}', "TypeError: entry.get is not a function"),
    ],
)
def test_a_failed_check_clears_the_last_references(
    api_form_page: tuple[Page, Offline], status: int, body: str, expected: str
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    open_example(page)
    wait_clean(page)
    held: list[Route] = []
    page.route("**/api/editor/check", lambda route: held.append(route))
    fill(page, "abc", "seed")
    wait_until_held(page, held)
    stubbed_rename(page, answer(409, REFUSAL))
    rename_person(page)
    expect(problem(page)).to_have_text("refused")
    assert page.locator(".editor-problem li").all_inner_texts() == ["household.persons.0.id"]
    answer(status, body)(held[0])
    expect(problem(page)).to_have_text(expected)
    expect(page.locator(".editor-problem li")).to_have_count(0)
    assert errors == []


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            '{"mapping": null, "errors": 5, "notices": []}',
            'The parse call failed (200): {"mapping":null,"errors":5,"notices":[]}',
        ),
        (
            '{"mapping": null, "errors": [], "notices": []}',
            'The parse call failed (200): {"mapping":null,"errors":[],"notices":[]}',
        ),
        (
            '{"mapping": {"name": "x"}, "errors": [], "notices": "abc"}',
            'The parse call failed (200): {"mapping":{"name":"x"},"errors":[],"notices":"abc"}',
        ),
    ],
)
def test_a_parse_result_with_the_wrong_shape_is_a_failed_call(
    api_form_page: tuple[Page, Offline], body: str, expected: str
) -> None:
    page, _ = api_form_page
    errors = watch_errors(page)
    page.route("**/api/editor/parse", answer(200, body))
    textarea(page).fill(MINIMAL)
    button(page, "Edit in form").click()
    expect(problem(page)).to_have_text(expected)
    settle_frames(page)
    expect(page.locator("#form")).to_be_hidden()
    expect(textarea(page)).to_be_visible()
    expect(button(page, "Back to YAML")).to_be_hidden()
    assert page.locator(".editor-notices li").count() == 0
    assert page.locator(".editor-errors li").count() == 0
    assert errors == []

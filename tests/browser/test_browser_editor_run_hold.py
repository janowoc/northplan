# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The scenario form while a run is out, on both pages.

On the Pyodide page one worker answers every Python call in the order it was sent, so a run
holds it; while a run is held the form and the three open buttons are off, the state line says
why, and no editor call is sent. Run is off while the form is opening or a Python edit is
out. On the API page the calls are separate requests served alongside the run, so the form
stays usable during a run and Run stays on while a Python edit is out. A Pyodide run is held
by wrapping ``Worker.prototype.postMessage`` so the simulate message is kept back; an editor
call is parked the same way by name. A starting file's request is held with ``page.route``.
A disabled fieldset is read from its attribute, because Playwright reports it as enabled.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from form_helpers import (
    ENGINE_TIMEOUT_MS,
    RUN_TIMEOUT_MS,
    button,
    control,
    fill,
    goto_ready,
    open_example,
    problem,
    state,
    textarea,
    wait_clean,
)
from offline import Offline, Site
from playwright.sync_api import Page, Route, expect
from test_browser_api_page import wait_until_held
from test_browser_editor_names import PERSONS, commit

HELD_TEXT = "The form is off while the scenario runs."
OPEN_BUTTONS = ("Edit in form", "New scenario", "Open example")
EXITS = ("Back to YAML", "Download YAML", "Discard form")

HOLD_RUNS = """(() => {
  const post = Worker.prototype.postMessage;
  window.__sent = [];
  window.__held = [];
  window.__release = () => {
    for (const [worker, message] of window.__held.splice(0)) {
      post.call(worker, message);
    }
  };
  Worker.prototype.postMessage = function (message, ...rest) {
    const type = message && message.type;
    window.__sent.push(type === 'editor' ? 'editor:' + message.call : type);
    if (type === 'simulate') {
      window.__held.push([this, message]);
      return undefined;
    }
    return post.call(this, message, ...rest);
  };
})();"""

PARK_CALL = """(call) => {
  const post = Worker.prototype.postMessage;
  window.__parked = [];
  Worker.prototype.postMessage = function (message, ...rest) {
    if (message && message.type === 'editor' && message.call === call) {
      window.__parked.push([this, message]);
      return undefined;
    }
    return post.call(this, message, ...rest);
  };
  window.__unpark = () => {
    for (const [worker, message] of window.__parked.splice(0)) {
      post.call(worker, message);
    }
  };
}"""


def ready_page(pyodide_page: tuple[Page, Offline], pyodide_site: Site) -> Page:
    page, _ = pyodide_page
    page.add_init_script(HOLD_RUNS)
    goto_ready(page, pyodide_site)
    return page


def sent(page: Page) -> list[str]:
    result: list[str] = page.evaluate("window.__sent")
    return result


def hold_run(page: Page) -> int:
    """Start a run, wait until its message is held, and return how many messages were sent."""
    page.locator("#run").click()
    page.wait_for_function("window.__held.length === 1")
    return len(sent(page))


def release_and_finish(page: Page) -> None:
    page.evaluate("window.__release()")
    expect(page.locator("#run")).to_have_text("Run", timeout=RUN_TIMEOUT_MS)


def park(page: Page, call: str) -> None:
    page.evaluate(PARK_CALL, call)


def wait_parked(page: Page) -> None:
    page.wait_for_function("window.__parked.length === 1", timeout=ENGINE_TIMEOUT_MS)


def unpark(page: Page) -> None:
    page.evaluate("window.__unpark()")


def test_a_run_turns_off_the_open_buttons_and_says_why(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page = ready_page(pyodide_page, pyodide_site)
    count = hold_run(page)
    for name in OPEN_BUTTONS:
        expect(button(page, name)).to_be_disabled()
    expect(state(page)).to_be_visible()
    expect(state(page)).to_have_text(HELD_TEXT)
    page.wait_for_timeout(300)
    assert len(sent(page)) == count
    release_and_finish(page)
    for name in OPEN_BUTTONS:
        expect(button(page, name)).to_be_enabled()
    expect(state(page)).to_be_hidden()


def test_open_form_does_nothing_while_a_run_is_held(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page = ready_page(pyodide_page, pyodide_site)
    count = hold_run(page)
    page.evaluate("Alpine.$data(document.body).openForm()")
    page.wait_for_timeout(300)
    assert len(sent(page)) == count
    expect(textarea(page)).to_be_visible()
    release_and_finish(page)


def test_open_file_does_nothing_while_a_run_is_held(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page = ready_page(pyodide_page, pyodide_site)
    count = hold_run(page)
    requests: list[str] = []
    page.on("request", lambda request: requests.append(request.url))
    page.evaluate("Alpine.$data(document.body).openFile('example')")
    page.evaluate("Alpine.$data(document.body).openFile('minimal')")
    page.wait_for_timeout(300)
    assert len(sent(page)) == count
    assert not [url for url in requests if url.endswith(("example.yaml", "minimal.yaml"))]
    expect(textarea(page)).to_be_visible()
    release_and_finish(page)


def test_a_run_turns_off_the_form_but_not_its_three_exits(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page = ready_page(pyodide_page, pyodide_site)
    open_example(page)
    wait_clean(page)
    before = len(sent(page))
    count = hold_run(page)
    expect(page.locator("#form")).to_have_attribute("disabled", "disabled")
    expect(control(page, "name")).to_be_disabled()
    for name in EXITS:
        expect(button(page, name)).to_be_enabled()
    expect(state(page)).to_have_text(HELD_TEXT)
    page.wait_for_timeout(300)
    assert len(sent(page)) == count
    release_and_finish(page)
    expect(page.locator("#form")).not_to_have_attribute("disabled", "disabled")
    wait_clean(page)
    expect(page.locator("#run")).to_be_enabled()
    assert sent(page)[before:] == ["simulate"]


def test_discarding_the_form_during_a_run_leaves_the_open_buttons_off(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page = ready_page(pyodide_page, pyodide_site)
    open_example(page)
    wait_clean(page)
    hold_run(page)
    button(page, "Discard form").click()
    expect(textarea(page)).to_be_visible()
    for name in OPEN_BUTTONS:
        expect(button(page, name)).to_be_disabled()
    expect(state(page)).to_be_visible()
    expect(state(page)).to_have_text(HELD_TEXT)
    release_and_finish(page)
    for name in OPEN_BUTTONS:
        expect(button(page, name)).to_be_enabled()


def test_run_is_off_while_the_form_opens(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page = ready_page(pyodide_page, pyodide_site)
    park(page, "parse")
    button(page, "Edit in form").click()
    wait_parked(page)
    expect(page.locator("#run")).to_be_disabled()
    unpark(page)
    wait_clean(page)
    expect(page.locator("#run")).to_be_enabled()


@pytest.mark.parametrize(
    ("name", "file"),
    [("New scenario", "minimal"), ("Open example", "example")],
    ids=["new-scenario", "open-example"],
)
def test_run_is_off_while_a_starting_file_loads(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site, name: str, file: str
) -> None:
    page = ready_page(pyodide_page, pyodide_site)
    held: list[Route] = []
    page.route(f"**/{file}.yaml", lambda route: held.append(route))
    expect(page.locator("#run")).to_be_enabled()
    button(page, name).click()
    wait_until_held(page, held)
    expect(page.locator("#run")).to_be_disabled()
    held[0].fallback()
    wait_clean(page)
    expect(page.locator("#run")).to_be_enabled()


def test_run_comes_back_when_a_starting_file_fails_to_load(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page = ready_page(pyodide_page, pyodide_site)
    held: list[Route] = []
    page.route("**/minimal.yaml", lambda route: held.append(route))
    button(page, "New scenario").click()
    wait_until_held(page, held)
    expect(page.locator("#run")).to_be_disabled()
    held[0].fulfill(status=404, body="")
    expect(problem(page)).to_have_text("Could not open the file: minimal.yaml: 404 Not Found")
    expect(page.locator("#run")).to_be_enabled()
    for name in OPEN_BUTTONS:
        expect(button(page, name)).to_be_enabled()


def test_run_is_off_while_a_python_edit_is_out(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page = ready_page(pyodide_page, pyodide_site)
    open_example(page)
    wait_clean(page)
    park(page, "rename")
    commit(page, "z", *PERSONS, 0, "id")
    wait_parked(page)
    expect(state(page)).to_have_text("Working…")
    expect(page.locator("#run")).to_be_disabled()
    unpark(page)
    wait_clean(page)
    expect(page.locator("#run")).to_be_enabled()


def test_the_api_pages_form_opens_and_checks_during_a_run(
    api_page: tuple[Page, Offline], api_site: Site
) -> None:
    page, _ = api_page
    goto_ready(page, api_site)
    held: list[Route] = []
    page.route("**/api/simulate", lambda route: held.append(route))
    page.locator("#run").click()
    wait_until_held(page, held)
    expect(state(page)).to_be_hidden()
    expect(button(page, "Open example")).to_be_enabled()
    with page.expect_request("**/api/editor/parse"):
        button(page, "Open example").click()
    wait_clean(page)
    expect(page.locator("#form")).not_to_have_attribute("disabled", "disabled")
    with page.expect_request("**/api/editor/check"):
        fill(page, "other", "name")
    wait_clean(page)
    held[0].fallback()
    expect(page.locator("#run")).to_have_text("Run", timeout=RUN_TIMEOUT_MS)
    wait_clean(page)
    expect(control(page, "name")).to_be_visible()
    expect(control(page, "name")).to_have_value("other")


@pytest.mark.parametrize(
    ("handler", "label"),
    [
        pytest.param(lambda route: route.abort(), "Error network:", id="unreachable"),
        pytest.param(
            lambda route: route.fulfill(status=500, body='{"detail": "the detail"}'),
            "Error 500:",
            id="refused",
        ),
        pytest.param(
            lambda route: route.fulfill(status=200, body="not JSON"),
            "Error client:",
            id="unreadable",
        ),
    ],
)
def test_the_api_pages_form_stays_open_after_a_failed_run(
    api_page: tuple[Page, Offline],
    api_site: Site,
    handler: Callable[[Route], None],
    label: str,
) -> None:
    page, _ = api_page
    goto_ready(page, api_site)
    open_example(page)
    wait_clean(page)
    fill(page, "other", "name")
    expect(page.locator("#run")).to_be_disabled()
    wait_clean(page)
    page.route("**/api/simulate", handler)
    page.locator("#run").click()
    expect(page.locator(".error strong")).to_have_text(label)
    expect(page.locator("#run")).to_have_text("Run")
    wait_clean(page)
    expect(control(page, "name")).to_be_visible()
    expect(control(page, "name")).to_have_value("other")


def test_the_api_pages_run_is_on_while_a_python_edit_is_out(
    api_page: tuple[Page, Offline], api_site: Site
) -> None:
    page, _ = api_page
    goto_ready(page, api_site)
    open_example(page)
    wait_clean(page)
    held: list[Route] = []
    page.route("**/api/editor/rename", lambda route: held.append(route))
    commit(page, "z", *PERSONS, 0, "id")
    wait_until_held(page, held)
    expect(page.locator("#run")).to_be_enabled()
    held[0].fallback()
    wait_clean(page)

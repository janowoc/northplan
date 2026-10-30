# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The Pyodide page, built by browser/build.py and driven in Chromium offline.

What the page shows after a run is compared with what the native entry point returns for
the same scenario: the summary's policy line, the caption, the chart's years and a
refusal's status and detail. The estate and depletion lines are matched against their
pattern instead, because the two platforms' floats differ in the last places.

It also owns the engine stopping while no run is out, and the engine never starting; a stop
during a run is in test_browser_editor_run_hold.py. Once the engine has failed, the page keeps
its first report and ignores what the worker sends afterwards.

A test stops a loaded engine by sending the page's own handlers a ``failed`` message or an
``error`` event on the captured worker. The ``error`` kind terminates the worker first, so that
a message posted after the stop would never be answered.
"""

from __future__ import annotations

import importlib.util
import json
import re
import time
from pathlib import Path
from types import ModuleType

import pytest
from form_helpers import (
    button,
    control,
    fill,
    goto_ready,
    open_example,
    problem,
    settle_frames,
    state,
    wait_clean,
    wait_errors,
)
from offline import PYODIDE_BASE, Offline, Site
from playwright.sync_api import Locator, Page, Route, expect
from test_browser_editor_names import PERSONS, commit
from test_browser_editor_run_hold import PARK_CALL, wait_parked

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = (REPO_ROOT / "scenarios" / "example.yaml").read_text(encoding="utf-8")
MINIMAL = (REPO_ROOT / "scenarios" / "minimal.yaml").read_text(encoding="utf-8")
READY = "The engine is ready; it runs in this browser."
LOADING = "Loading the engine (about 11 MB on a first visit)…"
REPOSITORY = "https://github.com/janowoc/northplan"
REFUSED_TEXT = "- 1\n"
ENGINE_TIMEOUT_MS = 120_000
RUN_TIMEOUT_MS = 300_000
TRACE_NAMES = ["10th percentile", "90th percentile", "Median"]
OPEN_BUTTONS = ("Edit in form", "New scenario", "Open example")

CAPTURE_WORKER = """(() => {
  const W = window.Worker;
  window.__workers = [];
  window.__sent = [];
  window.Worker = class extends W {
    constructor(...args) { super(...args); window.__workers.push(this); }
    postMessage(message, ...rest) {
      window.__sent.push(message.type === 'editor' ? 'editor:' + message.call : message.type);
      return super.postMessage(message, ...rest);
    }
  };
})();"""

STOP_FAILED = (
    "(detail) => window.__workers[0].dispatchEvent("
    "new MessageEvent('message', { data: { type: 'failed', detail } }))"
)
STOP_ERROR = (
    "(detail) => { window.__workers[0].terminate(); "
    "window.__workers[0].dispatchEvent(new ErrorEvent('error', { message: detail })); }"
)


@pytest.fixture(scope="module")
def entry() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "northplan_browser_under_test", REPO_ROOT / "browser" / "northplan_browser.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def status_line(page: Page) -> Locator:
    return page.locator(".status p").first


def years_line(page: Page) -> Locator:
    return page.locator(".status p").nth(1)


def version_line(page: Page) -> Locator:
    return page.locator(".status p").nth(2)


def banner(page: Page) -> Locator:
    return page.locator(".banner")


def disclaimer_links(page: Page) -> Locator:
    return page.locator("#disclaimer a")


def expect_disclaimer_links(page: Page, revision: str) -> None:
    for index, target in enumerate(("docs/limitations.md", "README.md#license", "LICENSE")):
        expect(disclaimer_links(page).nth(index)).to_have_attribute(
            "href", f"{REPOSITORY}/blob/{revision}/{target}"
        )


def run_button(page: Page) -> Locator:
    return page.locator("#run")


def textarea(page: Page) -> Locator:
    return page.locator("#scenario")


def error_box(page: Page) -> Locator:
    return page.locator(".error")


def stop(page: Page, how: str, detail: str = "the detail") -> None:
    """Stop the captured worker as the page sees it: ``"failed"`` delivers a ``failed`` message
    through the page's own ``onmessage``; ``"error"`` terminates the worker and then delivers an
    ``error`` event through its ``onerror``. Any other ``how`` raises ``KeyError``."""
    scripts = {"failed": STOP_FAILED, "error": STOP_ERROR}
    page.evaluate(scripts[how], detail)


def sent(page: Page) -> list[str]:
    result: list[str] = page.evaluate("window.__sent")
    return result


def wait_until_ready(page: Page) -> None:
    expect(status_line(page)).to_have_text(READY, timeout=ENGINE_TIMEOUT_MS)


def wait_until_held(page: Page, held: list[Route]) -> None:
    deadline = time.monotonic() + 30
    while not held:
        if time.monotonic() > deadline:
            pytest.fail("the request was not made within 30 seconds")
        page.wait_for_timeout(50)


def chart_names(page: Page) -> list[str]:
    return page.evaluate("document.getElementById('chart').data.map((trace) => trace.name)")


def chart_xs(page: Page) -> list[list[int]]:
    return page.evaluate(
        "document.getElementById('chart').data.map((trace) => Array.from(trace.x))"
    )


def assert_chart_purged(page: Page) -> None:
    assert page.evaluate("document.getElementById('chart').data === undefined") is True
    assert page.evaluate("document.getElementById('chart').children.length") == 0


def test_it_loads_the_engine_and_the_example(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, _ = pyodide_page
    assert pyodide_site.directory is not None
    info = json.loads((pyodide_site.directory / "build-info.json").read_text(encoding="utf-8"))
    page.goto(pyodide_site.url)
    wait_until_ready(page)
    expect(textarea(page)).to_have_value(EXAMPLE)
    expect(years_line(page)).to_have_text("Tax parameters: " + ", ".join(info["parameter_years"]))
    expect(run_button(page)).to_be_enabled()
    expect(run_button(page)).to_have_text("Run")
    settle_frames(page)
    expect(error_box(page)).to_be_hidden()
    expect(page.locator("#disclaimer")).to_be_visible()
    expect(page.locator(".source a").nth(0)).to_have_attribute(
        "href", f"{REPOSITORY}/blob/{info['commit']}/LICENSE"
    )
    expect(page.locator(".source a").nth(1)).to_have_text("the source of this build")
    expect(page.locator(".source a").nth(1)).to_have_attribute(
        "href", f"{REPOSITORY}/tree/{info['commit']}"
    )
    expect(version_line(page)).to_have_text(
        "northplan " + info["version"] if info["released"] else "development build"
    )
    if info["released"]:
        expect(banner(page)).to_be_hidden()
    else:
        expect(banner(page)).to_be_visible()


def test_run_waits_for_the_engine(pyodide_page: tuple[Page, Offline], pyodide_site: Site) -> None:
    page, _ = pyodide_page
    held: list[Route] = []
    page.route(PYODIDE_BASE + "pyodide.mjs", lambda route: held.append(route))
    page.goto(pyodide_site.url)
    wait_until_held(page, held)
    expect(status_line(page)).to_have_text(LOADING)
    expect(run_button(page)).to_be_disabled()
    held[0].fallback()
    wait_until_ready(page)
    expect(run_button(page)).to_be_enabled()


def test_run_shows_the_chart_and_the_summary(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site, entry: ModuleType
) -> None:
    page, _ = pyodide_page
    page.goto(pyodide_site.url)
    wait_until_ready(page)
    native = json.loads(entry.simulate(EXAMPLE, "{}"))["document"]
    run_button(page).click()
    expect(run_button(page)).to_have_text("Running…")
    expect(run_button(page)).to_be_disabled()
    expect(page.locator(".summary")).to_be_visible(timeout=RUN_TIMEOUT_MS)
    meta = native["meta"]
    lines = page.locator(".summary p").all_inner_texts()
    assert lines[2] == f"Policy {meta['policy']} · {meta['paths']:,} paths · seed {meta['seed']}"
    assert re.fullmatch(
        rf"Median after-tax estate: \$-?\d{{1,3}}(,\d{{3}})* \({re.escape(meta['estate_dollars'])}\)",
        lines[0],
    )
    assert re.fullmatch(
        r"Depletion probability: \d+\.\d% — the share of paths that run out of money at some "
        r"point before the second death\.",
        lines[1],
    )
    expect(page.locator(".caption")).to_have_text(meta["statistics"])
    years = [row["year"] for row in native["years"]]
    assert chart_names(page) == TRACE_NAMES
    assert chart_xs(page) == [years, years, years]
    expect(error_box(page)).to_be_hidden()
    expect(run_button(page)).to_have_text("Run")
    expect(run_button(page)).to_be_enabled()


def test_each_run_clears_the_last_result_or_error(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site, entry: ModuleType
) -> None:
    page, _ = pyodide_page
    page.goto(pyodide_site.url)
    wait_until_ready(page)
    textarea(page).fill(MINIMAL)
    run_button(page).click()
    expect(page.locator(".summary")).to_be_visible(timeout=RUN_TIMEOUT_MS)
    textarea(page).fill(REFUSED_TEXT)
    run_button(page).click()
    expected = json.loads(entry.simulate(REFUSED_TEXT, "{}"))
    expect(page.locator(".error strong")).to_have_text(
        f"Error {expected['status']}:", timeout=RUN_TIMEOUT_MS
    )
    expect(page.locator(".error pre")).to_have_text(expected["detail"])
    settle_frames(page)
    expect(page.locator(".summary")).to_be_hidden()
    expect(page.locator(".caption")).to_be_hidden()
    assert_chart_purged(page)
    textarea(page).fill(MINIMAL)
    run_button(page).click()
    expect(page.locator(".summary")).to_be_visible(timeout=RUN_TIMEOUT_MS)
    expect(error_box(page)).to_be_hidden()


def test_the_engine_failing_to_load_is_shown(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, _ = pyodide_page
    page.route("**/northplan.zip", lambda route: route.fulfill(status=404, body=""))
    page.goto(pyodide_site.url)
    expect(status_line(page)).to_have_text("The engine did not load.", timeout=ENGINE_TIMEOUT_MS)
    expect(page.locator(".error strong")).to_have_text("Error load:")
    expect(page.locator(".error pre")).to_have_text("Error: northplan.zip: HTTP 404")
    expect(run_button(page)).to_be_disabled()
    expect(textarea(page)).to_have_value(EXAMPLE)


@pytest.mark.parametrize("how", ["failed", "error"])
def test_the_engine_stopping_after_it_loaded_is_shown(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site, how: str
) -> None:
    page, _ = pyodide_page
    page.add_init_script(CAPTURE_WORKER)
    goto_ready(page, pyodide_site)
    stop(page, how)
    expect(status_line(page)).to_have_text("The engine stopped.")
    expect(page.locator(".error strong")).to_have_text("Error error:")
    expect(page.locator(".error pre")).to_have_text("the detail")
    expect(run_button(page)).to_be_disabled()
    expect(run_button(page)).to_have_text("Run")
    for name in OPEN_BUTTONS:
        expect(button(page, name)).to_be_disabled()
    expect(textarea(page)).to_have_value(EXAMPLE)


def test_a_worker_that_does_not_start_is_shown(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, _ = pyodide_page
    page.route("**/worker.js", lambda route: route.fulfill(status=404, body=""))
    page.goto(pyodide_site.url)
    expect(status_line(page)).to_have_text("The engine did not load.", timeout=ENGINE_TIMEOUT_MS)
    expect(page.locator(".error strong")).to_have_text("Error load:")
    expect(page.locator(".error pre")).to_have_text("the worker could not start")
    expect(run_button(page)).to_be_disabled()
    expect(textarea(page)).to_have_value(EXAMPLE)


def test_a_worker_that_cannot_be_made_is_shown(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, _ = pyodide_page
    page.add_init_script("window.Worker = function () { throw new Error('the detail'); };")
    page.goto(pyodide_site.url)
    expect(status_line(page)).to_have_text("The engine did not load.", timeout=ENGINE_TIMEOUT_MS)
    expect(page.locator(".error strong")).to_have_text("Error load:")
    expect(page.locator(".error pre")).to_have_text("Error: the detail")
    expect(run_button(page)).to_be_disabled()
    expect(textarea(page)).to_have_value(EXAMPLE)


def test_opening_the_form_fails_when_the_engine_stops(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, _ = pyodide_page
    page.add_init_script(CAPTURE_WORKER)
    goto_ready(page, pyodide_site)
    page.evaluate(PARK_CALL, "parse")
    button(page, "Edit in form").click()
    wait_parked(page)
    stop(page, "error")
    expect(problem(page)).to_have_text("The parse call failed (error): the detail")
    expect(status_line(page)).to_have_text("The engine stopped.")
    settle_frames(page)
    expect(textarea(page)).to_be_visible()
    expect(page.locator("#form")).to_be_hidden()
    expect(state(page)).to_be_hidden()


def test_a_check_fails_when_the_engine_stops(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, _ = pyodide_page
    page.add_init_script(CAPTURE_WORKER)
    goto_ready(page, pyodide_site)
    open_example(page)
    wait_clean(page)
    page.evaluate(PARK_CALL, "check")
    fill(page, "other", "name")
    wait_parked(page)
    stop(page, "error")
    expect(problem(page)).to_have_text("The check call failed (error): the detail")
    wait_errors(page)
    expect(control(page, "name")).to_be_visible()
    expect(control(page, "name")).to_have_value("other")
    expect(button(page, "Back to YAML")).to_be_disabled()
    expect(button(page, "Download YAML")).to_be_disabled()


def test_a_python_edit_fails_when_the_engine_stops(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, _ = pyodide_page
    page.add_init_script(CAPTURE_WORKER)
    goto_ready(page, pyodide_site)
    open_example(page)
    wait_clean(page)
    page.evaluate(PARK_CALL, "rename")
    commit(page, "z", *PERSONS, 0, "id")
    wait_parked(page)
    stop(page, "error")
    expect(problem(page)).to_have_text("The rename call failed (error): the detail")
    expect(control(page, *PERSONS, 0, "id")).to_have_value("a")
    wait_clean(page)
    expect(button(page, "Back to YAML")).to_be_enabled()


def test_a_check_after_the_engine_stopped_fails_at_once(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, _ = pyodide_page
    page.add_init_script(CAPTURE_WORKER)
    goto_ready(page, pyodide_site)
    open_example(page)
    wait_clean(page)
    stop(page, "error")
    before = len(sent(page))
    fill(page, "other", "name")
    expect(problem(page)).to_have_text("The check call failed (error): the detail")
    wait_errors(page)
    assert sent(page)[before:] == []


def test_a_python_edit_after_the_engine_stopped_fails_at_once(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, _ = pyodide_page
    page.add_init_script(CAPTURE_WORKER)
    goto_ready(page, pyodide_site)
    open_example(page)
    wait_clean(page)
    stop(page, "error")
    before = len(sent(page))
    commit(page, "z", *PERSONS, 0, "id")
    expect(problem(page)).to_have_text("The rename call failed (error): the detail")
    expect(control(page, *PERSONS, 0, "id")).to_have_value("a")
    wait_clean(page)
    assert sent(page)[before:] == []


def test_a_ready_after_the_engine_failed_is_ignored(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, _ = pyodide_page
    page.add_init_script(CAPTURE_WORKER)
    held: list[Route] = []
    page.route(PYODIDE_BASE + "pyodide.mjs", lambda route: held.append(route))
    page.goto(pyodide_site.url)
    wait_until_held(page, held)
    page.evaluate(
        "window.__workers[0].dispatchEvent(new ErrorEvent('error', { message: 'the detail' }))"
    )
    page.evaluate(
        "window.__workers[0].dispatchEvent(new MessageEvent('message', { data: { type: 'ready' } }))"
    )
    settle_frames(page)
    expect(status_line(page)).to_have_text("The engine did not load.")
    expect(page.locator(".error strong")).to_have_text("Error load:")
    expect(page.locator(".error pre")).to_have_text("the detail")
    expect(run_button(page)).to_be_disabled()
    for name in OPEN_BUTTONS:
        expect(button(page, name)).to_be_disabled()


def test_a_second_failure_keeps_the_first_report(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, _ = pyodide_page
    page.add_init_script(CAPTURE_WORKER)
    goto_ready(page, pyodide_site)
    stop(page, "error")
    expect(status_line(page)).to_have_text("The engine stopped.")
    stop(page, "error", "another detail")
    settle_frames(page)
    expect(status_line(page)).to_have_text("The engine stopped.")
    expect(page.locator(".error strong")).to_have_text("Error error:")
    expect(page.locator(".error pre")).to_have_text("the detail")


def test_a_missing_example_is_shown(pyodide_page: tuple[Page, Offline], pyodide_site: Site) -> None:
    page, _ = pyodide_page
    page.route("**/example.yaml", lambda route: route.fulfill(status=404, body="not the example"))
    page.goto(pyodide_site.url)
    expect(page.locator(".error strong")).to_have_text("Error 404:")
    expect(page.locator(".error pre")).to_have_text("example.yaml: Not Found")
    expect(textarea(page)).to_have_value("")
    wait_until_ready(page)


def test_an_unreachable_example_is_shown(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, _ = pyodide_page
    page.route("**/example.yaml", lambda route: route.abort())
    page.goto(pyodide_site.url)
    expect(page.locator(".error strong")).to_have_text("Error network:")
    expect(page.locator(".error pre")).to_have_text("TypeError: Failed to fetch")


def test_without_build_info_the_years_are_unknown(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, _ = pyodide_page
    page.route("**/build-info.json", lambda route: route.fulfill(status=404, body=""))
    page.goto(pyodide_site.url)
    wait_until_ready(page)
    expect(years_line(page)).to_have_text("Tax parameters: unknown")
    expect(page.locator(".source a").nth(0)).to_have_attribute(
        "href", f"{REPOSITORY}/blob/main/LICENSE"
    )
    expect(page.locator(".source a").nth(1)).to_have_text("the source")
    expect(page.locator(".source a").nth(1)).to_have_attribute("href", REPOSITORY)
    settle_frames(page)
    expect(error_box(page)).to_be_hidden()
    expect(banner(page)).to_be_hidden()
    expect(version_line(page)).to_be_hidden()
    expect_disclaimer_links(page, "main")


BANNER_TEXT = (
    "Development build, commit aaaaaaa. Its results can change from one commit to the next. "
    "The released version is at https://janowoc.github.io/northplan/."
)
BUILD_COMMIT = "a" * 40


def serve_build_info(page: Page, pyodide_site: Site, **changes: object) -> None:
    """Serve the built site's build-info.json with the commit ``a * 40``, the version
    ``9.9.9`` and ``changes`` applied; a ``changes`` value of ``None`` removes the key."""
    assert pyodide_site.directory is not None
    info = json.loads((pyodide_site.directory / "build-info.json").read_text(encoding="utf-8"))
    info.update(commit=BUILD_COMMIT, version="9.9.9")
    for key, value in changes.items():
        if value is None:
            info.pop(key, None)
        else:
            info[key] = value
    body = json.dumps(info)
    page.route(
        "**/build-info.json",
        lambda route: route.fulfill(status=200, content_type="application/json", body=body),
    )


def test_a_released_build_names_its_version_and_has_no_banner(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, _ = pyodide_page
    serve_build_info(page, pyodide_site, released=True)
    page.goto(pyodide_site.url)
    wait_until_ready(page)
    settle_frames(page)
    expect(error_box(page)).to_be_hidden()
    expect(version_line(page)).to_have_text("northplan 9.9.9")
    expect(banner(page)).to_be_hidden()
    expect_disclaimer_links(page, BUILD_COMMIT)


def test_a_development_build_says_so_in_a_banner_above_the_title(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, _ = pyodide_page
    serve_build_info(page, pyodide_site, released=False)
    page.goto(pyodide_site.url)
    wait_until_ready(page)
    settle_frames(page)
    expect(error_box(page)).to_be_hidden()
    expect(version_line(page)).to_have_text("development build")
    expect(banner(page)).to_be_visible()
    expect(banner(page)).to_have_text(BANNER_TEXT)
    expect(page.get_by_role("note")).to_have_text(BANNER_TEXT)
    expect(banner(page).locator("code")).to_have_text("aaaaaaa")
    link = banner(page).locator("a")
    expect(link).to_have_attribute("href", "https://janowoc.github.io/northplan/")
    expect(link).to_have_text("https://janowoc.github.io/northplan/")
    assert page.evaluate(
        "document.body.firstElementChild === document.querySelector('.banner')"
        " && document.querySelector('.banner').compareDocumentPosition("
        "document.querySelector('h1')) === Node.DOCUMENT_POSITION_FOLLOWING"
    )
    expect_disclaimer_links(page, BUILD_COMMIT)


@pytest.mark.parametrize("released", ["true", 1, None], ids=["string", "number", "absent"])
def test_a_released_field_that_is_not_true_is_a_development_build(
    released: object, pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, _ = pyodide_page
    serve_build_info(page, pyodide_site, released=released)
    page.goto(pyodide_site.url)
    wait_until_ready(page)
    settle_frames(page)
    expect(error_box(page)).to_be_hidden()
    expect(version_line(page)).to_have_text("development build")
    expect(banner(page)).to_be_visible()


def test_a_request_away_from_the_site_is_refused(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, offline = pyodide_page
    page.goto(pyodide_site.url)
    wait_until_ready(page)
    page.evaluate(
        "(async () => { for (const u of ['https://example.invalid/', '/health']) "
        "{ try { await fetch(u); } catch (e) {} } })()"
    )
    origin = pyodide_site.url.removesuffix("northplan/").rstrip("/")
    assert offline.refused == ["https://example.invalid/", origin + "/health"]
    offline.refused.clear()

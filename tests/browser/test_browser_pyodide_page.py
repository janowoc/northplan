# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The Pyodide page, built by browser/build.py and driven in Chromium offline.

What the page shows after a run is compared with what the native entry point returns for
the same scenario: the summary's policy line, the caption, the chart's years and a
refusal's status and detail. The estate and depletion lines are matched against their
pattern instead, because the two platforms' floats differ in the last places.
"""

from __future__ import annotations

import importlib.util
import json
import re
import time
from pathlib import Path
from types import ModuleType

import pytest
from offline import PYODIDE_BASE, Offline, Site
from playwright.sync_api import Locator, Page, Route, expect

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


def run_button(page: Page) -> Locator:
    return page.locator("#run")


def textarea(page: Page) -> Locator:
    return page.locator("#scenario")


def error_box(page: Page) -> Locator:
    return page.locator(".error")


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
    expect(error_box(page)).to_be_hidden()
    expect(page.locator("#disclaimer")).to_be_visible()
    expect(page.locator(".source a").nth(0)).to_have_attribute(
        "href", f"{REPOSITORY}/blob/{info['commit']}/LICENSE"
    )
    expect(page.locator(".source a").nth(1)).to_have_text("the source of this build")
    expect(page.locator(".source a").nth(1)).to_have_attribute(
        "href", f"{REPOSITORY}/tree/{info['commit']}"
    )


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
    expect(error_box(page)).to_be_hidden()


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

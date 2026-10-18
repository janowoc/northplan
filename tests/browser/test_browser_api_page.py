# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The API page, served by ``api.main.app`` under uvicorn on loopback and driven in Chromium
offline.

A run is compared with the response the page received, read from the network; a refusal
with the same body posted through the test client. The estate line is the response's
median rounded as JavaScript rounds, and the depletion line is matched against its pattern
and compared to the response's last year within the rounding of one decimal.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from offline import Offline, Site
from playwright.sync_api import Locator, Page, Route, expect

from api.main import app

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = (REPO_ROOT / "scenarios" / "example.yaml").read_text(encoding="utf-8")
MINIMAL = (REPO_ROOT / "scenarios" / "minimal.yaml").read_text(encoding="utf-8")
REFUSED_TEXT = "- 1\n"
RUN_TIMEOUT_MS = 300_000
TRACE_NAMES = ["10th percentile", "90th percentile", "Median"]
DEPLETION = re.compile(
    r"Depletion probability: (\d+\.\d)% — the share of paths that run out of money at some "
    r"point before the second death\."
)


def run_button(page: Page) -> Locator:
    return page.get_by_role("button")


def textarea(page: Page) -> Locator:
    return page.locator("#scenario")


def error_box(page: Page) -> Locator:
    return page.locator(".error")


def open_page(page: Page, site: Site) -> None:
    """Go to the page and wait until ``init`` has put the example in the textarea."""
    page.goto(site.url)
    expect(textarea(page)).to_have_value(EXAMPLE)


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


def test_it_shows_the_health_and_the_example(
    api_page: tuple[Page, Offline], api_site: Site
) -> None:
    page, _ = api_page
    page.goto(api_site.url)
    expect(page.locator(".status strong")).to_have_text("ok")
    expect(textarea(page)).to_have_value(EXAMPLE)
    expect(error_box(page)).to_be_hidden()


def test_run_shows_the_chart_and_the_summary(
    api_page: tuple[Page, Offline], api_site: Site
) -> None:
    page, _ = api_page
    open_page(page, api_site)
    held: list[Route] = []
    page.route("**/api/simulate", lambda route: held.append(route))
    run_button(page).click()
    wait_until_held(page, held)
    expect(run_button(page)).to_have_text("Running…")
    expect(run_button(page)).to_be_disabled()
    with page.expect_response("**/api/simulate", timeout=RUN_TIMEOUT_MS) as info:
        held[0].fallback()
    doc = info.value.json()
    meta = doc["meta"]
    expect(page.locator(".summary")).to_be_visible(timeout=RUN_TIMEOUT_MS)
    lines = page.locator(".summary p").all_inner_texts()
    estate = math.floor(doc["final"]["estate_after_tax_p50"] + 0.5)
    assert lines[0] == f"Median after-tax estate: ${estate:,} ({meta['estate_dollars']})"
    shown = DEPLETION.fullmatch(lines[1])
    assert shown is not None
    assert abs(float(shown.group(1)) - doc["years"][-1]["depletion_probability"] * 100) <= (
        0.05 + 1e-9
    )
    assert lines[2] == f"Policy {meta['policy']} · {meta['paths']:,} paths · seed {meta['seed']}"
    expect(page.locator(".caption")).to_have_text(meta["statistics"])
    years = [row["year"] for row in doc["years"]]
    assert chart_names(page) == TRACE_NAMES
    assert chart_xs(page) == [years, years, years]
    expect(run_button(page)).to_have_text("Run")
    expect(run_button(page)).to_be_enabled()


def test_a_refused_scenario_shows_the_detail_and_clears_the_last_result(
    api_page: tuple[Page, Offline], api_site: Site
) -> None:
    page, _ = api_page
    expected = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000)).post(
        "/api/simulate",
        content=REFUSED_TEXT.encode(),
        headers={"Content-Type": "application/yaml"},
    )
    open_page(page, api_site)
    textarea(page).fill(MINIMAL)
    run_button(page).click()
    expect(page.locator(".summary")).to_be_visible(timeout=RUN_TIMEOUT_MS)
    textarea(page).fill(REFUSED_TEXT)
    run_button(page).click()
    expect(page.locator(".error strong")).to_have_text(
        f"Error {expected.status_code}:", timeout=RUN_TIMEOUT_MS
    )
    expect(page.locator(".error pre")).to_have_text(expected.json()["detail"])
    expect(page.locator(".summary")).to_be_hidden()
    expect(page.locator(".caption")).to_be_hidden()
    assert_chart_purged(page)


@pytest.mark.parametrize(
    ("body", "shown"),
    [
        pytest.param('{"detail": "the detail"}', "the detail", id="string-detail"),
        pytest.param('{"detail": {"a": [1]}}', '{\n  "a": [\n    1\n  ]\n}', id="object-detail"),
        pytest.param("not JSON", "not JSON", id="not-json"),
    ],
)
def test_an_error_body_is_shown_as_the_page_reads_it(
    api_page: tuple[Page, Offline], api_site: Site, body: str, shown: str
) -> None:
    page, _ = api_page
    open_page(page, api_site)
    page.route("**/api/simulate", lambda route: route.fulfill(status=500, body=body))
    run_button(page).click()
    expect(page.locator(".error strong")).to_have_text("Error 500:")
    expect(page.locator(".error pre")).to_have_text(shown)
    expect(run_button(page)).to_have_text("Run")
    expect(run_button(page)).to_be_enabled()


def test_an_unreachable_api_is_shown_on_run(api_page: tuple[Page, Offline], api_site: Site) -> None:
    page, _ = api_page
    open_page(page, api_site)
    page.route("**/api/simulate", lambda route: route.abort())
    run_button(page).click()
    expect(page.locator(".error strong")).to_have_text("Error network:")
    expect(page.locator(".error pre")).to_have_text("TypeError: Failed to fetch")
    expect(run_button(page)).to_be_enabled()


@pytest.mark.parametrize(
    ("answer", "line"),
    [
        pytest.param("abort", "unreachable", id="unreachable"),
        pytest.param("error", "error 500", id="server-error"),
    ],
)
def test_the_health_line(
    api_page: tuple[Page, Offline], api_site: Site, answer: str, line: str
) -> None:
    page, _ = api_page
    if answer == "abort":
        page.route("**/health", lambda route: route.abort())
    else:
        page.route("**/health", lambda route: route.fulfill(status=500, body=""))
    page.goto(api_site.url)
    expect(page.locator(".status strong")).to_have_text(line)


def test_a_refused_example_is_shown(api_page: tuple[Page, Offline], api_site: Site) -> None:
    page, _ = api_page
    page.route(
        "**/api/example",
        lambda route: route.fulfill(status=404, body='{"detail": "no example here"}'),
    )
    page.goto(api_site.url)
    expect(page.locator(".error strong")).to_have_text("Error 404:")
    expect(page.locator(".error pre")).to_have_text("no example here")
    expect(textarea(page)).to_have_value("")


def test_an_unreachable_example_is_shown(api_page: tuple[Page, Offline], api_site: Site) -> None:
    page, _ = api_page
    page.route("**/api/example", lambda route: route.abort())
    page.goto(api_site.url)
    expect(page.locator(".error strong")).to_have_text("Error network:")
    expect(page.locator(".error pre")).to_have_text("TypeError: Failed to fetch")

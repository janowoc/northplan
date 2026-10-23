# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Opening a scenario in the form and coming back: the YAML that returns is what Python writes
for the same mapping, on both pages. It covers the committed scenarios, values JavaScript's
numbers cannot hold, number text the form sends as text, the two starting files and what happens
when one is missing, text the form cannot open, the download, Run on the form's scenario, and the
buttons that wait for the engine.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from form_helpers import (
    COUPLE,
    EXAMPLE,
    MINIMAL,
    RUN_TIMEOUT_MS,
    back_to_yaml,
    button,
    control,
    fill,
    is_pyodide,
    lines,
    mapping_of,
    open_example,
    open_new,
    open_text,
    problem,
    shown_errors,
    state,
    textarea,
    wait_clean,
)
from offline import PYODIDE_BASE, Offline, Site
from playwright.sync_api import Page, Route, expect
from test_browser_pyodide_page import wait_until_held, wait_until_ready

from engine.params.loader import DEFAULT_PARAMS_ROOT
from report import editor


def synthetic() -> dict:
    mapping = copy.deepcopy(mapping_of(MINIMAL))
    mapping["seed"] = 2**64 + 1
    zeros = {
        "real_mean": 0.0,
        "vol": 0.0,
        "interest_yield": 0.0,
        "dividend_yield": 0.0,
        "distributed_gains_yield": 0.0,
    }
    mapping["assumptions"]["asset_classes"] = {
        "b": {**zeros, "real_mean": 1e-9},
        "2": {**zeros, "real_mean": -0.0},
        "10": {**zeros, "real_mean": 0.1 + 0.2},
    }
    mapping["assumptions"]["correlation"] = [[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]]
    mapping["assumptions"]["allocations"] = {"default": {"b": 0.5, "2": 0.25, "10": 0.25}}
    mapping["spending"]["schedule"] = [{"from_year": 2026, "annual": 95000}]
    mapping["name"] = "naïve “quoted” \x85 'x'"
    return mapping


SYNTHETIC = synthetic()
SYNTHETIC_TEXT = editor.emit(SYNTHETIC)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(EXAMPLE, id="example"),
        pytest.param(COUPLE, id="couple"),
        pytest.param(MINIMAL, id="minimal"),
    ],
)
def test_each_committed_scenario_comes_back_as_python_writes_it(
    form_page: tuple[Page, Offline], text: str
) -> None:
    page, _ = form_page
    open_text(page, text)
    wait_clean(page)
    assert back_to_yaml(page) == editor.emit(mapping_of(text))
    expect(page.locator("#form")).to_be_hidden()


def test_values_javascript_numbers_cannot_hold_come_back_unchanged(
    form_page: tuple[Page, Offline],
) -> None:
    check = editor.check(SYNTHETIC, params_root=DEFAULT_PARAMS_ROOT)
    assert check["errors"] == []
    assert editor.emit(mapping_of(SYNTHETIC_TEXT)) == SYNTHETIC_TEXT
    assert "seed: 18446744073709551617" in SYNTHETIC_TEXT
    page, _ = form_page
    open_text(page, SYNTHETIC_TEXT)
    wait_clean(page)
    assert back_to_yaml(page) == SYNTHETIC_TEXT


def test_number_text_comes_back_as_the_number(form_page: tuple[Page, Offline]) -> None:
    sent = mapping_of(MINIMAL)
    sent["seed"] = "1_000"
    assert "seed: '1_000'\n" in editor.emit(sent)
    expected = mapping_of(MINIMAL)
    expected["seed"] = 1000
    page, _ = form_page
    open_text(page, MINIMAL)
    fill(page, "1_000", "seed")
    wait_clean(page)
    assert back_to_yaml(page) == editor.emit(expected)


def test_new_scenario_and_open_example_leave_the_textarea(
    form_page: tuple[Page, Offline],
) -> None:
    page, _ = form_page
    textarea(page).fill("custom: text\n")
    open_new(page)
    wait_clean(page)
    button(page, "Discard form").click()
    expect(textarea(page)).to_have_value("custom: text\n")
    open_example(page)
    wait_clean(page)
    assert back_to_yaml(page) == editor.emit(mapping_of(EXAMPLE))


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("- 1\n", id="list"),
        pytest.param(MINIMAL.replace("birth_month: 1\n", "birth_month: 13\n"), id="month"),
    ],
)
def test_text_the_form_cannot_open_stays_with_its_errors(
    form_page: tuple[Page, Offline], text: str
) -> None:
    assert text == "- 1\n" or "birth_month: 13" in text
    page, _ = form_page
    expected = lines(editor.parse(text, params_root=DEFAULT_PARAMS_ROOT)["errors"])
    assert expected
    textarea(page).fill(text)
    button(page, "Edit in form").click()
    expect(page.locator(".editor-errors li")).to_have_text(expected)
    expect(page.locator(".editor-errors p")).to_have_text("The form cannot open this text:")
    assert shown_errors(page) == expected
    expect(page.locator("#form")).to_be_hidden()
    expect(button(page, "Back to YAML")).to_be_hidden()
    expect(textarea(page)).to_have_value(text)


def test_a_missing_starting_file_is_shown(form_page: tuple[Page, Offline]) -> None:
    page, _ = form_page
    if is_pyodide(page):
        page.route("**/minimal.yaml", lambda route: route.fulfill(status=404, body=""))
        expected = "Could not open the file: minimal.yaml: 404 Not Found"
    else:
        page.route(
            "**/api/minimal",
            lambda route: route.fulfill(status=404, body='{"detail": "no minimal here"}'),
        )
        expected = "Could not open the file: /api/minimal: 404 no minimal here"
    button(page, "New scenario").click()
    expect(problem(page)).to_have_text(expected)
    expect(page.locator("#form")).to_be_hidden()
    expect(button(page, "Back to YAML")).to_be_hidden()


def test_download_saves_the_checked_yaml(form_page: tuple[Page, Offline]) -> None:
    page, _ = form_page
    open_new(page)
    wait_clean(page)
    with page.expect_download() as info:
        button(page, "Download YAML").click()
    assert info.value.suggested_filename == "new-scenario.yaml"
    assert Path(info.value.path()).read_text(encoding="utf-8") == editor.emit(mapping_of(MINIMAL))
    fill(page, "", "name")
    wait_clean(page)
    with page.expect_download() as info:
        button(page, "Download YAML").click()
    assert info.value.suggested_filename == "scenario.yaml"
    unnamed = {**mapping_of(MINIMAL), "name": ""}
    assert Path(info.value.path()).read_text(encoding="utf-8") == editor.emit(unnamed)


def test_run_in_the_form_runs_the_checked_yaml(form_page: tuple[Page, Offline]) -> None:
    page, _ = form_page
    open_new(page)
    wait_clean(page)
    fill(page, "7", "seed")
    expect(page.locator("#run")).to_be_disabled()
    wait_clean(page)
    page.locator("#run").click()
    expect(page.locator(".summary")).to_be_visible(timeout=RUN_TIMEOUT_MS)
    assert page.locator(".summary p").all_inner_texts()[2].endswith("seed 7")


def test_the_editor_buttons_wait_for_the_engine(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site
) -> None:
    page, _ = pyodide_page
    held: list[Route] = []
    page.route(PYODIDE_BASE + "pyodide.mjs", lambda route: held.append(route))
    page.goto(pyodide_site.url)
    wait_until_held(page, held)
    names = ["Edit in form", "New scenario", "Open example"]
    for name in names:
        expect(button(page, name)).to_be_disabled()
    held[0].fallback()
    wait_until_ready(page)
    for name in names:
        expect(button(page, name)).to_be_enabled()


def test_anchors_open_with_a_notice(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    text = MINIMAL.replace("survivor_share: 1.0", "survivor_share: &share 1.0")
    assert text != MINIMAL
    open_text(page, text)
    wait_clean(page)
    assert page.locator(".editor-notices li").all_inner_texts() == [editor.FLATTENED]
    expect(state(page)).to_have_text("No errors.")
    assert control(page, "spending", "survivor_share").input_value() == "1.0"

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Helpers the scenario form's browser tests share: locators for the form's elements by the
path they edit, the engine's own answers to compare the page with, and the steps that open a
text in the form and wait for its check. It holds no test; ``conftest.py`` builds the
``form_page`` fixture from :func:`goto_ready`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from offline import Site
from playwright.sync_api import Locator, Page, expect

from engine.params.loader import DEFAULT_PARAMS_ROOT
from report import editor

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = (REPO_ROOT / "scenarios" / "example.yaml").read_text(encoding="utf-8")
COUPLE = (REPO_ROOT / "scenarios" / "late_life_couple.yaml").read_text(encoding="utf-8")
MINIMAL = (REPO_ROOT / "scenarios" / "minimal.yaml").read_text(encoding="utf-8")
ERRORS_TEXT = (
    "The scenario has errors; Run, Back to YAML and Download YAML stay off until they are fixed."
)
SETTLED = re.compile(r"^(No errors\.|The scenario has errors.*)$")
ENGINE_TIMEOUT_MS = 120_000
RUN_TIMEOUT_MS = 300_000


def mapping_of(text: str) -> Any:
    mapping = editor.parse(text, params_root=DEFAULT_PARAMS_ROOT)["mapping"]
    assert mapping is not None
    return mapping


def goto_ready(page: Page, site: Site) -> None:
    """Go to the site and wait until the page has its engine and the example in the box."""
    page.goto(site.url)
    expect(page.locator("#scenario")).to_have_value(EXAMPLE, timeout=ENGINE_TIMEOUT_MS)
    if site.directory is not None:
        expect(page.locator(".status p").first).to_have_text(
            "The engine is ready; it runs in this browser.", timeout=ENGINE_TIMEOUT_MS
        )


def is_pyodide(page: Page) -> bool:
    return page.url.endswith("/northplan/")


def field(page: Page, *path: str | int) -> Locator:
    loc = json.dumps(list(path), separators=(",", ":"), ensure_ascii=False)
    return page.locator(f"[data-loc='{loc}']")


def control(page: Page, *path: str | int) -> Locator:
    return field(page, *path).locator(":scope > .editor-value")


def own(page: Page, cls: str, *path: str | int) -> Locator:
    return field(page, *path).locator(f":scope > .{cls}")


def button(page: Page, name: str) -> Locator:
    return page.get_by_role("button", name=name, exact=True)


def state(page: Page) -> Locator:
    return page.locator(".editor-state")


def textarea(page: Page) -> Locator:
    return page.locator("#scenario")


def shown_errors(page: Page) -> list[str]:
    return page.locator(".editor-errors li").all_inner_texts()


def field_errors(page: Page, *path: str | int) -> list[str]:
    return own(page, "editor-field-errors", *path).locator("li").all_inner_texts()


def problem(page: Page) -> Locator:
    return page.locator(".editor-problem p")


def lines(errors: list[dict[str, Any]]) -> list[str]:
    return [
        ".".join(str(step) for step in e["loc"]) + ": " + e["message"] if e["loc"] else e["message"]
        for e in errors
    ]


def native_errors(mapping: Any) -> list[str]:
    return lines(editor.check(mapping, params_root=DEFAULT_PARAMS_ROOT)["errors"])


def _wait_state(page: Page, text: str | re.Pattern[str]) -> None:
    """Wait until the form's state line is shown and reads ``text``. The line is hidden while the
    form is closed and nothing is opening it or holding it off for a run, and it then still reads
    "No errors.", so a wait on its text alone would pass."""
    expect(state(page).filter(visible=True)).to_have_text(text)


def wait_settled(page: Page) -> None:
    _wait_state(page, SETTLED)


def wait_clean(page: Page) -> None:
    _wait_state(page, "No errors.")


def wait_errors(page: Page) -> None:
    _wait_state(page, ERRORS_TEXT)


def open_text(page: Page, text: str) -> None:
    textarea(page).fill(text)
    button(page, "Edit in form").click()
    wait_settled(page)


def open_new(page: Page) -> None:
    button(page, "New scenario").click()
    wait_settled(page)


def open_example(page: Page) -> None:
    button(page, "Open example").click()
    wait_settled(page)


def fill(page: Page, value: str, *path: str | int) -> None:
    control(page, *path).fill(value)


def tick(page: Page, cls: str, *path: str | int) -> None:
    """Click a control whose change re-draws the form, which Playwright's ``check`` would race."""
    own(page, cls, *path).click()


def back_to_yaml(page: Page) -> str:
    button(page, "Back to YAML").click()
    expect(textarea(page)).to_be_visible()
    return textarea(page).input_value()

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The web page: what it is served as, and what it loads and calls."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import FinalRow, SimulationMeta, SimulationResponse, YearRow

REPO_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = [
    "https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js",
    "https://cdn.plot.ly/plotly-2.35.2.min.js",
]


@pytest.fixture(scope="module")
def page() -> tuple[str, str]:
    response = TestClient(app, base_url="http://localhost").get("/")
    assert response.status_code == 200
    return response.headers["content-type"], response.text


def test_the_page_is_html(page: tuple[str, str]) -> None:
    assert page[0].startswith("text/html")


def test_it_has_a_scenario_textarea_and_a_run_button(page: tuple[str, str]) -> None:
    html = page[1]
    assert re.search(r'<textarea[^>]*id="scenario"', html)
    assert re.search(r"<button[^>]*>\s*Run\s*</button>", html)


def test_it_loads_exactly_the_two_pinned_scripts(page: tuple[str, str]) -> None:
    assert re.findall(r'<script[^>]*\ssrc="([^"]*)"', page[1]) == SCRIPTS


def test_it_calls_the_example_and_simulate_endpoints(page: tuple[str, str]) -> None:
    assert "/api/example" in page[1]
    assert "/api/simulate" in page[1]


def test_it_no_longer_says_the_endpoints_are_not_implemented(page: tuple[str, str]) -> None:
    assert "501" not in page[1]


def test_the_pages_field_names_are_fields_of_the_response_models() -> None:
    html = (REPO_ROOT / "web" / "index.html").read_text(encoding="utf-8")
    meta = set(re.findall(r"doc\.meta\.(\w+)", html))
    final = set(re.findall(r"doc\.final\.(\w+)", html))
    top = set(re.findall(r"doc\.(\w+)", html))
    year = set(re.findall(r"column\('(\w+)'\)", html))
    year |= set(re.findall(r"row\.(\w+)", html)) | set(re.findall(r"last\.(\w+)", html))

    assert meta and final and top and year
    assert {"after_tax_net_worth_p50", "depletion_probability"} <= year
    assert year <= set(YearRow.model_fields)
    assert meta <= set(SimulationMeta.model_fields)
    assert final <= set(FinalRow.model_fields)
    assert top <= set(SimulationResponse.model_fields)

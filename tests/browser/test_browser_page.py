# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The browser page and its worker, read as text: what they load, request and say."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pytest
from offline import CDN_FILES

from api.schemas import FinalRow, SimulationMeta, SimulationResponse, YearRow

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = (REPO_ROOT / "browser" / "index.html").read_text(encoding="utf-8")
API_PAGE = (REPO_ROOT / "api" / "web" / "index.html").read_text(encoding="utf-8")
WORKER = (REPO_ROOT / "browser" / "worker.js").read_text(encoding="utf-8")
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

SCRIPT_URLS = [
    "https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js",
    "https://cdn.plot.ly/plotly-2.35.2.min.js",
]
PYODIDE_BASE = "https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/"
REPOSITORY = "https://github.com/janowoc/northplan"


def scripts(text: str) -> list[str]:
    return re.findall(r'<script[^>]*\ssrc="([^"]*)"', text)


def show_function(text: str) -> str:
    """``show(doc) {`` to its matching closing brace, by brace counting."""
    start = text.index("show(doc) {")
    depth = 0
    for position in range(text.index("{", start), len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start : position + 1]
    raise AssertionError("show(doc) is not closed")


def test_the_two_pages_load_the_same_scripts() -> None:
    assert scripts(PAGE) == scripts(API_PAGE) == [*SCRIPT_URLS, "editor.js"]


def test_show_is_textually_identical_in_both_pages() -> None:
    assert show_function(PAGE) == show_function(API_PAGE)
    assert show_function(PAGE).count("\n") > 20


def test_the_pages_field_names_are_fields_of_the_response_models() -> None:
    meta = set(re.findall(r"doc\.meta\.(\w+)", PAGE))
    final = set(re.findall(r"doc\.final\.(\w+)", PAGE))
    top = set(re.findall(r"doc\.(\w+)", PAGE))
    year = set(re.findall(r"column\('(\w+)'\)", PAGE))
    year |= set(re.findall(r"row\.(\w+)", PAGE)) | set(re.findall(r"last\.(\w+)", PAGE))

    assert meta and final and top and year
    assert {"after_tax_net_worth_p50", "depletion_probability"} <= year
    assert year <= set(YearRow.model_fields)
    assert meta <= set(SimulationMeta.model_fields)
    assert final <= set(FinalRow.model_fields)
    assert top <= set(SimulationResponse.model_fields)


def test_it_has_a_scenario_textarea_a_run_button_an_error_box_and_a_chart() -> None:
    assert re.search(r'<textarea[^>]*id="scenario"', PAGE)
    assert re.search(r"<button[^>]*>\s*Run\s*</button>", PAGE)
    assert "'Running…'" in PAGE
    assert re.search(r'<div class="error"', PAGE)
    assert "'Error ' + (error && error.status) + ':'" in PAGE
    assert re.search(r'<div id="chart"', PAGE)


def test_run_is_disabled_until_the_engine_is_ready() -> None:
    assert ':disabled="running || !ready || formBusy || (formOpen && formYaml === null)"' in PAGE
    assert "Loading the engine (about 11 MB on a first visit)…" in PAGE


def test_every_url_is_one_the_page_may_request() -> None:
    allowed = [*SCRIPT_URLS, PYODIDE_BASE]
    for name, text in (("index.html", PAGE), ("worker.js", WORKER)):
        for url in re.findall(r"https?://[^\s\"'<>)`]+", text):
            assert url in allowed or url.startswith(REPOSITORY), f"{name}: {url}"


def test_no_request_starts_at_the_servers_root() -> None:
    for name, text in (("index.html", PAGE), ("worker.js", WORKER)):
        rooted = re.findall(r"(?:fetch|new URL|import)\(\s*[`'\"]/", text)
        assert not rooted, f"{name}: {rooted}"


def test_the_page_names_its_files_relatively() -> None:
    assert "fetch('build-info.json')" in PAGE
    assert "fetch('example.yaml')" in PAGE
    assert "new URL('worker.js', location.href)" in PAGE
    assert "{ type: 'module' }" in PAGE
    assert 'new URL("northplan.zip", self.location)' in WORKER


def test_the_pyodide_version_is_named_once_and_agrees_with_the_npm_pins() -> None:
    assert len(re.findall(r"PYODIDE_VERSION = ", WORKER)) == 1
    version = re.search(r'const PYODIDE_VERSION = "([^"]+)";', WORKER)
    assert version is not None
    tests = REPO_ROOT / "tests" / "browser"
    package = json.loads((tests / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((tests / "package-lock.json").read_text(encoding="utf-8"))
    assert package["dependencies"]["pyodide"] == version.group(1)
    assert lock["packages"][""]["dependencies"]["pyodide"] == version.group(1)
    assert lock["packages"]["node_modules/pyodide"]["version"] == version.group(1)
    for text in (
        PAGE,
        *(p.read_text(encoding="utf-8") for p in (REPO_ROOT / "browser").glob("*.py")),
    ):
        assert version.group(1) not in text


def test_the_offline_scripts_are_the_pages_scripts_at_the_npm_pins() -> None:
    assert list(CDN_FILES) == SCRIPT_URLS
    tests = REPO_ROOT / "tests" / "browser"
    package = json.loads((tests / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((tests / "package-lock.json").read_text(encoding="utf-8"))
    for name in ("alpinejs", "plotly.js-dist-min"):
        pinned = package["dependencies"][name]
        assert lock["packages"][""]["dependencies"][name] == pinned
        assert lock["packages"][f"node_modules/{name}"]["version"] == pinned


def squeeze(text: str) -> str:
    return " ".join(text.split())


def test_the_disclaimer_is_the_readmes_section() -> None:
    element = re.search(r'<div id="disclaimer">(.*?)</div>', PAGE, re.DOTALL)
    assert element is not None
    shown = squeeze(html.unescape(re.sub(r"<[^>]+>", "", element.group(1))))
    section = README[
        README.index("## Disclaimer") + len("## Disclaimer") : README.index("## License")
    ]
    section = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", section).replace("**", "")
    assert shown
    assert shown == squeeze(section)


@pytest.mark.parametrize(
    ("target", "url"),
    [
        ("docs/limitations.md", f"{REPOSITORY}/blob/main/docs/limitations.md"),
        ("LICENSE", f"{REPOSITORY}/blob/main/LICENSE"),
        ("#license", f"{REPOSITORY}#license"),
    ],
)
def test_each_disclaimer_link_is_absolute(target: str, url: str) -> None:
    assert f"]({target})" in README
    assert re.search(rf'<a\s+href="{re.escape(url)}">', PAGE)


def editor_block(text: str) -> str:
    start = text.index("<!-- editor -->")
    return text[start : text.index("<!-- /editor -->") + len("<!-- /editor -->")]


def test_the_editor_block_is_textually_identical_in_both_pages() -> None:
    assert editor_block(PAGE) == editor_block(API_PAGE)
    assert editor_block(PAGE).count("\n") > 15


def test_editor_js_makes_no_request() -> None:
    text = (REPO_ROOT / "api" / "web" / "editor.js").read_text(encoding="utf-8")
    for forbidden in (
        "fetch(",
        "XMLHttpRequest",
        "import(",
        "new Worker",
        "new URL(",
        "WebSocket",
        "sendBeacon",
        "http://",
        "https://",
    ):
        assert forbidden not in text, forbidden


def test_the_editor_files_are_named_relatively() -> None:
    assert "fetch(name + '.yaml')" in PAGE
    assert '<script src="editor.js"></script>' in PAGE
    assert '<script src="editor.js"></script>' in API_PAGE


def test_the_worker_answers_both_message_types() -> None:
    assert 'type === "editor"' in WORKER
    assert "entry.editor_call(event.data.call, event.data.arguments)" in WORKER
    assert 'type: "reply"' in WORKER

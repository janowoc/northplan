# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The page's editor calls against the API's: the same arguments, under Pyodide and natively.

The site is built with browser/build.py, and the Node harness runs the whole corpus once
under Pyodide. The API answers the same arguments in process. The browser's string must be
the status envelope around the API's body text, equal after the same normalisation as the
run differential applies, and nothing else.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from node_gate import BROWSER_TESTS, require_node
from test_browser_differential import BROWSER_PARAMS_ROOT, normalise

import api.main
from api.main import app
from engine.params.loader import DEFAULT_PARAMS_ROOT
from report import editor

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = (REPO_ROOT / "scenarios" / "example.yaml").read_text(encoding="utf-8")
COUPLE = (REPO_ROOT / "scenarios" / "late_life_couple.yaml").read_text(encoding="utf-8")
MINIMAL = (REPO_ROOT / "scenarios" / "minimal.yaml").read_text(encoding="utf-8")
GRID_LINE = "elections.cpp_start_age_years.a: [60, 65, 70]"
MAPPING = editor.parse(EXAMPLE, params_root=DEFAULT_PARAMS_ROOT)["mapping"]
WITH_GOLD = editor.add_asset_class(MAPPING, "gold")


@dataclass(frozen=True)
class Case:
    """One call, its arguments as text, and the status the API must give."""

    id: str
    call: str
    arguments: str
    status: int


def replaced(text: str, old: str, new: str, count: int = -1) -> str:
    assert old in text
    return text.replace(old, new, count)


def parse_of(text: str) -> str:
    return json.dumps({"text": text})


def mapped(**arguments: object) -> str:
    return json.dumps({"mapping": MAPPING, **arguments})


def start_year_mapping(year: int) -> dict:
    return {**MAPPING, "start_year": year}


CASES = [
    Case("parse-example", "parse", parse_of(EXAMPLE), 200),
    Case("parse-couple", "parse", parse_of(COUPLE), 200),
    Case("parse-minimal", "parse", parse_of(MINIMAL), 200),
    Case("parse-invalid-yaml", "parse", parse_of("name: [unclosed\n"), 200),
    Case(
        "parse-schema-refusal",
        "parse",
        parse_of(replaced(EXAMPLE, "n_paths: 10000", "n_paths: 0")),
        200,
    ),
    Case(
        "parse-start-year-2031",
        "parse",
        parse_of(replaced(EXAMPLE, "start_year: 2026", "start_year: 2031")),
        200,
    ),
    Case(
        "parse-grid-value-refused",
        "parse",
        parse_of(replaced(EXAMPLE, GRID_LINE, "elections.rrif_conversion.fraction: [-1.0]")),
        200,
    ),
    Case(
        "parse-binary-key",
        "parse",
        parse_of(replaced(EXAMPLE, "    default:", "    !!binary ZGVmYXVsdA==:", 1)),
        200,
    ),
    Case(
        "parse-merge-key-flattened",
        "parse",
        parse_of(
            replaced(
                EXAMPLE,
                "    withdrawal:\n",
                "    withdrawal:\n      <<: {fill_pension_credit: true}\n",
                1,
            )
        ),
        200,
    ),
    Case("check-example", "check", json.dumps({"mapping": MAPPING}), 200),
    Case(
        "check-start-year-2040",
        "check",
        json.dumps({"mapping": start_year_mapping(2040)}),
        200,
    ),
    Case("check-empty-mapping", "check", '{"mapping": {}}', 200),
    Case("check-null", "check", '{"mapping": null}', 200),
    Case("schema", "schema", "{}", 200),
    Case("references-person", "references", mapped(kind="person", name="a"), 200),
    Case("references-asset-class", "references", mapped(kind="asset-class", name="equity"), 200),
    Case("rename-person", "rename", mapped(kind="person", old="a", new="b"), 200),
    Case(
        "rename-asset-class",
        "rename",
        mapped(kind="asset-class", old="equity", new="stocks"),
        200,
    ),
    Case(
        "rename-person-lone-surrogate", "rename", mapped(kind="person", old="a", new="\ud800"), 200
    ),
    Case(
        "remove-asset-class",
        "remove",
        json.dumps({"mapping": WITH_GOLD, "kind": "asset-class", "name": "gold"}),
        200,
    ),
    Case("add-asset-class", "add-asset-class", mapped(name="gold"), 200),
    Case("move-asset-class", "move-asset-class", mapped(name="bonds", index=0), 200),
    Case("grid-paths", "grid-paths", mapped(), 200),
    Case("rename-missing-person", "rename", mapped(kind="person", old="zz", new="b"), 409),
    Case("remove-person", "remove", mapped(kind="person", name="a"), 409),
    Case("move-asset-class-out-of-range", "move-asset-class", mapped(name="bonds", index=9), 409),
    Case("unknown-call", "nope", "{}", 404),
    Case("not-json", "parse", "not json", 422),
    Case("nan", "check", '{"mapping": NaN}', 422),
    Case("float-too-large", "check", '{"mapping": 1e999}', 422),
    Case("integer-too-long", "parse", '{"text": 1' + "0" * 5000 + "}", 422),
    Case("key-given-twice", "parse", '{"text": "a", "text": "b"}', 422),
    Case(
        "nested-100000-deep",
        "check",
        '{"mapping": ' + "[" * 100_000 + "]" * 100_000 + "}",
        422,
    ),
    Case("not-an-object", "parse", "[]", 422),
    Case("unknown-argument", "schema", '{"a": 1}', 422),
    Case("missing-argument", "rename", '{"mapping": null, "kind": "person"}', 422),
    Case("bad-kind", "references", '{"mapping": null, "kind": "people", "name": "a"}', 422),
    Case("index-true", "move-asset-class", '{"mapping": null, "name": "a", "index": true}', 422),
    Case("name-5", "add-asset-class", '{"mapping": null, "name": 5}', 422),
    Case(
        "mapping-nested-400-deep",
        "check",
        '{"mapping": ' + "[" * 400 + "]" * 400 + "}",
        422,
    ),
]
BY_ID = {case.id: case for case in CASES}


@dataclass(frozen=True)
class Outcome:
    """What the API answered and the raw string the browser returned."""

    api_status: int
    api_text: str
    api_content_type: str
    browser_raw: str


def api_answer(case: Case) -> tuple[int, str, str]:
    client = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000))
    response = client.post(
        f"/api/editor/{case.call}",
        content=case.arguments.encode(),
        headers={"Content-Type": "application/json"},
    )
    return response.status_code, response.text, response.headers["content-type"]


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Outcome]:
    """The corpus run once under Pyodide in Node, and once through the API."""
    node = require_node()
    work = tmp_path_factory.mktemp("editor-differential")
    site = work / "site"
    built = subprocess.run(
        ["python3", str(REPO_ROOT / "browser" / "build.py"), str(site)],
        capture_output=True,
        text=True,
    )
    if built.returncode != 0:
        pytest.fail(
            f"browser/build.py exited {built.returncode}\nstdout:\n{built.stdout}\n"
            f"stderr:\n{built.stderr}"
        )
    corpus = work / "corpus.json"
    corpus.write_text(
        json.dumps([{"call": case.call, "arguments": case.arguments} for case in CASES]),
        encoding="utf-8",
    )
    results = work / "results.json"
    with corpus.open("rb") as stdin:
        completed = subprocess.run(
            [node, str(BROWSER_TESTS / "harness.mjs"), str(site), str(results)],
            stdin=stdin,
            cwd=BROWSER_TESTS,
            capture_output=True,
            text=True,
            timeout=900,
        )
    assert completed.returncode == 0, completed.stderr
    raw = json.loads(results.read_text(encoding="utf-8"))
    assert len(raw) == len(CASES)
    found: dict[str, Outcome] = {}
    for case, text in zip(CASES, raw, strict=True):
        status, body, content_type = api_answer(case)
        found[case.id] = Outcome(status, body, content_type, text)
    return found


def browser_body(outcome: Outcome) -> str:
    """The body text inside the browser's envelope, once the envelope is checked."""
    start = f'{{"status": {outcome.api_status}, "body": '
    assert outcome.browser_raw.startswith(start), outcome.browser_raw[:200]
    assert outcome.browser_raw.endswith("}")
    return outcome.browser_raw[len(start) : -1]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_the_browser_answers_as_the_api_does(outcomes: dict[str, Outcome], case: Case) -> None:
    outcome = outcomes[case.id]
    assert outcome.api_status == case.status, outcome.api_text
    assert outcome.api_content_type == "application/json"
    assert normalise(browser_body(outcome), BROWSER_PARAMS_ROOT) == normalise(
        outcome.api_text, str(api.main.PARAMS_ROOT)
    )


def test_the_normalisation_is_needed_for_the_parameter_year_error(
    outcomes: dict[str, Outcome],
) -> None:
    outcome = outcomes["parse-start-year-2031"]
    browser = browser_body(outcome)
    assert browser != outcome.api_text
    assert str(api.main.PARAMS_ROOT) in outcome.api_text
    assert BROWSER_PARAMS_ROOT in browser

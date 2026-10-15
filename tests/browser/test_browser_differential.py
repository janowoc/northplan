# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The page's engine against the API: the same scenario, run natively and under Pyodide.

The site is built with browser/build.py, and the Node harness runs the whole corpus once
under Pyodide, which computes what a browser computes. The API runs the same bodies in
process. A deterministic document must be equal to the last float; a stochastic one may
differ in dollar floats by rounding (the platforms' libm differ) and in a share by whole
paths. A refusal must match in status and in detail, after the two normalisations that
:func:`normalise` names and nothing else.

The two differences between the entry points are tested on the browser side alone: its
25,000-path cap, and that it counts a body's keys and strings, not its comments, so a
body the API refuses as too large runs in the browser.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from node_gate import BROWSER_TESTS, require_node

import api.main
from api.main import app
from engine.params.loader import DEFAULT_PARAMS_ROOT
from report.run import PATHS_WITH_DETERMINISTIC, open_run

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = (REPO_ROOT / "scenarios" / "example.yaml").read_text(encoding="utf-8")
COUPLE = (REPO_ROOT / "scenarios" / "late_life_couple.yaml").read_text(encoding="utf-8")
GRID_LINE = "elections.cpp_start_age_years.a: [60, 65, 70]"
YAML = {"Content-Type": "application/yaml"}
BROWSER_PARAMS_ROOT = "/northplan/params"
PYDANTIC_URL = re.compile(r"errors\.pydantic\.dev/\d+\.\d+/")


# --- The comparison ----------------------------------------------------------------------


def is_probability_over_all_paths(key: str) -> bool:
    return key in ("depletion_probability", "estate_zero_share") or key.startswith(
        "death_probability_"
    )


def compare(api_document: object, browser_document: object, *, deterministic: bool) -> list[str]:
    """The differences between the API's document and the browser's, empty when equal.

    Structure is exact: key order, key sets, value types (an int is not a float, a bool is
    neither), strings, ints, bools, None and list lengths. A float in a deterministic
    document is exact. In a stochastic one, ``depletion_probability``, ``estate_zero_share``
    and each ``death_probability_<id>`` may differ by 5 / ``meta.paths``, ``gis_exposure`` by
    5 / ``paths_alive`` of its own year row, and any other float must be
    ``math.isclose(rel_tol=1e-9, abs_tol=1e-6)``.
    """
    differences: list[str] = []
    paths = None
    if isinstance(api_document, dict) and isinstance(api_document.get("meta"), dict):
        paths = api_document["meta"].get("paths")

    def walk(a: object, b: object, where: str, key: str, row: dict | None) -> None:
        if type(a) is not type(b):
            differences.append(
                f"{where}: {type(a).__name__} {a!r} against {type(b).__name__} {b!r}"
            )
        elif isinstance(a, dict):
            assert isinstance(b, dict)
            if list(a) != list(b):
                differences.append(f"{where}: keys {list(a)!r} against {list(b)!r}")
                return
            for name in a:
                walk(a[name], b[name], f"{where}.{name}", name, a)
        elif isinstance(a, list):
            assert isinstance(b, list)
            if len(a) != len(b):
                differences.append(f"{where}: {len(a)} items against {len(b)}")
                return
            for index, (left, right) in enumerate(zip(a, b, strict=True)):
                walk(left, right, f"{where}[{index}]", key, row)
        elif isinstance(a, float):
            assert isinstance(b, float)
            if not close(a, b, key, row):
                differences.append(f"{where}: {a!r} against {b!r}")
        elif a != b:
            differences.append(f"{where}: {a!r} against {b!r}")

    def close(a: float, b: float, key: str, row: dict | None) -> bool:
        if deterministic:
            return a == b
        if is_probability_over_all_paths(key):
            return paths is not None and abs(a - b) <= 5 / paths
        if key == "gis_exposure":
            alive = row.get("paths_alive") if row is not None else None
            return bool(alive) and abs(a - b) <= 5 / alive
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)

    walk(api_document, browser_document, "document", "", None)
    return differences


def doc(share: float = 0.25, **overrides: object) -> dict:
    """A small synthetic document: 64 paths and a year row of 32 alive."""
    row = {
        "year": 2030,
        "paths_alive": 32,
        "net_worth_p50": 1_000_000.0,
        "depletion_probability": 0.25,
        "gis_exposure": 0.25,
        "death_probability_a": 0.25,
    }
    row.update(overrides)
    return {
        "meta": {"paths": 64, "seed": 42},
        "years": [row],
        "final": {"x": 0.0, "estate_zero_share": share},
    }


def test_equal_documents_have_no_differences() -> None:
    assert compare(doc(), doc(), deterministic=True) == []
    assert compare(doc(), doc(), deterministic=False) == []


def test_key_order_differing_fails() -> None:
    reordered = doc()
    reordered["years"][0] = dict(reversed(list(reordered["years"][0].items())))
    assert compare(doc(), reordered, deterministic=False)


def test_a_missing_key_fails() -> None:
    short = doc()
    del short["final"]
    assert compare(doc(), short, deterministic=False)


def test_an_int_against_a_float_fails() -> None:
    assert compare(doc(net_worth_p50=1.0), doc(net_worth_p50=1), deterministic=False)


def test_a_bool_against_an_int_fails() -> None:
    assert compare(doc(year=1), doc(year=True), deterministic=False)


def test_none_against_a_float_fails() -> None:
    assert compare(doc(gis_exposure=None), doc(gis_exposure=0.25), deterministic=False)
    assert compare(doc(gis_exposure=0.25), doc(gis_exposure=None), deterministic=False)


def test_a_differing_string_fails() -> None:
    a, b = doc(), doc()
    b["meta"]["scenario"] = "x"
    a["meta"]["scenario"] = "y"
    assert compare(a, b, deterministic=False)


def test_a_differing_int_fails() -> None:
    assert compare(doc(year=2030), doc(year=2031), deterministic=False)


def test_a_list_length_differing_fails() -> None:
    longer = doc()
    longer["years"].append(dict(longer["years"][0]))
    assert compare(doc(), longer, deterministic=False)


def test_a_deterministic_one_ulp_float_difference_fails() -> None:
    nudged = math.nextafter(1_000_000.0, math.inf)
    assert compare(doc(), doc(net_worth_p50=nudged), deterministic=True)
    assert compare(doc(), doc(net_worth_p50=nudged), deterministic=False) == []


def test_a_dollar_float_within_1e_9_relative_passes() -> None:
    assert compare(doc(), doc(net_worth_p50=1_000_000.0005), deterministic=False) == []


def test_a_dollar_float_beyond_1e_9_relative_fails() -> None:
    assert compare(doc(), doc(net_worth_p50=1_000_000.002), deterministic=False)


def test_a_float_near_zero_within_1e_6_absolute_passes() -> None:
    assert compare(doc(net_worth_p50=0.0), doc(net_worth_p50=5e-7), deterministic=False) == []


def test_a_float_near_zero_beyond_1e_6_absolute_fails() -> None:
    assert compare(doc(net_worth_p50=0.0), doc(net_worth_p50=2e-6), deterministic=False)


def test_depletion_probability_five_paths_apart_passes() -> None:
    five = 0.25 + 5 / 64
    assert compare(doc(), doc(depletion_probability=five), deterministic=False) == []


def test_depletion_probability_six_paths_apart_fails() -> None:
    six = 0.25 + 6 / 64
    assert compare(doc(), doc(depletion_probability=six), deterministic=False)


def test_estate_zero_share_five_paths_apart_passes() -> None:
    assert compare(doc(), doc(share=0.25 + 5 / 64), deterministic=False) == []


def test_estate_zero_share_six_paths_apart_fails() -> None:
    assert compare(doc(), doc(share=0.25 + 6 / 64), deterministic=False)


def test_estate_zero_share_is_exact_in_a_deterministic_document() -> None:
    assert compare(doc(), doc(share=0.25 + 1 / 64), deterministic=True)


def test_death_probability_five_paths_apart_passes() -> None:
    assert compare(doc(), doc(death_probability_a=0.25 + 5 / 64), deterministic=False) == []


def test_death_probability_six_paths_apart_fails() -> None:
    assert compare(doc(), doc(death_probability_a=0.25 + 6 / 64), deterministic=False)


def test_gis_exposure_five_persons_apart_passes() -> None:
    assert compare(doc(), doc(gis_exposure=0.25 + 5 / 32), deterministic=False) == []


def test_gis_exposure_six_persons_apart_fails() -> None:
    assert compare(doc(), doc(gis_exposure=0.25 + 6 / 32), deterministic=False)


def test_gis_exposure_none_on_both_sides_is_equal() -> None:
    assert compare(doc(gis_exposure=None), doc(gis_exposure=None), deterministic=False) == []


# --- The corpus --------------------------------------------------------------------------


def comment_padded(text: str) -> str:
    """``text`` followed by a YAML comment that takes the body past the API's size limit."""
    padding = "#" + "x" * 1023 + "\n"
    return text + "\n" + padding * (api.main.MAX_BODY_BYTES // len(padding) + 1)


@dataclass(frozen=True)
class Case:
    """One body and its options, as the browser receives them and as the API's query."""

    id: str
    kind: str
    text: str
    options: dict[str, object] = field(default_factory=dict)
    query: dict[str, str] = field(default_factory=dict)


SECOND_POLICY = (
    open_run(EXAMPLE, "yaml", paths=2, deterministic=False, params_root=DEFAULT_PARAMS_ROOT)[1]
    .scenario.policies[1]
    .name
)

MANY_CANDIDATES = EXAMPLE.replace(
    GRID_LINE,
    "elections.cpp_start_age_years.a: [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]\n"
    "  elections.oas_start_age_years.a: [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]\n"
    "  elections.rrif_conversion.age_years: [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]",
)

EXACT = [
    Case(
        "example-deterministic",
        "exact",
        EXAMPLE,
        {"deterministic": True},
        {"deterministic": "true"},
    ),
    Case(
        "couple-deterministic",
        "exact",
        COUPLE,
        {"deterministic": True},
        {"deterministic": "true"},
    ),
]
TOLERANT = [
    Case("example-200-paths", "tolerant", EXAMPLE, {"paths": 200}, {"paths": "200"}),
    Case("couple-1000-paths", "tolerant", COUPLE, {"paths": 1000}, {"paths": "1000"}),
    Case(
        "example-100-paths-nominal",
        "tolerant",
        EXAMPLE,
        {"paths": 100, "nominal": True},
        {"paths": "100", "nominal": "true"},
    ),
    Case(
        "example-100-paths-second-policy",
        "tolerant",
        EXAMPLE,
        {"paths": 100, "policy": SECOND_POLICY},
        {"paths": "100", "policy": SECOND_POLICY},
    ),
    Case("example-no-options", "tolerant", EXAMPLE),
    Case("couple-10000-paths", "tolerant", COUPLE, {"paths": 10000}, {"paths": "10000"}),
]
REFUSALS = [
    Case(
        "unknown-policy",
        "refusal",
        EXAMPLE,
        {"paths": 2, "policy": "nope"},
        {"paths": "2", "policy": "nope"},
    ),
    Case("invalid-yaml", "refusal", "name: [unclosed\n"),
    Case("schema-refusal", "refusal", EXAMPLE.replace("n_paths: 10000", "n_paths: 0")),
    Case(
        "grid-value-refused-at-prepare",
        "refusal",
        EXAMPLE.replace(GRID_LINE, "elections.rrif_conversion.fraction: [-1.0]"),
        {"paths": 2},
        {"paths": "2"},
    ),
    Case(
        "start-age-refused",
        "refusal",
        EXAMPLE.replace(GRID_LINE, "elections.cpp_start_age_years.a: [50]"),
        {"paths": 2},
        {"paths": "2"},
    ),
    Case(
        "too-many-candidates",
        "refusal",
        MANY_CANDIDATES,
        {"deterministic": True},
        {"deterministic": "true"},
    ),
    Case(
        "start-year-2031",
        "refusal",
        EXAMPLE.replace("start_year: 2026", "start_year: 2031"),
        {"paths": 2},
        {"paths": "2"},
    ),
    Case(
        "paths-with-deterministic",
        "refusal",
        EXAMPLE,
        {"paths": 2, "deterministic": True},
        {"paths": "2", "deterministic": "true"},
    ),
]
BROWSER_ONLY = [
    Case("over-the-cap", "browser", EXAMPLE, {"paths": 25001}),
    Case("comment-padded-body", "browser", comment_padded(EXAMPLE), {"paths": 50}),
]
CASES = [*EXACT, *TOLERANT, *REFUSALS, *BROWSER_ONLY]
BY_ID = {case.id: case for case in CASES}


@dataclass(frozen=True)
class Outcome:
    """What the API answered (None for a browser-only case) and what the browser returned."""

    api_status: int | None
    api_body: object
    browser: dict
    browser_raw: str


def api_answer(case: Case) -> tuple[int, object]:
    response = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000)).post(
        "/api/simulate", params=case.query, content=case.text.encode(), headers=YAML
    )
    return response.status_code, response.json()


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Outcome]:
    """The corpus run once under Pyodide in Node, and once through the API."""
    node = require_node()
    work = tmp_path_factory.mktemp("differential")
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
        json.dumps([{"text": case.text, "options": case.options} for case in CASES]),
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
        status, body = (None, None) if case.kind == "browser" else api_answer(case)
        found[case.id] = Outcome(status, body, json.loads(text), text)
    return found


# --- Documents ---------------------------------------------------------------------------


def check_document(outcome: Outcome, *, deterministic: bool) -> None:
    assert outcome.api_status == 200, outcome.api_body
    assert set(outcome.browser) == {"document"}, outcome.browser
    differences = compare(
        outcome.api_body, outcome.browser["document"], deterministic=deterministic
    )
    assert not differences, "\n".join(differences[:20])


@pytest.mark.parametrize("case", EXACT, ids=lambda case: case.id)
def test_a_deterministic_document_is_exactly_the_apis(
    outcomes: dict[str, Outcome], case: Case
) -> None:
    outcome = outcomes[case.id]
    check_document(outcome, deterministic=True)
    assert outcome.browser["document"] == outcome.api_body


@pytest.mark.parametrize("case", TOLERANT, ids=lambda case: case.id)
def test_a_stochastic_document_is_the_apis_within_the_stated_tolerances(
    outcomes: dict[str, Outcome], case: Case
) -> None:
    check_document(outcomes[case.id], deterministic=False)


# --- Refusals ----------------------------------------------------------------------------


def normalise(detail: str, params_root: str) -> str:
    """``detail`` with its own params root as ``<params>`` and the pydantic docs version
    as ``<v>``; the only two ways the platforms' refusals differ."""
    return PYDANTIC_URL.sub("errors.pydantic.dev/<v>/", detail.replace(params_root, "<params>"))


def api_detail(outcome: Outcome) -> str:
    assert isinstance(outcome.api_body, dict)
    return outcome.api_body["detail"]


@pytest.mark.parametrize("case", REFUSALS, ids=lambda case: case.id)
def test_a_refusal_has_the_apis_status_and_detail_after_normalising(
    outcomes: dict[str, Outcome], case: Case
) -> None:
    outcome = outcomes[case.id]
    assert set(outcome.browser) == {"status", "detail"}, outcome.browser
    assert outcome.browser["status"] == outcome.api_status
    assert normalise(outcome.browser["detail"], BROWSER_PARAMS_ROOT) == normalise(
        api_detail(outcome), str(api.main.PARAMS_ROOT)
    )


def test_the_normalisation_is_needed_for_the_parameter_year_refusal(
    outcomes: dict[str, Outcome],
) -> None:
    outcome = outcomes["start-year-2031"]
    assert outcome.browser["status"] == outcome.api_status == 400
    assert outcome.browser["detail"] != api_detail(outcome)
    assert str(api.main.PARAMS_ROOT) in api_detail(outcome)
    assert BROWSER_PARAMS_ROOT in outcome.browser["detail"]


def test_the_normalisation_is_needed_for_a_validation_refusal(
    outcomes: dict[str, Outcome],
) -> None:
    outcome = outcomes["schema-refusal"]
    assert outcome.browser["status"] == outcome.api_status == 422
    assert outcome.browser["detail"] != api_detail(outcome)
    assert PYDANTIC_URL.search(api_detail(outcome))
    assert PYDANTIC_URL.search(outcome.browser["detail"])


def test_the_normalisation_makes_the_parameter_year_refusals_equal(
    outcomes: dict[str, Outcome],
) -> None:
    outcome = outcomes["start-year-2031"]
    assert normalise(outcome.browser["detail"], BROWSER_PARAMS_ROOT) == normalise(
        api_detail(outcome), str(api.main.PARAMS_ROOT)
    )


def test_the_normalisation_makes_the_validation_refusals_equal(
    outcomes: dict[str, Outcome],
) -> None:
    outcome = outcomes["schema-refusal"]
    assert normalise(outcome.browser["detail"], BROWSER_PARAMS_ROOT) == normalise(
        api_detail(outcome), str(api.main.PARAMS_ROOT)
    )


def test_the_paths_with_deterministic_text_is_the_same_on_both_sides(
    outcomes: dict[str, Outcome],
) -> None:
    outcome = outcomes["paths-with-deterministic"]
    assert outcome.browser == {"status": 422, "detail": PATHS_WITH_DETERMINISTIC}
    assert outcome.api_status == 422
    assert api_detail(outcome) == PATHS_WITH_DETERMINISTIC


def test_the_normalisation_changes_only_the_root_and_the_docs_version() -> None:
    assert normalise("a /r/params/2031 b", "/r/params") == "a <params>/2031 b"
    assert normalise("see errors.pydantic.dev/2.13/v/x", "/r") == "see errors.pydantic.dev/<v>/v/x"
    assert normalise("nothing to change", "/r") == "nothing to change"


# --- Where the browser differs from the API, by design -----------------------------------


def test_a_run_over_the_browsers_path_cap_is_refused_there(outcomes: dict[str, Outcome]) -> None:
    """The API would run 25,001 paths (its limit is 100,000); that side is not run here."""
    browser = outcomes["over-the-cap"].browser
    assert browser == {
        "status": 422,
        "detail": (
            "the run asks for 25,001 paths, and a run here may have at most 25,000. "
            "Lower n_paths, or use the local install, which allows up to 100,000."
        ),
    }


def test_a_body_padded_with_a_comment_runs_in_the_browser(outcomes: dict[str, Outcome]) -> None:
    """The API answers 413 for this body, asserted below without running it; the browser
    counts the scenario's keys and strings, not its comments, so it runs."""
    case = BY_ID["comment-padded-body"]
    assert len(case.text.encode()) > api.main.MAX_BODY_BYTES
    browser = outcomes[case.id].browser
    assert set(browser) == {"document"}
    assert browser["document"]["meta"]["paths"] == 50


def test_the_api_refuses_the_padded_body_as_too_large() -> None:
    case = BY_ID["comment-padded-body"]
    status, body = api_answer(case)
    assert status == 413
    assert isinstance(body, dict) and "larger than" in body["detail"]

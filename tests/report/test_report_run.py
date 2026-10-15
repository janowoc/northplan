# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The shared simulate pipeline: each refusal, each branch of the path cap, and that its
document is the one the API returns."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from engine.core.indexation import RoutedParameterError
from engine.mc.prepare import MAX_PATHS
from engine.params.loader import DEFAULT_PARAMS_ROOT, MissingParameterError
from engine.scenario.load import ScenarioError
from report import run as report_run
from report.run import (
    PATHS_WITH_DETERMINISTIC,
    RunRefusedError,
    open_run,
    simulate_document,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = (REPO_ROOT / "scenarios" / "example.yaml").read_text(encoding="utf-8")
GRID_LINE = "elections.cpp_start_age_years.a: [60, 65, 70]"
YAML = {"Content-Type": "application/yaml"}


def refusal(text: str, **overrides: object) -> RunRefusedError:
    """The RunRefusedError ``simulate_document`` raises for ``text``."""
    options: dict[str, object] = {
        "policy": None,
        "paths": 2,
        "deterministic": False,
        "nominal": False,
        "params_root": DEFAULT_PARAMS_ROOT,
    }
    options.update(overrides)
    with pytest.raises(RunRefusedError) as caught:
        simulate_document(text, "yaml", **options)  # type: ignore[arg-type]
    return caught.value


def opened(text: str = EXAMPLE, **overrides: object):
    options: dict[str, object] = {
        "paths": 2,
        "deterministic": False,
        "params_root": DEFAULT_PARAMS_ROOT,
    }
    options.update(overrides)
    return open_run(text, "yaml", **options)  # type: ignore[arg-type]


@pytest.fixture
def prepare_must_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        pytest.fail("prepare_run was called")

    monkeypatch.setattr(report_run, "prepare_run", fail)


def raising(error: Exception):
    def raise_it(*_args: object, **_kwargs: object) -> None:
        raise error

    return raise_it


def test_the_error_carries_its_status_and_detail() -> None:
    error = RunRefusedError(422, "the detail")
    assert (error.status, error.detail, str(error)) == (422, "the detail", "the detail")


def test_a_scenario_that_does_not_parse_is_a_422_with_the_scenario_error() -> None:
    error = refusal("- 1\n- 2\n")
    assert error.status == 422
    assert "this parses to list." in error.detail


@pytest.mark.usefixtures("prepare_must_not_run")
class TestPathCapRefuses:
    """Each refuses before the run is opened: prepare_run, patched to fail, is not reached."""

    def test_asked_over_the_cap_is_refused_with_the_message(self) -> None:
        error = refusal(EXAMPLE, paths=1_001, path_cap=1_000)
        assert error.status == 422
        assert error.detail == (
            "the run asks for 1,001 paths, and a run here may have at most 1,000. "
            f"Lower n_paths, or use the local install, which allows up to {MAX_PATHS:,}."
        )

    def test_the_message_names_the_engines_maximum_formatted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(report_run, "MAX_PATHS", 7)
        error = refusal(EXAMPLE, paths=5, path_cap=4)
        assert error.detail.endswith("allows up to 7.")

    def test_an_absent_paths_counts_the_scenarios_n_paths(self) -> None:
        error = refusal(EXAMPLE, paths=None, path_cap=9_999)
        assert error.detail.startswith("the run asks for 10,000 paths, and")

    def test_a_deterministic_run_over_a_cap_of_zero_is_refused_as_one_path(self) -> None:
        error = refusal(EXAMPLE, paths=None, deterministic=True, path_cap=0)
        assert error.detail.startswith("the run asks for 1 paths, and")


class TestPathCapPasses:
    def test_asked_equal_to_the_cap_passes(self) -> None:
        scenario, prepared = opened(paths=5, path_cap=5)
        assert scenario.name == prepared.scenario.name

    def test_no_cap_never_refuses(self) -> None:
        opened(paths=50, path_cap=None)

    def test_a_deterministic_run_counts_as_one_path(self) -> None:
        opened(paths=None, deterministic=True, path_cap=1)

    def test_paths_override_the_scenarios_n_paths(self) -> None:
        assert "n_paths: 10000" in EXAMPLE
        opened(paths=5, path_cap=5)


def test_a_grid_value_the_schema_refuses_is_a_422_with_the_validation_message() -> None:
    body = EXAMPLE.replace(GRID_LINE, "elections.rrif_conversion.fraction: [-1.0]")
    assert body != EXAMPLE
    error = refusal(body)
    assert error.status == 422
    assert error.detail.startswith("1 validation error for Scenario")
    assert "rrif_conversion.fraction" in error.detail


def test_a_start_age_the_checks_refuse_is_a_422() -> None:
    body = EXAMPLE.replace(GRID_LINE, "elections.cpp_start_age_years.a: [50]")
    assert body != EXAMPLE
    error = refusal(body)
    assert error.status == 422
    assert "CPP start age 50" in error.detail


def test_too_many_candidates_is_a_422() -> None:
    body = EXAMPLE.replace(
        GRID_LINE,
        "elections.cpp_start_age_years.a: [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]\n"
        "  elections.oas_start_age_years.a: [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]\n"
        "  elections.rrif_conversion.age_years: [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]",
    )
    assert body != EXAMPLE
    error = refusal(body, paths=None, deterministic=True)
    assert error.status == 422
    assert "1,331 candidate policies, and a run may have at most 1,000" in error.detail


def test_a_scenario_error_from_prepare_run_is_a_422(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report_run, "prepare_run", raising(ScenarioError("a refusal")))
    error = refusal(EXAMPLE)
    assert (error.status, error.detail) == (422, "a refusal")


def test_a_parameter_error_at_prepare_run_is_a_400() -> None:
    error = refusal(EXAMPLE.replace("start_year: 2026", "start_year: 2031"))
    assert error.status == 400
    assert error.detail.startswith("No parameters for tax year 2031")


def test_a_parameter_error_from_evaluate_is_a_400(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report_run, "evaluate", raising(MissingParameterError("missing")))
    error = refusal(EXAMPLE)
    assert (error.status, error.detail) == (400, "missing")


def test_a_routed_parameter_error_from_prepare_run_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(report_run, "prepare_run", raising(RoutedParameterError("x")))
    with pytest.raises(RoutedParameterError):
        opened()


def test_a_routed_parameter_error_from_evaluate_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(report_run, "evaluate", raising(RoutedParameterError("x")))
    with pytest.raises(RoutedParameterError):
        simulate_document(
            EXAMPLE,
            "yaml",
            policy=None,
            paths=2,
            deterministic=False,
            nominal=False,
            params_root=DEFAULT_PARAMS_ROOT,
        )


def test_an_unknown_policy_is_a_422() -> None:
    error = refusal(EXAMPLE, policy="nope")
    assert error.status == 422
    assert error.detail.startswith("unknown policy 'nope'")


def test_the_deterministic_and_paths_text_is_the_one_the_api_uses() -> None:
    assert PATHS_WITH_DETERMINISTIC == (
        "paths and deterministic cannot both be given; "
        "the deterministic run is always exactly one path."
    )
    response = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000)).post(
        "/api/simulate?paths=2&deterministic=true", content=EXAMPLE.encode(), headers=YAML
    )
    assert response.json()["detail"] == PATHS_WITH_DETERMINISTIC


@pytest.mark.parametrize(
    ("query", "options"),
    [
        ("deterministic=true", {"paths": None, "deterministic": True}),
        ("paths=5", {"paths": 5, "deterministic": False}),
    ],
    ids=["deterministic", "five-paths"],
)
def test_the_document_equals_the_api_response(query: str, options: dict[str, object]) -> None:
    document = simulate_document(
        EXAMPLE, "yaml", policy=None, nominal=False, params_root=DEFAULT_PARAMS_ROOT, **options
    )  # type: ignore[arg-type]
    response = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000)).post(
        f"/api/simulate?{query}", content=EXAMPLE.encode(), headers=YAML
    )
    assert response.status_code == 200
    assert response.json() == document
    assert list(response.json()) == list(document)

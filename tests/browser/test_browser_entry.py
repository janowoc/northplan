# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The browser entry point, run natively: one named test per branch of ``simulate``."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from engine.core.indexation import RoutedParameterError
from engine.params.loader import DEFAULT_PARAMS_ROOT
from report.run import PATHS_WITH_DETERMINISTIC, simulate_document

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = (REPO_ROOT / "scenarios" / "example.yaml").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def entry() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "northplan_browser_under_test", REPO_ROOT / "browser" / "northplan_browser.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parsed(result: str) -> dict:
    assert isinstance(result, str)
    value = json.loads(result)
    assert isinstance(value, dict)
    return value


def error_result(entry: ModuleType, options: str) -> dict:
    result = parsed(entry.simulate(EXAMPLE, options))
    assert set(result) == {"status", "detail"}
    return result


BAD_OPTIONS = [
    ("unknown-key", '{"nope": 1}', "ValueError: unknown option 'nope'"),
    ("paths-string", '{"paths": "5"}', "ValueError: paths must be"),
    ("paths-float", '{"paths": 5.0}', "ValueError: paths must be"),
    ("paths-zero", '{"paths": 0}', "ValueError: paths must be"),
    ("paths-negative", '{"paths": -3}', "ValueError: paths must be"),
    ("paths-bool", '{"paths": true}', "ValueError: paths must be"),
    ("deterministic-int", '{"deterministic": 1}', "ValueError: deterministic must be"),
    ("deterministic-string", '{"deterministic": "true"}', "ValueError: deterministic must be"),
    ("deterministic-null", '{"deterministic": null}', "ValueError: deterministic must be"),
    ("nominal-int", '{"nominal": 1}', "ValueError: nominal must be"),
    ("nominal-null", '{"nominal": null}', "ValueError: nominal must be"),
    ("policy-int", '{"policy": 5}', "ValueError: policy must be"),
    ("policy-bool", '{"policy": true}', "ValueError: policy must be"),
    ("options-list", "[]", "ValueError: options must be a JSON object, not list"),
    ("options-string", '"paths"', "ValueError: options must be a JSON object, not str"),
    ("options-null", "null", "ValueError: options must be a JSON object, not NoneType"),
    ("options-number", "5", "ValueError: options must be a JSON object, not int"),
    ("options-not-json", "not json", "JSONDecodeError: Expecting value"),
]


@pytest.mark.parametrize(
    ("options", "message"),
    [pytest.param(options, message, id=name) for name, options, message in BAD_OPTIONS],
)
def test_a_bad_option_is_refused_by_the_entry_point_itself(
    entry: ModuleType, monkeypatch: pytest.MonkeyPatch, options: str, message: str
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        pytest.fail("the pipeline ran")

    monkeypatch.setattr(entry, "simulate_document", fail)
    result = error_result(entry, options)
    assert result["status"] == "error"
    assert result["detail"].startswith(message), result["detail"]


def test_paths_with_deterministic_is_a_422_before_anything_runs(
    entry: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        pytest.fail("the pipeline ran")

    monkeypatch.setattr(entry, "simulate_document", fail)
    result = error_result(entry, '{"paths": 5, "deterministic": true}')
    assert result == {"status": 422, "detail": PATHS_WITH_DETERMINISTIC}


def test_a_run_over_the_cap_is_a_422_at_25001(entry: ModuleType) -> None:
    assert entry.PATH_CAP == 25_000
    result = error_result(entry, '{"paths": 25001}')
    assert result["status"] == 422
    assert result["detail"].startswith("the run asks for 25,001 paths, and a run here may have at")
    assert "at most 25,000." in result["detail"]


def test_a_refused_run_is_its_status_and_detail(entry: ModuleType) -> None:
    result = error_result(entry, '{"policy": "nope", "paths": 2}')
    assert result["status"] == 422
    assert result["detail"].startswith("unknown policy 'nope'")


def test_a_parameter_error_is_a_400(entry: ModuleType) -> None:
    text = EXAMPLE.replace("start_year: 2026", "start_year: 2031")
    result = parsed(entry.simulate(text, '{"paths": 2}'))
    assert result["status"] == 400
    assert result["detail"].startswith("No parameters for tax year 2031")


def test_a_memory_error_is_reported_as_memory(
    entry: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def exhaust(*_args: object, **_kwargs: object) -> None:
        raise MemoryError

    monkeypatch.setattr(entry, "simulate_document", exhaust)
    assert error_result(entry, "{}") == {"status": "memory", "detail": entry.OUT_OF_MEMORY}
    assert entry.OUT_OF_MEMORY == (
        "the run ran out of the browser's memory: WebAssembly gives a page at most 4 GB. "
        "Lower n_paths, or use the local install."
    )


def test_a_routed_parameter_error_is_reported_as_an_error(
    entry: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def route(*_args: object, **_kwargs: object) -> None:
        raise RoutedParameterError("an engine bug")

    monkeypatch.setattr(entry, "simulate_document", route)
    assert error_result(entry, "{}") == {
        "status": "error",
        "detail": "RoutedParameterError: an engine bug",
    }


def test_any_other_exception_is_reported_as_an_error(
    entry: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def break_(*_args: object, **_kwargs: object) -> None:
        raise KeyError("k")

    monkeypatch.setattr(entry, "simulate_document", break_)
    assert error_result(entry, "{}") == {"status": "error", "detail": "KeyError: 'k'"}


def test_the_pipeline_is_called_with_the_options_the_cap_and_yaml(
    entry: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict = {}

    def record(*args: object, **kwargs: object) -> dict:
        seen["args"], seen["kwargs"] = args, kwargs
        return {}

    monkeypatch.setattr(entry, "simulate_document", record)
    entry.simulate("text", '{"policy": "p", "paths": 7, "nominal": true}')
    assert seen["args"] == ("text", "yaml")
    assert seen["kwargs"] == {
        "policy": "p",
        "paths": 7,
        "deterministic": False,
        "nominal": True,
        "params_root": DEFAULT_PARAMS_ROOT,
        "path_cap": 25_000,
    }


@pytest.mark.parametrize(
    ("options", "kwargs"),
    [
        ("{}", {"paths": 3}),
        ('{"paths": 3, "nominal": true}', {"paths": 3, "nominal": True}),
        ('{"deterministic": true}', {"paths": None, "deterministic": True}),
    ],
    ids=["no-options", "nominal", "deterministic"],
)
def test_a_success_is_the_documents_the_pipeline_returns(
    entry: ModuleType, options: str, kwargs: dict
) -> None:
    text = EXAMPLE.replace("n_paths: 10000", "n_paths: 3")
    expected = simulate_document(
        text,
        "yaml",
        **{
            "policy": None,
            "paths": None,
            "deterministic": False,
            "nominal": False,
            "params_root": DEFAULT_PARAMS_ROOT,
            **kwargs,
        },
    )
    result = parsed(entry.simulate(text, options))
    assert set(result) == {"document"}
    assert result["document"] == expected
    assert list(result["document"]) == list(expected)


def test_an_empty_options_object_runs_with_the_scenarios_own_options(entry: ModuleType) -> None:
    text = EXAMPLE.replace("n_paths: 10000", "n_paths: 3")
    assert text != EXAMPLE
    assert parsed(entry.simulate(text, "{}"))["document"]["meta"]["paths"] == 3

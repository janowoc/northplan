# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The HTTP API against the command line, and its refusals.

The main oracle is differential: the same scenario bytes go through ``cli.main.main``
to a JSON file and through ``POST /api/...`` to a response, and the two documents must
be equal, key order included, once the command line's scenario label and option
wording are restated as the API's. The rest pins each status code the API promises and the
response models against the tables they mirror.
"""

from __future__ import annotations

import asyncio
import base64
import errno
import importlib.metadata
import ipaddress
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

import api.main
from api.main import app
from api.schemas import (
    EvaluationRow,
    FinalRow,
    OptimizeMeta,
    SimulationMeta,
    YearRow,
)
from cli.main import main
from engine.core.indexation import RoutedParameterError
from engine.mc.prepare import prepare_run
from engine.optimize.objective import median_estate_after_tax
from engine.optimize.search import search
from engine.params.loader import MissingParameterError, parse_yaml
from engine.scenario.lifespan import LifespanNotRepresentableError
from engine.scenario.load import load_scenario
from engine.scenario.schema import Scenario
from report import editor
from report import run as report_run
from report.editor_calls import answer

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "scenarios" / "example.yaml"
COUPLE = REPO_ROOT / "scenarios" / "late_life_couple.yaml"

YAML = {"Content-Type": "application/yaml"}
JSON = {"Content-Type": "application/json"}
PATHS = 20
GRID_LINE = "elections.cpp_start_age_years.a: [60, 65, 70]"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000))


def example_text() -> str:
    return EXAMPLE.read_text(encoding="utf-8")


def cli_document(
    tmp_path: Path,
    scenario: Path,
    command: str,
    flags: list[str],
    scenario_name: str,
    *,
    api_wording: bool = True,
) -> dict:
    out = tmp_path / "x.json"
    assert main([command, str(scenario), *flags, "--out", str(out)]) == 0
    document = json.loads(out.read_text(encoding="utf-8"))
    if command == "optimize":
        document["best"]["meta"]["scenario"] = scenario_name
        if api_wording:
            to_api_wording(document["meta"])
    document["meta"]["scenario"] = scenario_name
    return document


def to_api_wording(meta: dict) -> None:
    """Restate the optimize meta's command-line wording as the API's."""
    meta["dollars"] = meta["dollars"].replace(
        "; --nominal does not apply", "; nominal=true does not apply"
    )
    for key in ("risk_aversion", "estate_utility_shift"):
        if meta[key].endswith(" (flag)"):
            meta[key] = meta[key][: -len(" (flag)")] + " (query)"


def api_document(
    client: TestClient, path: str, params: dict[str, str], body: bytes, headers: dict[str, str]
) -> dict:
    response = client.post(path, params=params, content=body, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


# =============================================================================
# Differential against the command line
# =============================================================================


class TestAgainstTheCommandLine:
    def check(
        self,
        client: TestClient,
        tmp_path: Path,
        *,
        command: str,
        flags: list[str],
        params: dict[str, str],
        scenario: Path = EXAMPLE,
        as_json: bool = False,
    ) -> dict:
        text = scenario.read_text(encoding="utf-8")
        name = parse_yaml(text)["name"]
        cli_doc = cli_document(tmp_path, scenario, command, flags, name)
        if as_json:
            body, headers = json.dumps(parse_yaml(text)).encode(), JSON
        else:
            body, headers = text.encode(), YAML
        api_doc = api_document(client, f"/api/{command}", params, body, headers)
        assert json.dumps(api_doc) == json.dumps(cli_doc)
        return api_doc

    def test_simulate_yaml(self, client: TestClient, tmp_path: Path) -> None:
        document = self.check(
            client,
            tmp_path,
            command="simulate",
            flags=["--paths", str(PATHS)],
            params={"paths": str(PATHS)},
        )
        cells = [v for row in document["years"] for v in row.values()]
        assert None in cells, "the example's rows have a null cell; the null path is untested"

    def test_simulate_json(self, client: TestClient, tmp_path: Path) -> None:
        self.check(
            client,
            tmp_path,
            command="simulate",
            flags=["--paths", str(PATHS)],
            params={"paths": str(PATHS)},
            as_json=True,
        )

    def test_simulate_nominal(self, client: TestClient, tmp_path: Path) -> None:
        nominal = self.check(
            client,
            tmp_path,
            command="simulate",
            flags=["--paths", str(PATHS), "--nominal"],
            params={"paths": str(PATHS), "nominal": "true"},
        )
        real = api_document(
            client,
            "/api/simulate",
            {"paths": str(PATHS)},
            example_text().encode(),
            YAML,
        )
        assert nominal != real
        assert nominal["years"] != real["years"]

    def test_simulate_deterministic(self, client: TestClient, tmp_path: Path) -> None:
        self.check(
            client,
            tmp_path,
            command="simulate",
            flags=["--deterministic"],
            params={"deterministic": "true"},
        )

    def test_simulate_a_named_policy(self, client: TestClient, tmp_path: Path) -> None:
        name = "taxable-first[elections.cpp_start_age_years.a=70]"
        document = self.check(
            client,
            tmp_path,
            command="simulate",
            flags=["--paths", str(PATHS), "--policy", name],
            params={"paths": str(PATHS), "policy": name},
        )
        assert document["meta"]["policy"] == name

    def test_simulate_the_couple(self, client: TestClient, tmp_path: Path) -> None:
        document = self.check(
            client,
            tmp_path,
            command="simulate",
            flags=["--paths", str(PATHS)],
            params={"paths": str(PATHS)},
            scenario=COUPLE,
        )
        deaths = [k for k in document["years"][0] if k.startswith("death_probability_")]
        assert len(deaths) == 2

    def test_optimize_median_estate(self, client: TestClient, tmp_path: Path) -> None:
        document = self.check(
            client,
            tmp_path,
            command="optimize",
            flags=["--paths", str(PATHS), "--objective", "median_estate_after_tax"],
            params={"paths": str(PATHS), "objective": "median_estate_after_tax"},
        )
        assert "elections.cpp_start_age_years.a" in document["evaluation"][0]

    def test_optimize_certainty_equivalent_with_the_scenario_preferences(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        self.check(
            client,
            tmp_path,
            command="optimize",
            flags=["--paths", str(PATHS), "--objective", "certainty_equivalent_estate"],
            params={"paths": str(PATHS), "objective": "certainty_equivalent_estate"},
        )

    def test_optimize_certainty_equivalent_with_overrides_in_nominal_dollars(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        flags = [
            "--paths",
            str(PATHS),
            "--objective",
            "certainty_equivalent_estate",
            "--risk-aversion",
            "3",
            "--estate-utility-shift",
            "1000",
            "--nominal",
        ]
        self.check(
            client,
            tmp_path,
            command="optimize",
            flags=flags,
            params={
                "paths": str(PATHS),
                "objective": "certainty_equivalent_estate",
                "risk_aversion": "3",
                "estate_utility_shift": "1000",
                "nominal": "true",
            },
        )
        raw = cli_document(tmp_path, EXAMPLE, "optimize", flags, "ignored", api_wording=False)[
            "meta"
        ]
        assert "--nominal" in raw["dollars"]
        assert raw["risk_aversion"].endswith(" (flag)")
        assert raw["estate_utility_shift"].endswith(" (flag)")

    def test_optimize_deterministic(self, client: TestClient, tmp_path: Path) -> None:
        self.check(
            client,
            tmp_path,
            command="optimize",
            flags=["--deterministic", "--objective", "success_probability"],
            params={"deterministic": "true", "objective": "success_probability"},
        )


# =============================================================================
# Response models mirror the tables
# =============================================================================


@pytest.fixture(scope="module")
def simulate_doc(client: TestClient) -> dict:
    return api_document(client, "/api/simulate", {"paths": "5"}, example_text().encode(), YAML)


@pytest.fixture(scope="module")
def optimize_doc(client: TestClient) -> dict:
    return api_document(
        client,
        "/api/optimize",
        {"paths": "5", "objective": "median_estate_after_tax"},
        example_text().encode(),
        YAML,
    )


class TestSchemaDrift:
    def test_year_row_fields_are_the_rows_keys_without_the_death_probabilities(
        self, simulate_doc: dict
    ) -> None:
        keys = list(simulate_doc["years"][0])
        plain = [k for k in keys if not k.startswith("death_probability_")]
        assert len(plain) < len(keys)
        assert keys[: len(plain)] == plain
        assert list(YearRow.model_fields) == plain

    def test_final_row_fields(self, simulate_doc: dict) -> None:
        assert list(FinalRow.model_fields) == list(simulate_doc["final"])

    def test_simulation_meta_fields(self, simulate_doc: dict) -> None:
        assert list(SimulationMeta.model_fields) == list(simulate_doc["meta"])

    def test_optimize_meta_fields(self, optimize_doc: dict) -> None:
        assert list(OptimizeMeta.model_fields) == list(optimize_doc["meta"])

    def test_evaluation_row_fields_then_free_parameters(self, optimize_doc: dict) -> None:
        keys = list(optimize_doc["evaluation"][0])
        assert list(EvaluationRow.model_fields) == keys[:6]
        found = search(prepare_run(load_scenario(EXAMPLE), n_paths=5), median_estate_after_tax)
        free: list[str] = []
        for report in found.evaluated:
            free.extend(key for key in report.parameters if key not in free)
        assert "elections.cpp_start_age_years.a" in free
        assert keys[6:] == free

    def test_openapi_request_bodies_and_responses(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        for path, model in (
            ("/api/simulate", "SimulationResponse"),
            ("/api/optimize", "OptimizeResponse"),
        ):
            post = paths[path]["post"]
            body = post["requestBody"]["content"]
            assert set(body) == {"application/json", "application/yaml"}
            for content in body.values():
                assert content["schema"] == {"$ref": "#/components/schemas/Scenario"}
            schema = post["responses"]["200"]["content"]["application/json"]["schema"]
            assert schema["$ref"].endswith(f"/{model}")

    def test_openapi_version_is_the_installed_package_version(self, client: TestClient) -> None:
        version = client.get("/openapi.json").json()["info"]["version"]
        assert version == importlib.metadata.version("northplan")

    def test_every_openapi_reference_resolves(self, client: TestClient) -> None:
        document = client.get("/openapi.json").json()
        references: list[str] = []

        def collect(node: object) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "$ref" and isinstance(value, str):
                        references.append(value)
                    else:
                        collect(value)
            elif isinstance(node, list):
                for item in node:
                    collect(item)

        collect(document)
        assert references
        for reference in references:
            assert reference.startswith("#/")
            target = document
            for part in reference[2:].split("/"):
                target = target[part.replace("~1", "/").replace("~0", "~")]

    def test_the_scenario_component_is_the_models_schema(self, client: TestClient) -> None:
        expected = Scenario.model_json_schema(ref_template="#/components/schemas/{model}")
        expected.pop("$defs")
        schemas = client.get("/openapi.json").json()["components"]["schemas"]
        assert schemas["Scenario"] == expected

    def test_the_openapi_override_refuses_a_duplicate_component(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        schema = {"$defs": {"YearRow": {"type": "object"}}, "type": "object"}
        monkeypatch.setattr(Scenario, "model_json_schema", lambda *_a, **_k: schema.copy())
        monkeypatch.setattr(api.main.app, "openapi_schema", None)
        with pytest.raises(RuntimeError, match=r"OpenAPI component 'YearRow' is defined twice\."):
            api.main.app.openapi()

    def test_the_openapi_document_is_built_once(self) -> None:
        assert api.main.app.openapi() is api.main.app.openapi()


# =============================================================================
# Content types
# =============================================================================


UNSUPPORTED = (
    "unsupported Content-Type {header!r}; send application/json, or YAML as "
    "application/yaml, application/x-yaml or text/yaml, in UTF-8."
)


class TestContentType:
    def post(self, client: TestClient, text: str, headers: dict[str, str]):
        return client.post("/api/simulate?paths=2", content=text.encode(), headers=headers)

    @pytest.mark.parametrize(
        "content_type",
        [
            "application/yaml",
            "application/x-yaml",
            "text/yaml",
            "application/yaml; charset=UTF-8",
            'application/yaml; charset="utf-8"',
        ],
    )
    def test_yaml_types_are_accepted(self, client: TestClient, content_type: str) -> None:
        assert self.post(client, example_text(), {"Content-Type": content_type}).status_code == 200

    def test_json_is_accepted(self, client: TestClient) -> None:
        body = json.dumps(parse_yaml(example_text()))
        assert self.post(client, body, JSON).status_code == 200

    def test_text_plain_is_unsupported(self, client: TestClient) -> None:
        header = "text/plain"
        response = self.post(client, example_text(), {"Content-Type": header})
        assert response.status_code == 415
        assert response.json()["detail"] == UNSUPPORTED.format(header=header)

    def test_a_missing_content_type_is_unsupported(self, client: TestClient) -> None:
        response = self.post(client, example_text(), {})
        assert response.status_code == 415
        assert response.json()["detail"] == UNSUPPORTED.format(header=None)

    def test_another_charset_is_unsupported(self, client: TestClient) -> None:
        header = "application/yaml; charset=latin-1"
        response = self.post(client, example_text(), {"Content-Type": header})
        assert response.status_code == 415
        assert response.json()["detail"] == UNSUPPORTED.format(header=header)


# =============================================================================
# 422: what the command line refuses with exit 2
# =============================================================================


class TestUnprocessable:
    def refused(
        self,
        client: TestClient,
        body: bytes | str,
        *,
        path: str = "/api/simulate",
        params: dict[str, str] | None = None,
        headers: dict[str, str] = YAML,
    ) -> str:
        content = body.encode() if isinstance(body, str) else body
        response = client.post(
            path, params={"paths": "2", **(params or {})}, content=content, headers=headers
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert isinstance(detail, str)
        return detail

    def test_a_body_that_is_not_utf8(self, client: TestClient) -> None:
        detail = self.refused(client, b"\xff\xfe\x00")
        assert detail.startswith("request body is not valid UTF-8: ")
        assert detail.endswith(". Send it as UTF-8.")

    def test_a_grid_over_an_integer_too_large_for_a_float(self, client: TestClient) -> None:
        old = "cpp_start_age_years: {a: 65}"
        assert example_text().count(old) == 1
        body = example_text().replace(old, f"cpp_start_age_years: {{a: {10**400}}}")
        detail = self.refused(client, body)
        assert (
            "grid['elections.cpp_start_age_years.a']: elections.cpp_start_age_years.a "
            "is an integer too large to convert to a float." in detail
        )

    def test_malformed_yaml(self, client: TestClient) -> None:
        assert "request body is not valid YAML" in self.refused(client, "a: [1, 2\n")

    def test_malformed_json(self, client: TestClient) -> None:
        detail = self.refused(client, '{"a": ', headers=JSON)
        assert "request body is not valid JSON" in detail

    def test_a_repeated_key_in_yaml(self, client: TestClient) -> None:
        detail = self.refused(client, "seed: 1\nseed: 2\n")
        assert detail.startswith("request body: ")
        assert "key 'seed' appears more than once" in detail

    def test_a_repeated_key_in_json(self, client: TestClient) -> None:
        detail = self.refused(client, '{"seed": 1, "seed": 2}', headers=JSON)
        assert detail.startswith("request body: key 'seed'")

    def test_a_top_level_list(self, client: TestClient) -> None:
        assert "this parses to list." in self.refused(client, "- 1\n- 2\n")

    def test_an_invalid_scenario(self, client: TestClient) -> None:
        body = example_text().replace("n_paths: 10000", "n_paths: 0")
        assert body != example_text()
        assert self.refused(client, body).startswith("request body is not a valid scenario:")

    def test_a_start_age_the_checks_refuse(self, client: TestClient) -> None:
        body = example_text().replace(GRID_LINE, "elections.cpp_start_age_years.a: [50]")
        assert body != example_text()
        detail = self.refused(client, body)
        assert "CPP start age 50" in detail

    def test_a_lifespan_the_checks_refuse(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*_args: object, **_kwargs: object) -> None:
            raise LifespanNotRepresentableError("a lifespan this run cannot represent")

        monkeypatch.setattr(report_run, "prepare_run", refuse)
        assert self.refused(client, example_text()) == "a lifespan this run cannot represent"

    def test_a_grid_value_the_schema_refuses(self, client: TestClient) -> None:
        body = example_text().replace(GRID_LINE, "elections.rrif_conversion.fraction: [-1.0]")
        assert body != example_text()
        detail = self.refused(client, body)
        assert detail.startswith("1 validation error for Scenario")
        assert "rrif_conversion.fraction" in detail

    def test_a_json_integer_over_the_digit_limit(self, client: TestClient) -> None:
        detail = self.refused(client, "1" * 5000, headers=JSON)
        assert "request body is not valid JSON" in detail

    def test_a_yaml_alias_bomb_is_refused_quickly(self, client: TestClient) -> None:
        lines = ["l0: &l0 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"]
        for level in range(1, 7):
            lines.append(f"l{level}: &l{level} [{', '.join([f'*l{level - 1}'] * 10)}]")
        lines.append("name: *l6")
        started = time.perf_counter()
        detail = self.refused(client, "\n".join(lines) + "\n")
        assert time.perf_counter() - started < 1.0
        assert "more than 100,000 values" in detail
        assert detail.startswith("request body: ")

    def test_an_unknown_policy(self, client: TestClient) -> None:
        detail = self.refused(client, example_text(), params={"policy": "nope"})
        assert detail.startswith("unknown policy 'nope'")

    def test_a_grid_base_name(self, client: TestClient) -> None:
        detail = self.refused(client, example_text(), params={"policy": "taxable-first"})
        assert "was expanded by the grid" in detail

    def test_a_missing_preference(self, client: TestClient) -> None:
        body = example_text().replace("risk_aversion: 2.0", "")
        assert body != example_text()
        detail = self.refused(
            client,
            body,
            path="/api/optimize",
            params={"objective": "certainty_equivalent_estate"},
        )
        assert detail.endswith(
            " Either can be given for this run with the risk_aversion or "
            "estate_utility_shift query parameter."
        )

    def test_an_unknown_query_parameter(self, client: TestClient) -> None:
        response = client.post(
            "/api/simulate?nominl=true", content=example_text().encode(), headers=YAML
        )
        assert response.status_code == 422
        assert response.json()["detail"] == (
            "unknown query parameter 'nominl'; /api/simulate accepts "
            "['deterministic', 'nominal', 'paths', 'policy']."
        )

    def test_paths_with_deterministic(self, client: TestClient) -> None:
        detail = self.refused(client, example_text(), params={"deterministic": "true"})
        assert detail == (
            "paths and deterministic cannot both be given; "
            "the deterministic run is always exactly one path."
        )

    def test_deterministic_false_with_paths_is_accepted(self, client: TestClient) -> None:
        response = client.post(
            "/api/simulate",
            params={"deterministic": "false", "paths": "3"},
            content=example_text().encode(),
            headers=YAML,
        )
        assert response.status_code == 200


class TestCheckOrder:
    """Unknown parameter, then paths with deterministic, then Content-Type, then body."""

    UNKNOWN = (
        "unknown query parameter {name!r}; {path} accepts "
        "['deterministic', 'nominal', 'paths', 'policy']."
    )
    PLAIN: ClassVar[dict[str, str]] = {"Content-Type": "text/plain"}

    def post(
        self, client: TestClient, query: str, headers: dict[str, str], body: bytes | None = None
    ):
        content = example_text().encode() if body is None else body
        return client.post(f"/api/simulate?{query}", content=content, headers=headers)

    def test_the_accepted_names_are_the_endpoints_own_parameters(self) -> None:
        simulate_names = api.main._SIMULATE_QUERY
        optimize_names = api.main._OPTIMIZE_QUERY
        assert simulate_names == {"nominal", "policy", "paths", "deterministic"}
        assert optimize_names == {
            "objective",
            "nominal",
            "paths",
            "deterministic",
            "risk_aversion",
            "estate_utility_shift",
        }

    def test_an_unknown_parameter_wins_over_the_content_type(self, client: TestClient) -> None:
        response = self.post(client, "nominl=1", self.PLAIN)
        assert response.status_code == 422
        assert response.json()["detail"] == self.UNKNOWN.format(name="nominl", path="/api/simulate")

    def test_an_unknown_parameter_wins_over_paths_with_deterministic(
        self, client: TestClient
    ) -> None:
        response = self.post(client, "zz=1&deterministic=true&paths=2", YAML)
        assert response.status_code == 422
        assert response.json()["detail"] == self.UNKNOWN.format(name="zz", path="/api/simulate")

    def test_paths_with_deterministic_wins_over_the_content_type(self, client: TestClient) -> None:
        response = self.post(client, "deterministic=true&paths=2", self.PLAIN)
        assert response.status_code == 422
        assert response.json()["detail"].startswith("paths and deterministic cannot both")

    def test_the_content_type_wins_over_the_body(self, client: TestClient) -> None:
        response = self.post(client, "paths=2", self.PLAIN, body=b"\xff\xfe\x00")
        assert response.status_code == 415

    def test_the_first_of_several_unknown_parameters_is_named(self, client: TestClient) -> None:
        response = self.post(client, "b=1&a=1", YAML)
        assert response.status_code == 422
        assert response.json()["detail"] == self.UNKNOWN.format(name="a", path="/api/simulate")

    def test_the_optimize_message_lists_its_own_parameters(self, client: TestClient) -> None:
        response = client.post(
            "/api/optimize?objective=success_probability&nominl=1",
            content=example_text().encode(),
            headers=YAML,
        )
        assert response.status_code == 422
        assert response.json()["detail"] == (
            "unknown query parameter 'nominl'; /api/optimize accepts "
            "['deterministic', 'estate_utility_shift', 'nominal', 'objective', 'paths', "
            "'risk_aversion']."
        )


class TestQueryValidation:
    @pytest.mark.parametrize(
        ("path", "query"),
        [
            ("/api/simulate", "paths=0"),
            ("/api/simulate", "paths=abc"),
            ("/api/optimize", "objective=nope"),
            ("/api/optimize", ""),
            ("/api/optimize", "objective=success_probability&risk_aversion=-1"),
            ("/api/optimize", "objective=success_probability&risk_aversion=nan"),
            ("/api/optimize", "objective=success_probability&estate_utility_shift=0"),
            ("/api/optimize", "objective=success_probability&estate_utility_shift=inf"),
        ],
    )
    def test_a_bad_option_is_refused_with_a_list_of_problems(
        self, client: TestClient, path: str, query: str
    ) -> None:
        response = client.post(f"{path}?{query}", content=example_text().encode(), headers=YAML)
        assert response.status_code == 422
        assert isinstance(response.json()["detail"], list)


# =============================================================================
# 400, 500, /api/example, /health
# =============================================================================


def test_a_year_without_parameters_is_a_400(client: TestClient) -> None:
    body = example_text().replace("start_year: 2026", "start_year: 2031")
    assert body != example_text()
    response = client.post("/api/simulate?paths=2", content=body.encode(), headers=YAML)
    assert response.status_code == 400
    assert response.json()["detail"].startswith("No parameters for tax year 2031")


class TestRoutedParameterError:
    @staticmethod
    def raise_routed(*_args: object, **_kwargs: object) -> None:
        raise RoutedParameterError("x")

    def test_it_is_a_500(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(report_run, "prepare_run", self.raise_routed)
        quiet = TestClient(
            app,
            base_url="http://localhost",
            client=("127.0.0.1", 50000),
            raise_server_exceptions=False,
        )
        response = quiet.post(
            "/api/simulate?paths=2", content=example_text().encode(), headers=YAML
        )
        assert response.status_code == 500

    def test_it_propagates_to_the_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(report_run, "prepare_run", self.raise_routed)
        with pytest.raises(RoutedParameterError):
            TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000)).post(
                "/api/simulate?paths=2", content=example_text().encode(), headers=YAML
            )


class TestParameterErrorsAnywhereInTheRun:
    @staticmethod
    def raise_missing(*_args: object, **_kwargs: object) -> None:
        raise MissingParameterError("missing for this test")

    def post(self, path: str, quiet: bool = False):
        client = TestClient(
            app,
            base_url="http://localhost",
            client=("127.0.0.1", 50000),
            raise_server_exceptions=not quiet,
        )
        return client.post(path, content=example_text().encode(), headers=YAML)

    def test_simulate_evaluate_is_a_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(report_run, "evaluate", self.raise_missing)
        response = self.post("/api/simulate?paths=2")
        assert response.status_code == 400
        assert response.json()["detail"] == "missing for this test"

    def test_optimize_search_is_a_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(api.main, "search", self.raise_missing)
        response = self.post("/api/optimize?paths=2&objective=median_estate_after_tax")
        assert response.status_code == 400
        assert response.json()["detail"] == "missing for this test"

    def test_optimize_re_evaluation_is_a_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(api.main, "evaluate", self.raise_missing)
        response = self.post("/api/optimize?paths=2&objective=median_estate_after_tax")
        assert response.status_code == 400
        assert response.json()["detail"] == "missing for this test"

    def test_simulate_routed_parameter_error_is_a_500(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_routed(*_args: object, **_kwargs: object) -> None:
            raise RoutedParameterError("x")

        monkeypatch.setattr(report_run, "evaluate", raise_routed)
        assert self.post("/api/simulate?paths=2", quiet=True).status_code == 500


class TestWording:
    def test_the_apis_own_wording(self, client: TestClient) -> None:
        document = api_document(
            client,
            "/api/optimize",
            {
                "objective": "certainty_equivalent_estate",
                "risk_aversion": "3",
                "estate_utility_shift": "1000",
                "nominal": "true",
                "paths": "5",
            },
            example_text().encode(),
            YAML,
        )
        meta = document["meta"]
        assert meta["dollars"] == (
            "real, January 2026 dollars; nominal=true does not apply to this table"
        )
        assert meta["risk_aversion"] == "3.0 (query)"
        assert meta["estate_utility_shift"] == "1000.0 (query)"

    def test_without_overrides_or_nominal(self, client: TestClient) -> None:
        document = api_document(
            client,
            "/api/optimize",
            {"objective": "certainty_equivalent_estate", "paths": "5"},
            example_text().encode(),
            YAML,
        )
        assert document["meta"]["risk_aversion"] == "2.0 (scenario)"
        assert document["meta"]["dollars"] == "real, January 2026 dollars"


class TestExample:
    def test_it_serves_the_file_as_it_is(self, client: TestClient) -> None:
        response = client.get("/api/example")
        assert response.status_code == 200
        assert response.content == EXAMPLE.read_bytes()
        assert response.headers["content-type"] == "application/yaml; charset=utf-8"

    def test_it_can_be_posted_back(self, client: TestClient) -> None:
        served = client.get("/api/example")
        response = client.post(
            "/api/simulate?paths=2",
            content=served.content,
            headers={"Content-Type": served.headers["content-type"]},
        )
        assert response.status_code == 200

    def test_a_missing_file_is_a_404(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api.main, "EXAMPLE_PATH", tmp_path / "absent.yaml")
        response = client.get("/api/example")
        assert response.status_code == 404
        assert isinstance(response.json()["detail"], str)

    def test_a_file_of_mode_zero_is_a_503(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        lock: Callable[[Path, int], None],
    ) -> None:
        path = tmp_path / "example.yaml"
        path.write_text("x: 1\n", encoding="utf-8")
        lock(path, 0o000)
        monkeypatch.setattr(api.main, "EXAMPLE_PATH", path)
        response = client.get("/api/example")
        reason = os.strerror(errno.EACCES)
        assert response.status_code == 503
        assert response.json()["detail"] == f"cannot read the example scenario at {path}: {reason}."

    def test_a_file_in_a_directory_of_mode_zero_is_a_503(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        lock: Callable[[Path, int], None],
    ) -> None:
        directory = tmp_path / "d"
        directory.mkdir()
        path = directory / "example.yaml"
        path.write_text("x: 1\n", encoding="utf-8")
        lock(directory, 0o000)
        monkeypatch.setattr(api.main, "EXAMPLE_PATH", path)
        response = client.get("/api/example")
        reason = os.strerror(errno.EACCES)
        assert response.status_code == 503
        assert response.json()["detail"] == f"cannot read the example scenario at {path}: {reason}."

    def test_a_symlink_to_itself_is_a_503(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "loop.yaml"
        path.symlink_to(path)
        monkeypatch.setattr(api.main, "EXAMPLE_PATH", path)
        response = client.get("/api/example")
        reason = os.strerror(errno.ELOOP)
        assert response.status_code == 503
        assert response.json()["detail"] == f"cannot read the example scenario at {path}: {reason}."

    def test_a_directory_is_a_404(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api.main, "EXAMPLE_PATH", tmp_path)
        response = client.get("/api/example")
        assert response.status_code == 404
        assert response.json()["detail"] == f"{tmp_path} is a directory, not a scenario file."

    def test_a_path_through_a_regular_file_is_a_404(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        through = tmp_path / "f"
        through.write_text("x\n", encoding="utf-8")
        path = through / "s.yaml"
        monkeypatch.setattr(api.main, "EXAMPLE_PATH", path)
        response = client.get("/api/example")
        assert response.status_code == 404
        assert response.json()["detail"] == f"no example scenario at {path}."

    def test_an_error_without_a_reason_is_a_503(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_args: object, **_kwargs: object) -> bytes:
            raise OSError("boom")

        path = tmp_path / "example.yaml"
        path.write_text("x: 1\n", encoding="utf-8")
        monkeypatch.setattr(api.main, "EXAMPLE_PATH", path)
        monkeypatch.setattr(Path, "read_bytes", boom)
        response = client.get("/api/example")
        assert response.status_code == 503
        assert response.json()["detail"] == f"cannot read the example scenario at {path}: boom."

    def test_a_fifo_is_a_404_and_is_not_read(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unread(*_args: object, **_kwargs: object) -> bytes:
            pytest.fail("the example was read")

        path = tmp_path / "example.yaml"
        os.mkfifo(path)
        monkeypatch.setattr(api.main, "EXAMPLE_PATH", path)
        monkeypatch.setattr(Path, "read_bytes", unread)
        response = client.get("/api/example")
        assert response.status_code == 404
        assert response.json()["detail"] == f"{path} is not a regular file."

    def test_a_character_device_is_a_404_and_is_not_read(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unread(*_args: object, **_kwargs: object) -> bytes:
            pytest.fail("the example was read")

        path = Path(os.devnull)
        monkeypatch.setattr(api.main, "EXAMPLE_PATH", path)
        monkeypatch.setattr(Path, "read_bytes", unread)
        response = client.get("/api/example")
        assert response.status_code == 404
        assert response.json()["detail"] == f"{path} is not a regular file."


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


# =============================================================================
# The Host check, the body cap, the run lock, and the engine's size limits
# =============================================================================


@pytest.mark.parametrize(
    "host",
    ["localhost", "localhost:8000", "127.0.0.1:8000", "[::1]", "[::1]:8000", "LOCALHOST:8000"],
)
def test_host_check_allows_loopback(client: TestClient, host: str) -> None:
    response = client.get("/health", headers={"Host": host})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "host",
    [
        "testserver",
        "evil.example:8000",
        "192.168.1.5:8000",
        "localhost.evil.example",
        "[::2]:8000",
        "[::1]evil",
        "[::1]:x:y",
        "localhost:abc",
        "localhost:",
    ],
)
@pytest.mark.parametrize(
    ("method", "path"), [("get", "/health"), ("get", "/"), ("post", "/api/simulate")]
)
def test_host_check_refuses_other_hosts(
    client: TestClient, host: str, method: str, path: str
) -> None:
    response = getattr(client, method)(path, headers={"Host": host})
    assert response.status_code == 400
    assert response.json() == {
        "detail": (
            f"host {host!r} is not served: this server answers only localhost, "
            "127.0.0.1 and [::1], because it is for this machine alone."
        )
    }


REPEATED_HOST = (
    "the request carries more than one Host header; this server answers a request with exactly one."
)


def _drive(
    headers: list[tuple[bytes, bytes]],
    path: str = "/health",
    *,
    client: tuple[str, int] | None = ("127.0.0.1", 50000),
    with_client: bool = True,
) -> tuple[int, dict]:
    """Call the ASGI app with exactly these headers; the status and the JSON body."""
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "scheme": "http",
        "http_version": "1.1",
    }
    if with_client:
        scope["client"] = client
    asyncio.run(app(scope, receive, send))
    body = b"".join(m.get("body", b"") for m in sent[1:])
    return sent[0]["status"], json.loads(body)


def test_host_check_refuses_an_empty_host() -> None:
    status, body = _drive([(b"host", b"")])
    assert status == 400
    assert body["detail"].startswith("host '' is not served")


@pytest.mark.parametrize(("first", "second"), [("localhost", "evil"), ("evil", "localhost")])
def test_host_check_refuses_repeated_host_headers(first: str, second: str) -> None:
    status, body = _drive([(b"host", first.encode()), (b"host", second.encode())])
    assert status == 400
    assert body["detail"] == REPEATED_HOST


def test_host_check_refuses_a_repeated_localhost() -> None:
    status, body = _drive([(b"host", b"localhost"), (b"host", b"localhost")])
    assert status == 400
    assert body["detail"] == REPEATED_HOST


def test_host_is_checked_before_the_query(client: TestClient) -> None:
    response = client.post(
        "/api/simulate?nominl=1",
        content=example_text().encode(),
        headers={**YAML, "Host": "evil.example"},
    )
    assert response.status_code == 400
    assert response.json()["detail"].startswith("host 'evil.example' is not served")


def test_host_check_refuses_a_missing_host() -> None:
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/health",
        "raw_path": b"/health",
        "query_string": b"",
        "headers": [],
        "scheme": "http",
        "http_version": "1.1",
        "client": ("127.0.0.1", 50000),
    }
    asyncio.run(app(scope, receive, send))

    assert sent[0]["status"] == 400
    body = b"".join(m.get("body", b"") for m in sent[1:])
    assert json.loads(body)["detail"].startswith("host None is not served")


# =============================================================================
# The token: a client that is not on this machine
# =============================================================================

SYNTHETIC_TOKEN = "synthetic-token-0123"
LAN_PEER = ("192.168.1.5", 40000)
CHALLENGE = (
    "this server needs its token: send it as the password of HTTP Basic authentication, "
    "with any user name. It is the NORTHPLAN_TOKEN the server was started with or, when "
    "that was not set, the one it wrote to its log as it started."
)
NO_TOKEN = (
    "this server was started without a token, so it answers only a client on its own "
    "machine that connects to localhost, 127.0.0.1 or [::1]."
)


def _basic(user: str, password: str) -> dict[str, str]:
    raw = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {raw}"}


def _lan_client(peer: tuple[str, int] = LAN_PEER) -> TestClient:
    return TestClient(app, base_url="http://localhost", client=peer)


@pytest.fixture
def token(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(api.main, "TOKEN", SYNTHETIC_TOKEN)
    return SYNTHETIC_TOKEN


@pytest.mark.parametrize(
    ("method", "path"), [("get", "/health"), ("get", "/"), ("post", "/api/simulate")]
)
@pytest.mark.usefixtures("token")
def test_a_lan_client_without_the_token_is_challenged(method: str, path: str) -> None:
    response = getattr(_lan_client(), method)(path)
    assert response.status_code == 401
    assert response.json() == {"detail": CHALLENGE}
    assert response.headers["www-authenticate"] == 'Basic realm="northplan"'


@pytest.mark.usefixtures("token")
def test_a_lan_client_with_a_wrong_token_is_challenged() -> None:
    response = _lan_client().get("/health", headers=_basic("u", "wrong-token-0123456"))
    assert response.status_code == 401
    assert response.json() == {"detail": CHALLENGE}
    assert response.headers["www-authenticate"] == 'Basic realm="northplan"'


@pytest.mark.parametrize(
    "host", ["localhost", "evil.example:8000", "192.168.1.5:8123", "testserver"]
)
def test_a_lan_client_with_the_token_is_served_under_any_host(token: str, host: str) -> None:
    response = _lan_client().get("/health", headers={"Host": host, **_basic("u", token)})
    assert response.status_code == 200


def test_a_lan_client_with_the_token_is_served_with_no_host(token: str) -> None:
    status, body = _drive(
        [(b"authorization", _basic("u", token)["Authorization"].encode())], client=LAN_PEER
    )
    assert status == 200
    assert body == {"status": "ok"}


def test_a_lan_client_with_the_token_can_run_a_simulation(token: str) -> None:
    response = _lan_client().post(
        "/api/simulate?paths=2",
        content=example_text().encode(),
        headers={**YAML, **_basic("u", token)},
    )
    assert response.status_code == 200


@pytest.mark.parametrize("user", ["", "alice"])
def test_any_user_name_is_accepted(token: str, user: str) -> None:
    response = _lan_client().get("/health", headers=_basic(user, token))
    assert response.status_code == 200


def test_a_token_may_contain_a_colon(monkeypatch: pytest.MonkeyPatch) -> None:
    colon_token = "colon:in:the:token-0"
    monkeypatch.setattr(api.main, "TOKEN", colon_token)
    served = _lan_client().get("/health", headers=_basic("u", colon_token))
    assert served.status_code == 200
    # Without a user name the first colon splits the token itself: user "colon".
    raw = base64.b64encode(colon_token.encode()).decode()
    refused = _lan_client().get("/health", headers={"Authorization": f"Basic {raw}"})
    assert refused.status_code == 401


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


@pytest.mark.parametrize(
    "authorization",
    [
        f"Bearer {_b64('u:' + SYNTHETIC_TOKEN)}",
        "Basic",
        "Basic !!!",
        f"Basic {_b64(SYNTHETIC_TOKEN)}",
        f"Basic {_b64(SYNTHETIC_TOKEN + ':')}",
    ],
)
@pytest.mark.usefixtures("token")
def test_a_malformed_authorization_is_no_token(authorization: str) -> None:
    response = _lan_client().get("/health", headers={"Authorization": authorization})
    assert response.status_code == 401


def test_two_authorization_headers_are_no_token(token: str) -> None:
    valid = _basic("u", token)["Authorization"].encode()
    status, body = _drive([(b"authorization", valid), (b"authorization", valid)], client=LAN_PEER)
    assert status == 401
    assert body == {"detail": CHALLENGE}


@pytest.mark.parametrize("scheme", ["basic", "BASIC"])
def test_the_basic_scheme_is_case_insensitive(token: str, scheme: str) -> None:
    raw = _b64(f"u:{token}")
    response = _lan_client().get("/health", headers={"Authorization": f"{scheme} {raw}"})
    assert response.status_code == 200


@pytest.mark.parametrize("authorized", [False, True])
def test_a_lan_client_is_forbidden_when_the_server_has_no_token(
    monkeypatch: pytest.MonkeyPatch, authorized: bool
) -> None:
    monkeypatch.setattr(api.main, "TOKEN", None)
    headers = _basic("u", SYNTHETIC_TOKEN) if authorized else {}
    response = _lan_client().get("/health", headers=headers)
    assert response.status_code == 403
    assert response.json() == {"detail": NO_TOKEN}
    assert "www-authenticate" not in response.headers


@pytest.mark.parametrize("authorized", [False, True])
@pytest.mark.usefixtures("token")
def test_a_rebinding_request_is_refused_without_a_challenge(authorized: bool) -> None:
    headers = {"Host": "evil.example"}
    if authorized:
        headers.update(_basic("u", "wrong-token-0123456"))
    response = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000)).get(
        "/health", headers=headers
    )
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "host 'evil.example' is not served: this server answers only localhost, "
        "127.0.0.1 and [::1], because it is for this machine alone."
    )
    assert "www-authenticate" not in response.headers


def test_a_loopback_client_with_a_foreign_host_and_the_token_is_served(token: str) -> None:
    response = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000)).get(
        "/health", headers={"Host": "evil.example", **_basic("u", token)}
    )
    assert response.status_code == 200


@pytest.mark.parametrize("has_token", [False, True])
def test_a_loopback_client_needs_no_token(monkeypatch: pytest.MonkeyPatch, has_token: bool) -> None:
    monkeypatch.setattr(api.main, "TOKEN", SYNTHETIC_TOKEN if has_token else None)
    response = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000)).get(
        "/health"
    )
    assert response.status_code == 200


def test_a_repeated_host_is_refused_before_the_token(token: str) -> None:
    status, body = _drive(
        [
            (b"host", b"localhost"),
            (b"host", b"localhost"),
            (b"authorization", _basic("u", token)["Authorization"].encode()),
        ],
        client=LAN_PEER,
    )
    assert status == 400
    assert body["detail"] == REPEATED_HOST


@pytest.mark.parametrize("peer", ["127.0.0.1", "127.5.5.5", "::1", "::ffff:127.0.0.1"])
@pytest.mark.usefixtures("token")
def test_loopback_peers_need_no_token(peer: str) -> None:
    assert _lan_client((peer, 40000)).get("/health").status_code == 200


@pytest.mark.parametrize(
    "peer",
    ["172.17.0.1", "::ffff:172.17.0.1", "::2", "192.168.1.209", "testclient", "localhost"],
)
@pytest.mark.usefixtures("token")
def test_other_peers_need_the_token(peer: str) -> None:
    assert _lan_client((peer, 40000)).get("/health").status_code == 401


@pytest.mark.usefixtures("token")
def test_a_scope_with_no_client_needs_the_token() -> None:
    status, _ = _drive([(b"host", b"localhost")], client=None)
    assert status == 401
    status, _ = _drive([(b"host", b"localhost")], with_client=False)
    assert status == 401


@pytest.mark.parametrize(
    "text", ["127.0.0.1", "127.255.255.255", "::1", "::ffff:127.0.0.1", "::ffff:127.9.9.9"]
)
def test_is_loopback_address_true(text: str) -> None:
    assert api.main.is_loopback_address(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "testclient",
        "localhost",
        "172.17.0.1",
        "::ffff:172.17.0.1",
        "::2",
        "192.168.1.209",
        "",
        "not an address",
    ],
)
def test_is_loopback_address_false(text: str) -> None:
    assert api.main.is_loopback_address(text) is False


def test_a_mapped_loopback_address_is_loopback_where_python_says_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ipaddress.IPv6Address, "is_loopback", property(lambda _self: False))
    assert api.main.is_loopback_address("::ffff:127.0.0.1") is True
    assert api.main.is_loopback_address("::ffff:172.17.0.1") is False


@pytest.mark.usefixtures("token")
def test_a_non_ascii_authorization_is_no_token() -> None:
    status, body = _drive(
        [(b"host", b"localhost"), (b"authorization", b"Basic \xe9")], client=LAN_PEER
    )
    assert status == 401
    assert body == {"detail": CHALLENGE}


@pytest.mark.usefixtures("token")
def test_a_non_ascii_authorization_on_a_rebinding_request_is_a_400() -> None:
    status, body = _drive([(b"host", b"evil.example"), (b"authorization", b"Basic \xe9")])
    assert status == 400
    assert body == {
        "detail": "host 'evil.example' is not served: this server answers only localhost, "
        "127.0.0.1 and [::1], because it is for this machine alone."
    }


@pytest.mark.usefixtures("token")
def test_a_lan_client_is_checked_before_the_query() -> None:
    response = _lan_client().post(
        "/api/simulate?nominl=1", content=example_text().encode(), headers=YAML
    )
    assert response.status_code == 401


OVERSIZE = "request body is larger than 1,048,576 bytes; a scenario is a few kilobytes."


def test_body_over_the_limit_is_413_by_content_length(client: TestClient) -> None:
    body = b"#" * (api.main.MAX_BODY_BYTES + 1)
    response = client.post("/api/simulate", content=body, headers=YAML)
    assert response.status_code == 413
    assert response.json()["detail"] == OVERSIZE


def test_content_length_alone_refuses_before_any_byte_is_read(client: TestClient) -> None:
    """A declared length over the limit is refused although the bytes sent are few."""
    declared = {**YAML, "Content-Length": str(api.main.MAX_BODY_BYTES + 1)}
    response = client.post("/api/simulate", content=b"#", headers=declared)
    assert response.status_code == 413
    assert response.json()["detail"] == OVERSIZE


def test_body_over_the_limit_is_413_when_streamed(client: TestClient) -> None:
    def chunks():
        for _ in range(api.main.MAX_BODY_BYTES // 65_536 + 1):
            yield b"#" * 65_536

    request = client.build_request("POST", "/api/simulate", content=chunks(), headers=YAML)
    assert "content-length" not in request.headers
    response = client.send(request)
    assert response.status_code == 413
    assert response.json()["detail"] == OVERSIZE


def test_body_at_the_limit_is_read(client: TestClient) -> None:
    body = b"#" * api.main.MAX_BODY_BYTES
    response = client.post("/api/simulate", content=body, headers=YAML)
    assert response.status_code == 422
    assert "request body" in response.json()["detail"]


def test_content_type_is_checked_before_the_body(client: TestClient) -> None:
    body = b"#" * (api.main.MAX_BODY_BYTES + 1)
    response = client.post("/api/simulate", content=body, headers={"Content-Type": "text/plain"})
    assert response.status_code == 415
    assert response.json()["detail"].startswith("unsupported Content-Type 'text/plain'")


def test_body_is_checked_before_the_run_lock(client: TestClient) -> None:
    body = b"#" * (api.main.MAX_BODY_BYTES + 1)
    assert api.main._RUN_LOCK.acquire(blocking=False)
    try:
        response = client.post("/api/simulate", content=body, headers=YAML)
    finally:
        api.main._RUN_LOCK.release()
    assert response.status_code == 413
    assert response.json()["detail"] == OVERSIZE


def test_content_length_with_thousands_of_digits_is_413() -> None:
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"#", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/simulate",
        "raw_path": b"/api/simulate",
        "query_string": b"",
        "headers": [
            (b"host", b"localhost"),
            (b"content-type", b"application/yaml"),
            (b"content-length", b"9" * 5_000),
        ],
        "scheme": "http",
        "http_version": "1.1",
        "client": ("127.0.0.1", 50000),
    }
    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["detail"] == OVERSIZE


RUN_BUSY = (
    "a run is already in progress; this server runs one at a time, so that two cannot "
    "exhaust memory together. Try again when it finishes."
)


def test_second_run_is_refused_with_429(client: TestClient) -> None:
    assert api.main._RUN_LOCK.acquire(blocking=False)
    try:
        simulated = client.post(
            "/api/simulate?paths=2", content=example_text().encode(), headers=YAML
        )
        optimized = client.post(
            "/api/optimize?paths=2&objective=median_estate_after_tax",
            content=example_text().encode(),
            headers=YAML,
        )
    finally:
        api.main._RUN_LOCK.release()
    for response in (simulated, optimized):
        assert response.status_code == 429
        assert response.json()["detail"] == RUN_BUSY


def _lock_is_free() -> bool:
    if not api.main._RUN_LOCK.acquire(blocking=False):
        return False
    api.main._RUN_LOCK.release()
    return True


def test_run_lock_is_released_after_a_run(client: TestClient) -> None:
    response = client.post("/api/simulate?paths=2", content=example_text().encode(), headers=YAML)
    assert response.status_code == 200
    assert _lock_is_free()


def test_run_lock_is_released_after_a_refused_run(client: TestClient) -> None:
    response = client.post(
        "/api/simulate?paths=2&policy=nope", content=example_text().encode(), headers=YAML
    )
    assert response.status_code == 422
    assert response.json()["detail"].startswith("unknown policy 'nope'")
    assert _lock_is_free()


def test_run_over_the_path_limit_is_422(client: TestClient) -> None:
    response = client.post(
        "/api/simulate?paths=100001", content=example_text().encode(), headers=YAML
    )
    assert response.status_code == 422
    assert "100,001 paths" in response.json()["detail"]
    assert "at most 100,000" in response.json()["detail"]
    assert _lock_is_free()


def test_simulate_is_bound_by_candidates_too(client: TestClient) -> None:
    body = example_text().replace(
        GRID_LINE,
        "elections.cpp_start_age_years.a: [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]\n"
        "  elections.oas_start_age_years.a: [65, 66, 67, 68, 69, 70, 71, 72, 73, 74]",
    )
    assert body != example_text()
    response = client.post("/api/simulate?paths=100000", content=body.encode(), headers=YAML)
    assert response.status_code == 422
    assert "110 candidate policies on 100,000 paths each" in response.json()["detail"]


def test_deterministic_run_is_bound_by_candidates(client: TestClient) -> None:
    body = example_text().replace(
        GRID_LINE,
        "elections.cpp_start_age_years.a: [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]\n"
        "  elections.oas_start_age_years.a: [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]\n"
        "  elections.rrif_conversion.age_years: [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]",
    )
    assert body != example_text()
    response = client.post("/api/simulate?deterministic=true", content=body.encode(), headers=YAML)
    assert response.status_code == 422
    assert "1,331 candidate policies, and a run may have at most 1,000" in response.json()["detail"]


def test_huge_validation_error_is_capped(client: TestClient) -> None:
    values = parse_yaml(example_text())
    values["policies"] = ["x"] * 5_000
    response = client.post(
        "/api/simulate?paths=2", content=json.dumps(values).encode(), headers=JSON
    )
    assert response.status_code == 422
    assert len(response.json()["detail"]) < 10_000
    assert "more errors, not shown" in response.json()["detail"]


def test_content_length_with_leading_zeros_within_the_limit_is_read() -> None:
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"#" * 10, "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/simulate",
        "raw_path": b"/api/simulate",
        "query_string": b"",
        "headers": [
            (b"host", b"localhost"),
            (b"content-type", b"application/yaml"),
            (b"content-length", b"0" * 5_000 + b"10"),
        ],
        "scheme": "http",
        "http_version": "1.1",
        "client": ("127.0.0.1", 50000),
    }
    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == 422


def _negative_grid_body() -> bytes:
    grid = ", ".join(f"-{i}.0" for i in range(1, 31))
    body = example_text().replace(GRID_LINE, f"contribution.weights.rrsp: [{grid}]")
    assert body != example_text()
    return body.encode()


def test_api_caps_a_validation_error_from_the_grid(client: TestClient) -> None:
    response = client.post(
        "/api/simulate?deterministic=true", content=_negative_grid_body(), headers=YAML
    )
    assert response.status_code == 422
    assert response.json()["detail"].endswith("... and 10 more errors, not shown.")


# =============================================================================
# The scenario editor's calls and the minimal scenario
# =============================================================================

MINIMAL = REPO_ROOT / "scenarios" / "minimal.yaml"
NOT_JSON = "unsupported Content-Type {header!r}; send the arguments as application/json, in UTF-8."


def editor_mapping() -> dict:
    return editor.parse(example_text(), params_root=api.main.PARAMS_ROOT)["mapping"]


class TestEditor:
    def test_it_answers_with_the_answers_status_and_body(self, client: TestClient) -> None:
        text = json.dumps({"text": example_text()})
        response = client.post("/api/editor/parse", content=text.encode(), headers=JSON)
        expected = answer("parse", text, params_root=api.main.PARAMS_ROOT)
        assert (response.status_code, response.text) == expected
        assert expected[0] == 200
        assert response.headers["content-type"] == "application/json"

    def test_a_refused_edit_is_answered_with_its_409(self, client: TestClient) -> None:
        text = json.dumps({"mapping": editor_mapping(), "kind": "person", "name": "a"})
        response = client.post("/api/editor/remove", content=text.encode(), headers=JSON)
        expected = answer("remove", text, params_root=api.main.PARAMS_ROOT)
        assert (response.status_code, response.text) == expected
        assert expected[0] == 409
        assert response.headers["content-type"] == "application/json"

    @pytest.mark.parametrize(
        "headers",
        [
            {"Content-Type": "application/yaml"},
            {},
            {"Content-Type": "application/json; charset=latin-1"},
            {"Content-Type": "text/plain"},
        ],
        ids=["yaml", "none", "latin-1", "text-plain"],
    )
    def test_a_content_type_other_than_json_is_a_415(
        self, client: TestClient, headers: dict[str, str]
    ) -> None:
        response = client.post("/api/editor/schema", content=b"{}", headers=headers)
        assert response.status_code == 415
        assert response.json()["detail"] == NOT_JSON.format(header=headers.get("Content-Type"))

    def test_a_query_parameter_is_a_422(self, client: TestClient) -> None:
        response = client.post("/api/editor/schema?x=1", content=b"{}", headers=JSON)
        assert response.status_code == 422
        assert response.json()["detail"] == (
            "unknown query parameter 'x'; /api/editor/schema accepts []."
        )

    def test_a_body_over_the_limit_is_a_413(self, client: TestClient) -> None:
        body = b" " * (api.main.MAX_BODY_BYTES + 1)
        response = client.post("/api/editor/schema", content=body, headers=JSON)
        assert response.status_code == 413
        assert response.json()["detail"] == OVERSIZE

    def test_a_body_that_is_not_utf8_is_a_422(self, client: TestClient) -> None:
        response = client.post("/api/editor/parse", content=b'{"text": "\xff"}', headers=JSON)
        assert response.status_code == 422
        assert response.json()["detail"].startswith("request body is not valid UTF-8")

    def test_it_answers_while_a_run_holds_the_lock(self, client: TestClient) -> None:
        answered: list[int] = []

        def request() -> None:
            response = client.post("/api/editor/schema", content=b"{}", headers=JSON)
            answered.append(response.status_code)

        thread = threading.Thread(target=request, daemon=True)
        assert api.main._RUN_LOCK.acquire(blocking=False)
        try:
            thread.start()
            thread.join(timeout=30)
            finished = not thread.is_alive()
        finally:
            api.main._RUN_LOCK.release()
        thread.join()
        assert finished
        assert answered == [200]

    def test_an_exception_in_the_call_is_a_500(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_args: object, **_kwargs: object) -> tuple[int, str]:
            raise RoutedParameterError("boom")

        monkeypatch.setattr(api.main, "answer", boom)
        quiet = TestClient(
            app,
            base_url="http://localhost",
            client=("127.0.0.1", 50000),
            raise_server_exceptions=False,
        )
        response = quiet.post("/api/editor/schema", content=b"{}", headers=JSON)
        assert response.status_code == 500

    def test_it_reads_params_root_per_request(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api.main, "PARAMS_ROOT", tmp_path)
        text = json.dumps({"text": example_text()})
        response = client.post("/api/editor/parse", content=text.encode(), headers=JSON)
        assert response.status_code == 200
        errors = response.json()["errors"]
        assert len(errors) == 1
        assert str(tmp_path) in errors[0]["message"]

    def test_check_reads_params_root_per_request(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mapping = editor_mapping()
        monkeypatch.setattr(api.main, "PARAMS_ROOT", tmp_path)
        text = json.dumps({"mapping": mapping})
        response = client.post("/api/editor/check", content=text.encode(), headers=JSON)
        assert response.status_code == 200
        errors = response.json()["errors"]
        assert len(errors) == 1
        assert str(tmp_path) in errors[0]["message"]

    @pytest.mark.parametrize(
        ("path", "headers", "body", "status", "detail"),
        [
            (
                "/api/editor/schema?x=1",
                {"Content-Type": "application/yaml"},
                b"{}",
                422,
                "unknown query parameter 'x'; /api/editor/schema accepts [].",
            ),
            (
                "/api/editor/schema",
                {"Content-Type": "application/yaml"},
                b" " * (api.main.MAX_BODY_BYTES + 1),
                415,
                NOT_JSON.format(header="application/yaml"),
            ),
            (
                "/api/editor/schema",
                JSON,
                b"\xff" * (api.main.MAX_BODY_BYTES + 1),
                413,
                OVERSIZE,
            ),
        ],
        ids=["query-before-content-type", "content-type-before-body", "size-before-utf8"],
    )
    def test_the_refusals_come_in_order(
        self,
        client: TestClient,
        path: str,
        headers: dict[str, str],
        body: bytes,
        status: int,
        detail: str,
    ) -> None:
        response = client.post(path, content=body, headers=headers)
        assert response.status_code == status
        assert response.json()["detail"] == detail

    @pytest.mark.usefixtures("token")
    def test_a_remote_client_without_the_token_is_refused(self) -> None:
        response = _lan_client(("10.0.0.5", 5)).post(
            "/api/editor/schema", content=b"{}", headers=JSON
        )
        assert response.status_code == 401

    def test_a_remote_client_is_forbidden_when_the_server_has_no_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api.main, "TOKEN", None)
        response = _lan_client(("10.0.0.5", 5)).post(
            "/api/editor/schema", content=b"{}", headers=JSON
        )
        assert response.status_code == 403


def _setup_absent(tmp_path: Path, _lock: Callable[[Path, int], None]) -> Path:
    return tmp_path / "absent.yaml"


def _setup_directory(tmp_path: Path, _lock: Callable[[Path, int], None]) -> Path:
    return tmp_path


def _setup_through_a_file(tmp_path: Path, _lock: Callable[[Path, int], None]) -> Path:
    through = tmp_path / "f"
    through.write_text("x\n", encoding="utf-8")
    return through / "s.yaml"


def _setup_loop(tmp_path: Path, _lock: Callable[[Path, int], None]) -> Path:
    path = tmp_path / "loop.yaml"
    path.symlink_to(path)
    return path


def _setup_mode_zero(tmp_path: Path, lock: Callable[[Path, int], None]) -> Path:
    path = tmp_path / "s.yaml"
    path.write_text("x: 1\n", encoding="utf-8")
    lock(path, 0o000)
    return path


def _setup_locked_directory(tmp_path: Path, lock: Callable[[Path, int], None]) -> Path:
    directory = tmp_path / "d"
    directory.mkdir()
    path = directory / "s.yaml"
    path.write_text("x: 1\n", encoding="utf-8")
    lock(directory, 0o000)
    return path


def _setup_fifo(tmp_path: Path, _lock: Callable[[Path, int], None]) -> Path:
    path = tmp_path / "s.yaml"
    os.mkfifo(path)
    return path


def _setup_devnull(_tmp_path: Path, _lock: Callable[[Path, int], None]) -> Path:
    return Path(os.devnull)


def _setup_regular(tmp_path: Path, _lock: Callable[[Path, int], None]) -> Path:
    path = tmp_path / "s.yaml"
    path.write_text("x: 1\n", encoding="utf-8")
    return path


MINIMAL_SETUPS = {
    "absent": _setup_absent,
    "directory": _setup_directory,
    "through-a-file": _setup_through_a_file,
    "symlink-to-itself": _setup_loop,
    "mode-zero": _setup_mode_zero,
    "directory-of-mode-zero": _setup_locked_directory,
    "fifo": _setup_fifo,
    "devnull": _setup_devnull,
    "regular": _setup_regular,
    "read-raises": _setup_regular,
}


class TestMinimal:
    def test_it_serves_the_file_as_it_is(self, client: TestClient) -> None:
        response = client.get("/api/minimal")
        assert response.status_code == 200
        assert response.content == MINIMAL.read_bytes()
        assert response.headers["content-type"] == "application/yaml; charset=utf-8"

    @pytest.mark.parametrize("setup", MINIMAL_SETUPS, ids=list(MINIMAL_SETUPS))
    def test_it_answers_as_the_example_does(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        request: pytest.FixtureRequest,
        setup: str,
    ) -> None:
        if setup in ("mode-zero", "directory-of-mode-zero"):
            lock = request.getfixturevalue("lock")
        else:

            def lock(_path: Path, _mode: int) -> None:
                pytest.fail("this setup locks nothing")

        path = MINIMAL_SETUPS[setup](tmp_path, lock)
        monkeypatch.setattr(api.main, "EXAMPLE_PATH", path)
        monkeypatch.setattr(api.main, "MINIMAL_PATH", path)
        if setup == "read-raises":

            def boom(*_args: object, **_kwargs: object) -> bytes:
                raise OSError("boom")

            monkeypatch.setattr(Path, "read_bytes", boom)
        example = client.get("/api/example")
        minimal = client.get("/api/minimal")
        assert minimal.status_code == example.status_code
        if example.status_code == 200:
            assert minimal.content == example.content
            return
        stated = example.json()["detail"]
        if setup in ("absent", "through-a-file"):
            assert "example scenario" in stated
            assert example.status_code == 404
        if setup in ("mode-zero", "directory-of-mode-zero", "symlink-to-itself", "read-raises"):
            assert "example scenario" in stated
            assert example.status_code == 503
        assert minimal.json()["detail"] == stated.replace("example scenario", "minimal scenario")

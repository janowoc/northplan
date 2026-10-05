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

import json
import time
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

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "scenarios" / "example.yaml"
COUPLE = REPO_ROOT / "scenarios" / "late_life_couple.yaml"

YAML = {"Content-Type": "application/yaml"}
JSON = {"Content-Type": "application/json"}
PATHS = 20
GRID_LINE = "elections.cpp_start_age_years.a: [60, 65, 70]"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


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

        monkeypatch.setattr(api.main, "prepare_run", refuse)
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
        monkeypatch.setattr(api.main, "prepare_run", self.raise_routed)
        quiet = TestClient(app, raise_server_exceptions=False)
        response = quiet.post(
            "/api/simulate?paths=2", content=example_text().encode(), headers=YAML
        )
        assert response.status_code == 500

    def test_it_propagates_to_the_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(api.main, "prepare_run", self.raise_routed)
        with pytest.raises(RoutedParameterError):
            TestClient(app).post(
                "/api/simulate?paths=2", content=example_text().encode(), headers=YAML
            )


class TestParameterErrorsAnywhereInTheRun:
    @staticmethod
    def raise_missing(*_args: object, **_kwargs: object) -> None:
        raise MissingParameterError("missing for this test")

    def post(self, path: str, quiet: bool = False):
        client = TestClient(app, raise_server_exceptions=not quiet)
        return client.post(path, content=example_text().encode(), headers=YAML)

    def test_simulate_evaluate_is_a_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(api.main, "evaluate", self.raise_missing)
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

        monkeypatch.setattr(api.main, "evaluate", raise_routed)
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


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}

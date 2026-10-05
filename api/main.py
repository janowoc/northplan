# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""FastAPI application: one container, API plus static files.

``POST /api/simulate`` and ``POST /api/optimize`` take a scenario as the request
body, JSON or YAML, and the run's options in the query string, and return the
document ``--out x.json`` writes, except that ``meta.scenario`` names the scenario
alone and the notes name query parameters, not flags. ``GET /api/example``
returns ``scenarios/example.yaml`` as it is on disk. ``GET /health`` does not touch
the engine.

Status codes: 422 for whatever the command line refuses with exit 2 (a body that
does not parse or validate, a scenario the load checks refuse, an unknown policy or
query parameter, a missing preference, a bad option); 400 with the message for a
:class:`~engine.params.loader.ParamError`; 415 for a Content-Type that is neither JSON
nor YAML, or a charset other than UTF-8. A
:class:`~engine.core.indexation.RoutedParameterError` signals an engine bug and is
not caught, so it surfaces as a 500. The engine work runs in a worker thread, off
the event loop.

The API trusts its caller as the command line trusts its user: it bounds neither a
run's size nor a request's, so it is served on loopback, for one user on this
machine. The scenario parser's cap on alias repetition
(:data:`engine.scenario.load.MAX_DOCUMENT_VALUES`) is not a defence against a
hostile caller.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from api.schemas import OptimizeResponse, SimulationResponse
from engine.core.indexation import RoutedParameterError
from engine.mc.prepare import PreparedRun, evaluate, prepare_run
from engine.optimize.objective import OBJECTIVE_NAMES, select_objective
from engine.optimize.search import search
from engine.params.loader import ParamError
from engine.scenario.load import ScenarioError, scenario_from_text
from engine.scenario.schema import Scenario
from report.tables import RunReport, UnknownPolicyError, preference, select_policy

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"
EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "scenarios" / "example.yaml"

_JSON_TYPES = frozenset({"application/json"})
_YAML_TYPES = frozenset({"application/yaml", "application/x-yaml", "text/yaml"})

_SCENARIO_REF = {"$ref": "#/components/schemas/Scenario"}
_REQUEST_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {"schema": _SCENARIO_REF},
            "application/yaml": {"schema": _SCENARIO_REF},
        },
    }
}

app = FastAPI(
    title="northplan",
    description="Canadian household financial planning engine",
    version="0.0.0",
)


def _openapi() -> dict[str, object]:
    """The OpenAPI document, with the Scenario schema and its sub-models as components."""
    if app.openapi_schema:
        return app.openapi_schema
    document = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        servers=app.servers,
    )
    scenario_schema = Scenario.model_json_schema(ref_template="#/components/schemas/{model}")
    definitions = scenario_schema.pop("$defs", {})
    schemas = document.setdefault("components", {}).setdefault("schemas", {})
    for name, schema in [*definitions.items(), ("Scenario", scenario_schema)]:
        if name in schemas:
            raise RuntimeError(f"OpenAPI component {name!r} is defined twice.")
        schemas[name] = schema
    app.openapi_schema = document
    return document


app.openapi = _openapi  # type: ignore[method-assign]


@contextmanager
def _parameter_errors() -> Iterator[None]:
    """A ParamError is a 400; a RoutedParameterError, an engine bug, passes unchanged."""
    try:
        yield
    except RoutedParameterError:
        raise
    except ParamError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check. Does not touch the engine."""
    return {"status": "ok"}


def _query_names(endpoint: Callable[..., object]) -> frozenset[str]:
    """The query parameter names ``endpoint`` accepts: its parameters but ``request``."""
    return frozenset(inspect.signature(endpoint).parameters) - {"request"}


def _check_query(
    request: Request, path: str, accepted: frozenset[str], deterministic: bool, paths: int | None
) -> None:
    """Refuse an unknown query parameter, then ``paths`` together with ``deterministic``."""
    unknown = sorted(set(request.query_params) - accepted)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"unknown query parameter {unknown[0]!r}; {path} accepts {sorted(accepted)!r}."
            ),
        )
    if deterministic and paths is not None:
        raise HTTPException(
            status_code=422,
            detail=(
                "paths and deterministic cannot both be given; "
                "the deterministic run is always exactly one path."
            ),
        )


def _syntax(content_type: str | None) -> Literal["yaml", "json"]:
    """The body's syntax from its Content-Type, or a 415."""
    unsupported = HTTPException(
        status_code=415,
        detail=(
            f"unsupported Content-Type {content_type!r}; send application/json, or YAML as "
            "application/yaml, application/x-yaml or text/yaml, in UTF-8."
        ),
    )
    if content_type is None:
        raise unsupported
    media, *parameters = content_type.split(";")
    media = media.strip().lower()
    for parameter in parameters:
        key, _, value = parameter.partition("=")
        if key.strip().lower() == "charset" and value.strip().strip("\"'").lower() != "utf-8":
            raise unsupported
    if media in _JSON_TYPES:
        return "json"
    if media in _YAML_TYPES:
        return "yaml"
    raise unsupported


def _open_run(
    body: bytes, content_type: str | None, paths: int | None, deterministic: bool
) -> tuple[Scenario, PreparedRun]:
    """Decode and load the body, then open the run; every refusal is an HTTPException."""
    syntax = _syntax(content_type)
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422, detail=f"request body is not valid UTF-8: {exc}. Send it as UTF-8."
        ) from exc
    try:
        scenario = scenario_from_text(text, "request body", syntax=syntax)
    except ScenarioError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    try:
        with _parameter_errors():
            prepared = prepare_run(scenario, n_paths=paths, deterministic=deterministic)
    except (ValidationError, ScenarioError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return scenario, prepared


@app.post("/api/simulate", response_model=SimulationResponse, openapi_extra=_REQUEST_BODY)
async def simulate(
    request: Request,
    nominal: bool = False,
    policy: str | None = None,
    paths: int | None = Query(None, ge=1),
    deterministic: bool = False,
) -> dict[str, object]:
    """Run one policy on the scenario in the request body.

    Query parameters: ``nominal`` (dollars at each year end instead of real), ``policy``
    (an expanded policy name; default the first), ``paths`` (overrides ``n_paths``) and
    ``deterministic`` (one path; not with ``paths``).

    Returns:
        The document ``northplan simulate --out x.json`` writes, except that
        ``meta.scenario`` names the scenario alone and the notes name query
        parameters, not flags.
    """
    _check_query(request, "/api/simulate", _SIMULATE_QUERY, deterministic, paths)
    body = await request.body()
    content_type = request.headers.get("content-type")

    def work() -> dict[str, object]:
        scenario, prepared = _open_run(body, content_type, paths, deterministic)
        try:
            spec = select_policy(prepared, scenario, policy)
        except UnknownPolicyError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        with _parameter_errors():
            result = evaluate(prepared, spec)
        report = RunReport(
            "simulate",
            prepared.scenario.name,
            prepared,
            deterministic=deterministic,
            nominal=nominal,
            nominal_option="nominal=true",
        )
        return report.simulate_document(result, spec.name)

    return await run_in_threadpool(work)


_SIMULATE_QUERY = _query_names(simulate)


@app.post("/api/optimize", response_model=OptimizeResponse, openapi_extra=_REQUEST_BODY)
async def optimize(
    request: Request,
    objective: Literal[*OBJECTIVE_NAMES],
    nominal: bool = False,
    paths: int | None = Query(None, ge=1),
    deterministic: bool = False,
    risk_aversion: float | None = Query(None, ge=0, allow_inf_nan=False),
    estate_utility_shift: float | None = Query(None, gt=0, allow_inf_nan=False),
) -> dict[str, object]:
    """Evaluate every expanded policy of the scenario in the request body, and rank them.

    Query parameters: ``objective`` (required), ``nominal``, ``paths``, ``deterministic``
    (as for simulate), and ``risk_aversion`` and ``estate_utility_shift`` (override the
    scenario's own).

    Returns:
        The document ``northplan optimize --out x.json`` writes, except that
        ``meta.scenario`` names the scenario alone and the notes name query
        parameters, not flags.
    """
    _check_query(request, "/api/optimize", _OPTIMIZE_QUERY, deterministic, paths)
    body = await request.body()
    content_type = request.headers.get("content-type")

    def work() -> dict[str, object]:
        _, prepared = _open_run(body, content_type, paths, deterministic)
        scenario = prepared.scenario
        risk_value, risk_text = preference(risk_aversion, scenario.risk_aversion, source="query")
        shift_value, shift_text = preference(
            estate_utility_shift, scenario.estate_utility_shift, source="query"
        )
        try:
            objective_fn = select_objective(
                objective, risk_aversion=risk_value, estate_utility_shift=shift_value
            )
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{error} Either can be given for this run with the risk_aversion or "
                    "estate_utility_shift query parameter."
                ),
            ) from error
        with _parameter_errors():
            found = search(prepared, objective_fn)
            result = evaluate(prepared, found.best)
        report = RunReport(
            "optimize",
            scenario.name,
            prepared,
            deterministic=deterministic,
            nominal=nominal,
            nominal_option="nominal=true",
        )
        return report.optimize_document(
            found,
            result,
            objective=objective,
            risk_aversion_text=risk_text,
            estate_utility_shift_text=shift_text,
        )

    return await run_in_threadpool(work)


_OPTIMIZE_QUERY = _query_names(optimize)


@app.get("/api/example")
def example() -> Response:
    """The committed example scenario, as it is on disk."""
    try:
        content = EXAMPLE_PATH.read_bytes()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"no example scenario at {EXAMPLE_PATH}."
        ) from exc
    return Response(content=content, media_type="application/yaml; charset=utf-8")


# Mounted last so that /health and /api/* win over any file of the same name.
app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")

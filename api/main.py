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
query parameter, a missing preference, a bad option, a run over the engine's limits
(:class:`~engine.mc.prepare.RunTooLargeError`)); 400 with the message for a
:class:`~engine.params.loader.ParamError`, and for a Host other than loopback; 415 for
a Content-Type that is neither JSON nor YAML, or a charset other than UTF-8; 413 for a
body over ``MAX_BODY_BYTES``; 429 while another run is in progress. A
:class:`~engine.core.indexation.RoutedParameterError` signals an engine bug and is
not caught, so it surfaces as a 500. The engine work runs in a worker thread, off
the event loop.

The server is for one user on this machine. It bounds what one request can cost (the
body's size, the run's paths and candidates, one run at a time, and the scenario
parser's caps on aliases, merge keys and text), so that a mistake cannot exhaust a
laptop, and it answers only a loopback Host, so that a web page cannot reach it by DNS
rebinding. It has no authentication: whoever can reach its port can run the engine, so
it is served on loopback.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Awaitable, Callable, Iterator, MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from api.schemas import OptimizeResponse, SimulationResponse
from engine.core.indexation import RoutedParameterError
from engine.mc.prepare import PreparedRun, evaluate, prepare_run
from engine.optimize.objective import OBJECTIVE_NAMES, select_objective
from engine.optimize.search import search
from engine.params.loader import ParamError
from engine.scenario.load import ScenarioError, scenario_from_text, validation_message
from engine.scenario.schema import Scenario
from report.tables import RunReport, UnknownPolicyError, preference, select_policy

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"
EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "scenarios" / "example.yaml"

MAX_BODY_BYTES = 1 << 20
_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]"})
_RUN_LOCK = threading.Lock()

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


def _loopback_name(raw: str) -> str | None:
    """The lower-case host of a Host header value, or None if its port, or a bracketed
    host's closing bracket, is malformed.

    A bracketed host is followed by nothing or by ``:`` and digits; an unbracketed
    host with a ``:`` has digits after the last one.
    """
    if raw.startswith("["):
        end = raw.find("]")
        if end < 0:
            return None
        host, rest = raw[: end + 1], raw[end + 1 :]
        if rest and not (rest.startswith(":") and _is_digits(rest[1:])):
            return None
        return host.lower()
    if ":" in raw:
        host, _, port = raw.rpartition(":")
        return host.lower() if _is_digits(port) else None
    return raw.lower()


def _is_digits(text: str) -> bool:
    return bool(text) and text.isascii() and text.isdigit()


class _LoopbackHostMiddleware:
    """Answer 400 to an HTTP request whose Host header is not a loopback name.

    Pure ASGI, so that it covers every route and the static files. The port is
    ignored and the name is compared in lower case; other scope types pass through.
    """

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        hosts = [value.decode("latin-1") for name, value in scope["headers"] if name == b"host"]
        raw = hosts[0] if hosts else None
        detail: str | None = None
        if len(hosts) > 1:
            detail = (
                "the request carries more than one Host header; this server answers a "
                "request with exactly one."
            )
        elif not hosts or _loopback_name(hosts[0]) not in _ALLOWED_HOSTS:
            detail = (
                f"host {raw!r} is not served: this server answers only localhost, "
                "127.0.0.1 and [::1], because it is for this machine alone."
            )
        if detail is not None:
            response = JSONResponse({"detail": detail}, status_code=400)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


app.add_middleware(_LoopbackHostMiddleware)


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


async def _read_body(request: Request) -> bytes:
    """The request body, or a 413 once it passes ``MAX_BODY_BYTES``; read no further."""
    too_large = HTTPException(
        status_code=413,
        detail=(
            f"request body is larger than {MAX_BODY_BYTES:,} bytes; a scenario is a few kilobytes."
        ),
    )
    declared = request.headers.get("content-length")
    significant = (declared or "").lstrip("0")
    # More significant digits than the limit has is over it, and int() refuses past 4,300
    # digits; leading zeros are not significant.
    if (
        declared is not None
        and _is_digits(declared)
        and (
            len(significant) > len(str(MAX_BODY_BYTES)) or int(significant or "0") > MAX_BODY_BYTES
        )
    ):
        raise too_large
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            raise too_large
        chunks.append(chunk)
    return b"".join(chunks)


_RUN_BUSY = (
    "a run is already in progress; this server runs one at a time, so that two cannot "
    "exhaust memory together. Try again when it finishes."
)


def _open_run(
    body: bytes, syntax: Literal["yaml", "json"], paths: int | None, deterministic: bool
) -> tuple[Scenario, PreparedRun]:
    """Decode and load the body, then open the run; every refusal is an HTTPException."""
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
        detail = validation_message(error) if isinstance(error, ValidationError) else str(error)
        raise HTTPException(status_code=422, detail=detail) from error
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
    syntax = _syntax(request.headers.get("content-type"))
    body = await _read_body(request)

    def work() -> dict[str, object]:
        if not _RUN_LOCK.acquire(blocking=False):
            raise HTTPException(status_code=429, detail=_RUN_BUSY)
        try:
            scenario, prepared = _open_run(body, syntax, paths, deterministic)
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
        finally:
            _RUN_LOCK.release()

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
    syntax = _syntax(request.headers.get("content-type"))
    body = await _read_body(request)

    def work() -> dict[str, object]:
        if not _RUN_LOCK.acquire(blocking=False):
            raise HTTPException(status_code=429, detail=_RUN_BUSY)
        try:
            _, prepared = _open_run(body, syntax, paths, deterministic)
            scenario = prepared.scenario
            risk_value, risk_text = preference(
                risk_aversion, scenario.risk_aversion, source="query"
            )
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
        finally:
            _RUN_LOCK.release()

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

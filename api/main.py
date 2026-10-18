# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""FastAPI application: one container, API plus static files.

``POST /api/simulate`` and ``POST /api/optimize`` take a scenario as the request
body, JSON or YAML, and the run's options in the query string, and return the
document ``--out x.json`` writes, except that ``meta.scenario`` names the scenario
alone and the notes name query parameters, not flags. ``GET /api/example``
returns the file at ``EXAMPLE_PATH`` as it is on disk: ``scenarios/example.yaml``
beside the source tree, unless ``northplan-serve`` was given ``--example``.
``GET /api/minimal`` returns the file at ``MINIMAL_PATH`` the same way:
``scenarios/minimal.yaml`` beside the source tree, which no flag changes.
``POST /api/editor/{call}`` takes the call's arguments as a JSON body and answers with
the status and body :func:`report.editor_calls.answer` gives. Runs and the editor load
their parameters from ``PARAMS_ROOT``, the engine's default unless ``northplan-serve``
was given ``--params``; the three paths are read per request, but the editor keeps each
parameter year it has loaded until the process ends. ``GET /health`` does not touch the
engine.

Status codes: 422 for whatever the command line refuses with exit 2 (a body that does
not parse or validate, a scenario the load checks refuse, an unknown policy or query
parameter, a missing preference, a bad option, a run over the engine's limits
(:class:`~engine.mc.prepare.RunTooLargeError`)); 400 with the message for a
:class:`~engine.params.loader.ParamError`, for a request with more than one Host header,
and for a loopback client that names a Host other than loopback and carries no valid
token; 401, with a Basic challenge, for any other client without the valid token, and
403 instead when the server has no token; 415 for a Content-Type that is neither JSON
nor YAML, or a charset other than UTF-8, and on ``/api/editor/{call}`` for one that is
not JSON; 413 for a body over ``MAX_BODY_BYTES``; 429 while another run is in progress.
``GET /api/example`` and ``GET /api/minimal`` answer 404 when no file is found at their
path (ENOENT or ENOTDIR) or what is there is not a regular file, and 503 when the lookup
or the read fails with any other ``OSError``, such as EACCES. A
:class:`~engine.core.indexation.RoutedParameterError` signals an engine bug and is not
caught, so it surfaces as a 500. The engine work runs in a worker thread, off the event
loop.

The server is for one household. It bounds what one request can cost (the body's size,
the run's paths and candidates, one run at a time, and the scenario parser's caps on
aliases, merge keys and text), so that a mistake cannot exhaust a laptop. A request is
served when its connection comes from a loopback address and its Host is localhost,
127.0.0.1 or [::1], or when it carries ``TOKEN`` as the password of HTTP Basic
authentication, under any user name and any Host. The Host condition stops a web page
from reaching the server by DNS rebinding. The peer address is the socket's:
``northplan-serve`` runs uvicorn with proxy headers off. ``TOKEN`` is None unless
``northplan-serve`` set it, and then only a loopback client is served. Over plain HTTP
the token and the scenario cross the network unencrypted.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import importlib.metadata
import inspect
import ipaddress
import stat
import threading
from collections.abc import Awaitable, Callable, Iterator, MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from api.schemas import OptimizeResponse, SimulationResponse
from engine.core.indexation import RoutedParameterError
from engine.mc.prepare import PreparedRun, evaluate
from engine.optimize.objective import OBJECTIVE_NAMES, select_objective
from engine.optimize.search import search
from engine.params.loader import DEFAULT_PARAMS_ROOT, ParamError
from engine.scenario.schema import Scenario
from report.editor_calls import answer
from report.run import (
    PATHS_WITH_DETERMINISTIC,
    RunRefusedError,
    open_run,
    simulate_document,
)
from report.tables import RunReport, preference

WEB_ROOT = Path(__file__).resolve().parent / "web"
EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "scenarios" / "example.yaml"
MINIMAL_PATH = Path(__file__).resolve().parents[1] / "scenarios" / "minimal.yaml"
PARAMS_ROOT: Path = DEFAULT_PARAMS_ROOT
TOKEN: str | None = None

MAX_BODY_BYTES = 1 << 20
_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]"})
_RUN_LOCK = threading.Lock()

_CHALLENGE_DETAIL = (
    "this server needs its token: send it as the password of HTTP Basic authentication, "
    "with any user name. It is the NORTHPLAN_TOKEN the server was started with or, when "
    "that was not set, the one it wrote to its log as it started."
)
_NO_TOKEN_DETAIL = (
    "this server was started without a token, so it answers only a client on its own "
    "machine that connects to localhost, 127.0.0.1 or [::1]."
)

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
    version=importlib.metadata.version("northplan"),
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


def is_loopback_address(text: str) -> bool:
    """Whether ``text`` is an IP address in a loopback range.

    Takes the text of an address; returns False for text that is not one. All of
    127.0.0.0/8, ``::1`` and an IPv4-mapped IPv6 address of 127.0.0.0/8 are loopback.
    The mapped form is tested explicitly, because some Python 3.12 releases, 3.12.3
    among them, answer False for it.
    """
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped.is_loopback
    return address.is_loopback


def _peer_is_loopback(scope: MutableMapping[str, Any]) -> bool:
    """Whether the socket's peer, ``scope["client"]``, is a loopback address.

    A missing or None client is not loopback. No forwarded header is read.
    """
    client = scope.get("client")
    return client is not None and is_loopback_address(client[0])


def _carries_token(scope: MutableMapping[str, Any]) -> bool:
    """Whether the request carries ``TOKEN`` as the password of HTTP Basic authentication.

    False when ``TOKEN`` is None, when there is not exactly one Authorization header, or
    when it is not a well-formed Basic credential. The user name is ignored; the password
    is everything after the first colon.
    """
    if TOKEN is None:
        return False
    values = [value for name, value in scope["headers"] if name == b"authorization"]
    if len(values) != 1:
        return False
    scheme, _, credentials = values[0].partition(b" ")
    if scheme.lower() != b"basic":
        return False
    try:
        decoded = base64.b64decode(credentials.strip(), validate=True)
    except binascii.Error:
        return False
    _, colon, password = decoded.partition(b":")
    return bool(colon) and hmac.compare_digest(password, TOKEN.encode("ascii"))


class _AccessMiddleware:
    """Serve an HTTP request from a loopback client naming a loopback Host, or carrying
    the token; refuse the rest.

    Pure ASGI, so that it covers every route and the static files; other scope types
    pass through. Refusals, in order: 400 for more than one Host header; 400 for a
    loopback client without a loopback Host or the token, with no challenge; 403 when
    ``TOKEN`` is None; else 401 with ``WWW-Authenticate: Basic realm="northplan"``. The
    Host port is ignored and the name is compared in lower case.
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
        refusal: JSONResponse | None = None
        if len(hosts) > 1:
            refusal = JSONResponse(
                {
                    "detail": "the request carries more than one Host header; this server "
                    "answers a request with exactly one."
                },
                status_code=400,
            )
        else:
            loopback = _peer_is_loopback(scope)
            served = (
                loopback and len(hosts) == 1 and _loopback_name(hosts[0]) in _ALLOWED_HOSTS
            ) or _carries_token(scope)
            if served:
                pass
            elif loopback:
                raw = hosts[0] if hosts else None
                refusal = JSONResponse(
                    {
                        "detail": f"host {raw!r} is not served: this server answers only "
                        "localhost, 127.0.0.1 and [::1], because it is for this machine alone."
                    },
                    status_code=400,
                )
            elif TOKEN is None:
                refusal = JSONResponse({"detail": _NO_TOKEN_DETAIL}, status_code=403)
            else:
                refusal = JSONResponse(
                    {"detail": _CHALLENGE_DETAIL},
                    status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="northplan"'},
                )
        if refusal is not None:
            await refusal(scope, receive, send)
            return
        await self.app(scope, receive, send)


app.add_middleware(_AccessMiddleware)


@contextmanager
def _parameter_errors() -> Iterator[None]:
    """A ParamError is a 400; a RoutedParameterError, an engine bug, passes unchanged."""
    try:
        yield
    except RoutedParameterError:
        raise
    except ParamError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@contextmanager
def _refusals() -> Iterator[None]:
    """A RunRefusedError is an HTTPException with its status and detail."""
    try:
        yield
    except RunRefusedError as error:
        raise HTTPException(status_code=error.status, detail=error.detail) from error


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
            detail=PATHS_WITH_DETERMINISTIC,
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


def _decode(body: bytes) -> str:
    """The body as UTF-8 text, or a 422."""
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422, detail=f"request body is not valid UTF-8: {exc}. Send it as UTF-8."
        ) from exc


def _open_run(
    body: bytes, syntax: Literal["yaml", "json"], paths: int | None, deterministic: bool
) -> tuple[Scenario, PreparedRun]:
    """Decode and load the body, then open the run; every refusal is an HTTPException."""
    text = _decode(body)
    with _refusals():
        return open_run(
            text, syntax, paths=paths, deterministic=deterministic, params_root=PARAMS_ROOT
        )


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
            text = _decode(body)
            with _refusals():
                return simulate_document(
                    text,
                    syntax,
                    policy=policy,
                    paths=paths,
                    deterministic=deterministic,
                    nominal=nominal,
                    params_root=PARAMS_ROOT,
                )
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


def _scenario_file(path: Path, name: str) -> Response:
    """The file at ``path``, as it is on disk, for the ``name`` scenario's route.

    Raises:
        HTTPException: As :func:`example` documents, with ``name`` in place of "example"
            where the detail names the scenario.
    """
    try:
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            raise HTTPException(
                status_code=404, detail=f"{path} is a directory, not a scenario file."
            )
        if not stat.S_ISREG(mode):
            raise HTTPException(status_code=404, detail=f"{path} is not a regular file.")
        content = path.read_bytes()
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(status_code=404, detail=f"no {name} scenario at {path}.") from exc
    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"cannot read the {name} scenario at {path}: {exc.strerror or exc}.",
        ) from exc
    return Response(content=content, media_type="application/yaml; charset=utf-8")


@app.get("/api/example")
def example() -> Response:
    """The file at ``EXAMPLE_PATH``, as it is on disk.

    Raises:
        HTTPException: 404 when no file is found there (ENOENT or ENOTDIR) or what is there
            is not a regular file, such as a directory or a FIFO, which is not read; 503
            when the lookup or the read fails with any other ``OSError``, such as EACCES,
            ELOOP or ENAMETOOLONG, with the operating system's reason in the detail, or the
            exception's own text when it carries none.
    """
    return _scenario_file(EXAMPLE_PATH, "example")


@app.get("/api/minimal")
def minimal() -> Response:
    """The file at ``MINIMAL_PATH``, as it is on disk; refused as :func:`example` refuses."""
    return _scenario_file(MINIMAL_PATH, "minimal")


@app.post("/api/editor/{call}")
async def editor_call(request: Request, call: str) -> Response:
    """Answer the scenario editor's call ``call`` on the JSON object in the request body.

    Runs off the event loop and does not wait for a run in progress.

    Returns:
        The status and body :func:`report.editor_calls.answer` gives, as ``application/json``.

    Raises:
        HTTPException: 422 for any query parameter; 415 for a Content-Type other than JSON in
            UTF-8; 413 for a body over ``MAX_BODY_BYTES``; 422 for a body that is not UTF-8.
    """
    _check_query(request, f"/api/editor/{call}", frozenset(), False, None)
    content_type = request.headers.get("content-type")
    try:
        syntax = _syntax(content_type)
    except HTTPException:
        syntax = None
    if syntax != "json":
        raise HTTPException(
            status_code=415,
            detail=(
                f"unsupported Content-Type {content_type!r}; "
                "send the arguments as application/json, in UTF-8."
            ),
        )
    text = _decode(await _read_body(request))
    status, body = await run_in_threadpool(lambda: answer(call, text, params_root=PARAMS_ROOT))
    return Response(content=body, status_code=status, media_type="application/json")


# Mounted last so that /health and /api/* win over any file of the same name.
app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")

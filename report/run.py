# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Opening a run from scenario text and producing the simulate document.

Owns the pipeline the HTTP API and the browser entry point share: parse the scenario,
check the run's size against an optional path cap, open the run, select the policy,
evaluate it and build the document. It imports neither a web framework nor the API.

Every refusal is a :class:`RunRefusedError` carrying the HTTP status the API answers
with, 422 or 400. A :class:`~engine.core.indexation.RoutedParameterError` signals an
engine bug and passes through unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final, Literal

from pydantic import ValidationError

from engine.core.indexation import RoutedParameterError
from engine.mc.prepare import MAX_PATHS, PreparedRun, evaluate, prepare_run
from engine.params.loader import ParamError
from engine.scenario.load import ScenarioError, scenario_from_text, validation_message
from engine.scenario.schema import Scenario
from report.tables import RunReport, UnknownPolicyError, select_policy

__all__ = [
    "PATHS_WITH_DETERMINISTIC",
    "RunRefusedError",
    "open_run",
    "parameter_errors",
    "simulate_document",
]

PATHS_WITH_DETERMINISTIC: Final[str] = (
    "paths and deterministic cannot both be given; "
    "the deterministic run is always exactly one path."
)


class RunRefusedError(Exception):
    """A refused run: ``status`` is the HTTP status the API answers with, ``detail`` the text."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


@contextmanager
def parameter_errors() -> Iterator[None]:
    """A ParamError is a 400 :class:`RunRefusedError`; a RoutedParameterError passes unchanged."""
    try:
        yield
    except RoutedParameterError:
        raise
    except ParamError as error:
        raise RunRefusedError(400, str(error)) from error


def open_run(
    text: str,
    syntax: Literal["yaml", "json"],
    *,
    paths: int | None,
    deterministic: bool,
    params_root: Path,
    path_cap: int | None = None,
) -> tuple[Scenario, PreparedRun]:
    """Parse ``text`` and open the run it describes.

    Args:
        text: The scenario, in ``syntax``.
        paths: Overrides the scenario's ``n_paths`` when not None.
        deterministic: One path with the returns at their expected values.
        params_root: The directory the run's parameters are loaded from.
        path_cap: When not None, the most paths the run may ask for, checked before
            the run is opened; a deterministic run asks for one.

    Returns:
        The scenario and the prepared run.

    Raises:
        RunRefusedError: 422 for a text that does not parse or validate, a run over
            ``path_cap`` or over the engine's limits, or a scenario the load checks
            refuse; 400 for a ParamError.
    """
    try:
        scenario = scenario_from_text(text, "request body", syntax=syntax)
    except ScenarioError as error:
        raise RunRefusedError(422, str(error)) from error
    if path_cap is not None:
        asked = 1 if deterministic else paths if paths is not None else scenario.n_paths
        if asked > path_cap:
            raise RunRefusedError(
                422,
                f"the run asks for {asked:,} paths, and a run here may have at most "
                f"{path_cap:,}. Lower n_paths, or use the local install, which allows "
                f"up to {MAX_PATHS:,}.",
            )
    try:
        with parameter_errors():
            prepared = prepare_run(
                scenario, n_paths=paths, deterministic=deterministic, params_root=params_root
            )
    except (ValidationError, ScenarioError) as error:
        detail = validation_message(error) if isinstance(error, ValidationError) else str(error)
        raise RunRefusedError(422, detail) from error
    return scenario, prepared


def simulate_document(
    text: str,
    syntax: Literal["yaml", "json"],
    *,
    policy: str | None,
    paths: int | None,
    deterministic: bool,
    nominal: bool,
    params_root: Path,
    path_cap: int | None = None,
) -> dict[str, object]:
    """Run one policy on the scenario in ``text`` and return the simulate document.

    The arguments are those of :func:`open_run`, plus ``policy`` (an expanded policy
    name; None for the first) and ``nominal`` (dollars at each year end, not real).

    Returns:
        The document ``northplan simulate --out x.json`` writes, except that
        ``meta.scenario`` names the scenario alone and the notes name query parameters.

    Raises:
        RunRefusedError: As :func:`open_run`, and 422 for an unknown policy.
    """
    scenario, prepared = open_run(
        text,
        syntax,
        paths=paths,
        deterministic=deterministic,
        params_root=params_root,
        path_cap=path_cap,
    )
    try:
        spec = select_policy(prepared, scenario, policy)
    except UnknownPolicyError as error:
        raise RunRefusedError(422, str(error)) from error
    with parameter_errors():
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

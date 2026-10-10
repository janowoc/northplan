# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``northplan`` command line interface: ``simulate`` and ``optimize``.

Reads a scenario YAML file, runs it through :func:`engine.mc.prepare.prepare_run`, and writes
the results as CSV or JSON, chosen by the ``--out`` extension. The tables, and the
real-to-nominal conversion when ``--nominal`` asks for it, come from :mod:`report.tables`,
which the HTTP API shares; the engine never converts a figure to nominal.

The run opens on 1 January of the scenario's start year and closes each December, so a row
here is a year: net worth at 31 December, the year's total spending, and the tax assessed on
that year rather than the cash paid during it. The last row is the year of the latest second
death across paths.

Each table is written with ``# key: value`` header lines; in CSV the other tables of a run
go to sibling files beside ``--out`` (``.final``, ``.trace``, ``.best``); in JSON they sit
in the one document, except the trace, which goes to a ``.trace`` sibling. No file is
written until every table is computed. In the trace a NaN — the engine's
``estate_after_tax`` before the second death — is written as missing; any other non-finite
value is refused.

Exit codes: 0 on success; 2 on a scenario the engine refuses or a usage error, with the
message on stderr; 1 on a parameter that is missing or cannot be read, or a failed write.
A missing parameters directory, when ``--params`` was not given, is exit 1 with a line
saying to pass ``--params``. Any other exception propagates.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

import report.tables
from engine.core.indexation import RoutedParameterError
from engine.mc.prepare import PreparedRun, evaluate, prepare_run
from engine.mc.trace import trace_rows
from engine.optimize.objective import OBJECTIVE_NAMES, select_objective
from engine.optimize.search import search
from engine.params.loader import DEFAULT_PARAMS_ROOT, ParamError, ParamYearMissingError
from engine.scenario.load import ScenarioError, load_scenario, validation_message
from engine.scenario.schema import PolicySpec, Scenario
from report.tables import (
    Meta,
    RunReport,
    UnknownPolicyError,
    checked,
    preference,
    real_dollars,
    select_policy,
)

__all__ = ["build_parser", "main"]

_PREFERENCE_HINT = (
    "northplan: either can be given for this run with --risk-aversion or --estate-utility-shift."
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} must be at least 1")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be at least 0")
    return value


def _finite_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"{text!r} must be finite")
    return value


def _non_negative_float(text: str) -> float:
    value = _finite_float(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be at least 0")
    return value


def _positive_float(text: str) -> float:
    value = _finite_float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be above 0")
    return value


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("scenario", metavar="SCENARIO", help="scenario YAML file")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output file; its .csv or .json extension picks the format "
        "(default: <scenario name>.csv, or .optimize.csv for optimize, in the working directory)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--paths",
        type=_positive_int,
        default=None,
        help="Monte Carlo paths for this run, overriding the scenario's n_paths",
    )
    group.add_argument(
        "--deterministic",
        action="store_true",
        help="one path with zero volatility: returns compound at each asset class's arithmetic "
        "mean, and every person lives to the life table's terminal age",
    )
    parser.add_argument(
        "--nominal",
        action="store_true",
        help="convert dollar columns of the year and final tables to nominal dollars at the "
        "year end each row reports",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        metavar="DIR",
        help="parameters directory, holding one subdirectory per tax year "
        "(default: params/ beside the source tree)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        A parser with the ``simulate`` and ``optimize`` subcommands; a subcommand is required.
    """
    parser = argparse.ArgumentParser(prog="northplan", description="Canadian financial planning.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    sim = commands.add_parser("simulate", help="run one policy and write a row per year")
    _add_common(sim)
    sim.add_argument(
        "--policy",
        default=None,
        help="expanded policy name (default: the scenario's first policy)",
    )
    sim.add_argument(
        "--trace-path",
        type=_non_negative_int,
        default=None,
        metavar="N",
        help="also write the monthly record of path N to a trace file",
    )

    opt = commands.add_parser(
        "optimize", help="evaluate every expanded policy and write the evaluation table"
    )
    _add_common(opt)
    opt.add_argument(
        "--objective", required=True, choices=OBJECTIVE_NAMES, help="the score to maximise"
    )
    opt.add_argument(
        "--risk-aversion",
        type=_non_negative_float,
        default=None,
        metavar="X",
        help="override the scenario's risk_aversion for this run (finite, at least 0)",
    )
    opt.add_argument(
        "--estate-utility-shift",
        type=_positive_float,
        default=None,
        metavar="X",
        help="override the scenario's estate_utility_shift for this run (finite, above 0)",
    )
    return parser


# ---------------------------------------------------------------------------
# Headers and files
# ---------------------------------------------------------------------------


def _sibling(path: Path, tag: str) -> Path:
    return path.with_name(f"{path.stem}.{tag}{path.suffix}")


def _render_csv(meta: Meta, columns: Sequence[str], rows: Sequence[dict[str, object]]) -> str:
    buffer = io.StringIO(newline="")
    for key, value in meta.items():
        buffer.write(f"# {key}: {value}\n")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(["" if row[c] is None else row[c] for c in columns])
    return buffer.getvalue()


def _render_json(payload: object) -> str:
    return json.dumps(payload, indent=2, allow_nan=False) + "\n"


def _columns(rows: Sequence[dict[str, object]]) -> list[str]:
    return list(rows[0]) if rows else []


# ---------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------


def _prepare(loaded: Scenario, args: argparse.Namespace) -> PreparedRun:
    """``prepare_run`` with ``--params`` passed only when given."""
    if args.params is None:
        return prepare_run(loaded, n_paths=args.paths, deterministic=args.deterministic)
    return prepare_run(
        loaded, n_paths=args.paths, deterministic=args.deterministic, params_root=args.params
    )


def _simulate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[tuple[Path, str]]:
    loaded = load_scenario(args.scenario)
    out = args.out if args.out is not None else Path(f"{loaded.name}.csv")
    try:
        prepared = _prepare(loaded, args)
    except ValidationError as error:
        raise _ScenarioRefusedError(validation_message(error)) from error
    try:
        spec: PolicySpec = select_policy(prepared, loaded, args.policy)
    except UnknownPolicyError as error:
        parser.error(str(error))
    n_paths = prepared.draws.n_paths
    if args.trace_path is not None and args.trace_path >= n_paths:
        parser.error(
            f"--trace-path {args.trace_path} is outside the run's paths: "
            f"it must be in [0, {n_paths})."
        )

    run = RunReport(
        "simulate",
        f"{prepared.scenario.name} ({args.scenario})",
        prepared,
        deterministic=args.deterministic,
        nominal=args.nominal,
        nominal_option="--nominal",
    )
    result = evaluate(prepared, spec, trace_path=args.trace_path)

    as_json = out.suffix.lower() == ".json"
    if as_json:
        document = run.simulate_document(result, spec.name)
    else:
        year_meta, years, final_meta, final = run.tables(result, spec.name)

    files: list[tuple[Path, str]] = []
    trace_meta: Meta | None = None
    trace: list[dict[str, object]] = []
    if args.trace_path is not None:
        trace = checked("trace", trace_rows(result.trace), nan_is_missing=True)
        trace_meta = run.meta(
            {"policy": spec.name},
            {"dollars": real_dollars(run.start_year, args.nominal, nominal_option="--nominal")},
        )
        trace_meta = _with_trace_path(trace_meta, args.trace_path)

    if as_json:
        files.append((out, _render_json(document)))
        if trace_meta is not None:
            files.append(
                (_sibling(out, "trace"), _render_json({"meta": trace_meta, "rows": trace}))
            )
    else:
        files.append((out, _render_csv(year_meta, _columns(years), years)))
        files.append((_sibling(out, "final"), _render_csv(final_meta, _columns([final]), [final])))
        if trace_meta is not None:
            files.append((_sibling(out, "trace"), _render_csv(trace_meta, _columns(trace), trace)))
    return files


def _with_trace_path(meta: Meta, trace_path: int) -> Meta:
    """Insert ``trace_path`` after ``deterministic``, before ``dollars``."""
    out: Meta = {}
    for key, value in meta.items():
        if key == "dollars":
            out["trace_path"] = trace_path
        out[key] = value
    return out


def _optimize(args: argparse.Namespace) -> list[tuple[Path, str]]:
    loaded = load_scenario(args.scenario)
    out = args.out if args.out is not None else Path(f"{loaded.name}.optimize.csv")
    try:
        prepared = _prepare(loaded, args)
    except ValidationError as error:
        raise _ScenarioRefusedError(validation_message(error)) from error
    scenario = prepared.scenario
    risk_aversion, risk_text = preference(args.risk_aversion, scenario.risk_aversion, source="flag")
    shift, shift_text = preference(
        args.estate_utility_shift, scenario.estate_utility_shift, source="flag"
    )
    try:
        objective = select_objective(
            args.objective, risk_aversion=risk_aversion, estate_utility_shift=shift
        )
    except ValueError as error:
        raise _ScenarioRefusedError(str(error), _PREFERENCE_HINT) from error

    found = search(prepared, objective)
    best = found.best
    result = evaluate(prepared, best)
    run = RunReport(
        "optimize",
        f"{prepared.scenario.name} ({args.scenario})",
        prepared,
        deterministic=args.deterministic,
        nominal=args.nominal,
        nominal_option="--nominal",
    )

    if out.suffix.lower() == ".json":
        payload = run.optimize_document(
            found,
            result,
            objective=args.objective,
            risk_aversion_text=risk_text,
            estate_utility_shift_text=shift_text,
        )
        return [(out, _render_json(payload))]
    evaluation = checked("evaluation table", report.tables.evaluation_rows(found))
    evaluation_meta = run.evaluation_meta(args.objective, risk_text, shift_text, best.name)
    year_meta, years, final_meta, final = run.tables(result, best.name)
    return [
        (out, _render_csv(evaluation_meta, _columns(evaluation), evaluation)),
        (_sibling(out, "best"), _render_csv(year_meta, _columns(years), years)),
        (
            _sibling(out, "best.final"),
            _render_csv(final_meta, _columns([final]), [final]),
        ),
    ]


class _ScenarioRefusedError(Exception):
    """A refusal reported as exit 2: the message, then any extra stderr lines."""

    def __init__(self, message: str, *extra: str) -> None:
        super().__init__(message)
        self.extra = extra


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success. 2 on a :class:`~engine.scenario.load.ScenarioError`, a grid value the
        schema refuses, or a missing preference, after printing the message to stderr;
        1 on any other :class:`~engine.params.loader.ParamError` (a
        :class:`~engine.core.indexation.RoutedParameterError` propagates) or an ``OSError``
        writing outputs; when the error is a missing tax year, ``--params`` was not given and
        the default parameters directory does not exist, a second line says to pass
        ``--params``.

    Raises:
        RoutedParameterError: Propagated with its traceback, not reported: it signals an
            engine bug, unlike every other :class:`~engine.params.loader.ParamError`.
        SystemExit: With code 2, on a usage error (argparse's own, a bad ``--out`` extension,
            an unknown ``--policy``, or a ``--trace-path`` outside the run's paths).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.out is not None and args.out.suffix.lower() not in (".csv", ".json"):
        parser.error(f"--out {str(args.out)!r} must end in .csv or .json.")

    try:
        files = _simulate(parser, args) if args.command == "simulate" else _optimize(args)
    except (ScenarioError, _ScenarioRefusedError) as error:
        print(f"northplan: error: {error}", file=sys.stderr)
        for line in getattr(error, "extra", ()):
            print(line, file=sys.stderr)
        return 2
    except RoutedParameterError:
        raise
    except ParamError as error:
        print(f"northplan: error: {error}", file=sys.stderr)
        if (
            isinstance(error, ParamYearMissingError)
            and args.params is None
            and not DEFAULT_PARAMS_ROOT.is_dir()
        ):
            print(
                "northplan: pass --params DIR, the directory holding one subdirectory "
                "per tax year.",
                file=sys.stderr,
            )
        return 1

    try:
        for path, text in files:
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(text)
    except OSError as error:
        print(f"northplan: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``northplan`` command line interface: ``simulate`` and ``optimize``.

Reads a scenario YAML file, runs it through :func:`engine.mc.prepare.prepare_run`, and writes
the results as CSV or JSON, chosen by the ``--out`` extension. Real-to-nominal conversion, if
requested, happens here — the engine never converts a figure to nominal.

The run opens on 1 January of the scenario's start year and closes each December, so a row
here is a year: net worth at 31 December, the year's total spending, and the tax assessed on
that year rather than the cash paid during it. The last row is the year of the latest second
death across paths. Dollar percentiles in a row are over the paths with anyone alive at that
December close; ``depletion_probability`` and each ``death_probability_<id>`` are over all
paths.

Each table is written with ``# key: value`` header lines; in CSV the other tables of a run
go to sibling files beside ``--out`` (``.final``, ``.trace``, ``.best``); in JSON they sit
in the one document, except the trace, which goes to a ``.trace`` sibling. No file is
written until every table is computed. In the trace a NaN — the engine's
``estate_after_tax`` before the second death — is written as missing; any other non-finite
value is refused.

Exit codes: 0 on success; 2 on a scenario the engine refuses or a usage error, with the
message on stderr; 1 on a missing parameter or a failed write. Any other exception propagates.
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

import numpy as np
from pydantic import ValidationError

from engine.core.indexation import RoutedParameterError
from engine.mc.prepare import PreparedRun, evaluate, prepare_run
from engine.mc.simulate import SimulationResult
from engine.mc.trace import trace_rows
from engine.optimize.objective import OBJECTIVE_NAMES, select_objective
from engine.optimize.search import SearchResult, search
from engine.params.loader import ParamError
from engine.scenario.load import ScenarioError, load_scenario
from engine.scenario.schema import PolicySpec, Scenario

__all__ = ["build_parser", "evaluation_rows", "final_row", "main", "year_rows"]

_DOLLAR_FIELDS = ("net_worth", "after_tax_net_worth", "spending_achieved", "tax_assessed")
_PERCENTILES = (10, 25, 50, 75, 90)
_DETERMINISTIC_NOTE = (
    "yes: one path; returns compound at each asset class's arithmetic mean real return; "
    "every person lives to the life table's terminal age"
)
_PREFERENCE_HINT = (
    "northplan: either can be given for this run with --risk-aversion or --estate-utility-shift."
)

Cell = float | int | None
Meta = dict[str, str | int]


def year_rows(
    result: SimulationResult,
    *,
    start_year: int,
    inflation: float,
    person_ids: Sequence[str],
    nominal: bool,
) -> list[dict[str, Cell]]:
    """One row per simulated year: percentiles, probabilities and GIS exposure.

    Dollar percentiles (linear method) are over the paths with anyone alive at that
    December close, all ``None`` when no path is. ``depletion_probability`` and each
    ``death_probability_<id>`` are over all paths; ``gis_exposure`` pools living persons.

    Args:
        result: One policy's result; arrays are ``(n_years, n_paths)``.
        start_year: The scenario's start year.
        inflation: The scenario's annual inflation rate, a fraction.
        person_ids: Person ids in household order, one per ``result.death_year`` row.
        nominal: Multiply each path's dollars by ``(1 + inflation)`` raised to the years from
            January of ``start_year`` to 31 December of the row's year, before the percentile
            (L61). Real dollars are left untouched when false.

    Returns:
        Dicts with the columns ``year, paths_alive``, then ``<field>_p10`` to ``_p90`` for
        ``net_worth``, ``after_tax_net_worth``, ``spending_achieved``, ``tax_assessed`` in
        that order, then ``depletion_probability, gis_exposure``, then
        ``death_probability_<id>`` per person in ``person_ids`` order.

    Raises:
        ValueError: ``len(person_ids)`` differs from ``result.death_year.shape[0]``.
    """
    if len(person_ids) != result.death_year.shape[0]:
        raise ValueError(
            f"year_rows: {len(person_ids)} person ids for a result with "
            f"{result.death_year.shape[0]} persons."
        )
    rows: list[dict[str, Cell]] = []
    for i, year in enumerate(int(y) for y in result.years):
        alive = result.living_count[i] > 0
        row: dict[str, Cell] = {"year": year, "paths_alive": int(alive.sum())}
        factor = (1 + inflation) ** (year - start_year + 1)
        for field in _DOLLAR_FIELDS:
            values = getattr(result, field)[i][alive]
            if nominal:
                values = values * factor  # L61
            for q in _PERCENTILES:
                row[f"{field}_p{q}"] = float(np.percentile(values, q)) if alive.any() else None
        row["depletion_probability"] = float(result.depleted[i].mean())
        living = result.living_count[i].sum()
        row["gis_exposure"] = float(result.gis_band_count[i].sum() / living) if living else None
        for k, person_id in enumerate(person_ids):
            row[f"death_probability_{person_id}"] = float((result.death_year[k] == year).mean())
        rows.append(row)
    return rows


def final_row(
    result: SimulationResult, *, start_year: int, inflation: float, nominal: bool
) -> dict[str, float]:
    """The estate distribution over all paths.

    Args:
        result: One policy's result; ``estate_after_tax`` is ``(n_paths,)``.
        start_year: The scenario's start year.
        inflation: The scenario's annual inflation rate, a fraction.
        nominal: Multiply each path's estate, before the statistics, by ``(1 + inflation)``
            raised to the years from January of ``start_year`` to 31 December of that path's
            second-death year (L61). ``estate_zero_share`` is the same either way.

    Returns:
        Dict of ``estate_after_tax_p10 .. _p90``, ``estate_after_tax_mean`` and
        ``estate_zero_share``, in that order.
    """
    estate = result.estate_after_tax
    if nominal:
        second_death_year = result.death_year.max(axis=0)
        estate = estate * (1 + inflation) ** (second_death_year - start_year + 1)  # L61
    row = {f"estate_after_tax_p{q}": float(np.percentile(estate, q)) for q in _PERCENTILES}
    row["estate_after_tax_mean"] = float(np.mean(estate))
    row["estate_zero_share"] = float((result.estate_after_tax == 0.0).mean())
    return row


def evaluation_rows(search_result: SearchResult) -> list[dict[str, object]]:
    """One row per evaluated candidate, in evaluation order, always in real dollars.

    Args:
        search_result: The outcome of :func:`engine.optimize.search.search`.

    Returns:
        Dicts of ``name, score, median_estate_after_tax, success_probability, gis_exposure,
        best``, then one column per free-parameter key (the union over candidates, first-seen
        order), ``None`` where a candidate lacks the key.
    """
    keys: list[str] = []
    for report in search_result.evaluated:
        for key in report.parameters:
            if key not in keys:
                keys.append(key)
    rows: list[dict[str, object]] = []
    for report in search_result.evaluated:
        row: dict[str, object] = {
            "name": report.name,
            "score": report.score,
            "median_estate_after_tax": report.median_estate_after_tax,
            "success_probability": report.success_probability,
            "gis_exposure": report.gis_exposure,
            "best": report.name == search_result.best.name,
        }
        for key in keys:
            row[key] = report.parameters.get(key)
        rows.append(row)
    return rows


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


def _real_dollars(start_year: int, nominal: bool) -> str:
    note = "; --nominal does not apply to this table" if nominal else ""
    return f"real, January {start_year} dollars{note}"


def _year_dollars(start_year: int, inflation: float, nominal: bool) -> str:
    if not nominal:
        return f"real, January {start_year} dollars"
    return (
        f"nominal: real x (1 + {inflation!r}) ** (years from January {start_year} "
        "to 31 December of the row's year)"
    )


def _final_dollars(start_year: int, inflation: float, nominal: bool) -> str:
    if not nominal:
        return f"real, January {start_year} dollars"
    return (
        f"nominal: each path's estate x (1 + {inflation!r}) ** (years from January "
        f"{start_year} to 31 December of its second death's year)"
    )


_YEAR_STATISTICS = (
    "dollar percentiles over paths with anyone alive at the December close (paths_alive); "
    "depletion_probability and death_probability_<id> over all paths; "
    "gis_exposure over living persons"
)


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


def _checked(
    table: str, rows: Sequence[dict[str, object]], *, nan_is_missing: bool = False
) -> list[dict[str, object]]:
    """``rows`` with every non-finite float refused; with ``nan_is_missing``, a NaN becomes None.

    Raises:
        ValueError: A float that is not finite (a NaN too, unless ``nan_is_missing``); names
            the table and the column.
    """
    out: list[dict[str, object]] = []
    for row in rows:
        clean = dict(row)
        for column, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                if nan_is_missing and math.isnan(value):
                    clean[column] = None
                else:
                    raise ValueError(
                        f"the {table} has a non-finite value {value!r} in column {column!r}."
                    )
        out.append(clean)
    return out


def _columns(rows: Sequence[dict[str, object]]) -> list[str]:
    return list(rows[0]) if rows else []


# ---------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------


class _Run:
    """The invocation's fixed context, so each table's header is built the same way."""

    def __init__(
        self,
        command: str,
        label: str,
        prepared: PreparedRun,
        *,
        deterministic: bool,
        nominal: bool,
    ) -> None:
        self.command = command
        self.label = label
        self.prepared = prepared
        self.scenario = prepared.scenario
        self.deterministic = deterministic
        self.nominal = nominal
        self.start_year = self.scenario.start_year
        self.inflation = self.scenario.assumptions.inflation

    def meta(self, head: dict[str, str | int], tail: dict[str, str | int]) -> Meta:
        meta: Meta = {
            "command": self.command,
            "scenario": f"{self.scenario.name} ({self.label})",
        }
        meta.update(head)
        meta["seed"] = self.prepared.draws.seed
        meta["paths"] = self.prepared.draws.n_paths
        meta["deterministic"] = _DETERMINISTIC_NOTE if self.deterministic else "no"
        meta.update(tail)
        return meta

    def year_meta(self, policy: str) -> Meta:
        return self.meta(
            {"policy": policy},
            {
                "dollars": _year_dollars(self.start_year, self.inflation, self.nominal),
                "statistics": _YEAR_STATISTICS,
            },
        )

    def final_meta(self, policy: str) -> Meta:
        return self.meta(
            {"policy": policy},
            {
                "dollars": _final_dollars(self.start_year, self.inflation, self.nominal),
                "statistics": "over all paths",
            },
        )

    def tables(
        self, result: SimulationResult, policy: str
    ) -> tuple[Meta, list[dict[str, object]], Meta, dict[str, object]]:
        person_ids = [p.id for p in self.scenario.household.persons]
        years = year_rows(
            result,
            start_year=self.start_year,
            inflation=self.inflation,
            person_ids=person_ids,
            nominal=self.nominal,
        )
        final = final_row(
            result, start_year=self.start_year, inflation=self.inflation, nominal=self.nominal
        )
        (checked_final,) = _checked("final row", [final])
        return (
            self.year_meta(policy),
            _checked("year table", years),
            self.final_meta(policy),
            checked_final,
        )


def _policy_error_message(name: str, loaded: Scenario, expanded: Sequence[str]) -> str:
    message = f"unknown policy {name!r}; the scenario's expanded policies are {list(expanded)!r}."
    if name in [p.name for p in loaded.policies]:
        under = [n for n in expanded if n.startswith(f"{name}[")]
        message = (
            f"policy {name!r} was expanded by the grid; use one of {under!r} "
            f"(all expanded policies: {list(expanded)!r})."
        )
    return message


def _simulate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[tuple[Path, str]]:
    loaded = load_scenario(args.scenario)
    out = args.out if args.out is not None else Path(f"{loaded.name}.csv")
    try:
        prepared = prepare_run(loaded, n_paths=args.paths, deterministic=args.deterministic)
    except ValidationError as error:
        raise _ScenarioRefusedError(str(error)) from error
    names = [p.name for p in prepared.scenario.policies]
    spec: PolicySpec = prepared.scenario.policies[0]
    if args.policy is not None:
        if args.policy not in names:
            parser.error(_policy_error_message(args.policy, loaded, names))
        spec = prepared.scenario.policies[names.index(args.policy)]
    n_paths = prepared.draws.n_paths
    if args.trace_path is not None and args.trace_path >= n_paths:
        parser.error(
            f"--trace-path {args.trace_path} is outside the run's paths: "
            f"it must be in [0, {n_paths})."
        )

    run = _Run(
        "simulate",
        str(args.scenario),
        prepared,
        deterministic=args.deterministic,
        nominal=args.nominal,
    )
    result = evaluate(prepared, spec, trace_path=args.trace_path)
    year_meta, years, final_meta, final = run.tables(result, spec.name)

    files: list[tuple[Path, str]] = []
    trace_meta: Meta | None = None
    trace: list[dict[str, object]] = []
    if args.trace_path is not None:
        trace = _checked("trace", trace_rows(result.trace), nan_is_missing=True)
        trace_meta = run.meta(
            {"policy": spec.name},
            {"dollars": _real_dollars(run.start_year, args.nominal)},
        )
        trace_meta = _with_trace_path(trace_meta, args.trace_path)

    if out.suffix.lower() == ".json":
        meta = dict(year_meta)
        meta["estate_dollars"] = final_meta["dollars"]
        files.append((out, _render_json({"meta": meta, "years": years, "final": final})))
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


def _preference(flag: float | None, scenario_value: float | None) -> tuple[float | None, str]:
    if flag is not None:
        return flag, f"{flag!r} (flag)"
    if scenario_value is not None:
        return scenario_value, f"{scenario_value!r} (scenario)"
    return None, "none"


def _optimize(args: argparse.Namespace) -> list[tuple[Path, str]]:
    loaded = load_scenario(args.scenario)
    out = args.out if args.out is not None else Path(f"{loaded.name}.optimize.csv")
    try:
        prepared = prepare_run(loaded, n_paths=args.paths, deterministic=args.deterministic)
    except ValidationError as error:
        raise _ScenarioRefusedError(str(error)) from error
    scenario = prepared.scenario
    risk_aversion, risk_text = _preference(args.risk_aversion, scenario.risk_aversion)
    shift, shift_text = _preference(args.estate_utility_shift, scenario.estate_utility_shift)
    try:
        objective = select_objective(
            args.objective, risk_aversion=risk_aversion, estate_utility_shift=shift
        )
    except ValueError as error:
        raise _ScenarioRefusedError(str(error), _PREFERENCE_HINT) from error

    found = search(prepared, objective)
    best = found.best
    result = evaluate(prepared, best)
    run = _Run(
        "optimize",
        str(args.scenario),
        prepared,
        deterministic=args.deterministic,
        nominal=args.nominal,
    )
    evaluation = _checked("evaluation table", evaluation_rows(found))
    evaluation_meta = run.meta(
        {
            "objective": args.objective,
            "risk_aversion": risk_text,
            "estate_utility_shift": shift_text,
            "best": best.name,
        },
        {"dollars": _real_dollars(run.start_year, args.nominal)},
    )
    year_meta, years, final_meta, final = run.tables(result, best.name)

    if out.suffix.lower() == ".json":
        best_meta = dict(year_meta)
        best_meta["estate_dollars"] = final_meta["dollars"]
        payload = {
            "meta": evaluation_meta,
            "evaluation": evaluation,
            "best": {"meta": best_meta, "years": years, "final": final},
        }
        return [(out, _render_json(payload))]
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
        writing outputs.

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

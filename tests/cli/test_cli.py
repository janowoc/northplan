# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``cli.main`` against the committed example and late-life couple scenarios.

Runs that several tests share are made once each, in module-scoped fixtures. The oracle is
a recomputation: the statistics are rebuilt here, from ``evaluate`` and ``search`` on a
``prepare_run`` of the same scenario and path count, and compared to the written cells
with ``==``.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import cli.main
import report.tables
from cli.main import main
from engine.core.indexation import RoutedParameterError, UnroutedParameterError
from engine.mc.prepare import PreparedRun, evaluate, prepare_run
from engine.mc.simulate import SimulationResult
from engine.mc.trace import trace_rows
from engine.optimize.objective import select_objective, success_probability
from engine.optimize.search import CandidateReport, SearchResult, search
from engine.scenario import load_scenario
from engine.scenario.schema import PolicySpec
from report.tables import evaluation_rows, final_row, year_rows

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "scenarios" / "example.yaml"
START_YEAR = 2026
INFLATION = 0.02
PERCENTILES = (10, 25, 50, 75, 90)
DOLLAR_FIELDS = ("net_worth", "after_tax_net_worth", "spending_achieved", "tax_assessed")
FLAGS_LINE = (
    "northplan: either can be given for this run with --risk-aversion or --estate-utility-shift."
)
DETERMINISTIC_NOTE = (
    "yes: one path; returns compound at each asset class's arithmetic mean real return; "
    "every person lives to the life table's terminal age"
)
YEAR_COLUMNS = (
    ["year", "paths_alive"]
    + [f"{f}_p{q}" for f in DOLLAR_FIELDS for q in PERCENTILES]
    + ["depletion_probability", "gis_exposure", "death_probability_a"]
)
FINAL_COLUMNS = [f"estate_after_tax_p{q}" for q in PERCENTILES] + [
    "estate_after_tax_mean",
    "estate_zero_share",
]
REAL_DOLLARS = "real, January 2026 dollars"
NOT_APPLICABLE = "; --nominal does not apply to this table"


# =============================================================================
# Helpers
# =============================================================================


def run_main(argv: list[str]) -> int:
    """``main(argv)``, with a usage error's ``SystemExit`` turned into its code."""
    try:
        return main(argv)
    except SystemExit as exit_:
        assert isinstance(exit_.code, int)
        return exit_.code


@dataclass(frozen=True)
class Table:
    meta: dict[str, str]
    columns: list[str]
    cells: list[list[str]]

    def column(self, name: str) -> list[str]:
        index = self.columns.index(name)
        return [row[index] for row in self.cells]

    def floats(self, name: str) -> list[float | None]:
        return [None if cell == "" else float(cell) for cell in self.column(name)]


def read_table(path: Path) -> Table:
    meta: dict[str, str] = {}
    body: list[str] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.startswith("# "):
            key, _, value = line[2:].partition(": ")
            meta[key] = value
        elif line:
            body.append(line)
    rows = list(csv.reader(body))
    return Table(meta=meta, columns=rows[0], cells=rows[1:])


def run_ok(argv: list[str]) -> None:
    assert run_main(argv) == 0


@pytest.fixture(scope="module")
def prepared() -> PreparedRun:
    return prepare_run(load_scenario(EXAMPLE), n_paths=20)


@pytest.fixture(scope="module")
def specs(prepared: PreparedRun) -> tuple[PolicySpec, ...]:
    return prepared.scenario.policies


@pytest.fixture(scope="module")
def result(prepared: PreparedRun, specs: tuple[PolicySpec, ...]) -> SimulationResult:
    return evaluate(prepared, specs[0])


@pytest.fixture(scope="module")
def couple_prepared() -> PreparedRun:
    return prepare_run(load_scenario(REPO_ROOT / "scenarios" / "late_life_couple.yaml"), n_paths=20)


@pytest.fixture(scope="module")
def couple_result(couple_prepared: PreparedRun) -> SimulationResult:
    return evaluate(couple_prepared, couple_prepared.scenario.policies[0])


def expected_year_table(
    result: SimulationResult, *, nominal: bool, person_ids: tuple[str, ...] = ("a",)
) -> list[dict[str, float | int | None]]:
    """The year table recomputed from the arrays, one dict per year."""
    rows: list[dict[str, float | int | None]] = []
    for i, year in enumerate(result.years.tolist()):
        alive = result.living_count[i] > 0
        row: dict[str, float | int | None] = {"year": year, "paths_alive": int(alive.sum())}
        for field in DOLLAR_FIELDS:
            values = getattr(result, field)[i][alive]
            if nominal:
                values = values * (1 + INFLATION) ** (year - START_YEAR + 1)
            for q in PERCENTILES:
                row[f"{field}_p{q}"] = float(np.percentile(values, q)) if values.size else None
        row["depletion_probability"] = float(np.mean(result.depleted[i]))
        living = int(result.living_count[i].sum())
        row["gis_exposure"] = int(result.gis_band_count[i].sum()) / living if living else None
        for k, person_id in enumerate(person_ids):
            row[f"death_probability_{person_id}"] = float(np.mean(result.death_year[k] == year))
        rows.append(row)
    return rows


def expected_final(result: SimulationResult, *, nominal: bool) -> dict[str, float]:
    estate = result.estate_after_tax
    if nominal:
        second = result.death_year.max(axis=0)
        estate = estate * (1 + INFLATION) ** (second - START_YEAR + 1)
    row = {f"estate_after_tax_p{q}": float(np.percentile(estate, q)) for q in PERCENTILES}
    row["estate_after_tax_mean"] = float(np.mean(estate))
    row["estate_zero_share"] = float(np.mean(result.estate_after_tax == 0.0))
    return row


def assert_year_table_equals(table: Table, expected: list[dict[str, float | int | None]]) -> None:
    assert len(table.cells) == len(expected)
    for name in table.columns:
        assert table.floats(name) == [row[name] for row in expected], name


def assert_final_equals(table: Table, expected: dict[str, float]) -> None:
    assert table.columns == list(expected)
    assert len(table.cells) == 1
    for name, value in expected.items():
        assert table.floats(name) == [value], name


def mutated(tmp_path: Path, *replacements: tuple[str, str], name: str = "scenario.yaml") -> Path:
    """A copy of the example with each ``old`` replaced by ``new``; every ``old`` must exist."""
    text = EXAMPLE.read_text(encoding="utf-8")
    for old, new in replacements:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


GRID_BLOCK = "grid:\n  elections.cpp_start_age_years.a: [60, 65, 70]\n"


# =============================================================================
# simulate: the configurations, each run once
# =============================================================================


@dataclass(frozen=True)
class SimulateRun:
    out: Path
    years: Table
    final: Table
    trace: Table


def simulate(
    tmp_path_factory: pytest.TempPathFactory, label: str, *extra: str, trace: bool = True
) -> SimulateRun:
    directory = tmp_path_factory.mktemp(label)
    out = directory / "run.csv"
    argv = ["simulate", str(EXAMPLE), "--out", str(out), *extra]
    if trace:
        argv += ["--trace-path", "3"]
    run_ok(argv)
    return SimulateRun(
        out=out,
        years=read_table(out),
        final=read_table(directory / "run.final.csv"),
        trace=read_table(directory / "run.trace.csv") if trace else Table({}, [], []),
    )


@pytest.fixture(scope="module")
def sim_real(tmp_path_factory: pytest.TempPathFactory) -> SimulateRun:
    return simulate(tmp_path_factory, "real", "--paths", "20")


@pytest.fixture(scope="module")
def sim_nominal(tmp_path_factory: pytest.TempPathFactory) -> SimulateRun:
    return simulate(tmp_path_factory, "nominal", "--paths", "20", "--nominal")


@pytest.fixture(scope="module")
def sim_second_policy(
    tmp_path_factory: pytest.TempPathFactory, specs: tuple[PolicySpec, ...]
) -> SimulateRun:
    return simulate(
        tmp_path_factory, "second", "--paths", "20", "--policy", specs[1].name, trace=False
    )


@pytest.fixture(scope="module")
def sim_json(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("json")
    run_ok(["simulate", str(EXAMPLE), "--out", str(directory / "run.json"), "--paths", "20"])
    return directory


@pytest.fixture(scope="module")
def sim_json_trace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("json-trace")
    out = directory / "run.json"
    run_ok(["simulate", str(EXAMPLE), "--out", str(out), "--paths", "20", "--trace-path", "3"])
    return directory


YEAR_STATISTICS = (
    "dollar percentiles over paths with anyone alive at the December close (paths_alive); "
    "depletion_probability and death_probability_<id> over all paths; "
    "gis_exposure over living persons"
)


class TestSimulateTables:
    def test_statistics_header_is_exact_in_csv_and_json(
        self, sim_real: SimulateRun, sim_json: Path
    ) -> None:
        assert sim_real.years.meta["statistics"] == YEAR_STATISTICS
        document = json.loads((sim_json / "run.json").read_text(encoding="utf-8"))
        assert document["meta"]["statistics"] == YEAR_STATISTICS

    def test_scenario_header_records_the_argument_as_typed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(REPO_ROOT)
        out = tmp_path / "x.csv"
        run_ok(["simulate", "./scenarios/example.yaml", "--paths", "2", "--out", str(out)])
        assert read_table(out).meta["scenario"] == "example-household (./scenarios/example.yaml)"

    def test_year_table_columns_and_header_keys(self, sim_real: SimulateRun) -> None:
        assert sim_real.years.columns == YEAR_COLUMNS
        assert list(sim_real.years.meta) == [
            "command",
            "scenario",
            "policy",
            "seed",
            "paths",
            "deterministic",
            "dollars",
            "statistics",
        ]
        assert sim_real.years.meta["command"] == "simulate"
        assert sim_real.years.meta["scenario"] == f"example-household ({EXAMPLE})"
        assert sim_real.years.meta["seed"] == "42"
        assert sim_real.years.meta["paths"] == "20"
        assert sim_real.years.meta["deterministic"] == "no"
        assert sim_real.years.meta["dollars"] == REAL_DOLLARS
        assert list(sim_real.final.meta)[-2:] == ["dollars", "statistics"]
        assert sim_real.final.meta["statistics"] == "over all paths"
        assert sim_real.final.columns == FINAL_COLUMNS

    def test_every_year_cell_equals_the_recomputed_oracle(
        self, sim_real: SimulateRun, result: SimulationResult
    ) -> None:
        expected = expected_year_table(result, nominal=False)
        assert_year_table_equals(sim_real.years, expected)

        # Guards: the comparison is not between zeros or between empties.
        medians = [row["net_worth_p50"] for row in expected]
        assert any(m is not None and m != 0.0 for m in medians)
        assert len({m for m in medians if m is not None}) > 1
        assert medians[-1] is None  # nobody alive at the last row's December
        assert sim_real.years.cells[-1][sim_real.years.columns.index("net_worth_p50")] == ""
        assert any(g is not None for g in sim_real.years.floats("gis_exposure"))
        assert any(p > 0 for p in sim_real.years.floats("tax_assessed_p50") if p is not None)

    def test_last_depletion_probability_is_the_ever_depleted_share(
        self, sim_real: SimulateRun, result: SimulationResult
    ) -> None:
        last = sim_real.years.floats("depletion_probability")[-1]
        assert last == float(result.depleted.any(axis=0).mean())
        assert last == pytest.approx(1 - success_probability(result), abs=1e-12)
        assert last is not None and last > 0  # guard: some path depleted

    def test_death_probabilities_sum_to_one_and_end_at_the_last_death(
        self, sim_real: SimulateRun, result: SimulationResult
    ) -> None:
        probabilities = [p for p in sim_real.years.floats("death_probability_a") if p is not None]
        assert math.fsum(probabilities) == pytest.approx(1.0, abs=1e-12)
        assert sum(p > 0 for p in probabilities) > 1  # guard: deaths spread over years
        assert int(sim_real.years.column("year")[-1]) == int(result.death_year.max())

    def test_final_row_equals_the_oracle(
        self, sim_real: SimulateRun, result: SimulationResult
    ) -> None:
        expected = expected_final(result, nominal=False)
        assert_final_equals(sim_real.final, expected)
        # Guards: a zero share that is neither 0 nor 1, and non-zero statistics.
        assert 0.0 < expected["estate_zero_share"] < 1.0
        assert expected["estate_after_tax_p50"] > 0 and expected["estate_after_tax_mean"] > 0


class TestNominal:
    def test_year_dollars_use_the_year_end_factor_per_path(
        self, sim_nominal: SimulateRun, sim_real: SimulateRun, result: SimulationResult
    ) -> None:
        expected = expected_year_table(result, nominal=True)
        assert_year_table_equals(sim_nominal.years, expected)

        # Mutation guard: an off-by-one exponent gives a different cell somewhere.
        off_by_one = []
        for i, year in enumerate(result.years.tolist()):
            alive = result.living_count[i] > 0
            if alive.any():
                v = result.net_worth[i][alive] * (1 + INFLATION) ** (year - START_YEAR)
                off_by_one.append(float(np.percentile(v, 50)))
            else:
                off_by_one.append(None)
        cli_cells = sim_nominal.years.floats("net_worth_p50")
        assert any(c is not None and c != o for c, o in zip(cli_cells, off_by_one, strict=True))

        # Nominal differs from real in the dollar columns and in no other column.
        assert cli_cells != sim_real.years.floats("net_worth_p50")
        for name in (
            "year",
            "paths_alive",
            "depletion_probability",
            "gis_exposure",
            "death_probability_a",
        ):
            assert sim_nominal.years.column(name) == sim_real.years.column(name), name

    def test_final_row_uses_each_paths_second_death_year(
        self, sim_nominal: SimulateRun, sim_real: SimulateRun, result: SimulationResult
    ) -> None:
        second = result.death_year.max(axis=0)
        assert len(set(second.tolist())) >= 2  # guard: a single scalar factor would fail
        assert_final_equals(sim_nominal.final, expected_final(result, nominal=True))

        scalar = result.estate_after_tax * (1 + INFLATION) ** (int(second.max()) - START_YEAR + 1)
        assert sim_nominal.final.floats("estate_after_tax_p50") != [
            float(np.percentile(scalar, 50))
        ]
        assert sim_nominal.final.floats("estate_zero_share") == sim_real.final.floats(
            "estate_zero_share"
        )
        assert sim_nominal.final.floats("estate_zero_share")[0] > 0  # guard

    def test_headers_state_the_conversion(self, sim_nominal: SimulateRun) -> None:
        assert sim_nominal.years.meta["dollars"] == (
            "nominal: real x (1 + 0.02) ** (years from January 2026 to 31 December of the "
            "row's year)"
        )
        assert sim_nominal.final.meta["dollars"] == (
            "nominal: each path's estate x (1 + 0.02) ** (years from January 2026 to 31 "
            "December of its second death's year)"
        )
        assert sim_nominal.trace.meta["dollars"] == REAL_DOLLARS + NOT_APPLICABLE


class TestTrace:
    def test_trace_file_is_the_monthly_record_of_that_path(
        self, sim_real: SimulateRun, prepared: PreparedRun, specs: tuple[PolicySpec, ...]
    ) -> None:
        traced = evaluate(prepared, specs[0], trace_path=3)
        expected = trace_rows(traced.trace)
        trace = sim_real.trace

        assert trace.columns == list(expected[0])
        assert len(trace.cells) == len(traced.trace) == len(expected)
        assert trace.meta["trace_path"] == "3"
        assert list(trace.meta).index("trace_path") == list(trace.meta).index("dollars") - 1
        assert trace.meta["dollars"] == REAL_DOLLARS
        for row_cells, row in zip(trace.cells, expected, strict=True):
            assert row_cells == [
                "" if row[c] is None or row[c] != row[c] else str(row[c]) for c in trace.columns
            ]

        # A NaN is missing: estate_after_tax is empty exactly where the engine's value is NaN.
        estates = [row["estate_after_tax"] for row in expected]
        assert any(e != e for e in estates) and any(e == e for e in estates)  # both occur
        cells = trace.column("estate_after_tax")
        for cell, estate in zip(cells, estates, strict=True):
            assert cell == ("" if estate != estate else str(estate))

        # Guards: the file has empties, non-zero floats, and many months.
        assert len(expected) > 100
        assert any(cell == "" for row_cells in trace.cells for cell in row_cells)
        assert any(isinstance(v, float) and v != 0.0 for row in expected for v in row.values())

    def test_year_and_trace_files_have_no_trace_path_key_without_the_flag(
        self, sim_real: SimulateRun
    ) -> None:
        assert "trace_path" not in sim_real.years.meta
        assert "trace_path" not in sim_real.final.meta


class TestDeterministic:
    def test_two_runs_are_byte_identical(self, tmp_path: Path) -> None:
        outputs: list[dict[str, bytes]] = []
        for label in ("one", "two"):
            out = tmp_path / label / "run.csv"
            out.parent.mkdir()
            run_ok(
                [
                    "simulate",
                    str(EXAMPLE),
                    "--deterministic",
                    "--trace-path",
                    "0",
                    "--out",
                    str(out),
                ]
            )
            outputs.append({p.name: p.read_bytes() for p in sorted(out.parent.iterdir())})
        assert outputs[0] == outputs[1]
        assert set(outputs[0]) == {"run.csv", "run.final.csv", "run.trace.csv"}
        assert all(len(content) > 200 for content in outputs[0].values())  # guard

        table = read_table(tmp_path / "one" / "run.csv")
        assert table.meta["deterministic"] == DETERMINISTIC_NOTE
        assert table.meta["paths"] == "1"
        first = [table.floats(f"net_worth_p{q}")[0] for q in PERCENTILES]
        assert len(set(first)) == 1 and first[0] != 0.0  # one path: every percentile agrees


class TestPolicySelection:
    def test_default_is_the_first_expanded_policy(
        self, sim_real: SimulateRun, specs: tuple[PolicySpec, ...]
    ) -> None:
        assert sim_real.years.meta["policy"] == specs[0].name
        assert "[" in specs[0].name  # guard: an expanded name, not the written one

    def test_exact_expanded_name_selects_that_policy(
        self,
        sim_second_policy: SimulateRun,
        sim_real: SimulateRun,
        prepared: PreparedRun,
        specs: tuple[PolicySpec, ...],
    ) -> None:
        assert sim_second_policy.years.meta["policy"] == specs[1].name
        assert sim_second_policy.final.meta["policy"] == specs[1].name
        other = evaluate(prepared, specs[1])
        assert_year_table_equals(sim_second_policy.years, expected_year_table(other, nominal=False))
        assert_final_equals(sim_second_policy.final, expected_final(other, nominal=False))
        # Guard: the two policies produce different numbers.
        assert sim_second_policy.years.column("net_worth_p50") != sim_real.years.column(
            "net_worth_p50"
        )


class TestJson:
    def test_simulate_json_shape_and_values(self, sim_json: Path, sim_real: SimulateRun) -> None:
        document = json.loads((sim_json / "run.json").read_text(encoding="utf-8"))
        assert list(document) == ["meta", "years", "final"]
        meta = document["meta"]
        assert meta["estate_dollars"] == sim_real.final.meta["dollars"]
        assert type(meta["seed"]) is int and meta["seed"] == 42
        assert type(meta["paths"]) is int and meta["paths"] == 20
        for key, value in sim_real.years.meta.items():
            if key not in ("seed", "paths"):
                assert meta[key] == value, key

        years = document["years"]
        assert len(years) == len(sim_real.years.cells)
        assert all(list(row) == sim_real.years.columns for row in years)
        nulls = 0
        for column in sim_real.years.columns:
            csv_cells = sim_real.years.floats(column)
            json_cells = [row[column] for row in years]
            assert json_cells == csv_cells, column
            nulls += sum(cell is None for cell in json_cells)
        assert nulls > 0  # guard: the last row's empties arrive as null
        assert document["final"] == {c: sim_real.final.floats(c)[0] for c in sim_real.final.columns}

    def test_trace_json_shape_and_values(
        self, sim_json_trace: Path, prepared: PreparedRun, specs: tuple[PolicySpec, ...]
    ) -> None:
        sim_json = sim_json_trace
        document = json.loads((sim_json / "run.trace.json").read_text(encoding="utf-8"))
        assert list(document) == ["meta", "rows"]
        assert document["meta"]["trace_path"] == 3
        assert document["meta"]["dollars"] == REAL_DOLLARS
        expected = trace_rows(evaluate(prepared, specs[0], trace_path=3).trace)
        assert len(document["rows"]) == len(expected)
        for got, want in zip(document["rows"], expected, strict=True):
            assert list(got) == list(want)
            for column, value in want.items():
                assert got[column] == (None if value != value else value), column
        estates = [row["estate_after_tax"] for row in expected]
        assert any(e != e for e in estates) and any(e == e for e in estates)  # both occur
        assert [r["estate_after_tax"] is None for r in document["rows"]] == [
            e != e for e in estates
        ]
        assert any(v is None for row in expected for v in row.values())  # guard
        assert sorted(p.name for p in sim_json.iterdir()) == ["run.json", "run.trace.json"]


class TestDefaultOutputNames:
    def test_simulate_writes_beside_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        run_ok(["simulate", str(EXAMPLE), "--paths", "2"])
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "example-household.csv",
            "example-household.final.csv",
        ]

    def test_optimize_writes_beside_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        run_ok(["optimize", str(EXAMPLE), "--paths", "2", "--objective", "median_estate_after_tax"])
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "example-household.optimize.best.csv",
            "example-household.optimize.best.final.csv",
            "example-household.optimize.csv",
        ]


# =============================================================================
# optimize
# =============================================================================


@dataclass(frozen=True)
class OptimizeRun:
    directory: Path
    evaluation: Table
    best_years: Table
    best_final: Table


def optimize(
    tmp_path_factory: pytest.TempPathFactory, label: str, objective: str, *extra: str
) -> OptimizeRun:
    directory = tmp_path_factory.mktemp(label)
    out = directory / "opt.csv"
    run_ok(
        [
            "optimize",
            str(EXAMPLE),
            "--paths",
            "20",
            "--objective",
            objective,
            "--out",
            str(out),
            *extra,
        ]
    )
    return OptimizeRun(
        directory=directory,
        evaluation=read_table(out),
        best_years=read_table(directory / "opt.best.csv"),
        best_final=read_table(directory / "opt.best.final.csv"),
    )


CE = "certainty_equivalent_estate"


@pytest.fixture(scope="module")
def opt_file(tmp_path_factory: pytest.TempPathFactory) -> OptimizeRun:
    return optimize(tmp_path_factory, "opt-file", CE)


@pytest.fixture(scope="module")
def opt_risk_flag(tmp_path_factory: pytest.TempPathFactory) -> OptimizeRun:
    return optimize(tmp_path_factory, "opt-ra", CE, "--risk-aversion", "0.5", "--nominal")


@pytest.fixture(scope="module")
def opt_shift_flag(tmp_path_factory: pytest.TempPathFactory) -> OptimizeRun:
    return optimize(tmp_path_factory, "opt-shift", CE, "--estate-utility-shift", "10000")


def oracle_search(prepared: PreparedRun, risk_aversion: float, shift: float) -> SearchResult:
    objective = select_objective(CE, risk_aversion=risk_aversion, estate_utility_shift=shift)
    return search(prepared, objective)


def scores(run: OptimizeRun) -> list[float | None]:
    return run.evaluation.floats("score")


def assert_evaluation_equals(table: Table, found: SearchResult) -> None:
    expected = evaluation_rows(found)
    assert table.columns == list(expected[0])
    assert len(table.cells) == len(expected)
    for name in table.columns:
        assert table.column(name) == [
            "" if row[name] is None else str(row[name]) for row in expected
        ], name


class TestOptimize:
    def test_evaluation_table_and_best_tables_equal_the_oracle(
        self, opt_file: OptimizeRun, prepared: PreparedRun
    ) -> None:
        found = oracle_search(prepared, 2.0, 50000.0)
        assert_evaluation_equals(opt_file.evaluation, found)

        table = opt_file.evaluation
        assert table.columns[:6] == [
            "name",
            "score",
            "median_estate_after_tax",
            "success_probability",
            "gis_exposure",
            "best",
        ]
        flags = table.column("best")
        assert flags.count("True") == 1
        best_name = table.column("name")[flags.index("True")]
        assert best_name == table.meta["best"] == found.best.name
        assert list(table.meta) == [
            "command",
            "scenario",
            "objective",
            "risk_aversion",
            "estate_utility_shift",
            "best",
            "seed",
            "paths",
            "deterministic",
            "dollars",
        ]
        assert table.meta["objective"] == CE
        assert table.meta["dollars"] == REAL_DOLLARS

        best_result = evaluate(prepared, found.best)
        assert opt_file.best_years.meta["policy"] == found.best.name
        assert "objective" not in opt_file.best_years.meta
        assert_year_table_equals(
            opt_file.best_years, expected_year_table(best_result, nominal=False)
        )
        assert_final_equals(opt_file.best_final, expected_final(best_result, nominal=False))

        # Guards: scores differ across candidates, the best is the highest, parameters present.
        assert len(set(scores(opt_file))) == len(scores(opt_file)) == 3
        assert max(scores(opt_file)) == scores(opt_file)[flags.index("True")]
        assert any(cell != "" for cell in table.column(table.columns[6]))

    def test_risk_aversion_from_the_file(
        self, opt_file: OptimizeRun, prepared: PreparedRun
    ) -> None:
        meta = opt_file.evaluation.meta
        assert meta["risk_aversion"] == "2.0 (scenario)"
        assert meta["estate_utility_shift"] == "50000.0 (scenario)"
        found = oracle_search(prepared, 2.0, 50000.0)
        assert scores(opt_file) == [r.score for r in found.evaluated]
        assert all(s is not None and s > 0 for s in scores(opt_file))  # guard

    def test_risk_aversion_flag_overrides_the_file(
        self, opt_risk_flag: OptimizeRun, opt_file: OptimizeRun, prepared: PreparedRun
    ) -> None:
        meta = opt_risk_flag.evaluation.meta
        assert meta["risk_aversion"] == "0.5 (flag)"
        assert meta["estate_utility_shift"] == "50000.0 (scenario)"
        found = oracle_search(prepared, 0.5, 50000.0)
        assert scores(opt_risk_flag) == [r.score for r in found.evaluated]
        assert scores(opt_risk_flag) != scores(opt_file)  # guard: the override moved the score

    def test_estate_utility_shift_flag_overrides_the_file(
        self, opt_shift_flag: OptimizeRun, opt_file: OptimizeRun, prepared: PreparedRun
    ) -> None:
        meta = opt_shift_flag.evaluation.meta
        assert meta["estate_utility_shift"] == "10000.0 (flag)"
        assert meta["risk_aversion"] == "2.0 (scenario)"
        found = oracle_search(prepared, 2.0, 10000.0)
        assert scores(opt_shift_flag) == [r.score for r in found.evaluated]
        assert scores(opt_shift_flag) != scores(opt_file)  # guard

    def test_nominal_never_applies_to_the_evaluation_table(
        self, opt_risk_flag: OptimizeRun, prepared: PreparedRun
    ) -> None:
        assert opt_risk_flag.evaluation.meta["dollars"] == REAL_DOLLARS + NOT_APPLICABLE
        found = oracle_search(prepared, 0.5, 50000.0)
        assert_evaluation_equals(opt_risk_flag.evaluation, found)
        # The best policy's own tables are nominal under the flag.
        best_result = evaluate(prepared, found.best)
        assert_year_table_equals(
            opt_risk_flag.best_years, expected_year_table(best_result, nominal=True)
        )
        assert_final_equals(opt_risk_flag.best_final, expected_final(best_result, nominal=True))
        assert opt_risk_flag.best_years.meta["dollars"].startswith("nominal: ")

    def test_optimize_json_shape(self, tmp_path: Path) -> None:
        out = tmp_path / "opt.json"
        run_ok(
            [
                "optimize",
                str(EXAMPLE),
                "--paths",
                "4",
                "--objective",
                "median_estate_after_tax",
                "--out",
                str(out),
            ]
        )
        document = json.loads(out.read_text(encoding="utf-8"))
        assert list(document) == ["meta", "evaluation", "best"]
        assert list(document["best"]) == ["meta", "years", "final"]
        assert document["meta"]["risk_aversion"] == "2.0 (scenario)"
        assert type(document["meta"]["seed"]) is int
        assert document["meta"]["best"] == document["best"]["meta"]["policy"]
        assert "estate_dollars" in document["best"]["meta"]
        assert len(document["evaluation"]) == 3
        assert sum(row["best"] for row in document["evaluation"]) == 1
        assert document["best"]["years"] and document["best"]["final"]
        assert [p.name for p in tmp_path.iterdir()] == ["opt.json"]

    def test_missing_preference_exits_2_naming_the_field(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scenario = mutated(tmp_path, ("risk_aversion: 2.0\n", ""))
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        out = out_dir / "opt.csv"

        def no_search(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("search ran")

        monkeypatch.setattr(cli.main, "search", no_search)
        code = run_main(
            ["optimize", str(scenario), "--paths", "2", "--objective", CE, "--out", str(out)]
        )
        assert code == 2
        stderr = capsys.readouterr().err
        assert "Scenario.risk_aversion" in stderr
        assert "Scenario.estate_utility_shift" not in stderr
        assert stderr.splitlines()[-1] == FLAGS_LINE
        assert list(out_dir.iterdir()) == []
        assert sorted(p.name for p in tmp_path.iterdir()) == ["out", "scenario.yaml"]
        monkeypatch.undo()

        # The flag supplies it for the run, and the file stays without it.
        assert (
            run_main(
                [
                    "optimize",
                    str(scenario),
                    "--paths",
                    "2",
                    "--objective",
                    CE,
                    "--risk-aversion",
                    "1.5",
                    "--out",
                    str(tmp_path / "ok.csv"),
                ]
            )
            == 0
        )
        assert read_table(tmp_path / "ok.csv").meta["risk_aversion"] == "1.5 (flag)"


# =============================================================================
# Refusals that are not usage errors
# =============================================================================


class TestScenarioRefusals:
    def check_exit(
        self,
        argv_tail: list[str],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        code: int,
    ) -> str:
        out = tmp_path / "out.csv"
        assert run_main(["simulate", *argv_tail, "--paths", "2", "--out", str(out)]) == code
        stderr = capsys.readouterr().err
        assert stderr.startswith("northplan: error: ")
        assert sorted(p.name for p in tmp_path.iterdir() if p.name != "scenario.yaml") == []
        return stderr

    def test_missing_scenario_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stderr = self.check_exit([str(tmp_path / "absent.yaml")], tmp_path, capsys, 2)
        assert "No scenario file at" in stderr

    def test_malformed_yaml(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = tmp_path / "scenario.yaml"
        path.write_text("name: [unclosed\n", encoding="utf-8")
        stderr = self.check_exit([str(path)], tmp_path, capsys, 2)
        assert "is not valid YAML" in stderr

    def test_duplicate_key(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = tmp_path / "scenario.yaml"
        path.write_text("name: a\nname: b\n", encoding="utf-8")
        stderr = self.check_exit([str(path)], tmp_path, capsys, 2)
        assert "key 'name' appears more than once in this mapping" in stderr

    def test_schema_invalid_scenario(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = mutated(tmp_path, ("n_paths: 10000", "n_paths: 0"))
        stderr = self.check_exit([str(path)], tmp_path, capsys, 2)
        assert "is not a valid scenario" in stderr

    def test_start_age_not_allowed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = mutated(
            tmp_path,
            ("cpp_start_age_years: {a: 65}", "cpp_start_age_years: {a: 50}"),
            (GRID_BLOCK, ""),
        )
        stderr = self.check_exit([str(path)], tmp_path, capsys, 2)
        assert "50" in stderr

    def test_grid_value_the_schema_refuses(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = mutated(
            tmp_path,
            (
                "elections.cpp_start_age_years.a: [60, 65, 70]",
                "elections.rrif_conversion.fraction: [1.5]",
            ),
        )
        self.check_exit([str(path)], tmp_path, capsys, 2)

    def test_param_error_exits_1(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = mutated(tmp_path, ("start_year: 2026 ", "start_year: 2031 "))
        self.check_exit([str(path)], tmp_path, capsys, 1)

    def test_routed_parameter_error_propagates_with_its_traceback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raises(*_args: object, **_kwargs: object) -> None:
            raise RoutedParameterError("x")

        monkeypatch.setattr(cli.main, "prepare_run", raises)
        with pytest.raises(RoutedParameterError):
            main(["simulate", str(EXAMPLE), "--out", str(tmp_path / "x.csv")])
        assert list(tmp_path.iterdir()) == []

    def test_other_parameter_errors_are_one_line_exit_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def raises(*_args: object, **_kwargs: object) -> None:
            raise UnroutedParameterError("y")

        monkeypatch.setattr(cli.main, "prepare_run", raises)
        assert run_main(["simulate", str(EXAMPLE), "--out", str(tmp_path / "x.csv")]) == 1
        stderr = capsys.readouterr().err
        assert stderr.splitlines() == ["northplan: error: y"]
        assert "Traceback" not in stderr
        assert list(tmp_path.iterdir()) == []

    def test_oserror_on_write_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "no-such-directory" / "run.csv"
        assert run_main(["simulate", str(EXAMPLE), "--paths", "2", "--out", str(out)]) == 1
        assert capsys.readouterr().err.startswith("northplan: error: ")
        assert list(tmp_path.iterdir()) == []

    def test_unexpected_exception_propagates_and_nothing_is_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(prepared: PreparedRun, spec: PolicySpec, *, trace_path: int | None) -> None:
            raise RuntimeError(f"boom {prepared.draws.n_paths} {spec.name} {trace_path}")

        monkeypatch.setattr(cli.main, "evaluate", boom)
        out = tmp_path / "run.csv"
        with pytest.raises(RuntimeError, match="boom"):
            main(["simulate", str(EXAMPLE), "--paths", "2", "--out", str(out)])
        assert list(tmp_path.iterdir()) == []


# =============================================================================
# Non-finite values
# =============================================================================


class TestNonFinite:
    def refused(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        module: ModuleType,
        name: str,
        replacement: object,
        argv: list[str],
        table: str,
        column: str,
    ) -> None:
        monkeypatch.setattr(module, name, replacement)
        out = tmp_path / "out.csv"
        with pytest.raises(ValueError, match=f"the {table} .*{column}"):
            main([*argv, "--deterministic", "--out", str(out)])
        assert list(tmp_path.iterdir()) == []

    def test_infinite_trace_value_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self.refused(
            monkeypatch,
            tmp_path,
            cli.main,
            "trace_rows",
            lambda _trace: [{"context.month": 1, "estate_after_tax": math.inf}],
            ["simulate", str(EXAMPLE), "--trace-path", "0"],
            "trace",
            "estate_after_tax",
        )

    def test_nan_in_the_year_table_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self.refused(
            monkeypatch,
            tmp_path,
            report.tables,
            "year_rows",
            lambda *_args, **_kwargs: [{"year": 2026, "net_worth_p50": math.nan}],
            ["simulate", str(EXAMPLE)],
            "year table",
            "net_worth_p50",
        )

    def test_nan_in_the_final_row_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self.refused(
            monkeypatch,
            tmp_path,
            report.tables,
            "final_row",
            lambda *_args, **_kwargs: {"estate_after_tax_mean": math.nan},
            ["simulate", str(EXAMPLE)],
            "final row",
            "estate_after_tax_mean",
        )

    def test_nan_in_the_evaluation_table_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self.refused(
            monkeypatch,
            tmp_path,
            report.tables,
            "evaluation_rows",
            lambda _found: [{"name": "x", "score": math.nan}],
            ["optimize", str(EXAMPLE), "--objective", "median_estate_after_tax"],
            "evaluation table",
            "score",
        )

    def test_optimize_computes_the_best_tables_before_writing_the_evaluation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The evaluation table is complete and finite; the best policy's year table is
        # not, so a file written as soon as the evaluation is ready would exist here.
        self.refused(
            monkeypatch,
            tmp_path,
            report.tables,
            "year_rows",
            lambda *_args, **_kwargs: [{"year": 2026, "net_worth_p50": math.nan}],
            ["optimize", str(EXAMPLE), "--objective", "median_estate_after_tax"],
            "year table",
            "net_worth_p50",
        )


# =============================================================================
# Usage errors
# =============================================================================


def usage_cases() -> list[tuple[str, list[str]]]:
    simulate_ = ["simulate", str(EXAMPLE)]
    optimize_ = ["optimize", str(EXAMPLE), "--objective", "median_estate_after_tax"]
    return [
        ("paths-with-deterministic", [*simulate_, "--paths", "5", "--deterministic"]),
        ("zero-paths", [*simulate_, "--paths", "0"]),
        ("txt-extension", [*simulate_, "--out", "x.txt"]),
        ("trace-path-past-the-last-path", [*simulate_, "--paths", "20", "--trace-path", "20"]),
        ("trace-path-1-when-deterministic", [*simulate_, "--deterministic", "--trace-path", "1"]),
        ("optimize-without-objective", ["optimize", str(EXAMPLE)]),
        ("unknown-objective", ["optimize", str(EXAMPLE), "--objective", "bogus"]),
        ("risk-aversion-nan", [*optimize_, "--risk-aversion", "nan"]),
        ("risk-aversion-negative", [*optimize_, "--risk-aversion", "-1"]),
        ("estate-utility-shift-zero", [*optimize_, "--estate-utility-shift", "0"]),
        ("no-command", []),
    ]


class TestUsageErrors:
    @pytest.mark.parametrize(
        "argv", [case[1] for case in usage_cases()], ids=[case[0] for case in usage_cases()]
    )
    def test_exits_2_and_writes_nothing(
        self,
        argv: list[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert run_main(argv) == 2
        assert capsys.readouterr().err  # the reason is on stderr
        assert list(tmp_path.iterdir()) == []

    def test_unknown_policy_lists_the_expanded_names(
        self,
        capsys: pytest.CaptureFixture[str],
        specs: tuple[PolicySpec, ...],
        tmp_path: Path,
    ) -> None:
        argv = ["simulate", str(EXAMPLE), "--paths", "2", "--policy", "nope"]
        assert run_main([*argv, "--out", str(tmp_path / "x.csv")]) == 2
        stderr = capsys.readouterr().err
        assert len(specs) == 3
        assert all(spec.name in stderr for spec in specs)
        assert "expanded by the grid" not in stderr
        assert list(tmp_path.iterdir()) == []

    def test_written_policy_name_under_a_grid_lists_its_expansions(
        self,
        capsys: pytest.CaptureFixture[str],
        specs: tuple[PolicySpec, ...],
        tmp_path: Path,
    ) -> None:
        argv = ["simulate", str(EXAMPLE), "--paths", "2", "--policy", "taxable-first"]
        assert run_main([*argv, "--out", str(tmp_path / "x.csv")]) == 2
        stderr = capsys.readouterr().err
        assert "expanded by the grid" in stderr
        assert all(spec.name in stderr for spec in specs)
        assert list(tmp_path.iterdir()) == []

    def test_preference_flags_are_documented_as_overrides(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as raised:
            main(["optimize", "--help"])
        assert raised.value.code == 0
        text = " ".join(capsys.readouterr().out.split())
        assert "override the scenario's risk_aversion" in text
        assert "override the scenario's estate_utility_shift" in text


# =============================================================================
# The pieces
# =============================================================================


class TestPieces:
    def test_year_rows_refuses_a_person_id_count_that_does_not_match(
        self, result: SimulationResult
    ) -> None:
        with pytest.raises(ValueError, match="person ids"):
            year_rows(
                result,
                start_year=START_YEAR,
                inflation=INFLATION,
                person_ids=["a", "b"],
                nominal=False,
            )

    def test_year_rows_and_final_row_are_the_functions_the_files_come_from(
        self, sim_real: SimulateRun, result: SimulationResult
    ) -> None:
        rows = year_rows(
            result, start_year=START_YEAR, inflation=INFLATION, person_ids=["a"], nominal=False
        )
        assert list(rows[0]) == YEAR_COLUMNS
        assert sim_real.years.floats("net_worth_p50") == [r["net_worth_p50"] for r in rows]
        final = final_row(result, start_year=START_YEAR, inflation=INFLATION, nominal=False)
        assert list(final) == FINAL_COLUMNS

    def test_death_probability_columns_follow_household_order_on_a_couple(
        self, couple_prepared: PreparedRun, couple_result: SimulationResult
    ) -> None:
        outcome = couple_result
        ids = [person.id for person in couple_prepared.scenario.household.persons]
        assert len(ids) == 2
        rows = year_rows(
            outcome,
            start_year=couple_prepared.scenario.start_year,
            inflation=INFLATION,
            person_ids=ids,
            nominal=False,
        )
        assert [c for c in rows[0] if c.startswith("death_probability_")] == [
            f"death_probability_{i}" for i in ids
        ]
        for k, person_id in enumerate(ids):
            expected = [float((outcome.death_year[k] == y).mean()) for y in outcome.years.tolist()]
            assert [r[f"death_probability_{person_id}"] for r in rows] == expected
        # Guard: the two columns are not the same column.
        assert any(
            r[f"death_probability_{ids[0]}"] != r[f"death_probability_{ids[1]}"] for r in rows
        )

    def test_gis_exposure_is_the_pooled_share_of_living_persons(
        self, couple_prepared: PreparedRun, couple_result: SimulationResult
    ) -> None:
        # Synthetic counts, not engine output: everyone alive on every second path, so
        # every entry is at most living_count and some shares are strictly between 0 and 1.
        living = couple_result.living_count
        synthetic = dataclasses.replace(
            couple_result, gis_band_count=(living * (np.arange(20) % 2 == 0)).astype(np.int64)
        )
        assert synthetic.gis_band_count.shape == living.shape
        assert (synthetic.gis_band_count <= living).all()
        rows = year_rows(
            synthetic,
            start_year=couple_prepared.scenario.start_year,
            inflation=INFLATION,
            person_ids=[p.id for p in couple_prepared.scenario.household.persons],
            nominal=False,
        )
        shares = []
        for i, row in enumerate(rows):
            persons_alive = int(living[i].sum())
            if persons_alive:
                expected = int(synthetic.gis_band_count[i].sum()) / persons_alive
                assert row["gis_exposure"] == expected
                shares.append(expected)
            else:
                assert row["gis_exposure"] is None
        assert any(0.0 < share < 1.0 for share in shares)  # guard

        # Guard against a per-path count: in some year one path has both persons alive and
        # another has one, and the pooled value then differs from the path-count mutation.
        guard_years = [
            i
            for i in range(len(rows))
            if (living[i] == 2).any()
            and (living[i] == 1).any()
            and rows[i]["gis_exposure"]
            != int(synthetic.gis_band_count[i].sum()) / int((living[i] > 0).sum())
        ]
        assert guard_years

    def test_couple_death_columns_through_the_command_line(
        self, couple_prepared: PreparedRun, couple_result: SimulationResult, tmp_path: Path
    ) -> None:
        out = tmp_path / "couple.csv"
        run_ok(
            [
                "simulate",
                str(REPO_ROOT / "scenarios" / "late_life_couple.yaml"),
                "--paths",
                "20",
                "--out",
                str(out),
            ]
        )
        table = read_table(out)
        ids = [person.id for person in couple_prepared.scenario.household.persons]
        assert ids == ["a", "b"]
        for k, person_id in enumerate(ids):
            expected = [
                float((couple_result.death_year[k] == y).mean())
                for y in couple_result.years.tolist()
            ]
            assert table.floats(f"death_probability_{person_id}") == expected
        # Guard: the two columns differ in some row.
        assert table.column("death_probability_a") != table.column("death_probability_b")

    def test_evaluation_rows_unions_parameter_keys_in_first_seen_order(self) -> None:
        def report(name: str, parameters: dict[str, float]) -> CandidateReport:
            return CandidateReport(
                name=name,
                parameters=parameters,
                score=1.0,
                median_estate_after_tax=2.0,
                success_probability=0.5,
                gis_exposure=0.25,
            )

        synthetic = SearchResult(
            best=PolicySpec.model_construct(name="second"),
            best_score=1.0,
            evaluated=(
                report("first", {"x": 1.0}),
                report("second", {"y": 2.0, "x": 3.0}),
            ),
            seed=0,
        )
        rows = evaluation_rows(synthetic)
        assert list(rows[0])[6:] == ["x", "y"]
        assert [(r["x"], r["y"]) for r in rows] == [(1.0, None), (3.0, 2.0)]
        assert [r["best"] for r in rows] == [False, True]


class TestConsoleScript:
    def test_installed_script_runs_the_example(self, tmp_path: Path) -> None:
        script = Path(sys.executable).parent / "northplan"
        if not script.exists():
            pytest.fail(f"the console script {script} is missing; run `pip install -e .`.")
        out = tmp_path / "x.csv"
        completed = subprocess.run(
            [
                str(script),
                "simulate",
                "scenarios/example.yaml",
                "--paths",
                "20",
                "--out",
                str(out),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert out.exists()


class TestStatic:
    def test_nothing_in_cli_reads_a_start_month(self) -> None:
        files = sorted((REPO_ROOT / "cli").glob("*.py"))
        assert files
        for path in files:
            assert "start_month" not in path.read_text(encoding="utf-8"), path

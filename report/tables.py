# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A run's tables and JSON documents, from a result, in real or nominal dollars.

Owns building the year, final and evaluation tables and the documents the command
line writes with ``--out x.json`` and the HTTP API returns, from a
:class:`~engine.mc.simulate.SimulationResult` or a
:class:`~engine.optimize.search.SearchResult`. That includes the real-to-nominal
conversion when a caller asks for it (L61): the engine never converts a figure to
nominal.

A row here is a year: net worth at 31 December, the year's total spending, and the tax
assessed on that year rather than the cash paid during it. The last row is the year of
the latest second death across paths. Dollar percentiles in a row are over the paths
with anyone alive at that December close; ``depletion_probability`` and each
``death_probability_<id>`` are over all paths.

Invariants: nothing here changes a simulated number, and a non-finite float never
leaves :func:`checked`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from engine.mc.prepare import PreparedRun
from engine.mc.simulate import SimulationResult
from engine.optimize.search import SearchResult
from engine.scenario.schema import PolicySpec, Scenario

__all__ = [
    "Cell",
    "Meta",
    "RunReport",
    "UnknownPolicyError",
    "checked",
    "evaluation_rows",
    "final_row",
    "preference",
    "real_dollars",
    "select_policy",
    "year_rows",
]

_DOLLAR_FIELDS = ("net_worth", "after_tax_net_worth", "spending_achieved", "tax_assessed")
_PERCENTILES = (10, 25, 50, 75, 90)
_DETERMINISTIC_NOTE = (
    "yes: one path; returns compound at each asset class's arithmetic mean real return; "
    "every person lives to the life table's terminal age"
)
_YEAR_STATISTICS = (
    "dollar percentiles over paths with anyone alive at the December close (paths_alive); "
    "depletion_probability and death_probability_<id> over all paths; "
    "gis_exposure over living persons"
)

Cell = float | int | None
Meta = dict[str, str | int]


class UnknownPolicyError(ValueError):
    """A policy name that is not one of the scenario's expanded policies."""


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


def real_dollars(start_year: int, nominal: bool, *, nominal_option: str) -> str:
    """The ``dollars`` header of an always-real table; ``nominal_option`` spells the option."""
    note = f"; {nominal_option} does not apply to this table" if nominal else ""
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


def checked(
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


def preference(
    override: float | None, scenario_value: float | None, *, source: str
) -> tuple[float | None, str]:
    """A preference's value and the text naming its origin; ``source`` labels an override."""
    if override is not None:
        return override, f"{override!r} ({source})"
    if scenario_value is not None:
        return scenario_value, f"{scenario_value!r} (scenario)"
    return None, "none"


def select_policy(prepared: PreparedRun, loaded: Scenario, name: str | None) -> PolicySpec:
    """The expanded policy called ``name``, or the first when ``name`` is None.

    Args:
        prepared: The opened run; its scenario holds the expanded policies.
        loaded: The scenario as loaded, before the grid was expanded.
        name: An expanded policy name, or None.

    Raises:
        UnknownPolicyError: ``name`` is not an expanded policy; the message says so, and
            names the expansions when ``name`` is a grid's base name.
    """
    policies = prepared.scenario.policies
    if name is None:
        return policies[0]
    names = [p.name for p in policies]
    if name in names:
        return policies[names.index(name)]
    message = f"unknown policy {name!r}; the scenario's expanded policies are {names!r}."
    if name in [p.name for p in loaded.policies]:
        under = [n for n in names if n.startswith(f"{name}[")]
        message = (
            f"policy {name!r} was expanded by the grid; use one of {under!r} "
            f"(all expanded policies: {names!r})."
        )
    raise UnknownPolicyError(message)


class RunReport:
    """The invocation's fixed context, so each table's header is built the same way.

    ``scenario_label`` is what the ``scenario`` header says: the caller decides whether it
    names a file as well as the scenario. ``nominal_option`` is how the front end spells
    its nominal option in the notes.
    """

    def __init__(
        self,
        command: str,
        scenario_label: str,
        prepared: PreparedRun,
        *,
        deterministic: bool,
        nominal: bool,
        nominal_option: str,
    ) -> None:
        self.command = command
        self.scenario_label = scenario_label
        self.prepared = prepared
        self.scenario = prepared.scenario
        self.deterministic = deterministic
        self.nominal = nominal
        self.nominal_option = nominal_option
        self.start_year = self.scenario.start_year
        self.inflation = self.scenario.assumptions.inflation

    def meta(self, head: dict[str, str | int], tail: dict[str, str | int]) -> Meta:
        """The header: command and scenario, then ``head``, seed, paths, deterministic, ``tail``."""
        meta: Meta = {
            "command": self.command,
            "scenario": self.scenario_label,
        }
        meta.update(head)
        meta["seed"] = self.prepared.draws.seed
        meta["paths"] = self.prepared.draws.n_paths
        meta["deterministic"] = _DETERMINISTIC_NOTE if self.deterministic else "no"
        meta.update(tail)
        return meta

    def year_meta(self, policy: str) -> Meta:
        """The header of the year table for ``policy``."""
        return self.meta(
            {"policy": policy},
            {
                "dollars": _year_dollars(self.start_year, self.inflation, self.nominal),
                "statistics": _YEAR_STATISTICS,
            },
        )

    def final_meta(self, policy: str) -> Meta:
        """The header of the final row for ``policy``."""
        return self.meta(
            {"policy": policy},
            {
                "dollars": _final_dollars(self.start_year, self.inflation, self.nominal),
                "statistics": "over all paths",
            },
        )

    def evaluation_meta(
        self,
        objective: str,
        risk_aversion_text: str,
        estate_utility_shift_text: str,
        best: str,
    ) -> Meta:
        """The header of an optimize document: the objective, both preferences, the best policy."""
        return self.meta(
            {
                "objective": objective,
                "risk_aversion": risk_aversion_text,
                "estate_utility_shift": estate_utility_shift_text,
                "best": best,
            },
            {
                "dollars": real_dollars(
                    self.start_year, self.nominal, nominal_option=self.nominal_option
                )
            },
        )

    def tables(
        self, result: SimulationResult, policy: str
    ) -> tuple[Meta, list[dict[str, object]], Meta, dict[str, object]]:
        """``policy``'s result as year meta, year rows, final meta, final row, in that order.

        The rows have already passed through ``checked``.
        """
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
        (checked_final,) = checked("final row", [final])
        return (
            self.year_meta(policy),
            checked("year table", years),
            self.final_meta(policy),
            checked_final,
        )

    def simulate_document(self, result: SimulationResult, policy: str) -> dict[str, object]:
        """The JSON document of one policy's result: ``meta``, ``years`` and ``final``."""
        year_meta, years, final_meta, final = self.tables(result, policy)
        meta = dict(year_meta)
        meta["estate_dollars"] = final_meta["dollars"]
        return {"meta": meta, "years": years, "final": final}

    def optimize_document(
        self,
        found: SearchResult,
        result: SimulationResult,
        *,
        objective: str,
        risk_aversion_text: str,
        estate_utility_shift_text: str,
    ) -> dict[str, object]:
        """The JSON document of a search: ``meta``, ``evaluation`` and the ``best`` document."""
        return {
            "meta": self.evaluation_meta(
                objective, risk_aversion_text, estate_utility_shift_text, found.best.name
            ),
            "evaluation": checked("evaluation table", evaluation_rows(found)),
            "best": self.simulate_document(result, found.best.name),
        }

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Response models for the HTTP API.

The request has no model here: its body is the scenario itself,
:class:`engine.scenario.schema.Scenario`, as JSON or as YAML, and the run's options
travel in the query string. A scenario states no horizon; every run goes to the
second death.

The response speaks in **years** even though the engine steps in months. A row per
year is what a plan is discussed in, and a monthly response would be twelve times
the payload for detail nobody reads off a chart. A year has a row if and only if
its December close ran; a run's draws cover whole years, so every death year
closes, and the last row is the year of the latest second death across paths.

These models mirror, column for column, the tables :mod:`report.tables` builds and
the command line writes with ``--out x.json``. Columns named for a person
(``death_probability_<id>``) or for a candidate's free parameter are extra fields.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SimulationMeta(BaseModel):
    """The header of a simulation document: the run's context and what its numbers mean."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: str
    scenario: str
    policy: str
    seed: int
    paths: int
    deterministic: str
    dollars: str
    statistics: str
    estate_dollars: str


class YearRow(BaseModel):
    """One simulated year, summarised across paths.

    A year, aggregated from twelve simulated months: net worth is the position at
    31 December, spending is the year's total, and the tax figure is what was
    assessed on the year, not the cash paid during it. A percentile is ``None`` in a
    year no path has anyone alive. ``death_probability_<id>``, one per person, arrives
    as an extra field (``extra="allow"``) after the declared ones.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    year: int
    paths_alive: int
    net_worth_p10: float | None
    net_worth_p25: float | None
    net_worth_p50: float | None
    net_worth_p75: float | None
    net_worth_p90: float | None
    after_tax_net_worth_p10: float | None
    after_tax_net_worth_p25: float | None
    after_tax_net_worth_p50: float | None
    after_tax_net_worth_p75: float | None
    after_tax_net_worth_p90: float | None
    spending_achieved_p10: float | None
    spending_achieved_p25: float | None
    spending_achieved_p50: float | None
    spending_achieved_p75: float | None
    spending_achieved_p90: float | None
    tax_assessed_p10: float | None
    tax_assessed_p25: float | None
    tax_assessed_p50: float | None
    tax_assessed_p75: float | None
    tax_assessed_p90: float | None
    depletion_probability: float
    gis_exposure: float | None


class FinalRow(BaseModel):
    """The after-tax estate distribution over all paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    estate_after_tax_p10: float
    estate_after_tax_p25: float
    estate_after_tax_p50: float
    estate_after_tax_p75: float
    estate_after_tax_p90: float
    estate_after_tax_mean: float
    estate_zero_share: float


class SimulationResponse(BaseModel):
    """One policy's result: header, a row per year, and the estate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    meta: SimulationMeta
    years: list[YearRow]
    final: FinalRow


class OptimizeMeta(BaseModel):
    """The header of an optimization document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: str
    scenario: str
    objective: str
    risk_aversion: str
    estate_utility_shift: str
    best: str
    seed: int
    paths: int
    deterministic: str
    dollars: str


class EvaluationRow(BaseModel):
    """One evaluated candidate policy, always in real dollars.

    Each candidate's free parameters arrive as extra fields (``extra="allow"``), ``None``
    where a candidate lacks one.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    name: str
    score: float
    median_estate_after_tax: float
    success_probability: float
    gis_exposure: float
    best: bool


class OptimizeResponse(BaseModel):
    """The evaluation of every candidate, and the best one's full result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    meta: OptimizeMeta
    evaluation: list[EvaluationRow]
    best: SimulationResponse

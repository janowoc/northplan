"""Request and response models for the HTTP API.

Pydantic models, distinct from the engine's frozen dataclasses. Keeping them
separate means a wire-format change cannot quietly alter engine semantics, and
the engine stays importable without FastAPI.

The wire format speaks in **years** even though the engine steps in months. A
scenario states a horizon in years, and results come back a row per year. That
is what a plan is discussed in, and a monthly response for a long horizon is
twelve times the payload for detail nobody reads off a chart. The engine
expands the horizon to months internally and aggregates back at each year end.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScenarioRequest(BaseModel):
    """A scenario to simulate.

    The same shape as a scenario YAML file under ``scenarios/``. Regression
    fixtures and real user scenarios use one schema, so a fixture can be POSTed
    and a POSTed scenario can be saved as a fixture.
    """

    name: str = Field(description="Scenario name, used for the export filename.")
    n_paths: int = Field(description="Monte Carlo path count.")
    n_years: int = Field(description="Horizon in years; the engine steps 12x this in months.")
    seed: int = Field(description="Seed for the common random numbers.")
    base_year: int = Field(description="Tax year and the base year for real dollars.")
    start_month: int = Field(
        description=(
            "Month the simulation opens in, 1-12. A scenario that starts in "
            "September starts in September; its first tax year is a short one."
        )
    )


class YearRow(BaseModel):
    """One simulated year, summarised across paths.

    A year, aggregated from twelve simulated months: net worth is the position
    at 31 December, spending is the year's total, and the tax figure is what
    was assessed on the year — not what was paid in cash during it, which is
    mostly the prior year's liability settled in the filing month.

    Amounts are real dollars unless ``nominal`` was requested, in which case
    they have been converted on the way out of the API.
    """

    year: int
    net_worth_p10: float
    net_worth_p50: float
    net_worth_p90: float
    spending_p50: float
    tax_assessed_p50: float
    depletion_probability: float


class SimulationResponse(BaseModel):
    """Result of running one scenario."""

    scenario: str
    seed: int
    real_dollars: bool = Field(
        description="True if amounts are real dollars in the scenario's base year."
    )
    rows: list[YearRow]

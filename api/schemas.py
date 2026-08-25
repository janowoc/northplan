"""Request and response models for the HTTP API.

Pydantic models, distinct from the engine's frozen dataclasses. Keeping them
separate means a wire-format change cannot quietly alter engine semantics, and
the engine stays importable without FastAPI.
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
    n_years: int = Field(description="Years to simulate.")
    seed: int = Field(description="Seed for the common random numbers.")
    base_year: int = Field(description="Tax year and the base year for real dollars.")


class YearRow(BaseModel):
    """One simulated year, summarised across paths.

    Amounts are real dollars unless ``nominal`` was requested, in which case
    they have been converted on the way out of the API.
    """

    year: int
    net_worth_p10: float
    net_worth_p50: float
    net_worth_p90: float
    spending_p50: float
    tax_paid_p50: float
    depletion_probability: float


class SimulationResponse(BaseModel):
    """Result of running one scenario."""

    scenario: str
    seed: int
    real_dollars: bool = Field(
        description="True if amounts are real dollars in the scenario's base year."
    )
    rows: list[YearRow]

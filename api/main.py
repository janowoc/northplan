"""FastAPI application: one container, API plus static files.

Endpoints are deliberately few. Simulation and optimization endpoints return
501 until the engine behind them is implemented and verified — an endpoint that
returns a plausible number before the tax layer is verified is worse than one
that returns nothing.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from api.schemas import ScenarioRequest, SimulationResponse

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(
    title="northplan",
    description="Canadian household financial planning engine",
    version="0.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check. Does not touch the engine."""
    return {"status": "ok"}


@app.post("/api/simulate", response_model=SimulationResponse)
def simulate(request: ScenarioRequest) -> SimulationResponse:
    """Run one scenario under a fixed policy and return the distribution.

    Raises:
        HTTPException: 501 until the engine is implemented.
    """
    raise HTTPException(status_code=501, detail="Engine not implemented.")


@app.post("/api/optimize", response_model=SimulationResponse)
def optimize(request: ScenarioRequest) -> SimulationResponse:
    """Search policy space for a scenario and return the best policy's result.

    Raises:
        HTTPException: 501 until the engine is implemented.
    """
    raise HTTPException(status_code=501, detail="Engine not implemented.")


# Mounted last so that /health and /api/* win over any file of the same name.
app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")

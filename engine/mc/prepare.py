# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Opening a run: one function from a scenario to checked, shared inputs.

:func:`prepare_run` is the one entry point that turns a
:class:`~engine.scenario.schema.Scenario` into a :class:`PreparedRun`: it
expands the grid, loads the parameter year, runs the load-time guards
(:func:`~engine.scenario.start_ages.check_start_ages`,
:func:`~engine.scenario.lifespan.check_lifespan`) on the expansion, checks the
parameter year's shape -- ``resp``'s reach-back, via
:func:`~engine.accounts.resp.governing_income_year_offset` -- and builds the
market inputs and the common random numbers every candidate policy shares.
:func:`evaluate` then runs one candidate spec against a :class:`PreparedRun`.

Invariant: every spec :func:`evaluate` runs has passed the guards, because it
must be one of ``prepared.scenario.policies`` -- the grid-expanded, guard-passed
scenario carried on ``PreparedRun``. A spec taken from the scenario as written,
before expansion, is refused.

The lower layers this module composes --
:func:`engine.core.build.build_initial_state`,
:func:`engine.core.build.draw_deaths`, and :func:`engine.mc.simulate.run` --
stay callable directly; that is what the unit tests below this layer use, and
what this module is a convenience over, not a replacement for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from engine.accounts import resp
from engine.core.build import (
    build_deterministic_draws,
    build_draws,
    build_initial_state,
    build_market_inputs,
    draw_deaths,
    with_death_months,
)
from engine.core.indexation import RealParamYear, real_year
from engine.mc.market import MarketInputs
from engine.mc.returns import RandomDraws
from engine.mc.simulate import SimulationResult, run
from engine.params.loader import DEFAULT_PARAMS_ROOT, ParamYear, load_year
from engine.policy.build import build_policy, expand_grid
from engine.scenario.lifespan import check_lifespan
from engine.scenario.schema import PolicySpec, Scenario
from engine.scenario.start_ages import check_start_ages

__all__ = ["PreparedRun", "evaluate", "prepare_run"]


@dataclass(frozen=True, slots=True)
class PreparedRun:
    """The checked, shared inputs one or more policies can be evaluated against.

    Attributes:
        scenario: The grid-expanded scenario (``grid`` empty); every policy
            in it has passed the load-time guards.
        params: The parameter year matching ``scenario.start_year``.
        real_params: ``params`` in the scenario's real-dollar view.
        market: The scenario's capital-market assumptions as arrays, from
            :func:`~engine.core.build.build_market_inputs`.
        draws: Common random numbers, shared unchanged across every
            candidate policy.

    Built by :func:`prepare_run`, which runs the guards first; a hand-built
    ``PreparedRun`` skips them.
    """

    scenario: Scenario
    params: ParamYear
    real_params: RealParamYear
    market: MarketInputs
    draws: RandomDraws


def prepare_run(
    scenario: Scenario,
    *,
    n_paths: int | None = None,
    deterministic: bool = False,
    params_root: Path | str = DEFAULT_PARAMS_ROOT,
) -> PreparedRun:
    """Expand, check, and build the shared inputs for a scenario's run.

    The guards run on the *expanded* scenario, so a grid value is checked
    like a written one.

    Args:
        scenario: The scenario to prepare, as loaded (grid may be non-empty).
        n_paths: Monte Carlo paths; ``None`` uses ``scenario.n_paths``. Must
            not be given when ``deterministic`` is true.
        deterministic: Build zero-volatility, single-path draws instead of
            Monte Carlo draws.
        params_root: Root directory ``load_year`` loads under.

    Returns:
        A :class:`PreparedRun` ready for :func:`evaluate`.

    Raises:
        ValueError: ``deterministic`` and ``n_paths`` both given; whatever
            :func:`~engine.mc.returns.generate` refuses; or the RESP offset
            check's refusal.
        pydantic.ValidationError: A ``ValueError`` subclass, raised by
            :func:`~engine.policy.build.expand_grid` for a grid value the
            schema refuses.
        engine.params.loader.ParamError: A subclass, from ``load_year``.
        engine.scenario.start_ages.StartAgeNotAllowedError: A CPP, OAS, or
            RRIF conversion age the parameter year does not allow, including
            a CPP or OAS pension in pay for someone too young to receive it.
        engine.scenario.lifespan.LifespanNotRepresentableError: A person or
            household the life table cannot represent.
    """
    if deterministic and n_paths is not None:
        raise ValueError(
            f"prepare_run: deterministic=True and n_paths={n_paths!r} were both given; "
            "the deterministic run is always exactly one path."
        )

    expanded = expand_grid(scenario)
    params = load_year(expanded.start_year, params_root)
    check_start_ages(expanded, params)
    check_lifespan(expanded, params)
    real_params = real_year(params, expanded.assumptions.inflation)
    resp.governing_income_year_offset(real_params.resp)
    market = build_market_inputs(expanded.assumptions)
    mortality = params["mortality"]
    if deterministic:
        draws = build_deterministic_draws(expanded, market, mortality)
    else:
        draws = build_draws(
            expanded,
            market,
            n_paths if n_paths is not None else expanded.n_paths,
            mortality,
        )

    return PreparedRun(
        scenario=expanded, params=params, real_params=real_params, market=market, draws=draws
    )


def evaluate(
    prepared: PreparedRun,
    spec: PolicySpec,
    *,
    trace_path: int | None = None,
    death_months: Sequence[int | None] | None = None,
) -> SimulationResult:
    """Run one candidate policy against a :class:`PreparedRun`'s shared inputs.

    ``spec`` is matched to ``prepared.scenario.policies`` by value (pydantic
    equality), so an equal copy is accepted; but a spec taken from the
    scenario *as written*, before grid expansion, is refused whenever the
    scenario carries a grid, since expansion rewrites each candidate's name.

    Args:
        prepared: The checked, shared inputs from :func:`prepare_run`.
        spec: The candidate policy to evaluate; must equal one of
            ``prepared.scenario.policies``.
        trace_path: Forwarded to :func:`~engine.mc.simulate.run`.
        death_months: If given, forwarded to
            :func:`~engine.core.build.with_death_months` after
            :func:`~engine.core.build.draw_deaths`.

    Returns:
        The :class:`~engine.mc.simulate.SimulationResult` for ``spec``.

    Raises:
        ValueError: ``spec`` is not one of ``prepared.scenario.policies``
            (names ``spec.name`` and the expanded policy names); or whatever
            :func:`~engine.core.build.with_death_months` refuses about
            ``death_months``.
    """
    if spec not in prepared.scenario.policies:
        names = [policy.name for policy in prepared.scenario.policies]
        raise ValueError(
            f"evaluate: spec {spec.name!r} is not one of the expanded scenario's "
            f"policies {names!r}. A spec taken from the scenario as written, before "
            "expand_grid, is refused whenever the scenario carries a grid."
        )

    state = build_initial_state(prepared.scenario, prepared.draws.n_paths, spec)
    state = draw_deaths(state, prepared.draws, prepared.params["mortality"])
    if death_months is not None:
        state = with_death_months(state, death_months)
    policy = build_policy(spec, prepared.scenario.household)
    return run(
        state, policy, prepared.draws, prepared.market, prepared.real_params, trace_path=trace_path
    )

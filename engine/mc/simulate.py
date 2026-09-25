# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The path loop: run every month for all paths, under one policy.

This calls ``engine.core.step.advance_month_traced`` in a loop over months. It
contains no financial logic of its own and must never grow any.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from engine.core.context import MonthRecord
from engine.core.indexation import RealParamYear
from engine.core.state import DEATH_NOT_DRAWN, HouseholdState
from engine.core.step import advance_month_traced
from engine.mc.market import MarketInputs
from engine.mc.returns import RandomDraws
from engine.policy.base import Policy


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Per-year, per-path output of one policy evaluation.

    A row exists **if and only if** ``close_year`` ran for that year: ``n_years`` is
    the count of December closes the run reached, not a count derived from the
    calendar. The trailing months after the last December a run stops in are still
    simulated — ``run`` steps every one of ``draws.n_months`` — but produce no row,
    since the year they belong to never closes.

    All dollar amounts are real; conversion to nominal happens at display, in
    ``api/`` or ``cli/``, never here.

    Attributes:
        years: Calendar years simulated, ``(n_years,)``.
        net_worth: Real household net worth at each 31 December,
            ``(n_years, n_paths)``.
        spending_achieved: Real after-tax spending achieved over each year, summed
            from the twelve months, ``(n_years, n_paths)``.
        tax_assessed: Real household tax *assessed* on each year's income,
            ``(n_years, n_paths)`` -- not paid; the cash leaves in the
            following year's filing month, reflected in ``net_worth``.
        depleted: Whether the household ran out of money by each year end,
            ``(n_years, n_paths)``. Monotone in year once true; detected in
            the month it happens, reported at the year that contains it.
        seed: The seed of the draws used, for reproducibility.
        trace: The full monthly record for a single traced path — one
            :class:`~engine.core.context.MonthRecord` per month simulated, every
            array sliced to that one path — or ``()`` when ``run`` was not asked to
            trace one. See ``trace_path`` on :func:`run`.
    """

    years: NDArray[np.int64]
    net_worth: NDArray[np.float64]
    spending_achieved: NDArray[np.float64]
    tax_assessed: NDArray[np.float64]
    depleted: NDArray[np.bool_]
    seed: int
    trace: tuple[MonthRecord, ...] = ()


def run(
    initial_state: HouseholdState,
    policy: Policy,
    draws: RandomDraws,
    market: MarketInputs,
    real_params: RealParamYear,
    trace_path: int | None = None,
) -> SimulationResult:
    """Simulate every path under one policy, one month at a time.

    The loop is over ``draws.n_months`` months, a count the caller derived
    from the longest survival curve in the household
    (:func:`engine.core.build.build_draws`). The run opens on 1 January of
    the scenario's start year and has no separate horizon: it runs to the
    second death (``docs/limitations.md`` L4, L10).

    This is the only loop over time in the repository. It calls
    :func:`engine.core.step.advance_month_traced` once per month and appends a row to
    :class:`SimulationResult` **only** when that month's step ran ``close_year`` (see
    that class's docstring): ``n_years`` is the count of December closes, not a
    count derived from the calendar, and the months after the last one a run
    reaches are still stepped but produce no row. ``run`` contains no financial
    logic of its own.

    Args:
        initial_state: Opening state for the first simulated month.
        policy: The policy being evaluated.
        draws: Common random numbers, monthly, generated once and shared across
            every policy. Never regenerate inside this function.
        market: The asset-class allocation and yield mix each account's
            balance grows under.
        real_params: Parameters for the tax year the run opens in, in the
            scenario's real-dollar view.
        trace_path: If given, the index of a single path (``0 <= trace_path <
            n_paths``) to record a full monthly trace for, returned as
            ``SimulationResult.trace`` — one
            :class:`~engine.core.context.MonthRecord` per month simulated, every
            array sliced to that path. ``None`` (the default) traces nothing,
            and ``trace`` comes back empty. #36 extends what the record holds;
            this issue only wires the trace through.

    Returns:
        A :class:`SimulationResult`, recorded per year.

    Raises:
        ValueError: A state whose ``death_month_index`` is still
            :data:`engine.core.state.DEATH_NOT_DRAWN` on any person is
            refused — :func:`engine.core.build.draw_deaths` runs before
            ``run``, and a half-built opening state must fail loudly. Also
            refused: ``draws.n_paths`` disagreeing with ``initial_state.n_paths``,
            ``draws.n_months`` less than or equal to the largest
            ``death_month_index`` in the state (the draws are too short to reach
            the last death), ``policy.elections()`` disagreeing with
            ``initial_state.elections`` (the state was built for a different
            policy), ``draws.real_returns.shape[1]`` disagreeing with
            ``len(market.asset_class_names)``, and ``trace_path`` outside
            ``[0, n_paths)``.
    """
    for person in initial_state.persons:
        if np.any(person.death_month_index == DEATH_NOT_DRAWN):
            raise ValueError(
                f"person {person.person_id!r}: death_month_index is still "
                "DEATH_NOT_DRAWN on at least one path. engine.core.build.draw_deaths "
                "must run before simulate.run; a half-built opening state must fail "
                "loudly rather than simulate a household that never dies."
            )

    if draws.n_paths != initial_state.n_paths:
        raise ValueError(
            f"draws.n_paths ({draws.n_paths!r}) != initial_state.n_paths "
            f"({initial_state.n_paths!r})."
        )

    largest_death = max(int(np.max(person.death_month_index)) for person in initial_state.persons)
    if draws.n_months <= largest_death:
        raise ValueError(
            f"draws.n_months ({draws.n_months!r}) <= the largest death_month_index "
            f"in the state ({largest_death!r}); the draws are too short to reach the "
            "last death."
        )

    if policy.elections() != initial_state.elections:
        raise ValueError(
            "policy.elections() != initial_state.elections; initial_state was built "
            "for a different policy's elections (engine.core.build.build_initial_state "
            "is the only writer of Elections)."
        )

    if draws.real_returns.shape[1] != len(market.asset_class_names):
        raise ValueError(
            f"draws.real_returns.shape[1] ({draws.real_returns.shape[1]!r}) != "
            f"len(market.asset_class_names) ({len(market.asset_class_names)!r})."
        )

    if trace_path is not None and not 0 <= trace_path < initial_state.n_paths:
        raise ValueError(
            f"trace_path ({trace_path!r}) is out of range [0, {initial_state.n_paths})."
        )

    state = initial_state
    years: list[int] = []
    net_worth: list[NDArray[np.float64]] = []
    spending_achieved: list[NDArray[np.float64]] = []
    tax_assessed: list[NDArray[np.float64]] = []
    depleted: list[NDArray[np.bool_]] = []
    trace: list[MonthRecord] = []

    for m in range(draws.n_months):
        years_closed_before = len(state.history)
        state, record = advance_month_traced(
            state, draws.real_returns[m], policy, market, real_params
        )
        if trace_path is not None:
            trace.append(record.at_path(trace_path))
        if len(state.history) > years_closed_before:
            year_record = state.history[-1]
            years.append(year_record.year)
            net_worth.append(year_record.net_worth)
            spending_achieved.append(year_record.spending)
            tax_assessed.append(year_record.tax_assessed)
            depleted.append(year_record.depleted)

    n_paths = initial_state.n_paths
    if years:
        years_arr = np.array(years, dtype=np.int64)
        net_worth_arr = np.stack(net_worth).astype(np.float64)
        spending_achieved_arr = np.stack(spending_achieved).astype(np.float64)
        tax_assessed_arr = np.stack(tax_assessed).astype(np.float64)
        depleted_arr = np.stack(depleted).astype(np.bool_)
    else:
        years_arr = np.zeros((0,), dtype=np.int64)
        net_worth_arr = np.zeros((0, n_paths), dtype=np.float64)
        spending_achieved_arr = np.zeros((0, n_paths), dtype=np.float64)
        tax_assessed_arr = np.zeros((0, n_paths), dtype=np.float64)
        depleted_arr = np.zeros((0, n_paths), dtype=np.bool_)

    return SimulationResult(
        years=years_arr,
        net_worth=net_worth_arr,
        spending_achieved=spending_achieved_arr,
        tax_assessed=tax_assessed_arr,
        depleted=depleted_arr,
        seed=draws.seed,
        trace=tuple(trace),
    )

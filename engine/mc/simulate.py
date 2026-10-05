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
    calendar. ``run`` stops at the first December close reached at or after every
    path's second death, or when ``draws.n_months`` is exhausted, whichever comes
    first (see :func:`run`). Draws from :mod:`engine.core.build` are sized to whole years, so with
    them the close always comes first and every death year has a row; hand-built draws that end
    before a death year's December leave that year without one.

    All dollar amounts are real; conversion to nominal happens at display, in
    ``report/``, never here.

    Attributes:
        years: Calendar years simulated, ``(n_years,)``.
        net_worth: Real household net worth at each 31 December,
            ``(n_years, n_paths)``.
        after_tax_net_worth: Real household after-tax net worth at each 31 December,
            ``(n_years, n_paths)`` -- ``YearRecord.after_tax_net_worth``, the liquidation
            value as if every person died that day with no spousal rollover; never negative,
            and ``0`` on a path already finished.
        spending_achieved: Real after-tax spending achieved over each year, summed
            from the twelve months, ``(n_years, n_paths)``.
        tax_assessed: Real household tax *assessed* on each year's income,
            ``(n_years, n_paths)`` -- not paid; the cash leaves in the
            following year's filing month, reflected in ``net_worth``. December
            assessments only: the terminal return's tax at the second death is not
            included here, only in the trace
            (``engine.core.context.MonthContext.terminal_assessment``).
        depleted: Whether the household ran out of money by each year end,
            ``(n_years, n_paths)``. Monotone in year once true; detected in
            the month it happens, reported at the year that contains it.
        estate_after_tax: Real estate value after the terminal return, from the final
            simulated state -- never negative, and finite on every path, since ``run``
            never returns before every path's second death (see ``draws.n_months`` in
            :func:`run`), ``(n_paths,)``.
        death_year: Calendar year of each person's death, in
            ``initial_state.persons`` order, ``(n_persons, n_paths)`` int64 --
            ``initial_state.year + death_month_index // 12``, from the opening
            state's already-drawn ``death_month_index``.
        gis_band_count: Persons in the GIS band at each December close, the sum over
            persons of ``YearRecord.gis_band``, ``(n_years, n_paths)`` int64.
        living_count: Persons alive at each year ``y``'s December close
            (``death_month_index > (y - start_year) * 12 + 11``, compared only, never
            subtracted from), ``(n_years, n_paths)`` int64.
        seed: The seed of the draws used, for reproducibility.
        trace: The full monthly record for a single traced path — one
            :class:`~engine.core.context.MonthRecord` per month simulated, every
            array sliced to that one path — or ``()`` when ``run`` was not asked to
            trace one. See ``trace_path`` on :func:`run`.
    """

    years: NDArray[np.int64]
    net_worth: NDArray[np.float64]
    after_tax_net_worth: NDArray[np.float64]
    spending_achieved: NDArray[np.float64]
    tax_assessed: NDArray[np.float64]
    depleted: NDArray[np.bool_]
    estate_after_tax: NDArray[np.float64]
    death_year: NDArray[np.int64]
    gis_band_count: NDArray[np.int64]
    living_count: NDArray[np.int64]
    seed: int
    trace: tuple[MonthRecord, ...] = ()

    @property
    def gis_band_person_years(self) -> NDArray[np.int64]:
        """``gis_band_count`` summed over years, ``(n_paths,)`` int64.

        A count of person-years in the GIS band;
        :func:`engine.optimize.objective.gis_exposure` turns it into a rate.
        Read-only: computed from ``gis_band_count`` on each access.
        """
        return self.gis_band_count.sum(axis=0, dtype=np.int64)

    @property
    def living_person_years(self) -> NDArray[np.int64]:
        """``living_count`` summed over years, ``(n_paths,)`` int64.

        Read-only: computed from ``living_count`` on each access.
        """
        return self.living_count.sum(axis=0, dtype=np.int64)


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

    The loop breaks at the end of the first month, at or after every path's second death, in
    which ``close_year`` ran (``state.history`` grew) -- including a final death that itself
    falls in December, whose own close runs within the same step -- or when ``draws.n_months``
    is exhausted, whichever comes first. The trace, when asked for, ends at that month too. Draws
    from :mod:`engine.core.build` are sized to whole years, so with them every death year
    has a row; hand-built draws that end before a death year's December leave that year without
    one.

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
            and ``trace`` comes back empty.

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
    after_tax_net_worth: list[NDArray[np.float64]] = []
    spending_achieved: list[NDArray[np.float64]] = []
    tax_assessed: list[NDArray[np.float64]] = []
    depleted: list[NDArray[np.bool_]] = []
    gis_band: list[tuple[NDArray[np.bool_], ...]] = []
    trace: list[MonthRecord] = []

    finished_all = False
    for m in range(draws.n_months):
        years_closed_before = len(state.history)
        state, record = advance_month_traced(
            state, draws.real_returns[m], policy, market, real_params
        )
        if trace_path is not None:
            trace.append(record.at_path(trace_path))
        history_grew = len(state.history) > years_closed_before
        if history_grew:
            year_record = state.history[-1]
            years.append(year_record.year)
            net_worth.append(year_record.net_worth)
            after_tax_net_worth.append(year_record.after_tax_net_worth)
            spending_achieved.append(year_record.spending)
            tax_assessed.append(year_record.tax_assessed)
            depleted.append(year_record.depleted)
            gis_band.append(year_record.gis_band)
        if not finished_all and np.all(np.isfinite(state.estate_after_tax)):
            finished_all = True
        if finished_all and history_grew:
            break

    n_paths = initial_state.n_paths
    if years:
        years_arr = np.array(years, dtype=np.int64)
        net_worth_arr = np.stack(net_worth).astype(np.float64)
        after_tax_net_worth_arr = np.stack(after_tax_net_worth).astype(np.float64)
        spending_achieved_arr = np.stack(spending_achieved).astype(np.float64)
        tax_assessed_arr = np.stack(tax_assessed).astype(np.float64)
        depleted_arr = np.stack(depleted).astype(np.bool_)
        gis_band_count_arr = np.stack(
            [np.stack(year_bands).astype(np.int64).sum(axis=0) for year_bands in gis_band]
        )
    else:
        years_arr = np.zeros((0,), dtype=np.int64)
        net_worth_arr = np.zeros((0, n_paths), dtype=np.float64)
        after_tax_net_worth_arr = np.zeros((0, n_paths), dtype=np.float64)
        spending_achieved_arr = np.zeros((0, n_paths), dtype=np.float64)
        tax_assessed_arr = np.zeros((0, n_paths), dtype=np.float64)
        depleted_arr = np.zeros((0, n_paths), dtype=np.bool_)
        gis_band_count_arr = np.zeros((0, n_paths), dtype=np.int64)

    start_year = initial_state.year
    death_year = np.stack(
        [
            np.full(n_paths, start_year, dtype=np.int64) + (person.death_month_index // 12)
            for person in initial_state.persons
        ],
        axis=0,
    ).astype(np.int64)

    december_indices = (years_arr - start_year) * 12 + 11
    living_count_arr = np.zeros((len(years_arr), n_paths), dtype=np.int64)
    for person in initial_state.persons:
        alive_at_close = person.death_month_index[None, :] > december_indices[:, None]
        living_count_arr = living_count_arr + alive_at_close.astype(np.int64)

    return SimulationResult(
        years=years_arr,
        net_worth=net_worth_arr,
        after_tax_net_worth=after_tax_net_worth_arr,
        spending_achieved=spending_achieved_arr,
        tax_assessed=tax_assessed_arr,
        depleted=depleted_arr,
        estate_after_tax=state.estate_after_tax,
        death_year=death_year,
        gis_band_count=gis_band_count_arr,
        living_count=living_count_arr,
        seed=draws.seed,
        trace=tuple(trace),
    )

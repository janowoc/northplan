# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Return and mortality draw generation.

Generated once per scenario, from an explicit seed, and reused across every
policy evaluation. Nothing here may be called from inside the optimizer's loop.

The simulation steps monthly, so the draws are monthly and the first axis of
every array is a month. Scenario assumptions are still expressed *annually*,
because that is how return and inflation assumptions are stated and argued
about; the conversion to a monthly distribution happens exactly once, here, and
is never repeated downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RandomDraws:
    """The fixed random inputs to a scenario. Generated once, reused forever.

    Attributes:
        seed: The seed these were generated from. Recorded so a run can be
            reproduced exactly.
        real_returns: Real returns *per month*, shape
            ``(n_months, n_assets, n_paths)``. Real, not nominal, and monthly,
            not annual. The naming is deliberate: an array that is silently
            annual has the right shape and the wrong magnitude, and produces a
            plausible answer.
        mortality: Uniform draws for mortality, ``(n_months, n_persons,
            n_paths)``. Compared against a monthly hazard, so a death lands in
            a month rather than at a year boundary.
        n_months: Month count, for shape assertions at call sites.
        n_paths: Path count, for shape assertions at call sites.
    """

    seed: int
    real_returns: NDArray[np.float64]
    mortality: NDArray[np.float64]
    n_months: int
    n_paths: int


def generate(
    seed: int,
    n_years: int,
    n_paths: int,
    annual_means: NDArray[np.float64],
    annual_covariance: NDArray[np.float64],
    n_persons: int,
) -> RandomDraws:
    """Generate the full set of monthly random draws for a scenario.

    Called exactly once, before the optimizer starts. The returned draws are
    passed to every policy evaluation unchanged.

    The horizon is given in years and expanded to ``12 * n_years`` months here,
    so that a scenario file and the API keep talking in years while the engine
    steps in months.

    Converting an annual assumption to a monthly one is a modelling decision,
    not arithmetic, and the implementation must state which convention it uses
    and hold to it. The requirement it has to satisfy: **twelve monthly draws
    compounded together must reproduce the specified annual distribution** —
    not merely the annual mean divided by twelve, which understates compounding
    and misstates dispersion by a factor of the square root of twelve. That
    equality is a test, not a comment.

    Args:
        seed: Fixed seed. The same seed must reproduce identical draws.
        n_years: Horizon in years. Expanded to months internally.
        n_paths: Number of Monte Carlo paths.
        annual_means: Expected **real annual** return per asset class,
            ``(n_assets,)``, as bare fractions.
        annual_covariance: Covariance matrix of real **annual** returns,
            ``(n_assets, n_assets)``.
        n_persons: Number of persons in the household.

    Returns:
        A frozen :class:`RandomDraws` with monthly draws.

    Raises:
        ValueError: If ``annual_covariance`` is not square, not symmetric, or
            not positive semi-definite, or if its size does not match
            ``annual_means``.
    """
    raise NotImplementedError


def deterministic(
    n_years: int,
    annual_means: NDArray[np.float64],
    n_persons: int,
) -> RandomDraws:
    """Draws with zero volatility: one path, every month at the mean.

    The bridge between the deterministic single-path check and Monte Carlo.
    Running the simulator with these must reproduce the hand-checked
    spreadsheet exactly; that equality is the test that says the Monte Carlo
    wrapper introduced no error of its own.

    Every month gets the monthly return that compounds to ``annual_means`` over
    twelve months — not one twelfth of it. A spreadsheet that applies the
    annual figure once a year and one that applies the monthly equivalent twelve
    times agree only if the conversion compounds, and the check is worthless if
    the two disagree for that reason.

    Args:
        n_years: Horizon in years.
        annual_means: Expected real annual return per asset class,
            ``(n_assets,)``.
        n_persons: Number of persons in the household.

    Returns:
        A :class:`RandomDraws` with ``n_paths == 1`` and no dispersion.
    """
    raise NotImplementedError

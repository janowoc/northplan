"""Return and mortality draw generation.

Generated once per scenario, from an explicit seed, and reused across every
policy evaluation. Nothing here may be called from inside the optimizer's loop.
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
        real_returns: Real returns, shape ``(n_years, n_assets, n_paths)``.
            Real, not nominal.
        mortality: Uniform draws for mortality, ``(n_years, n_persons, n_paths)``.
        n_paths: Path count, for shape assertions at call sites.
    """

    seed: int
    real_returns: NDArray[np.float64]
    mortality: NDArray[np.float64]
    n_paths: int


def generate(
    seed: int,
    n_years: int,
    n_paths: int,
    means: NDArray[np.float64],
    covariance: NDArray[np.float64],
    n_persons: int,
) -> RandomDraws:
    """Generate the full set of random draws for a scenario.

    Called exactly once, before the optimizer starts. The returned draws are
    passed to every policy evaluation unchanged.

    Args:
        seed: Fixed seed. The same seed must reproduce identical draws.
        n_years: Number of simulated years.
        n_paths: Number of Monte Carlo paths.
        means: Expected **real** annual return per asset class, ``(n_assets,)``,
            as bare fractions.
        covariance: Covariance matrix of real returns, ``(n_assets, n_assets)``.
        n_persons: Number of persons in the household.

    Returns:
        A frozen :class:`RandomDraws`.

    Raises:
        ValueError: If ``covariance`` is not square, not symmetric, or not
            positive semi-definite, or if its size does not match ``means``.
    """
    raise NotImplementedError


def deterministic(
    n_years: int,
    means: NDArray[np.float64],
    n_persons: int,
) -> RandomDraws:
    """Draws with zero volatility: one path, returns equal to ``means``.

    The bridge between the deterministic single-path check and Monte Carlo.
    Running the simulator with these must reproduce the hand-checked
    spreadsheet exactly; that equality is the test that says the Monte Carlo
    wrapper introduced no error of its own.

    Args:
        n_years: Number of simulated years.
        means: Expected real annual return per asset class, ``(n_assets,)``.
        n_persons: Number of persons in the household.

    Returns:
        A :class:`RandomDraws` with ``n_paths == 1`` and no dispersion.
    """
    raise NotImplementedError

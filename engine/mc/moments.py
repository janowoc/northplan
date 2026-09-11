# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Covariance construction and moment matching for lognormal return draws.

This module owns three things, and the algebra for them exists nowhere
else: building the covariance of annual simple returns from per-class
volatilities and a correlation matrix; matching annual arithmetic moments
to the mean and covariance of the *monthly log-returns* a draw is actually
taken from; and the attainability checks on that matching.

**Invariant: this module imports nothing from ``engine.scenario``.**
``engine.scenario.schema`` imports this module, so the dependency between
the scenario package and Monte Carlo runs one way only —
``tests/test_layering.py`` pins this in both directions.

Arrays are positional, in the caller's asset-class order; ``names``, where
accepted, only labels messages and never changes which entry is which.

The one non-obvious fact a caller needs: a correlation matrix can be
perfectly valid (symmetric, unit diagonal, positive semi-definite) and
still have no lognormal distribution that realises it, because entrywise
``log(1 + x)`` does not preserve positive semi-definiteness. A correlation
of exactly -1 between two classes is never realisable; +1 is realisable
exactly when ``vol / (1 + real_mean)`` matches between the pair. Nothing
here is clipped, projected, or nudged to the nearest matrix that works.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Final

import numpy as np
from numpy.typing import NDArray

from engine.core.timeline import MONTHS_PER_YEAR

#: How negative the smallest eigenvalue of the moment-matched monthly
#: log-covariance may be and still count as positive semi-definite.
#:
#: Same value as ``engine.mc.returns._COVARIANCE_PSD_TOLERANCE`` and
#: ``engine.scenario.schema.PSD_TOLERANCE``, deliberately a separate copy.
_PSD_TOLERANCE: Final[float] = 1e-8


def covariance_from_correlation(
    vols: NDArray[np.float64], correlation: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Covariance of annual simple returns from per-class vols and correlation.

    ``covariance[i][j] = correlation[i][j] * vols[i] * vols[j]``. Checks
    shapes only. Neither this nor :func:`monthly_log_moments` checks
    ``correlation`` for symmetry, a unit diagonal, or positive
    semi-definiteness; validating it is the caller's job. A non-finite input
    passes through as a non-finite covariance, which
    :func:`monthly_log_moments` refuses.

    Args:
        vols: Annual volatilities, ``(n,)``.
        correlation: Correlation matrix, ``(n, n)``, same order as ``vols``.

    Returns:
        Covariance matrix, ``(n, n)``, same order.

    Raises:
        ValueError: If ``vols`` is not 1-D, or ``correlation``'s shape does
            not match ``vols``.
    """
    vols = np.asarray(vols, dtype=np.float64)
    correlation = np.asarray(correlation, dtype=np.float64)

    if vols.ndim != 1:
        raise ValueError(
            f"vols: expected a 1-D array of per-asset-class volatilities, got shape {vols.shape}."
        )
    n = vols.shape[0]
    if correlation.shape != (n, n):
        raise ValueError(
            f"correlation: expected shape ({n}, {n}) to match {n} vol(s), got {correlation.shape}."
        )

    with np.errstate(invalid="ignore", over="ignore"):
        return correlation * np.outer(vols, vols)


def _first_nonfinite_mean_message(
    annual_means: NDArray[np.float64], names: Sequence[str] | None
) -> str:
    i = int(np.flatnonzero(~np.isfinite(annual_means))[0])
    v = float(annual_means[i])
    if names is not None:
        return f"asset class {names[i]!r}: real_mean {v!r} is not finite (NaN or +/-inf)."
    return f"annual_means[{i}] = {v!r}: must be finite (not NaN or +/-inf)."


def _first_nonfinite_covariance_message(
    annual_covariance: NDArray[np.float64], names: Sequence[str] | None
) -> str:
    i, j = (int(x) for x in np.argwhere(~np.isfinite(annual_covariance))[0])
    v = float(annual_covariance[i, j])
    if names is not None:
        if i == j:
            return (
                f"asset class {names[i]!r}: annual variance {v!r} is not finite "
                "(NaN or +/-inf); check its vol."
            )
        a, b = min(i, j), max(i, j)
        return (
            f"asset classes {names[a]!r} and {names[b]!r}: annual covariance {v!r} is "
            "not finite (NaN or +/-inf); check their vols and correlation."
        )
    return f"annual_covariance[{i}][{j}] = {v!r}: must be finite (not NaN or +/-inf)."


def _growth_not_positive_message(
    index: int,
    annual_means: NDArray[np.float64],
    growth: NDArray[np.float64],
    names: Sequence[str] | None,
) -> str:
    if names is not None:
        return (
            f"asset class {names[index]!r}: real_mean {float(annual_means[index])!r} is at "
            f"or below -1, so 1 + real_mean = {float(growth[index])!r} is not strictly "
            "positive and log(1 + real_mean) is undefined. No lognormal return has this "
            "mean."
        )
    return (
        f"annual_means[{index}] = {annual_means[index]!r}: 1 + annual_means must be "
        "strictly positive. At or below -1, (1 + mu) raised to a fractional power is a "
        "complex number rather than an error, and log(1 + mu) is undefined."
    )


def _log_argument_not_positive_message(
    i: int,
    j: int,
    annual_covariance: NDArray[np.float64],
    log_argument: NDArray[np.float64],
    names: Sequence[str] | None,
) -> str:
    if names is None:
        return (
            f"annual_covariance[{i}][{j}] = {annual_covariance[i, j]!r}: the "
            f"moment-matching log argument 1 + Sigma[{i}][{j}] / "
            f"((1 + mu[{i}]) * (1 + mu[{j}])) = {log_argument[i, j]!r} is not "
            f"positive, so log(1 + Sigma_ij / ((1 + mu_i)(1 + mu_j))) is "
            f"undefined. No lognormal distribution realises this annual "
            f"covariance at these means."
        )
    if i == j:
        return (
            f"asset class {names[i]!r}: the moment-matching log argument "
            f"1 + variance / (1 + real_mean)**2 = {float(log_argument[i, i])!r} is not "
            "positive, so no lognormal distribution has this variance at this mean."
        )
    a, b = min(i, j), max(i, j)
    return (
        f"asset classes {names[a]!r} and {names[b]!r}: the moment-matching log argument "
        "1 + covariance / ((1 + real_mean) * (1 + real_mean)) = "
        f"{float(log_argument[i, j])!r} is not positive, so no lognormal distribution has "
        f"their annual covariance {float(annual_covariance[i, j])!r} at these means. A "
        "negative correlation this strong is not attainable at these means and "
        "volatilities."
    )


def _psd_failure_message(
    smallest: float, s_month: NDArray[np.float64], names: Sequence[str] | None
) -> str:
    if names is None:
        return (
            f"the moment-matched monthly log-covariance is not positive "
            f"semi-definite; its smallest eigenvalue is {smallest:g}. "
            f"Entrywise log(1 + x) does not preserve positive "
            f"semi-definiteness, so a correlation matrix that is perfectly "
            f"valid on its own can still have no lognormal distribution "
            f"that realises it at these annual means and volatilities -- "
            f"there is nothing to draw from, and it is not clipped, "
            f"projected, or nudged to the nearest matrix that works."
        )
    n = s_month.shape[0]
    for i, j in combinations(range(n), 2):
        pair_smallest = float(np.linalg.eigvalsh(s_month[np.ix_([i, j], [i, j])]).min())
        if pair_smallest < -_PSD_TOLERANCE:
            return (
                f"asset classes {names[i]!r} and {names[j]!r}: no lognormal distribution "
                "has these two classes' annual means, volatilities and correlation "
                "together; the moment-matched monthly log-covariance of the pair has "
                f"smallest eigenvalue {pair_smallest:g}. A correlation of -1 is never "
                "realisable, and +1 only when vol / (1 + real_mean) is the same for both "
                "classes. Nothing is clipped or nudged to the nearest matrix that works."
            )
    joined = ", ".join(repr(name) for name in names)
    return (
        f"asset classes {joined}: no lognormal distribution has these classes' annual "
        "means, volatilities and correlations together, although no one pair of them is "
        "unrealisable on its own; the moment-matched monthly log-covariance has smallest "
        f"eigenvalue {smallest:g}. Nothing is clipped or nudged to the nearest matrix that "
        "works."
    )


def monthly_log_moments(
    annual_means: NDArray[np.float64],
    annual_covariance: NDArray[np.float64],
    names: Sequence[str] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean and covariance of monthly log-returns matching annual moments.

    ``s_ij = log(1 + Sigma_ij / ((1 + mu_i)(1 + mu_j)))``,
    ``m_i = log(1 + mu_i) - s_ii / 2``, both then divided by
    ``MONTHS_PER_YEAR``, where ``mu = annual_means`` are annual arithmetic
    real means and ``Sigma = annual_covariance`` is the covariance of annual
    simple returns. The attainability check runs on ``s_month`` — the
    matrix a draw is actually taken from — not on ``annual_covariance``.

    With ``names`` given, every message names the asset class(es) involved;
    without it, messages use indices, matching ``engine.mc.returns.generate``.

    Args:
        annual_means: Annual arithmetic real means, ``(n,)``.
        annual_covariance: Covariance of annual simple returns, ``(n, n)``.
        names: Optional asset class names, same order and length as
            ``annual_means``, for messages only.

    Returns:
        ``(m_month, s_month)``: mean ``(n,)`` and covariance ``(n, n)`` of
        monthly log-returns.

    Raises:
        ValueError: In order: wrong shape or size of ``annual_means``,
            ``annual_covariance``, or ``names``; a non-finite mean or
            covariance entry; ``1 + annual_means`` not strictly positive; the
            moment-matching log argument not strictly positive; the
            moment-matched monthly log-covariance not positive
            semi-definite.
    """
    annual_means = np.asarray(annual_means, dtype=np.float64)
    annual_covariance = np.asarray(annual_covariance, dtype=np.float64)

    if annual_means.ndim != 1:
        raise ValueError(
            "annual_means: expected a 1-D array of per-asset-class means, got shape "
            f"{annual_means.shape}."
        )
    n = annual_means.shape[0]
    if n < 1:
        raise ValueError("annual_means: no asset classes given (empty array).")
    if annual_covariance.shape != (n, n):
        raise ValueError(
            f"annual_covariance: expected shape ({n}, {n}) to match annual_means, got "
            f"{annual_covariance.shape}."
        )
    if names is not None and len(names) != n:
        raise ValueError(
            f"names: expected {n} asset class name(s) to match annual_means, got {len(names)}."
        )

    if not np.all(np.isfinite(annual_means)):
        raise ValueError(_first_nonfinite_mean_message(annual_means, names))
    if not np.all(np.isfinite(annual_covariance)):
        raise ValueError(_first_nonfinite_covariance_message(annual_covariance, names))

    growth = 1.0 + annual_means
    offending = np.flatnonzero(growth <= 0.0)
    if offending.size:
        index = int(offending[0])
        raise ValueError(_growth_not_positive_message(index, annual_means, growth, names))

    log_argument = 1.0 + annual_covariance / np.outer(growth, growth)
    bad = np.argwhere(log_argument <= 0.0)
    if bad.size:
        i, j = (int(x) for x in bad[0])
        raise ValueError(
            _log_argument_not_positive_message(i, j, annual_covariance, log_argument, names)
        )

    s_annual = np.log(log_argument)
    m_annual = np.log(growth) - np.diag(s_annual) / 2.0
    m_month = m_annual / MONTHS_PER_YEAR
    s_month = s_annual / MONTHS_PER_YEAR

    smallest = float(np.linalg.eigvalsh(s_month).min())
    if smallest < -_PSD_TOLERANCE:
        raise ValueError(_psd_failure_message(smallest, s_month, names))

    return m_month, s_month

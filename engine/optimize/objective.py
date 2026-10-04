# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Scalar objectives that reduce a path distribution to one number.

The optimizer maximises one of these. This module owns the objectives a
scenario can select (:data:`OBJECTIVE_NAMES`), the exposure rate that is only
reported, and :func:`select_objective`, which binds a household preference to
an objective before any run starts.

Shape convention: the simulation steps monthly but
:class:`~engine.mc.simulate.SimulationResult` is recorded annually, so its
per-year fields are ``(n_years, n_paths)`` with a *year* on the first axis, not
a month, and ``estate_after_tax`` is ``(n_paths,)``. Nothing here divides or
multiplies by twelve. Each function says which axis it reduces; reducing over
the wrong one gives a number that moves plausibly and means nothing.

Invariants: every score is a plain ``float``; dollar scores are real dollars; a
household preference that is ``None`` is never replaced by a number here.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from engine.mc.simulate import SimulationResult

OBJECTIVE_NAMES: tuple[str, ...] = (
    "median_estate_after_tax",
    "certainty_equivalent_estate",
    "success_probability",
)


def median_estate_after_tax(result: SimulationResult) -> float:
    """Median over paths of the real after-tax estate.

    Args:
        result: Output of one policy evaluation.

    Returns:
        Real dollars.
    """
    return float(np.median(result.estate_after_tax))


def success_probability(result: SimulationResult) -> float:
    """Fraction of paths never depleted in any year.

    Reduces over years with "ever depleted", then over paths.

    Args:
        result: Output of one policy evaluation.

    Returns:
        Probability in ``[0, 1]``.
    """
    return float(1.0 - np.mean(result.depleted.any(axis=0)))


def gis_exposure(result: SimulationResult) -> float:
    """Share of living person-years spent in the GIS band, pooled over years and paths.

    Reported only; it is not a selectable objective. Pooling weights each
    person-year equally, so a path that lives long counts for more.

    Args:
        result: Output of one policy evaluation.

    Returns:
        A rate in ``[0, 1]``; ``0.0`` when nobody is alive at any December
        close, since nobody is then in the band.
    """
    living = int(result.living_person_years.sum())
    if living == 0:
        return 0.0
    return float(result.gis_band_person_years.sum() / living)


def _missing_preference_error(
    risk_aversion: float | None, estate_utility_shift: float | None
) -> ValueError | None:
    missing = []
    if risk_aversion is None:
        missing.append("Scenario.risk_aversion")
    if estate_utility_shift is None:
        missing.append("Scenario.estate_utility_shift")
    if not missing:
        return None
    return ValueError(
        f"certainty_equivalent_estate needs {' and '.join(missing)}, which "
        f"{'is' if len(missing) == 1 else 'are'} None. It will not substitute a value: "
        "state the preference in the scenario."
    )


def certainty_equivalent_estate(
    result: SimulationResult, risk_aversion: float | None, estate_utility_shift: float | None
) -> float:
    """Certainty equivalent of the after-tax estate under CRRA utility, over paths.

    With ``x = estate_after_tax + estate_utility_shift`` and ``g = risk_aversion``
    the result is ``(mean(x ** (1 - g))) ** (1 / (1 - g)) - estate_utility_shift``,
    and ``exp(mean(log(x))) - estate_utility_shift`` at ``g == 1``. The general
    case is computed in the log domain, with ``log1p`` and ``expm1``, so it stays
    finite for large ``g`` and accurate as ``g`` approaches 1.

    Args:
        result: Output of one policy evaluation.
        risk_aversion: Coefficient of relative risk aversion, ``>= 0``.
        estate_utility_shift: Real dollars added to every estate before the
            utility, ``> 0``.

    Returns:
        Real dollars.

    Raises:
        ValueError: Either preference is ``None``; naming each missing field.
    """
    error = _missing_preference_error(risk_aversion, estate_utility_shift)
    if error is not None:
        raise error
    assert risk_aversion is not None and estate_utility_shift is not None
    log_x = np.log(result.estate_after_tax + estate_utility_shift)
    if risk_aversion == 1.0:
        return float(np.exp(np.mean(log_x)) - estate_utility_shift)
    a = 1.0 - risk_aversion
    pivot = np.max(log_x) if a > 0 else np.min(log_x)  # so a * (log_x - pivot) <= 0
    q = np.log1p(np.mean(np.expm1(a * (log_x - pivot)))) / a
    return float(np.exp(pivot + q) - estate_utility_shift)


def select_objective(
    name: str, *, risk_aversion: float | None, estate_utility_shift: float | None
) -> Callable[[SimulationResult], float]:
    """Return the one-argument objective called ``name``.

    Args:
        name: One of :data:`OBJECTIVE_NAMES`.
        risk_aversion: Bound into ``certainty_equivalent_estate``.
        estate_utility_shift: Bound into ``certainty_equivalent_estate``.

    Returns:
        A callable from a :class:`~engine.mc.simulate.SimulationResult` to a float.

    Raises:
        ValueError: ``name`` is not in :data:`OBJECTIVE_NAMES`, or it is
            ``certainty_equivalent_estate`` and either preference is ``None``;
            raised here, before any run.
    """
    if name == "median_estate_after_tax":
        return median_estate_after_tax
    if name == "success_probability":
        return success_probability
    if name == "certainty_equivalent_estate":
        error = _missing_preference_error(risk_aversion, estate_utility_shift)
        if error is not None:
            raise error

        def certainty_equivalent(result: SimulationResult) -> float:
            return certainty_equivalent_estate(result, risk_aversion, estate_utility_shift)

        return certainty_equivalent
    raise ValueError(f"unknown objective {name!r}; choose one of {list(OBJECTIVE_NAMES)!r}.")

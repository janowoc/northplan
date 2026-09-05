# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Scalar objectives that reduce a path distribution to one number.

The optimizer maximises one of these. Each collapses ``(n_years, n_paths)``
output to a scalar, so the axis being reduced matters: reducing over the wrong
axis produces a number that moves plausibly and means nothing.

The simulation steps monthly but ``SimulationResult`` is recorded annually, so
the arrays these functions see have a *year* on the first axis, not a month.
Nothing here divides or multiplies by twelve.
"""

from __future__ import annotations

from engine.mc.simulate import SimulationResult


def success_probability(result: SimulationResult) -> float:
    """Fraction of paths that never deplete.

    Reduces over the path axis after collapsing years with "ever depleted".
    A blunt objective — it treats running out in year 2 the same as year 30 —
    and it is here as a reference point, not a recommendation.

    Args:
        result: Output of one policy evaluation.

    Returns:
        Probability in ``[0, 1]``.
    """
    raise NotImplementedError


def median_terminal_wealth(result: SimulationResult) -> float:
    """Median real net worth in the final simulated year.

    Args:
        result: Output of one policy evaluation.

    Returns:
        Median across paths, in real dollars.
    """
    raise NotImplementedError


def certainty_equivalent_spending(result: SimulationResult, risk_aversion: float) -> float:
    """Constant real spending level equivalent in utility to the distribution.

    Prices the whole distribution rather than one quantile of it, so a policy
    is not rewarded for a fat tail bought with a bad floor. The recommended
    default objective for the readiness and decumulation use cases.

    Args:
        result: Output of one policy evaluation.
        risk_aversion: Coefficient of relative risk aversion. A scenario input,
            not a tax parameter.

    Returns:
        Certainty-equivalent annual spending, in real dollars. Annual, because
        the spending it prices is the annual total ``SimulationResult`` records
        — not a monthly figure.
    """
    raise NotImplementedError

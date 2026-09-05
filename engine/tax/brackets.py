# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Progressive bracket arithmetic, shared by federal and provincial tax.

The one piece of bracket machinery in the repository. Federal and provincial
modules supply their own edges and rates from ``params/`` and call in here;
neither reimplements the marginal-rate walk.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def tax_on_income(
    income: ArrayLike,
    edges: tuple[float, ...],
    rates: tuple[float, ...],
) -> NDArray[np.float64]:
    """Progressive tax on ``income`` given bracket edges and marginal rates.

    Computed as a sum over brackets of ``rate * clip(income - lower, 0, width)``,
    which is branch-free and therefore vectorizes across paths.

    Args:
        income: Taxable income, real dollars, shape ``(n_paths,)`` or scalar.
            Negative income yields zero tax; loss carry-forward is not handled
            here.
        edges: Lower edge of each bracket in ascending order, starting at 0.
            Loaded from ``params/``, never inlined.
        rates: Marginal rate for each bracket, as bare fractions. Same length
            as ``edges``.

    Returns:
        Tax payable, shape broadcast from ``income``.

    Raises:
        ValueError: If ``edges`` and ``rates`` differ in length, or ``edges``
            is not ascending.
    """
    raise NotImplementedError


def marginal_rate(
    income: ArrayLike,
    edges: tuple[float, ...],
    rates: tuple[float, ...],
) -> NDArray[np.float64]:
    """Marginal rate applying to the next dollar of ``income``.

    At an exact bracket boundary the *higher* bracket's rate applies, since the
    next dollar falls into it. Policy code that compares a withdrawal against a
    bracket edge depends on this convention.

    Args:
        income: Taxable income, real dollars, shape ``(n_paths,)`` or scalar.
        edges: Lower edge of each bracket, ascending, from ``params/``.
        rates: Marginal rate per bracket, bare fractions.

    Returns:
        Marginal rate, shape broadcast from ``income``.
    """
    raise NotImplementedError


def room_below_edge(
    income: ArrayLike,
    edges: tuple[float, ...],
    target_edge_index: int,
) -> NDArray[np.float64]:
    """Dollars of additional income before ``income`` crosses a bracket edge.

    The primitive behind "withdraw up to the top of this bracket" policies.
    Returns zero where income is already at or above the edge.

    Args:
        income: Current taxable income, real dollars.
        edges: Bracket edges, ascending, from ``params/``.
        target_edge_index: Index into ``edges`` of the ceiling to measure to.

    Returns:
        Non-negative headroom, shape broadcast from ``income``.
    """
    raise NotImplementedError

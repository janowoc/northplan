# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Progressive bracket arithmetic, shared by federal and provincial tax.

The one piece of bracket machinery in the repository. Federal and provincial
modules supply their own ``brackets.edges_annual`` and ``brackets.rates`` from
``params/`` and call in here; neither reimplements the marginal-rate walk.

All three functions validate the table they are handed. That is deliberate
duplication rather than a caller's responsibility: ``marginal_rate`` is what
withdrawal policies call to decide how much to take out of a registered
account, and a table it accepted silently would steer a withdrawal with a
wrong rate and no error anywhere. The structural tests in
``tests/params/test_param_file_structure.py`` already pin the shipped files,
so these guards exist for tables assembled in code.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validated_edges(edges: tuple[float, ...]) -> NDArray[np.float64]:
    """``edges`` as a float array, or ``ValueError`` if it is not a valid ladder.

    An empty ``edges`` is valid and returns an empty array: a flat-rate
    jurisdiction has no edge and one rate, which
    ``test_bracket_tables_have_one_more_rate_than_edge`` states outright. The
    two checks below are therefore guarded on there being an edge at all,
    rather than indexing ``edges[0]`` unconditionally.
    """
    edges_arr = np.asarray(edges, dtype=np.float64)
    if edges_arr.size:
        if edges_arr[0] <= 0:
            raise ValueError(f"first bracket edge must be positive, got {edges_arr[0]}")
        if np.any(np.diff(edges_arr) <= 0):
            raise ValueError(f"bracket edges must strictly ascend, got {edges_arr}")
    return edges_arr


def _validated_table(
    edges: tuple[float, ...], rates: tuple[float, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The edges, the bracket lower bounds, and the rates, all validated.

    Returns the lower bounds alongside the edges because they are what the
    length invariant is expressed against: the bounds are 0 followed by every
    edge, so there is exactly one per bracket, and comparing against their
    count states ``len(rates) == len(edges) + 1`` without writing the literal.
    """
    edges_arr = _validated_edges(edges)
    rates_arr = np.asarray(rates, dtype=np.float64)
    lowers = np.insert(edges_arr, 0, 0)
    if rates_arr.size != lowers.size:
        raise ValueError(
            f"expected {lowers.size} rates for {edges_arr.size} edges "
            f"(len(rates) must be len(edges) + 1), got {rates_arr.size} rates"
        )
    return edges_arr, lowers, rates_arr


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
            here (L17).
        edges: Upper bound of every bracket except the last, strictly
            ascending and positive, annual real dollars. Loaded from
            ``params/`` as ``brackets.edges_annual``, never inlined. The last
            bracket is unbounded and has no edge.
        rates: Marginal rate for each bracket, as bare fractions, loaded from
            ``params/`` as ``brackets.rates``. ``len(rates) == len(edges) + 1``;
            a flat-rate jurisdiction has an empty ``edges`` and a single rate.

    Returns:
        Tax payable, ``float64``, shape broadcast from ``income``.

    Raises:
        ValueError: If ``edges`` and ``rates`` differ in length, ``edges`` is
            not strictly ascending, or the first edge is not positive.
    """
    _, lowers, rates_arr = _validated_table(edges, rates)
    income_arr = np.asarray(income, dtype=np.float64)
    # The top bracket has no edge, so it has no upper clip either. ``None``
    # rather than ``np.inf``: an infinite bound multiplied by a zero top rate
    # is a nan, and under the repository's filterwarnings = ["error"] the
    # RuntimeWarning that comes with it is a test failure.
    uppers: list[np.float64 | None] = [*np.diff(lowers), None]
    tax = np.zeros_like(income_arr)
    for lower, upper, rate in zip(lowers, uppers, rates_arr, strict=True):
        if rate == 0:
            # A zero-rate bracket contributes nothing, and saying so is not an
            # optimisation. The top bracket has no upper clip, so an infinite
            # income reaches this multiplication as 0.0 * inf — a nan and a
            # RuntimeWarning, which filterwarnings = ["error"] turns into a
            # failure. A zero top rate is how a jurisdiction with no top
            # bracket would be written, so nothing rules the pair out.
            continue
        tax = tax + rate * np.clip(income_arr - lower, 0, upper)
    return np.asarray(tax, dtype=np.float64)


def marginal_rate(
    income: ArrayLike,
    edges: tuple[float, ...],
    rates: tuple[float, ...],
) -> NDArray[np.float64]:
    """Marginal rate applying to the next dollar of ``income``.

    At an exact bracket boundary the *higher* bracket's rate applies, since the
    next dollar falls into it. Policy code that compares a withdrawal against a
    bracket edge depends on this convention.

    Below zero income this returns ``rates[0]``, which is NOT the slope of
    :func:`tax_on_income` there — that function is flat at zero below zero
    income, so the true rate on the next dollar is zero until income climbs
    back to it. The two disagree on ``(-inf, 0)`` by design, and a policy
    pricing a withdrawal for a path whose taxable income has gone negative
    will charge itself the bottom rate on dollars that are in fact untaxed.

    Args:
        income: Taxable income, real dollars, shape ``(n_paths,)`` or scalar.
        edges: Upper bound of every bracket except the last, strictly
            ascending and positive, from ``params/`` as ``brackets.edges_annual``.
        rates: Marginal rate per bracket, bare fractions, from ``params/`` as
            ``brackets.rates``. ``len(rates) == len(edges) + 1``.

    Returns:
        Marginal rate, ``float64``, shape broadcast from ``income``. ``nan``
        where ``income`` is ``nan``, matching :func:`tax_on_income`.

    Raises:
        ValueError: If ``edges`` and ``rates`` differ in length, ``edges`` is
            not strictly ascending, or the first edge is not positive.
    """
    edges_arr, _, rates_arr = _validated_table(edges, rates)
    income_arr = np.asarray(income, dtype=np.float64)
    # side="right" is load-bearing: at an exact edge it returns the index of
    # the bracket *above*, matching "the next dollar falls in the higher
    # bracket". side="left" would return the bracket below instead, applying
    # the wrong rate exactly on the boundary — the one case a table with
    # distinct adjacent rates would catch and nothing else would. It also
    # sends negative income to index 0, giving rates[0] with no special case.
    index = np.searchsorted(edges_arr, income_arr, side="right")
    # searchsorted sorts nan to the end, so a nan income would otherwise come
    # back as the top rate rather than as nan — invisible in a policy decision
    # while tax_on_income reports nan for the same path.
    rate = np.where(np.isnan(income_arr), np.nan, rates_arr[index])
    return np.asarray(rate, dtype=np.float64)


def room_below_edge(
    income: ArrayLike,
    edges: tuple[float, ...],
    edge_index: int,
) -> NDArray[np.float64]:
    """Dollars of additional income before ``income`` crosses a bracket edge.

    The primitive behind "withdraw up to the top of this bracket" policies.
    Returns zero where income is already at or above the edge.

    Args:
        income: Current taxable income, real dollars.
        edges: Upper bound of every bracket except the last, strictly
            ascending and positive, from ``params/`` as ``brackets.edges_annual``.
        edge_index: Index into ``edges`` of the ceiling to measure to. Must be
            non-negative: Python would read -1 as the *top* edge, and a policy
            computing an index as ``bracket - 1`` lands there for anyone in the
            bottom bracket, sizing a withdrawal against the wrong ceiling
            instead of raising.

    Returns:
        Non-negative headroom, ``float64``, shape broadcast from ``income``.

    Raises:
        IndexError: If ``edge_index`` is not a valid index into ``edges``,
            including a negative one and any index into an empty (flat-rate)
            ``edges``.
        ValueError: If ``edges`` is not strictly ascending, or the first edge
            is not positive. An unsorted ladder makes "the edge at index i"
            mean something other than the caller intends.
    """
    edges_arr = _validated_edges(edges)
    if not 0 <= edge_index < edges_arr.size:
        raise IndexError(
            f"edge_index {edge_index} is out of range for {edges_arr.size} edges; "
            f"it must be non-negative and below the edge count"
        )
    income_arr = np.asarray(income, dtype=np.float64)
    return np.asarray(np.clip(edges_arr[edge_index] - income_arr, 0, None), dtype=np.float64)

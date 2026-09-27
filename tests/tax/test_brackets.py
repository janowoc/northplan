# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bracket arithmetic against an obviously synthetic table.

``EDGES`` and ``RATES`` are not a real jurisdiction's brackets — they are
chosen so every boundary case can be checked by hand: brackets of ``[0, 100)``
at 10%, ``[100, 200)`` at 20%, and ``[200, inf)`` at 30%. ``FLAT_RATES`` with
an empty edge list exercises the flat-rate jurisdiction that
``test_bracket_tables_have_one_more_rate_than_edge`` in
``tests/params/test_param_file_structure.py`` says is legal.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from engine.tax.brackets import marginal_rate, room_below_edge, tax_on_income

#: Upper bound of every bracket but the last: [0, 100) at 10%, [100, 200) at
#: 20%, [200, inf) at 30%.
EDGES: tuple[float, ...] = (100.0, 200.0)
RATES: tuple[float, ...] = (0.1, 0.2, 0.3)

#: A flat-rate jurisdiction: no edges, one rate.
FLAT_EDGES: tuple[float, ...] = ()
FLAT_RATES: tuple[float, ...] = (0.1,)

#: Hand-computed points: negative, zero, mid-bracket, on each edge, above the
#: top.
#:
#: tax_on_income:
#:   -50  -> clipped to 0                                          -> 0
#:      0 -> 0                                                     -> 0
#:    150 -> 100*0.1 + 50*0.2 = 10 + 10                             -> 20
#:    100 -> 100*0.1 + 0*0.2 = 10 + 0                                -> 10
#:    200 -> 100*0.1 + 100*0.2 + 0*0.3 = 10 + 20                     -> 30
#:    300 -> 100*0.1 + 100*0.2 + 100*0.3 = 10 + 20 + 30               -> 60
#:
#: marginal_rate (side="right": exact edge takes the higher bracket's rate):
#:   -50 -> 0.1   0 -> 0.1   150 -> 0.2   100 -> 0.2   200 -> 0.3   300 -> 0.3
INCOMES: tuple[float, ...] = (-50.0, 0.0, 150.0, 100.0, 200.0, 300.0)
EXPECTED_TAX: tuple[float, ...] = (0.0, 0.0, 20.0, 10.0, 30.0, 60.0)
EXPECTED_MARGINAL_RATE: tuple[float, ...] = (0.1, 0.1, 0.2, 0.2, 0.3, 0.3)

#: room_below_edge(income, EDGES, edge_index=0) -> max(0, 100 - income):
#:   -50 -> 150   0 -> 100   150 -> 0   100 -> 0   200 -> 0   300 -> 0
EXPECTED_ROOM_EDGE_0: tuple[float, ...] = (150.0, 100.0, 0.0, 0.0, 0.0, 0.0)

#: room_below_edge(income, EDGES, edge_index=1) -> max(0, 200 - income):
#:   -50 -> 250   0 -> 200   150 -> 50   100 -> 100   200 -> 0   300 -> 0
EXPECTED_ROOM_EDGE_1: tuple[float, ...] = (250.0, 200.0, 50.0, 100.0, 0.0, 0.0)


def _assert_float64_array(result: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == expected_shape


# --- tax_on_income -----------------------------------------------------------


@pytest.mark.parametrize(("income", "expected"), list(zip(INCOMES, EXPECTED_TAX, strict=True)))
def test_tax_on_income_scalar(income: float, expected: float) -> None:
    result = tax_on_income(income, EDGES, RATES)
    _assert_float64_array(result, ())
    assert result == pytest.approx(expected)


def test_tax_on_income_array() -> None:
    result = tax_on_income(np.array(INCOMES), EDGES, RATES)
    _assert_float64_array(result, (len(INCOMES),))
    np.testing.assert_allclose(result, EXPECTED_TAX)


def test_tax_on_income_flat_rate_table() -> None:
    """An empty edge list is legal and taxes every dollar at the one rate."""
    result = tax_on_income(np.array(INCOMES), FLAT_EDGES, FLAT_RATES)
    _assert_float64_array(result, (len(INCOMES),))
    expected = np.clip(np.array(INCOMES), 0, None) * FLAT_RATES[0]
    np.testing.assert_allclose(result, expected)


def test_tax_on_income_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="rates"):
        tax_on_income(100.0, EDGES, (0.1, 0.2))


def test_tax_on_income_rejects_non_ascending_edges() -> None:
    with pytest.raises(ValueError, match="ascend"):
        tax_on_income(100.0, (200.0, 100.0), RATES)


def test_tax_on_income_rejects_non_positive_first_edge() -> None:
    with pytest.raises(ValueError, match="positive"):
        tax_on_income(100.0, (0.0, 200.0), RATES)


def test_tax_on_income_rejects_negative_first_edge() -> None:
    with pytest.raises(ValueError, match="positive"):
        tax_on_income(100.0, (-50.0, 200.0), RATES)


def test_tax_on_income_flat_rate_table_does_not_raise_on_empty_edges() -> None:
    """The empty-edge guard must not index ``edges[0]`` and raise IndexError."""
    result = tax_on_income(50.0, FLAT_EDGES, FLAT_RATES)
    _assert_float64_array(result, ())
    assert result == pytest.approx(5.0)


# --- marginal_rate -------------------------------------------------------------


@pytest.mark.parametrize(
    ("income", "expected"), list(zip(INCOMES, EXPECTED_MARGINAL_RATE, strict=True))
)
def test_marginal_rate_scalar(income: float, expected: float) -> None:
    result = marginal_rate(income, EDGES, RATES)
    _assert_float64_array(result, ())
    assert result == pytest.approx(expected)


def test_marginal_rate_array() -> None:
    result = marginal_rate(np.array(INCOMES), EDGES, RATES)
    _assert_float64_array(result, (len(INCOMES),))
    np.testing.assert_allclose(result, EXPECTED_MARGINAL_RATE)


def test_marginal_rate_flat_rate_table() -> None:
    result = marginal_rate(np.array(INCOMES), FLAT_EDGES, FLAT_RATES)
    _assert_float64_array(result, (len(INCOMES),))
    np.testing.assert_allclose(result, np.full(len(INCOMES), FLAT_RATES[0]))


# --- room_below_edge -----------------------------------------------------------


@pytest.mark.parametrize(
    ("income", "expected"), list(zip(INCOMES, EXPECTED_ROOM_EDGE_0, strict=True))
)
def test_room_below_edge_zero_scalar(income: float, expected: float) -> None:
    result = room_below_edge(income, EDGES, 0)
    _assert_float64_array(result, ())
    assert result == pytest.approx(expected)


def test_room_below_edge_zero_array() -> None:
    result = room_below_edge(np.array(INCOMES), EDGES, 0)
    _assert_float64_array(result, (len(INCOMES),))
    np.testing.assert_allclose(result, EXPECTED_ROOM_EDGE_0)


@pytest.mark.parametrize(
    ("income", "expected"), list(zip(INCOMES, EXPECTED_ROOM_EDGE_1, strict=True))
)
def test_room_below_edge_one_scalar(income: float, expected: float) -> None:
    result = room_below_edge(income, EDGES, 1)
    _assert_float64_array(result, ())
    assert result == pytest.approx(expected)


def test_room_below_edge_one_array() -> None:
    result = room_below_edge(np.array(INCOMES), EDGES, 1)
    _assert_float64_array(result, (len(INCOMES),))
    np.testing.assert_allclose(result, EXPECTED_ROOM_EDGE_1)


def test_room_below_edge_bad_index_raises_index_error() -> None:
    with pytest.raises(IndexError):
        room_below_edge(100.0, EDGES, 5)


def test_room_below_edge_on_flat_rate_table_raises_index_error() -> None:
    """An empty edge list has no index at all; even index 0 is out of range."""
    with pytest.raises(IndexError):
        room_below_edge(100.0, FLAT_EDGES, 0)


# --- table validation, in all three ------------------------------------------
#
# The three used to disagree: only tax_on_income validated, so the same bad
# table raised there, leaked a numpy IndexError out of marginal_rate for some
# incomes and not others, and returned a plausible wrong rate for others still.
# marginal_rate is what a withdrawal policy calls, so it is the one that most
# needs to refuse a table it cannot read.

BAD_TABLES: tuple[tuple[str, tuple[float, ...], tuple[float, ...]], ...] = (
    ("too few rates", (100.0, 200.0), (0.1, 0.2)),
    ("too many rates", (100.0, 200.0), (0.1, 0.2, 0.3, 0.4)),
    ("edges out of order", (200.0, 100.0), (0.1, 0.2, 0.3)),
    ("a zero first edge", (0.0, 200.0), (0.1, 0.2, 0.3)),
    ("a negative first edge", (-50.0, 200.0), (0.1, 0.2, 0.3)),
)


@pytest.mark.parametrize(
    ("_description", "edges", "rates"), BAD_TABLES, ids=[case[0] for case in BAD_TABLES]
)
def test_tax_on_income_rejects_a_bad_table(
    _description: str, edges: tuple[float, ...], rates: tuple[float, ...]
) -> None:
    """Every malformed table raises, at every income."""
    for income in INCOMES:
        with pytest.raises(ValueError):
            tax_on_income(income, edges, rates)


@pytest.mark.parametrize(
    ("_description", "edges", "rates"), BAD_TABLES, ids=[case[0] for case in BAD_TABLES]
)
def test_marginal_rate_rejects_a_bad_table(
    _description: str, edges: tuple[float, ...], rates: tuple[float, ...]
) -> None:
    """The same tables, refused the same way, at every income.

    Looping over incomes is the point rather than padding. With no validation
    a short rate table raised only once some path's income reached the last
    edge, so whether the bug appeared at all depended on the household's age.
    """
    for income in INCOMES:
        with pytest.raises(ValueError):
            marginal_rate(income, edges, rates)


@pytest.mark.parametrize(
    ("_description", "edges", "_rates"), BAD_TABLES, ids=[case[0] for case in BAD_TABLES]
)
def test_room_below_edge_rejects_a_bad_edge_ladder(
    _description: str, edges: tuple[float, ...], _rates: tuple[float, ...]
) -> None:
    """A bad ladder raises; a bad rate count is not this function's business.

    ``room_below_edge`` never sees ``rates``, so the two length-mismatch cases
    are well formed as far as it is concerned and must NOT raise. Asserting
    that split keeps a future author from validating rates it was not given.
    """
    ascending = all(lower < upper for lower, upper in pairwise(edges))
    if ascending and edges[0] > 0:
        assert room_below_edge(0.0, edges, 0) == pytest.approx(edges[0])
        return
    with pytest.raises(ValueError):
        room_below_edge(0.0, edges, 0)


def test_room_below_edge_rejects_a_negative_index() -> None:
    """-1 is the top edge to Python and a mistake to a caller.

    A policy computing its ceiling as ``bracket - 1`` lands on -1 for anyone in
    the bottom bracket. Plain tuple indexing would hand back headroom to the
    highest edge in the table — for a real federal ladder, an order of
    magnitude more room than intended — and size a meltdown withdrawal against
    it with nothing raised.
    """
    with pytest.raises(IndexError):
        room_below_edge(0.0, EDGES, -1)


# --- shared array contracts ---------------------------------------------------


def test_a_nan_income_stays_nan_in_both_functions() -> None:
    """A path whose income has gone nan must not read as the top bracket.

    ``searchsorted`` sorts nan to the end, so an unguarded ``marginal_rate``
    returns the top rate for it: the damage is invisible in the policy decision
    while ``tax_on_income`` reports nan for the same path.
    """
    assert np.isnan(tax_on_income(np.nan, EDGES, RATES))
    assert np.isnan(marginal_rate(np.nan, EDGES, RATES))
    assert np.isnan(room_below_edge(np.nan, EDGES, 0))


def test_every_function_accepts_a_read_only_array() -> None:
    """CLAUDE.md requires every state array to be read-only, so callers pass one."""
    income = np.array(INCOMES, dtype=np.float64)
    income.setflags(write=False)
    np.testing.assert_allclose(tax_on_income(income, EDGES, RATES), EXPECTED_TAX)
    np.testing.assert_allclose(marginal_rate(income, EDGES, RATES), EXPECTED_MARGINAL_RATE)
    np.testing.assert_allclose(room_below_edge(income, EDGES, 0), EXPECTED_ROOM_EDGE_0)


def test_no_function_mutates_or_aliases_its_input() -> None:
    """The caller's income array survives the call, and no result shares its memory."""
    income = np.array(INCOMES, dtype=np.float64)
    untouched = income.copy()
    results = (
        tax_on_income(income, EDGES, RATES),
        marginal_rate(income, EDGES, RATES),
        room_below_edge(income, EDGES, 0),
    )
    np.testing.assert_array_equal(income, untouched)
    for result in results:
        assert not np.shares_memory(result, income)


def test_an_integer_income_is_not_truncated() -> None:
    """An int array must be widened to float64, not floored into it."""
    income = np.array([150, 250], dtype=np.int64)
    np.testing.assert_allclose(tax_on_income(income, EDGES, RATES), [20.0, 45.0])
    _assert_float64_array(tax_on_income(income, EDGES, RATES), (2,))
    _assert_float64_array(marginal_rate(income, EDGES, RATES), (2,))
    _assert_float64_array(room_below_edge(income, EDGES, 0), (2,))


def test_an_empty_path_array_stays_empty() -> None:
    """Zero paths is a shape, not an error, and must not warn."""
    income = np.zeros(0, dtype=np.float64)
    for result in (
        tax_on_income(income, EDGES, RATES),
        marginal_rate(income, EDGES, RATES),
        room_below_edge(income, EDGES, 0),
    ):
        _assert_float64_array(result, (0,))


def test_an_infinite_income_with_a_zero_top_rate_does_not_warn() -> None:
    """The unbounded top bracket is clipped with None, not multiplied by inf.

    ``filterwarnings = ["error"]`` turns the RuntimeWarning from ``0.0 * inf``
    into a failure, and a zero top rate is how a jurisdiction with no top
    bracket would be written. Nothing rules out either input reaching here.
    """
    assert tax_on_income(np.inf, (100.0,), (0.1, 0.0)) == pytest.approx(10.0)
    assert np.isinf(tax_on_income(np.inf, EDGES, RATES))
    assert tax_on_income(-np.inf, EDGES, RATES) == pytest.approx(0.0)


def test_marginal_rate_is_the_slope_of_tax_on_income_above_zero() -> None:
    """The two agree on the next dollar wherever income is non-negative.

    Below zero they disagree by design — ``tax_on_income`` is flat at zero, so
    its slope is zero, while ``marginal_rate`` returns ``rates[0]`` as issue 9
    specifies. That disagreement is asserted here so it stays deliberate.
    """
    for income in (0.0, 50.0, 100.0, 150.0, 200.0, 300.0):
        step = tax_on_income(income + 1.0, EDGES, RATES) - tax_on_income(income, EDGES, RATES)
        assert step == pytest.approx(marginal_rate(income, EDGES, RATES))

    below = -50.0
    flat = tax_on_income(below + 1.0, EDGES, RATES) - tax_on_income(below, EDGES, RATES)
    assert flat == pytest.approx(0.0)
    assert marginal_rate(below, EDGES, RATES) == pytest.approx(RATES[0])

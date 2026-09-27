# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``engine.policy.withdrawal.fill_to_bracket``, on a SYNTHETIC single-edge table.

The edge (12000) and the background monthly income (500) below are made up to exercise the
arithmetic; neither is presented as a real bracket or income figure.

Two tests check that the year's own income plus December's fill lands exactly on the edge:
``test_the_year_ends_exactly_at_the_edge``, where December's single remaining month leaves
nothing to project, and the chained test, whose projection matches the months still to come.
The rest check the room arithmetic and its guards on their own, each with a zero projection
except ``test_zero_where_ytd_plus_projection_reaches_the_edge``, which exists to exercise
a non-zero one.

Every call below passes ``projected_income_rest_of_year`` explicitly. How the projection is
computed is tested in ``tests/policy/test_withdrawal.py``, against
``_projected_income_rest_of_year``, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.policy.withdrawal import fill_to_bracket

#: SYNTHETIC. A single edge, so ceiling_index=0 is the only legal index.
_EDGES = (12000.0,)
#: SYNTHETIC background monthly income, already recognised in the month it is earned -- so
#: at month m (1-indexed), ytd going into that month's fill call is _INCOME * m.
_INCOME = 500.0


def _ytd(month: int) -> float:
    return _INCOME * month


class TestTwelveMonths:
    """One calendar year of calls, month by month, with ample ``available``."""

    def test_january_is_the_edge_less_this_months_income_over_twelve(self) -> None:
        ytd = np.array([_ytd(1)])
        fill = fill_to_bracket(
            ytd,
            _EDGES,
            0,
            np.array([1e9]),
            months_remaining_in_year=12,
            projected_income_rest_of_year=np.zeros_like(ytd),
        )
        expected = (_EDGES[0] - _INCOME) / 12
        np.testing.assert_allclose(fill, [expected])

    def test_the_year_ends_exactly_at_the_edge(self) -> None:
        """December's own income plus December's fill totals exactly the edge.

        Both closed-form: December's ytd is ``_INCOME * 12`` and its ``months_remaining`` is
        1, so ``room_below_edge`` (the whole remaining headroom) is taken in one month --
        this holds independently of how the eleven months before it split their own room.
        """
        december_ytd = np.array([_ytd(12)])
        fill = fill_to_bracket(
            december_ytd,
            _EDGES,
            0,
            np.array([1e9]),
            months_remaining_in_year=1,
            projected_income_rest_of_year=np.zeros_like(december_ytd),
        )
        np.testing.assert_allclose(december_ytd + fill, [_EDGES[0]])

    def test_twelve_chained_calls_land_on_the_edge_with_a_matching_projection(self) -> None:
        """The circular closed-form checks above never chain one month's fill into the next
        month's ``ytd``, nor do they exercise the projection at all. This drives the same
        twelve months for real, on ``_EDGES`` itself (12000) rather than a larger synthetic
        edge: each call's ``ytd`` is the background income recognised so far *plus every
        earlier month's own fill*, exactly what ``OrderedWithdrawal`` produces by reading
        ``person.income`` after each month's withdrawal has posted to the ledger, and each
        call's projection is the run-rate formula computed from the background income
        alone -- the fills are registered withdrawals, so the projection excludes them from
        the run rate, exactly as it excludes ``rrsp_withdrawals``/``rrif_lif_withdrawals``
        from ``base``.

        With a projection that correctly anticipates the months still to come, the room
        below the edge is constant all year (``edge - 12 * _INCOME``), so every month's own
        fill is the same even share of it, and the running total never overshoots -- unlike
        the old rule, which measured room against ``ytd`` alone and let the months still to
        come push the year past the edge (see the ``projected = 0`` mutation below).
        """
        running_ytd = 0.0
        expected_fill = (_EDGES[0] - 12 * _INCOME) / 12
        for month in range(1, 13):
            months_remaining = 13 - month
            running_ytd += _INCOME  # this month's own background income, recognised first.
            background_so_far = _INCOME * month
            projected = background_so_far / month * (12 - month)
            fill = float(
                fill_to_bracket(
                    np.array([running_ytd]),
                    _EDGES,
                    0,
                    np.array([1e9]),
                    months_remaining_in_year=months_remaining,
                    projected_income_rest_of_year=np.array([projected]),
                )[0]
            )
            np.testing.assert_allclose(fill, expected_fill)
            running_ytd += fill  # this month's own fill, now part of every later ytd.
            assert running_ytd <= _EDGES[0] + 1e-6, (
                f"month {month}: running_ytd {running_ytd} overshot the edge {_EDGES[0]}"
            )

        np.testing.assert_allclose(running_ytd, _EDGES[0], rtol=1e-9)

    def test_every_month_matches_room_below_edge_over_months_remaining(self) -> None:
        """Every one of the twelve months, not only January and December, independently."""
        for month in range(1, 13):
            months_remaining = 13 - month
            ytd = np.array([_ytd(month)])
            fill = fill_to_bracket(
                ytd,
                _EDGES,
                0,
                np.array([1e9]),
                months_remaining_in_year=months_remaining,
                projected_income_rest_of_year=np.zeros_like(ytd),
            )
            expected = max(0.0, (_EDGES[0] - _ytd(month))) / months_remaining
            np.testing.assert_allclose(fill, [expected])


def test_zero_where_ytd_already_at_or_above_the_edge() -> None:
    ytd = np.array([_EDGES[0], _EDGES[0] + 1.0])
    fill = fill_to_bracket(
        ytd,
        _EDGES,
        0,
        np.array([1e9, 1e9]),
        months_remaining_in_year=6,
        projected_income_rest_of_year=np.zeros_like(ytd),
    )
    np.testing.assert_allclose(fill, [0.0, 0.0])


def test_zero_where_ytd_plus_projection_reaches_the_edge() -> None:
    """Year to date alone is under the edge, but year to date plus the projection already
    reaches or exceeds it -- exactly the case the projection exists to catch.
    """
    ytd = np.array([_EDGES[0] - 100.0, _EDGES[0] - 1.0])
    projected = np.array([100.0, 500.0])
    fill = fill_to_bracket(
        ytd,
        _EDGES,
        0,
        np.array([1e9, 1e9]),
        months_remaining_in_year=6,
        projected_income_rest_of_year=projected,
    )
    assert np.all(ytd < _EDGES[0]), "the case only tests something if ytd alone is under the edge"
    np.testing.assert_allclose(fill, [0.0, 0.0])


def test_capped_at_available() -> None:
    room = _EDGES[0] - 0.0
    monthly = room / 12
    available = monthly / 2
    ytd = np.array([0.0])
    fill = fill_to_bracket(
        ytd,
        _EDGES,
        0,
        np.array([available]),
        months_remaining_in_year=12,
        projected_income_rest_of_year=np.zeros_like(ytd),
    )
    np.testing.assert_allclose(fill, [available])


@pytest.mark.parametrize("bad_index", [-1, 1])
def test_bad_index_raises_index_error(bad_index: int) -> None:
    ytd = np.array([0.0])
    with pytest.raises(IndexError):
        fill_to_bracket(
            ytd,
            _EDGES,
            bad_index,
            np.array([1e9]),
            12,
            projected_income_rest_of_year=np.zeros_like(ytd),
        )

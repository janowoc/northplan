# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``engine.policy.withdrawal._gross_for_net``, on a SYNTHETIC two-edge withholding table.

``_EDGES``/``_RATES`` below are made up to exercise the band arithmetic; neither is presented
as a real withholding table (the real one is ``params/2026/rrif.yaml``'s ``withholding``
block, exercised separately in the gross-up end-to-end test).
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.policy.withdrawal import _gross_for_net

#: SYNTHETIC, and plainly not the real table (``params/2026/rrif.yaml``'s
#: ``withholding.edges_each`` and ``withholding.rates``).
_EDGES = (1234.0, 5678.0)
_RATES = (0.11, 0.23, 0.37)


def _net(gross: float, rate: float) -> float:
    return gross * (1 - rate)


class TestBandCrossing:
    """A target whose naive gross (band 0's rate) would land past band 0's own edge."""

    def test_naive_band_is_wrong_the_true_band_is_the_next_one_up(self) -> None:
        target = 1150.0
        naive = target / (1 - _RATES[0])
        assert naive > _EDGES[0], "the naive computation must actually cross the edge"

        result = _gross_for_net(
            np.array([target]), np.array([0.0]), np.array([0.0]), _EDGES, _RATES
        )
        expected = target / (1 - _RATES[1])
        assert _EDGES[0] < expected <= _EDGES[1]
        np.testing.assert_allclose(result, [expected])
        np.testing.assert_allclose(_net(result, _RATES[1]), [target])


def test_g0_is_respected_when_it_already_meets_the_target() -> None:
    target = 950.0
    g0 = 1500.0
    result = _gross_for_net(np.array([target]), np.array([g0]), np.array([0.0]), _EDGES, _RATES)
    np.testing.assert_allclose(result, [g0])
    assert _net(result[0], _RATES[1]) >= target


def test_below_m_there_is_no_withholding_to_gross_up() -> None:
    m = 2000.0
    target = 1000.0
    g0 = 500.0
    result = _gross_for_net(np.array([target]), np.array([g0]), np.array([m]), _EDGES, _RATES)
    np.testing.assert_allclose(result, [max(target, g0)])


def test_rates_not_ascending_raises() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        _gross_for_net(
            np.array([500.0]), np.array([0.0]), np.array([0.0]), _EDGES, (0.37, 0.11, 0.23)
        )


def test_m_greater_than_zero_with_a_target_above_it_grosses_up_above_the_floor() -> None:
    """``m > 0`` and the target above it -- ``G = m + c_j``, not the ``below_m`` shortcut."""
    m = 500.0
    target = 3000.0
    result = _gross_for_net(np.array([target]), np.array([0.0]), np.array([m]), _EDGES, _RATES)

    np.testing.assert_allclose(result, [500.0 + 2500.0 / (1 - _RATES[1])])
    assert result[0] > m
    np.testing.assert_allclose(m + (result - m) * (1 - _RATES[1]), [target])


def test_a_target_landing_exactly_on_an_edge_stays_in_the_lower_band() -> None:
    """The inclusive upper bound (``side="left"``) -- a candidate exactly at an edge is
    that band's own, not pushed into the next one up.
    """
    target = _EDGES[0] * (1 - _RATES[0])
    result = _gross_for_net(np.array([target]), np.array([0.0]), np.array([0.0]), _EDGES, _RATES)

    np.testing.assert_allclose(result, [_EDGES[0]])
    np.testing.assert_allclose(_net(result[0], _RATES[0]), target)


def test_one_call_with_paths_taking_different_branches() -> None:
    """Vectorised over paths that land in genuinely different branches -- path 0 is
    below its own ``m`` (the shortcut), path 1 needs the banded search.
    """
    target = np.array([1000.0, 3000.0])
    g0 = np.array([0.0, 0.0])
    m = np.array([3000.0, 500.0])

    result = _gross_for_net(target, g0, m, _EDGES, _RATES)

    np.testing.assert_allclose(result[0], 1000.0)  # below_m: G = max(target, g0).
    np.testing.assert_allclose(result[1], 500.0 + 2500.0 / (1 - _RATES[1]))

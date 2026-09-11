# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``engine.mc.moments``: covariance construction, moment matching, and the
attainability checks that let ``engine.scenario.schema.Assumptions`` refuse
a scenario at load time rather than at draw time.

Every expected value below is computed independently, by hand, in the test
itself -- never by calling the function under test to produce its own
expectation. All figures are synthetic, chosen to exercise the arithmetic
and the sign boundaries, never presented as real capital-market assumptions.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.mc.moments import covariance_from_correlation, monthly_log_moments
from engine.mc.returns import generate

MONTHS_PER_YEAR = 12


class TestCovarianceFromCorrelation:
    def test_matches_the_hand_formula_for_three_classes(self) -> None:
        vol = np.array([0.16, 0.05, 0.20])
        corr = np.array(
            [
                [1.0, 0.1, -0.2],
                [0.1, 1.0, 0.3],
                [-0.2, 0.3, 1.0],
            ]
        )

        result = covariance_from_correlation(vol, corr)

        expected = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                expected[i, j] = corr[i, j] * vol[i] * vol[j]
        np.testing.assert_allclose(result, expected, rtol=1e-15)

    def test_rejects_a_two_dimensional_vols(self) -> None:
        with pytest.raises(ValueError, match="vols"):
            covariance_from_correlation(np.array([[0.1, 0.2]]), np.eye(2))

    def test_rejects_a_mis_sized_correlation(self) -> None:
        with pytest.raises(ValueError, match="correlation"):
            covariance_from_correlation(np.array([0.1, 0.2, 0.3]), np.eye(2))


class TestMonthlyLogMomentsValid:
    def test_matches_the_hand_formula_for_two_classes(self) -> None:
        means = np.array([0.05, 0.01])
        vol = np.array([0.16, 0.05])
        corr = np.array([[1.0, 0.1], [0.1, 1.0]])
        cov = np.outer(vol, vol) * corr

        growth = 1.0 + means
        s_annual = np.log(1.0 + cov / np.outer(growth, growth))
        m_annual = np.log(growth) - np.diag(s_annual) / 2.0
        expected_m = m_annual / MONTHS_PER_YEAR
        expected_s = s_annual / MONTHS_PER_YEAR

        m_month, s_month = monthly_log_moments(means, cov)

        assert m_month.shape == (2,)
        assert s_month.shape == (2, 2)
        np.testing.assert_allclose(m_month, expected_m, rtol=1e-12)
        np.testing.assert_allclose(s_month, expected_s, rtol=1e-12)


class TestCorrelationMinusOne:
    def test_at_five_percent_vol_is_refused_and_names_both_classes(self) -> None:
        vol = np.array([0.05, 0.05])
        corr = np.array([[1.0, -1.0], [-1.0, 1.0]])
        cov = np.outer(vol, vol) * corr
        means = np.array([0.02, 0.02])

        with pytest.raises(ValueError) as excinfo:
            monthly_log_moments(means, cov, names=("equity", "bonds"))

        message = str(excinfo.value)
        assert "asset classes 'equity' and 'bonds'" in message
        assert (
            "A correlation of -1 is never realisable, and +1 only when vol / "
            "(1 + real_mean) is the same for both classes." in message
        )
        assert "annual_covariance[" not in message


class TestCorrelationPlusOne:
    def test_matched_ratio_loads_without_error(self) -> None:
        means = np.array([0.07, 0.20])
        vol_0 = 0.16
        vol_1 = vol_0 * (1.0 + means[1]) / (1.0 + means[0])
        vol = np.array([vol_0, vol_1])
        corr = np.array([[1.0, 1.0], [1.0, 1.0]])
        cov = np.outer(vol, vol) * corr

        monthly_log_moments(means, cov, names=("equity", "bonds"))

    def test_mismatched_ratio_is_refused_and_names_both_classes(self) -> None:
        means = np.array([0.07, 0.07])
        vol = np.array([0.16, 0.18])
        corr = np.array([[1.0, 1.0], [1.0, 1.0]])
        cov = np.outer(vol, vol) * corr

        with pytest.raises(ValueError) as excinfo:
            monthly_log_moments(means, cov, names=("equity", "bonds"))

        message = str(excinfo.value)
        assert "'equity'" in message
        assert "'bonds'" in message
        assert "asset classes 'equity' and 'bonds'" in message
        assert (
            "A correlation of -1 is never realisable, and +1 only when vol / "
            "(1 + real_mean) is the same for both classes." in message
        )


class TestOneBadPairAmongThree:
    def test_names_only_the_offending_pair(self) -> None:
        names = ("equity", "bonds", "gold")
        means = np.array([0.05, 0.01, 0.0])
        vol = np.array([0.16, 0.05, 0.05])
        corr = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, -1.0],
                [0.0, -1.0, 1.0],
            ]
        )
        cov = np.outer(vol, vol) * corr

        with pytest.raises(ValueError) as excinfo:
            monthly_log_moments(means, cov, names=names)

        message = str(excinfo.value)
        assert "'bonds'" in message
        assert "'gold'" in message
        assert "'equity'" not in message

    def test_two_bad_pairs_names_only_the_first(self) -> None:
        """When several pairs fail, the first ``i < j`` pair is named, not the
        worst one: here (1, 2) fails 3.3x worse than (0, 2), yet (0, 2) is
        the pair the message names."""
        names = ("a", "b", "c")
        means = np.zeros(3)
        vol = np.array([0.1, 0.3, 0.5])
        corr = np.array(
            [
                [1.0, 0.9, -1.0],
                [0.9, 1.0, -0.9],
                [-1.0, -0.9, 1.0],
            ]
        )
        cov = np.outer(vol, vol) * corr

        # Premise: pair (0, 1) passes on its own.
        monthly_log_moments(means[[0, 1]], cov[np.ix_([0, 1], [0, 1])])

        # Premise: each of (0, 2) and (1, 2) fails on its own, and fails on
        # the PSD branch specifically, not the log-argument branch.
        for i, j in ((0, 2), (1, 2)):
            with pytest.raises(ValueError, match="not positive semi-definite"):
                monthly_log_moments(means[[i, j]], cov[np.ix_([i, j], [i, j])])

        # Premise: (1, 2) fails strictly worse than (0, 2), so a search for
        # the worst pair would name 'b' and 'c' rather than 'a' and 'c'.
        growth = 1.0 + means
        s_month = np.log1p(cov / np.outer(growth, growth)) / MONTHS_PER_YEAR
        eigenvalues = {}
        for i, j in ((0, 2), (1, 2)):
            sub = s_month[np.ix_([i, j], [i, j])]
            eigenvalues[i, j] = np.linalg.eigvalsh(sub).min()
        assert eigenvalues[1, 2] < eigenvalues[0, 2]

        with pytest.raises(ValueError) as excinfo:
            monthly_log_moments(means, cov, names=names)

        message = str(excinfo.value)
        assert "asset classes 'a' and 'c'" in message
        assert "these two classes'" in message
        assert "'b'" not in message


class TestJointFailureNoSinglePairAtFault:
    def test_names_all_three_and_says_no_pair_is_at_fault(self) -> None:
        names = ("a1", "a2", "a3")
        means = np.array([0.0, 0.0, 0.0])
        vol = np.array([0.2, 0.2, 0.2])
        corr = np.array(
            [
                [1.0, -0.5, -0.5],
                [-0.5, 1.0, -0.5],
                [-0.5, -0.5, 1.0],
            ]
        )
        cov = np.outer(vol, vol) * corr

        # Premise: each 2-class sub-problem passes on its own.
        for i, j in ((0, 1), (0, 2), (1, 2)):
            sub_means = means[[i, j]]
            sub_cov = cov[np.ix_([i, j], [i, j])]
            monthly_log_moments(sub_means, sub_cov)

        with pytest.raises(ValueError) as excinfo:
            monthly_log_moments(means, cov, names=names)

        message = str(excinfo.value)
        for name in names:
            assert repr(name) in message
        assert "no one pair of them is unrealisable on its own" in message


class TestNamedFailureMessages:
    def test_growth_failure_names_the_class(self) -> None:
        with pytest.raises(ValueError, match="asset class 'x': real_mean"):
            monthly_log_moments(np.array([-1.0]), np.array([[0.01]]), names=("x",))

    def test_log_argument_failure_names_both_classes(self) -> None:
        vol = np.array([1.2, 1.2])
        corr = np.array([[1.0, -1.0], [-1.0, 1.0]])
        cov = np.outer(vol, vol) * corr
        means = np.array([0.0, 0.0])

        with pytest.raises(
            ValueError, match="asset classes 'x' and 'y': the moment-matching log argument"
        ):
            monthly_log_moments(means, cov, names=("x", "y"))

    def test_non_finite_mean_names_the_class(self) -> None:
        with pytest.raises(ValueError, match="asset class 'y': real_mean"):
            monthly_log_moments(np.array([0.05, np.nan]), np.eye(2) * 0.01, names=("x", "y"))

    def test_non_finite_variance_names_the_class(self) -> None:
        cov = np.array([[np.inf, 0.0], [0.0, 0.01]])
        with pytest.raises(ValueError, match="asset class 'x': annual variance"):
            monthly_log_moments(np.array([0.05, 0.01]), cov, names=("x", "y"))

    def test_non_finite_off_diagonal_covariance_names_the_pair(self) -> None:
        cov = np.array([[0.01, np.nan], [np.nan, 0.01]])
        with pytest.raises(ValueError, match="asset classes 'x' and 'y': annual covariance"):
            monthly_log_moments(np.array([0.05, 0.01]), cov, names=("x", "y"))

    def test_names_of_the_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="names"):
            monthly_log_moments(np.array([0.05, 0.01]), np.eye(2) * 0.01, names=("x",))

    def test_two_dimensional_means(self) -> None:
        with pytest.raises(ValueError, match="annual_means"):
            monthly_log_moments(np.array([[0.05]]), np.array([[[0.01]]]))

    def test_empty_means(self) -> None:
        with pytest.raises(ValueError, match="no asset classes"):
            monthly_log_moments(np.array([]), np.zeros((0, 0)))

    def test_mis_shaped_covariance(self) -> None:
        with pytest.raises(ValueError, match="annual_covariance"):
            monthly_log_moments(np.array([0.05, 0.01]), np.eye(3))


class TestUnnamedMessagesArePinned:
    """Without ``names``, the two messages that moved out of ``generate`` are
    the text ``generate`` raised before the move (commit ac73b8f), typed out
    here."""

    def test_log_argument_message(self) -> None:
        vol = np.array([1.2, 1.2])
        corr = np.array([[1.0, -1.0], [-1.0, 1.0]])
        cov = np.outer(vol, vol) * corr
        means = np.array([0.0, 0.0])

        cov_01 = cov[0, 1]
        log_argument_01 = 1.0 + cov_01 / ((1.0 + means[0]) * (1.0 + means[1]))
        expected = (
            f"annual_covariance[0][1] = {cov_01!r}: the "
            f"moment-matching log argument 1 + Sigma[0][1] / "
            f"((1 + mu[0]) * (1 + mu[1])) = {log_argument_01!r} is not "
            f"positive, so log(1 + Sigma_ij / ((1 + mu_i)(1 + mu_j))) is "
            f"undefined. No lognormal distribution realises this annual "
            f"covariance at these means."
        )

        with pytest.raises(ValueError) as from_moments:
            monthly_log_moments(means, cov)
        assert str(from_moments.value) == expected

        with pytest.raises(ValueError) as from_generate:
            generate(1, 3, 10, means, cov, n_persons=1)
        assert str(from_generate.value) == expected

    def test_psd_message(self) -> None:
        vol = np.array([0.05, 0.05])
        corr = np.array([[1.0, -1.0], [-1.0, 1.0]])
        cov = np.outer(vol, vol) * corr
        means = np.array([0.02, 0.02])

        growth = 1.0 + means
        s_month = np.log(1.0 + cov / np.outer(growth, growth)) / MONTHS_PER_YEAR
        smallest = float(np.linalg.eigvalsh(s_month).min())
        expected = (
            f"the moment-matched monthly log-covariance is not positive "
            f"semi-definite; its smallest eigenvalue is {smallest:g}. "
            f"Entrywise log(1 + x) does not preserve positive "
            f"semi-definiteness, so a correlation matrix that is perfectly "
            f"valid on its own can still have no lognormal distribution "
            f"that realises it at these annual means and volatilities -- "
            f"there is nothing to draw from, and it is not clipped, "
            f"projected, or nudged to the nearest matrix that works."
        )

        with pytest.raises(ValueError) as from_moments:
            monthly_log_moments(means, cov)
        assert str(from_moments.value) == expected

        with pytest.raises(ValueError) as from_generate:
            generate(1, 3, 10, means, cov, n_persons=1)
        assert str(from_generate.value) == expected

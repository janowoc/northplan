# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``engine.mc.returns``: monthly return draws and mortality uniforms.

The asset means, volatilities, and correlations used throughout are
synthetic — chosen to make one class's monthly mean obviously different from
another's, or to sit on a known side of the positive-semi-definite boundary,
never presented as a real capital-market assumption.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.core.mortality import death_month_index, survival_curve
from engine.mc.returns import DETERMINISTIC_SEED, deterministic, generate
from engine.params.loader import ParamSet, load_year

#: Two asset classes, far enough apart in mean that a transposed axis or a
#: mixed-up index is unmistakable. Not a real market assumption.
TWO_ASSET_MEANS = np.array([0.02, 0.12])

#: A modest, clearly-PSD covariance for the two classes above: independent
#: (zero off-diagonal), 5% and 15% annual volatility. Small enough that the
#: lognormal moment matching stays well away from any numerical edge case.
TWO_ASSET_COV = np.array([[0.05**2, 0.0], [0.0, 0.15**2]])

#: A covariance with a real, nonzero cross-asset correlation, for the
#: correlation-recovery test. corr = 0.3, vol = (5%, 15%).
_VOL = np.array([0.05, 0.15])
_CORR = np.array([[1.0, 0.3], [0.3, 1.0]])
CORRELATED_COV = np.outer(_VOL, _VOL) * _CORR


class TestReproducibility:
    """The one property common random numbers (design decision 2) depends on."""

    def test_same_seed_gives_identical_arrays(self) -> None:
        """If this fails, no policy comparison in the optimizer is meaningful:
        two evaluations of the same seed would each see a different market."""
        a = generate(7, 6, 100, TWO_ASSET_MEANS, TWO_ASSET_COV, 2)
        b = generate(7, 6, 100, TWO_ASSET_MEANS, TWO_ASSET_COV, 2)
        assert np.array_equal(a.real_returns, b.real_returns)
        assert np.array_equal(a.mortality, b.mortality)

    def test_different_seeds_differ(self) -> None:
        a = generate(7, 6, 100, TWO_ASSET_MEANS, TWO_ASSET_COV, 2)
        b = generate(8, 6, 100, TWO_ASSET_MEANS, TWO_ASSET_COV, 2)
        assert not np.array_equal(a.real_returns, b.real_returns)
        assert not np.array_equal(a.mortality, b.mortality)


class TestDeterministicCompounding:
    """Twelve monthly returns compounded must reproduce the annual mean.

    Would catch the classic error this module's docstrings warn against
    twice: applying ``mu / 12`` instead of ``(1 + mu) ** (1 / 12) - 1``, which
    understates compounding and would fail this test by roughly
    ``mu**2 / 24`` — for ``mu = 0.12`` that is about 6e-4, many orders of
    magnitude past the 1e-12 tolerance below.
    """

    def test_compounds_to_one_plus_mu_over_twelve_months(self) -> None:
        draws = deterministic(12, TWO_ASSET_MEANS, n_persons=1)
        for asset in range(TWO_ASSET_MEANS.size):
            compounded = np.prod(1.0 + draws.real_returns[:, asset, 0])
            assert compounded == pytest.approx(1.0 + TWO_ASSET_MEANS[asset], abs=1e-12)


class TestMonteCarloMoments:
    """Sample moments of the compounded annual return, at 800,000 paths.

    Tolerances are the ones the issue specifies (1% relative on mean and
    volatility, 0.02 absolute on cross-asset correlation) and the seed is
    fixed so the comparison cannot flake.

    The issue's own path count, 200,000, is not enough for its own tolerance:
    ``pytest.approx(mu, rel=0.01)`` at ``mu = 0.02`` is an absolute tolerance
    of ``2.0e-4``, while the standard error of that sample mean at 200,000
    paths is ``vol / sqrt(n) = 0.05 / sqrt(200_000) = 1.118e-4`` -- a margin
    of only 1.79 standard errors, under which a *correct* implementation
    fails about 7% of the time over the seed, and passes here only because
    the seed is fixed and numpy's ``Generator`` stream happens to be stable
    across the versions this repository has used so far (it is not a
    guarantee: ``multivariate_normal``'s sampling has been reworked before).
    At ``N_PATHS = 800_000`` the same arithmetic gives a margin of
    ``2.0e-4 / (0.05 / sqrt(800_000)) = 3.58`` standard errors, without
    touching the 1% tolerance itself, which is the issue's and must not be
    widened. Do not "helpfully" reduce this back down -- the margin above is
    tight, not slack.
    """

    N_PATHS = 800_000
    SEED = 20260913

    def test_sample_mean_and_std_match_within_one_percent(self) -> None:
        draws = generate(self.SEED, 12, self.N_PATHS, TWO_ASSET_MEANS, TWO_ASSET_COV, n_persons=1)
        annual = np.prod(1.0 + draws.real_returns, axis=0) - 1.0  # (n_assets, n_paths)

        vol = np.sqrt(np.diag(TWO_ASSET_COV))
        for asset in range(TWO_ASSET_MEANS.size):
            sample_mean = annual[asset].mean()
            sample_std = annual[asset].std(ddof=1)
            mu = TWO_ASSET_MEANS[asset]
            assert sample_mean == pytest.approx(mu, rel=0.01), (
                f"asset {asset}: sample mean {sample_mean} vs specified {mu}"
            )
            assert sample_std == pytest.approx(vol[asset], rel=0.01), (
                f"asset {asset}: sample std {sample_std} vs specified {vol[asset]}"
            )

    def test_cross_asset_correlation_matches_within_0_02(self) -> None:
        draws = generate(self.SEED, 12, self.N_PATHS, TWO_ASSET_MEANS, CORRELATED_COV, n_persons=1)
        annual = np.prod(1.0 + draws.real_returns, axis=0) - 1.0  # (n_assets, n_paths)

        sample_corr = float(np.corrcoef(annual[0], annual[1])[0, 1])
        assert sample_corr == pytest.approx(_CORR[0, 1], abs=0.02)


class TestAcceptedShapes:
    """Shapes worked out analytically as accepted and correct, but that
    nothing before this exercised: a single asset class, a class with zero
    volatility, and six classes at once. ``AssetClass.vol`` explicitly
    permits ``0.0`` as "a riskless class", so that one is a documented,
    supported input that had no coverage at all.

    Each test asserts the sample moments come back right, not merely that
    :func:`generate` does not raise -- a check that only asserts the absence
    of an exception would pass against a function that silently returned
    zeros. Tolerances are derived from the standard error of each sample
    statistic (``vol / sqrt(N_PATHS)`` for a sample mean, and, to leading
    order, ``vol / sqrt(2 * N_PATHS)`` for a sample standard deviation) at
    six standard errors, the same multiple ``tests/core/test_mortality.py``
    uses for its own sample-based checks.
    """

    N_PATHS = 200_000
    SEED = 20260916
    TOLERANCE_IN_SE = 6

    def _assert_moments(self, annual: np.ndarray, mu: np.ndarray, vol: np.ndarray) -> None:
        for i in range(mu.size):
            mean_se = vol[i] / np.sqrt(self.N_PATHS)
            std_se = vol[i] / np.sqrt(2 * self.N_PATHS)
            sample_mean = annual[i].mean()
            sample_std = annual[i].std(ddof=1)
            assert sample_mean == pytest.approx(mu[i], abs=self.TOLERANCE_IN_SE * mean_se), (
                f"asset {i}: sample mean {sample_mean} vs specified {mu[i]}"
            )
            assert sample_std == pytest.approx(vol[i], abs=self.TOLERANCE_IN_SE * std_se), (
                f"asset {i}: sample std {sample_std} vs specified {vol[i]}"
            )

    def test_a_single_asset_class(self) -> None:
        mu = np.array([0.06])
        vol = np.array([0.12])
        cov = vol.reshape(1, 1) ** 2

        draws = generate(self.SEED, 12, self.N_PATHS, mu, cov, n_persons=1)
        annual = np.prod(1.0 + draws.real_returns, axis=0) - 1.0  # (1, n_paths)

        self._assert_moments(annual, mu, vol)

    def test_a_riskless_class_alongside_a_risky_one(self) -> None:
        """``AssetClass.vol`` explicitly permits ``0.0`` as "a riskless
        class"; the riskless one must compound to exactly its mean, with no
        dispersion at all, on every path."""
        mu = np.array([0.02, 0.06])
        vol = np.array([0.0, 0.10])
        cov = np.diag(vol**2)

        draws = generate(self.SEED, 12, self.N_PATHS, mu, cov, n_persons=1)
        annual = np.prod(1.0 + draws.real_returns, axis=0) - 1.0  # (2, n_paths)

        assert np.all(annual[0] == annual[0][0]), "the riskless class has dispersion"
        assert annual[0][0] == pytest.approx(mu[0], abs=1e-9)
        self._assert_moments(annual[1:], mu[1:], vol[1:])

    def test_six_asset_classes(self) -> None:
        mu = np.array([0.01, 0.02, 0.04, 0.06, 0.08, 0.10])
        vol = np.array([0.01, 0.03, 0.06, 0.09, 0.13, 0.18])
        n = mu.size
        corr = np.full((n, n), 0.2)
        np.fill_diagonal(corr, 1.0)
        cov = np.outer(vol, vol) * corr

        draws = generate(self.SEED, 12, self.N_PATHS, mu, cov, n_persons=1)
        annual = np.prod(1.0 + draws.real_returns, axis=0) - 1.0  # (6, n_paths)

        self._assert_moments(annual, mu, vol)


class TestMortalityUniformRange:
    """The contract ``engine.core.mortality.death_month_index`` enforces."""

    def test_generate_uniforms_are_strictly_inside_zero_one(self) -> None:
        draws = generate(3, 6, 5_000, TWO_ASSET_MEANS, TWO_ASSET_COV, n_persons=3)
        assert np.all(draws.mortality > 0.0)
        assert np.all(draws.mortality < 1.0)

    def test_deterministic_uniforms_are_strictly_inside_zero_one(self) -> None:
        draws = deterministic(6, TWO_ASSET_MEANS, n_persons=3)
        assert np.all(draws.mortality > 0.0)
        assert np.all(draws.mortality < 1.0)


class TestAxisOrder:
    """The trap the spec calls out by name: a transposed ``moveaxis``.

    ``TWO_ASSET_MEANS`` differ tenfold, and the exact expected monthly mean
    of a *simple* return is ``(1 + mu) ** (1 / 12) - 1`` regardless of the
    moment-matching covariance term (the algebra is in
    ``engine.mc.returns.generate``'s docstring: the ``s_ii / 2`` term the
    monthly mean and monthly variance both carry cancels exactly). A
    transposed ``moveaxis`` would swap which of the two arrays below carries
    which class's mean and passes every other test in this file, since the
    shapes agree whenever the path count happens to differ from the asset
    count -- which it does here, on purpose.
    """

    N_PATHS = 50_000
    SEED = 20260914

    def test_first_axis_slot_carries_the_first_classs_mean(self) -> None:
        draws = generate(self.SEED, 12, self.N_PATHS, TWO_ASSET_MEANS, TWO_ASSET_COV, n_persons=1)
        for asset in range(TWO_ASSET_MEANS.size):
            sample = draws.real_returns[:, asset, :]
            expected = (1.0 + TWO_ASSET_MEANS[asset]) ** (1.0 / 12.0) - 1.0
            standard_error = sample.std(ddof=1) / np.sqrt(sample.size)
            tolerance = 8 * standard_error
            assert sample.mean() == pytest.approx(expected, abs=tolerance), (
                f"real_returns[:, {asset}, :] mean {sample.mean()} does not "
                f"match asset {asset}'s expected monthly mean {expected} "
                f"within {tolerance} -- the asset axis may be transposed"
            )


class TestDrawOrder:
    """Returns before mortality, from the same generator (design decision 2).

    Reconstructs the exact sequence a generator seeded identically must have
    produced if it drew the returns first and the mortality uniforms second,
    duplicating the transform in ``generate`` on purpose: this test exists to
    catch the *order* silently flipping, not to re-verify the moment
    matching, which :class:`TestMonteCarloMoments` and
    :class:`TestDeterministicCompounding` already do.
    """

    SEED = 20260915
    N_MONTHS = 3
    N_PATHS = 4
    N_PERSONS = 2

    def _expected(self, means: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        growth = 1.0 + means
        s_annual = np.log(1.0 + cov / np.outer(growth, growth))
        m_annual = np.log(growth) - np.diag(s_annual) / 2.0
        rng = np.random.default_rng(self.SEED)
        monthly_log_returns = rng.multivariate_normal(
            m_annual / 12.0, s_annual / 12.0, size=(self.N_MONTHS, self.N_PATHS)
        )
        real_returns = np.moveaxis(np.exp(monthly_log_returns) - 1.0, 2, 1)
        mortality = rng.random(size=(self.N_PERSONS, self.N_PATHS))
        mortality = np.where(mortality == 0.0, np.nextafter(0.0, 1.0), mortality)
        return real_returns, mortality

    def test_matches_a_generator_consumed_returns_then_mortality(self) -> None:
        draws = generate(
            self.SEED, self.N_MONTHS, self.N_PATHS, TWO_ASSET_MEANS, TWO_ASSET_COV, self.N_PERSONS
        )
        expected_returns, expected_mortality = self._expected(TWO_ASSET_MEANS, TWO_ASSET_COV)
        assert np.array_equal(draws.real_returns, expected_returns)
        assert np.array_equal(draws.mortality, expected_mortality)

    def test_reversed_order_would_not_match(self) -> None:
        """If the order silently flips, path 123 no longer has the same
        market under every policy; this is the check that would catch it."""
        draws = generate(
            self.SEED, self.N_MONTHS, self.N_PATHS, TWO_ASSET_MEANS, TWO_ASSET_COV, self.N_PERSONS
        )
        rng = np.random.default_rng(self.SEED)
        # Mortality drawn first, then returns: the wrong order.
        reversed_mortality = rng.random(size=(self.N_PERSONS, self.N_PATHS))
        growth = 1.0 + TWO_ASSET_MEANS
        s_annual = np.log(1.0 + TWO_ASSET_COV / np.outer(growth, growth))
        m_annual = np.log(growth) - np.diag(s_annual) / 2.0
        reversed_log_returns = rng.multivariate_normal(
            m_annual / 12.0, s_annual / 12.0, size=(self.N_MONTHS, self.N_PATHS)
        )
        reversed_returns = np.moveaxis(np.exp(reversed_log_returns) - 1.0, 2, 1)
        assert not np.array_equal(draws.real_returns, reversed_returns)
        assert not np.array_equal(draws.mortality, reversed_mortality)


class TestReadOnly:
    """Generated once and reused across every policy the optimizer evaluates."""

    def test_generate_real_returns_is_read_only(self) -> None:
        draws = generate(1, 3, 10, TWO_ASSET_MEANS, TWO_ASSET_COV, n_persons=1)
        with pytest.raises(ValueError):
            draws.real_returns[0, 0, 0] = 1.0

    def test_generate_mortality_is_read_only(self) -> None:
        draws = generate(1, 3, 10, TWO_ASSET_MEANS, TWO_ASSET_COV, n_persons=1)
        with pytest.raises(ValueError):
            draws.mortality[0, 0] = 0.5

    def test_deterministic_real_returns_is_read_only(self) -> None:
        draws = deterministic(3, TWO_ASSET_MEANS, n_persons=1)
        with pytest.raises(ValueError):
            draws.real_returns[0, 0, 0] = 1.0

    def test_deterministic_mortality_is_read_only(self) -> None:
        draws = deterministic(3, TWO_ASSET_MEANS, n_persons=1)
        with pytest.raises(ValueError):
            draws.mortality[0, 0] = 0.5


class TestCovarianceValueErrors:
    """One test each for the four ways ``annual_covariance`` can be unusable."""

    def test_not_square(self) -> None:
        cov = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
        with pytest.raises(ValueError, match="square"):
            generate(1, 3, 10, TWO_ASSET_MEANS, cov, n_persons=1)

    def test_size_mismatch(self) -> None:
        cov = np.eye(3) * 0.01
        with pytest.raises(ValueError, match="match annual_means"):
            generate(1, 3, 10, TWO_ASSET_MEANS, cov, n_persons=1)

    def test_asymmetric(self) -> None:
        cov = np.array([[0.01, 0.005], [0.004, 0.01]])
        with pytest.raises(ValueError, match="not symmetric"):
            generate(1, 3, 10, TWO_ASSET_MEANS, cov, n_persons=1)

    def test_non_psd(self) -> None:
        cov = np.array([[1.0, 2.0], [2.0, 1.0]])  # symmetric, eigenvalues 3 and -1
        with pytest.raises(ValueError, match="positive semi-definite"):
            generate(1, 3, 10, TWO_ASSET_MEANS, cov, n_persons=1)

    def test_no_asset_classes(self) -> None:
        """Without this guard, the symmetry check's ``np.abs(...).max()`` is a
        reduction over an empty array and raises numpy's own ``ValueError``,
        naming neither this function nor the reason."""
        with pytest.raises(ValueError, match="no asset classes"):
            generate(1, 3, 10, np.array([]), np.zeros((0, 0)), n_persons=1)


class TestMomentMatchedCovarianceGuards:
    """The guards on the matrix numpy actually draws from, ``s`` -- not on
    ``annual_covariance`` alone, which is what :class:`TestCovarianceValueErrors`
    exercises. Five of the cases below are reproduced through the public API
    (not constructed to fail): each one passes every check
    :class:`TestCovarianceValueErrors` runs and still cannot be drawn from,
    which is the whole reason these guards exist rather than being redundant
    with that class.

    The PSD guard's rule is asymmetric between the two signs of a perfect
    correlation, and it is not a gap in either direction: correlation -1
    between two classes is never realisable and is always rejected, at any
    volatility, while correlation +1 is rejected *unless* the pair has equal
    ``sigma / (1 + mu)`` -- in which case the two classes are the same
    lognormal distribution written twice and correctly accepted. The tests
    below pin both sides of that asymmetry, not just the rejecting one.
    """

    def test_growth_positive_rejects_a_percent_for_fraction_typo(self) -> None:
        """``real_mean = -5``: the log argument stays defined (it depends on
        ``growth ** 2``, positive even when ``growth`` is negative), but
        ``log(growth)`` in ``m_i`` is a log of a negative number and would
        silently become NaN without this guard."""
        with pytest.raises(ValueError, match="strictly positive"):
            generate(1, 3, 10, np.array([-5.0]), np.array([[0.01]]), n_persons=1)

    def test_growth_positive_rejects_exactly_minus_one(self) -> None:
        """``real_mean = -1.0`` exactly makes ``growth`` zero: a division by
        zero in the log argument's denominator."""
        with pytest.raises(ValueError, match="strictly positive"):
            generate(1, 3, 10, np.array([-1.0]), np.array([[0.01]]), n_persons=1)

    def test_deterministic_growth_positive_rejects_the_same_typo(self) -> None:
        """The same guard in ``deterministic``: an all-NaN ``real_returns``
        there fails the issue 21 spreadsheet comparison with nothing saying
        why."""
        with pytest.raises(ValueError, match="strictly positive"):
            deterministic(3, np.array([-5.0]), n_persons=1)

    def test_log_argument_rejects_extreme_negative_correlation(self) -> None:
        """Two assets at 120% volatility, correlation -1: the covariance
        passes ``_check_covariance`` (its eigenvalues are 0 and 2.88, both
        non-negative), but at zero mean the log argument is -0.44."""
        vol = np.array([1.2, 1.2])
        corr = np.array([[1.0, -1.0], [-1.0, 1.0]])
        cov = np.outer(vol, vol) * corr
        with pytest.raises(ValueError, match="log argument"):
            generate(1, 3, 10, np.array([0.0, 0.0]), cov, n_persons=1)

    def test_moment_matched_psd_rejects_correlation_minus_one_at_5_percent_vol(self) -> None:
        """Two classes at 5% vol, real_mean 2%, correlation -1: passes every
        check on ``annual_covariance`` itself, and still has no lognormal
        that realises it. Correlation -1 between two classes is never
        realisable by two increasing transforms of one normal and is always
        rejected, at any volatility and any means -- unlike +1, which is
        rejected only when ``sigma / (1 + mu)`` differs between the pair
        (see the two tests below)."""
        vol = np.array([0.05, 0.05])
        corr = np.array([[1.0, -1.0], [-1.0, 1.0]])
        cov = np.outer(vol, vol) * corr
        with pytest.raises(ValueError, match="moment-matched"):
            generate(1, 3, 10, np.array([0.02, 0.02]), cov, n_persons=1)

    def test_moment_matched_psd_rejects_correlation_one_at_16_18_percent_vol(self) -> None:
        """Two classes at 16% and 18% vol, real_mean 7%, correlation +1: the
        second reproduced case for the same failure mode, opposite sign of
        correlation from the case above.

        The unequal volatilities here are **load-bearing, not incidental**:
        it is ``sigma / (1 + mu)`` that has to differ for +1 to be rejected,
        not ``sigma`` alone, and at equal means (7% and 7%) equal
        volatilities would make that ratio equal too -- see
        ``test_moment_matched_psd_accepts_correlation_one_when_ratio_matches``,
        which is exactly that case and is correctly accepted. A maintainer
        who believes correlation +1 is rejected unconditionally and
        "simplifies" this fixture to equal vols will watch the guard stop
        firing here and conclude the guard is broken; it would not be -- the
        fixture would have moved into the accepting case.

        Also pins the *number* the guard reports: the smallest eigenvalue of
        the annual matrix ``s_annual`` is ``-2.0676e-06``, and of the monthly
        matrix ``s_month = s_annual / MONTHS_PER_YEAR`` -- the one
        ``rng.multivariate_normal`` is actually handed -- is
        ``-1.723e-07``, twelve times smaller. Both figures are computed here
        independently of ``engine.mc.returns``, from the same moment-matching
        formula spelled out by hand rather than imported, so a regression
        back to checking ``s_annual`` prints the wrong one of the two and
        this test catches it rather than only the (identical either way)
        fact that *some* ``ValueError`` was raised."""
        vol = np.array([0.16, 0.18])
        corr = np.array([[1.0, 1.0], [1.0, 1.0]])
        cov = np.outer(vol, vol) * corr
        means = np.array([0.07, 0.07])

        # Independent of engine.mc.returns: the monthly eigenvalue the
        # message must report, from the moment-matching formula spelled out
        # by hand.
        growth = 1.0 + means
        s_annual = np.log(1.0 + cov / np.outer(growth, growth))
        expected_monthly_eigenvalue = float(np.linalg.eigvalsh(s_annual / 12.0).min())

        with pytest.raises(ValueError, match="moment-matched") as excinfo:
            generate(1, 3, 10, means, cov, n_persons=1)

        assert f"{expected_monthly_eigenvalue:g}" in str(excinfo.value)

    def test_moment_matched_psd_accepts_correlation_one_when_ratio_matches(self) -> None:
        """The accepting side of the same asymmetric rule, pinned here
        rather than only asserted in prose: two classes with correlation +1
        and equal ``sigma / (1 + mu)`` are the same lognormal distribution
        written twice, ``s`` is exactly rank one with a smallest eigenvalue
        of exactly zero, and there is a perfectly good distribution to draw
        from -- this is correct behaviour, not a hole in the guard above.
        Different volatilities on purpose (16% and a derived ~17.94%, not
        16% and 16%): it is the ratio that has to match, not the volatility
        itself, and using two different means (7% and 20%) with equal vols
        would not demonstrate that."""
        annual_means = np.array([0.07, 0.20])
        vol_0 = 0.16
        # Chosen so sigma / (1 + mu) matches exactly, not approximately.
        vol_1 = vol_0 * (1.0 + annual_means[1]) / (1.0 + annual_means[0])
        vol = np.array([vol_0, vol_1])
        corr = np.array([[1.0, 1.0], [1.0, 1.0]])
        cov = np.outer(vol, vol) * corr

        draws = generate(1, 3, 10, annual_means, cov, n_persons=1)

        assert draws.real_returns.shape == (3, 2, 10)


class TestAnnualMeansShapeGuard:
    """A 0-d ``annual_means`` -- a bare scalar such as ``0.05`` passed in
    place of ``np.array([0.05])``, the one-asset-class scenario written the
    wrong way -- would otherwise reach ``annual_means.shape[0]`` and raise
    ``IndexError: tuple index out of range``, naming neither the function
    nor the field. Out of contract, loud rather than silent, but still not
    ours to say why."""

    def test_generate_rejects_a_bare_scalar(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            generate(1, 3, 10, 0.05, np.array([[0.01]]), n_persons=1)

    def test_deterministic_rejects_a_bare_scalar(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            deterministic(3, 0.05, n_persons=1)


class TestFiniteGuard:
    """Without this guard, NaN or +/-inf clears every other guard in this
    module silently: ``nan <= 0.0`` and ``nan > tolerance`` are both
    ``False``, and ``np.linalg.eigvalsh`` of a matrix containing NaN returns
    NaN, which fails no comparison either."""

    def test_generate_rejects_nan_in_annual_means(self) -> None:
        means = np.array([np.nan, 0.02])
        with pytest.raises(ValueError, match="must be finite"):
            generate(1, 3, 10, means, TWO_ASSET_COV, n_persons=1)

    def test_generate_rejects_infinite_entry_in_annual_covariance(self) -> None:
        cov = np.array([[np.inf, 0.0], [0.0, 0.01]])
        with pytest.raises(ValueError, match="must be finite"):
            generate(1, 3, 10, TWO_ASSET_MEANS, cov, n_persons=1)

    def test_deterministic_rejects_nan_in_annual_means(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            deterministic(3, np.array([np.nan]), n_persons=1)

    def test_deterministic_rejects_infinite_entry_in_annual_means(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            deterministic(3, np.array([np.inf]), n_persons=1)


class TestCountValueErrors:
    """An empty draw set is the failure issue 11's review found for
    ``n_paths=0``: every downstream shape assertion passes vacuously against
    it, so it must be refused here rather than left to be discovered later."""

    def test_n_months_below_one(self) -> None:
        with pytest.raises(ValueError, match="n_months"):
            generate(1, 0, 10, TWO_ASSET_MEANS, TWO_ASSET_COV, n_persons=1)

    def test_n_paths_below_one(self) -> None:
        with pytest.raises(ValueError, match="n_paths"):
            generate(1, 3, 0, TWO_ASSET_MEANS, TWO_ASSET_COV, n_persons=1)

    def test_n_persons_below_one(self) -> None:
        with pytest.raises(ValueError, match="n_persons"):
            generate(1, 3, 10, TWO_ASSET_MEANS, TWO_ASSET_COV, n_persons=0)


class TestDeterministicCountValueErrors:
    """``__post_init__`` cannot supply these: it infers the axis lengths it
    validates from the array it is validating, so a zero count on ``deterministic``
    itself produces a small-but-consistent array rather than an error.
    """

    def test_n_months_below_one(self) -> None:
        """A run with zero months, silently accepted before this guard."""
        with pytest.raises(ValueError, match="n_months"):
            deterministic(0, TWO_ASSET_MEANS, n_persons=1)

    def test_n_persons_below_one(self) -> None:
        """A household with no people, silently accepted before this guard."""
        with pytest.raises(ValueError, match="n_persons"):
            deterministic(6, TWO_ASSET_MEANS, n_persons=0)

    def test_no_asset_classes(self) -> None:
        """A scenario with no asset classes, which
        ``Assumptions._check_asset_classes_exist`` exists specifically to
        prevent at the schema level; ``deterministic`` needs its own copy of
        the guard since it never goes through the schema."""
        with pytest.raises(ValueError, match="no asset classes"):
            deterministic(6, np.array([]), n_persons=1)


class TestDeterministicNoDispersion:
    """Every path-month-asset entry for a given asset is the same number."""

    def test_every_month_matches_the_first(self) -> None:
        draws = deterministic(6, TWO_ASSET_MEANS, n_persons=1)
        for asset in range(TWO_ASSET_MEANS.size):
            column = draws.real_returns[:, asset, 0]
            assert np.all(column == column[0])


class TestDeterministicSeed:
    def test_seed_is_the_deterministic_marker(self) -> None:
        draws = deterministic(6, TWO_ASSET_MEANS, n_persons=1)
        assert draws.seed == DETERMINISTIC_SEED


class TestDeterministicMortalityCoupling:
    """The coupling between this module and ``engine.core.mortality``.

    ``deterministic``'s uniforms must be accepted by
    ``engine.core.mortality.death_month_index`` and must land on the very
    last index of the survival curve -- the latest death the curve admits,
    which is what "removes dispersion, not mortality" means. Nothing else in
    either module's test suite exercises the two together.
    """

    START_YEAR = 2026
    TERMINAL_AGE = 40
    Q = 0.4

    @pytest.fixture
    def table(self, tmp_path: Path) -> ParamSet:
        year_dir = tmp_path / str(self.START_YEAR)
        year_dir.mkdir(parents=True, exist_ok=True)
        rows = "\n".join(f"    {age}: {self.Q}" for age in range(self.TERMINAL_AGE))
        text = f"""\
# SYNTHETIC TEST FIXTURE -- flat q(x), not a real mortality curve.
terminal_age_years: {self.TERMINAL_AGE}
q_x:
  f:
{rows}
    {self.TERMINAL_AGE}: 1.0
  m:
{rows}
    {self.TERMINAL_AGE}: 1.0
"""
        (year_dir / "mortality.yaml").write_text(text, encoding="utf-8")
        return load_year(self.START_YEAR, tmp_path)["mortality"]

    def test_deterministic_uniform_dies_at_the_last_index_of_the_curve(
        self, table: ParamSet
    ) -> None:
        curve = survival_curve(self.START_YEAR, 1, "f", self.START_YEAR, table)
        draws = deterministic(6, TWO_ASSET_MEANS, n_persons=1)

        u = draws.mortality[0, :]  # this person's uniform, across the one path
        death_month = death_month_index(u, curve)

        assert death_month.shape == (1,)
        assert death_month[0] == len(curve) - 1

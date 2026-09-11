# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Return and mortality draw generation.

Generated once per scenario, from an explicit seed, and reused across every
policy evaluation. Nothing here may be called from inside the optimizer's loop.

The simulation steps monthly, so the draws are monthly and the first axis of
every array is a month. Scenario assumptions are still expressed *annually*,
because that is how return and inflation assumptions are stated and argued
about; the conversion to a monthly distribution happens exactly once, in
:func:`engine.mc.moments.monthly_log_moments`, which :func:`generate` calls,
and is never repeated downstream.

There is no horizon in years to convert here, and there is no ``n_years``
parameter anywhere in this module. The simulation runs every path to the
second death and has no separate horizon (``docs/limitations.md`` L10), so the
month count is a property of the household's ages and the life table, not of a
scenario field. The caller derives it — the longest
:func:`engine.core.mortality.survival_curve` across the household, since every
path is dead by the end of it — and hands the count in as ``n_months``. That
derivation lives in ``engine/mc/simulate.py`` (issue 19), not here: a reader
of :func:`generate` or :func:`deterministic` should not go looking for a
scenario field that supplies the horizon, because there isn't one.

The attainability of the moments — ``1 + annual_means > 0``, a positive
moment-matching log argument, and a positive semi-definite moment-matched
monthly log-covariance — is checked in :mod:`engine.mc.moments`, which
:func:`generate` calls and which ``engine.scenario.schema.Assumptions``
calls when a scenario loads. This module's own guards (the counts,
finiteness, ``1 + annual_means > 0``, and the shape, symmetry and positive
semi-definiteness of ``annual_covariance``) run first. They protect direct
calls and tests, and are not redundant with the schema's checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from engine.core.timeline import MONTHS_PER_YEAR
from engine.mc.moments import monthly_log_moments

#: Tolerance for how far a covariance matrix entry may sit from its mirror
#: and still be treated as symmetric.
#:
#: Same value as ``engine.scenario.schema.TOLERANCE``, but a deliberate second
#: copy rather than an import: by the time a scenario reaches this module its
#: correlation matrix has already been validated by
#: ``engine.scenario.schema.Assumptions._check_correlation``, so this guard
#: exists for direct construction of a :class:`RandomDraws` and for tests, not
#: for the scenario path. A numerical tolerance on a covariance check here is
#: free to diverge from a correlation check in the scenario layer even though
#: both currently hold the same number.
_COVARIANCE_SYMMETRY_TOLERANCE: Final[float] = 1e-9

#: How negative the smallest eigenvalue of the covariance matrix may be and
#: still be treated as positive semi-definite.
#:
#: Same value and same reasoning as ``engine.scenario.schema.PSD_TOLERANCE``:
#: looser than :data:`_COVARIANCE_SYMMETRY_TOLERANCE` because it is the output
#: of an eigenvalue decomposition rather than of one subtraction, and a matrix
#: that is positive semi-definite in exact arithmetic routinely produces an
#: eigenvalue a few units in the last place below zero. Duplicated from the
#: schema module for the same reason as the tolerance above; do not delete
#: either guard as redundant with the schema's.
_COVARIANCE_PSD_TOLERANCE: Final[float] = 1e-8

#: The seed recorded on a :class:`RandomDraws` built by :func:`deterministic`.
#:
#: No draw is ever made from it — :func:`deterministic` calls neither
#: ``numpy.random.default_rng`` nor anything downstream of it — so this marks
#: "no draw was made", not a real stream. ``RandomDraws.seed`` still has to
#: hold something, and the field is not optional: a scenario replayed with
#: ``deterministic`` output must not read as "seed unknown".
DETERMINISTIC_SEED: Final[int] = 0


@dataclass(frozen=True, slots=True)
class RandomDraws:
    """The fixed random inputs to a scenario. Generated once, reused forever.

    Generated once per scenario and handed unchanged to every policy the
    optimizer evaluates — this is what "common random numbers" (design
    decision 2) means: path 123 sees the same market and the same death month
    under every policy, so the optimizer is comparing policies against the
    same draws rather than against noise. **Both array fields are read-only**
    for exactly the reason ``engine/core/state.py`` freezes its own arrays: a
    mutation anywhere, by any policy, would silently corrupt every other
    policy's view of the same run. ``__post_init__`` marks each array
    non-writeable directly, rather than through
    ``engine.core.state.freeze`` — that helper also walks a view's whole
    ``.base`` chain to guard against a caller retaining a writeable handle on
    a buffer this object shares memory with, a case that does not arise here:
    every array a caller passes in is fresh from :func:`generate` or
    :func:`deterministic`, not a slice of something a caller kept a reference
    to.

    Attributes:
        seed: The seed these were generated from. Recorded so a run can be
            reproduced exactly. :data:`DETERMINISTIC_SEED` on a
            :class:`RandomDraws` built by :func:`deterministic`.
        real_returns: Real returns *per month*, shape
            ``(n_months, n_assets, n_paths)``. Real, not nominal, and monthly,
            not annual. The naming is deliberate: an array that is silently
            annual has the right shape and the wrong magnitude, and produces a
            plausible answer.
        mortality: One uniform draw per person per path, shape ``(n_persons,
            n_paths)``, each value strictly inside the open interval
            ``(0, 1)``. Inverted through
            :func:`engine.core.mortality.survival_curve` by
            :func:`engine.core.mortality.death_month_index`, once per person
            per path, into a death month — never compared against a monthly
            hazard. (An earlier version of this field was
            ``(n_months, n_persons, n_paths)`` and was compared against a
            per-month Bernoulli hazard so that a death landed in a month
            rather than at a year boundary; issue 12 replaced that mechanism
            with the inversion above, which needs only one uniform per person
            per path.)
        n_months: Month count, for shape assertions at call sites.
        n_paths: Path count, for shape assertions at call sites.
    """

    seed: int
    real_returns: NDArray[np.float64]
    mortality: NDArray[np.float64]
    n_months: int
    n_paths: int

    def __post_init__(self) -> None:
        if self.real_returns.ndim != 3:
            raise ValueError(
                f"RandomDraws.real_returns: expected 3 dimensions "
                f"(n_months, n_assets, n_paths), got shape "
                f"{self.real_returns.shape}."
            )
        n_assets = self.real_returns.shape[1]
        expected_returns_shape = (self.n_months, n_assets, self.n_paths)
        if self.real_returns.shape != expected_returns_shape:
            raise ValueError(
                f"RandomDraws.real_returns: expected shape "
                f"{expected_returns_shape} (n_months={self.n_months}, "
                f"n_assets={n_assets}, n_paths={self.n_paths}), got "
                f"{self.real_returns.shape}."
            )

        if self.mortality.ndim != 2:
            raise ValueError(
                f"RandomDraws.mortality: expected 2 dimensions "
                f"(n_persons, n_paths), got shape {self.mortality.shape}."
            )
        n_persons = self.mortality.shape[0]
        expected_mortality_shape = (n_persons, self.n_paths)
        if self.mortality.shape != expected_mortality_shape:
            raise ValueError(
                f"RandomDraws.mortality: expected shape "
                f"{expected_mortality_shape} (n_persons={n_persons}, "
                f"n_paths={self.n_paths}), got {self.mortality.shape}."
            )

        self.real_returns.flags.writeable = False
        self.mortality.flags.writeable = False


def _check_covariance(
    annual_means: NDArray[np.float64], annual_covariance: NDArray[np.float64]
) -> None:
    """Raise ``ValueError`` if ``annual_covariance`` cannot be used as-is.

    Checks the same four things :func:`generate`'s docstring promises, each
    with its own message naming the offending value: not square, size not
    matching ``annual_means``, not symmetric, not positive semi-definite.

    The symmetry and PSD checks are the same shape of test
    ``engine.scenario.schema.Assumptions._check_correlation`` uses — the
    largest gap between an entry and its mirror, and the smallest eigenvalue
    against a negative tolerance — deliberately duplicated rather than
    imported; see :data:`_COVARIANCE_SYMMETRY_TOLERANCE`. There is no
    diagonal-is-one check here, unlike that one: ``annual_covariance`` is a
    covariance matrix, whose diagonal is a variance, not a correlation matrix,
    whose diagonal is forced to one.

    Also refuses an ``annual_means`` that is not 1-D, and refuses it empty,
    before any of the four checks above run. A 0-d ``annual_means`` -- a bare
    scalar such as ``0.05`` passed in place of ``np.array([0.05])`` -- would
    otherwise reach ``annual_means.shape[0]`` below and raise ``IndexError:
    tuple index out of range``, naming neither this function nor the field.
    With zero asset classes, the symmetry check's ``np.abs(...).max()`` is a
    reduction over an empty array, which raises numpy's own ``ValueError``
    ("zero-size array to reduction operation maximum which has no identity")
    naming neither this function nor the reason. Same situation
    ``engine.scenario.schema.Assumptions._check_asset_classes_exist`` exists
    to prevent for a scenario; these are the same guards for direct
    construction and for tests.
    """
    if annual_means.ndim != 1:
        raise ValueError(
            f"annual_means: expected a 1-D array of per-asset-class means, "
            f"got {annual_means.ndim} dimension(s), shape "
            f"{annual_means.shape}. A bare scalar (e.g. 0.05) is not the "
            f"same as a one-element array (np.array([0.05]))."
        )
    n_assets = annual_means.shape[0]
    if n_assets < 1:
        raise ValueError(
            "annual_means: no asset classes given (empty array), so there is "
            "nothing for annual_covariance to be a covariance of."
        )
    if annual_covariance.ndim != 2 or annual_covariance.shape[0] != annual_covariance.shape[1]:
        raise ValueError(
            f"annual_covariance: expected a square matrix, got shape "
            f"{annual_covariance.shape}."
        )
    if annual_covariance.shape[0] != n_assets:
        raise ValueError(
            f"annual_covariance: expected a {n_assets}x{n_assets} matrix to "
            f"match annual_means's {n_assets} asset class(es), got "
            f"{annual_covariance.shape[0]}x{annual_covariance.shape[1]}."
        )

    off_diagonal = float(np.abs(annual_covariance - annual_covariance.T).max())
    if off_diagonal > _COVARIANCE_SYMMETRY_TOLERANCE:
        raise ValueError(
            f"annual_covariance: not symmetric; the largest gap between an "
            f"entry and its mirror is {off_diagonal:g}."
        )

    smallest = float(np.linalg.eigvalsh(annual_covariance).min())
    if smallest < -_COVARIANCE_PSD_TOLERANCE:
        raise ValueError(
            f"annual_covariance: not positive semi-definite; its smallest "
            f"eigenvalue is {smallest:g}."
        )


def _check_finite(name: str, array: NDArray[np.float64]) -> None:
    """Raise ``ValueError`` unless every entry of ``array`` is finite.

    NaN and +/-inf both clear every guard in this module silently: ``nan <=
    0.0`` and ``nan > tolerance`` are both ``False``, and
    ``np.linalg.eigvalsh`` of a matrix containing NaN returns NaN, which
    fails no comparison either.

    This is a guard for direct construction of a :class:`RandomDraws`, like
    every guard above it in this module. ``engine.mc.moments`` refuses
    non-finite moments when a scenario loads, naming the asset class.

    Args:
        name: The parameter name to report in the message, e.g.
            ``"annual_means"``.
        array: The array to check. Any shape, including 0-d.

    Raises:
        ValueError: If any entry of ``array`` is not finite, naming ``name``,
            the entry's index, and its value.
    """
    finite = np.isfinite(array)
    if not np.all(finite):
        # np.argwhere on a boolean array whose *value* array is 0-d gives a
        # result whose .size is always 0 regardless of content (shape
        # (1, 0)), which would silently miss a NaN scalar; ravelling first
        # avoids that degenerate case for every shape, including 0-d.
        flat_index = int(np.flatnonzero(~finite.ravel())[0])
        index = np.unravel_index(flat_index, array.shape)
        subscript = "".join(f"[{i}]" for i in index)
        value = array.reshape(-1)[flat_index]
        raise ValueError(f"{name}{subscript} = {value!r}: must be finite (not NaN or +/-inf).")


def _check_growth_positive(annual_means: NDArray[np.float64]) -> None:
    """Raise ``ValueError`` unless ``1 + annual_means`` is strictly positive
    everywhere.

    Same reasoning ``engine.scenario.schema.Assumptions.inflation`` carries as
    ``Field(gt=-1.0)``: at or below -1, ``(1 + mu)`` raised to a fractional
    power is a complex number rather than an error, and ``log(1 + mu)`` is
    undefined at ``mu <= -1``. ``engine.mc.moments`` enforces the same rule
    when a scenario loads; this copy guards direct calls to :func:`generate`
    and :func:`deterministic`.

    Args:
        annual_means: Expected real annual return per asset class.

    Raises:
        ValueError: If ``1 + annual_means[i] <= 0`` for any ``i``, naming the
            index and the value.
    """
    offending = np.flatnonzero(1.0 + annual_means <= 0.0)
    if offending.size:
        index = int(offending[0])
        raise ValueError(
            f"annual_means[{index}] = {annual_means[index]!r}: 1 + "
            f"annual_means must be strictly positive. At or below -1, "
            f"(1 + mu) raised to a fractional power is a complex number "
            f"rather than an error, and log(1 + mu) is undefined."
        )


def generate(
    seed: int,
    n_months: int,
    n_paths: int,
    annual_means: NDArray[np.float64],
    annual_covariance: NDArray[np.float64],
    n_persons: int,
) -> RandomDraws:
    """Generate the full set of monthly random draws for a scenario.

    Called exactly once, before the optimizer starts. The returned draws are
    passed to every policy evaluation unchanged.

    **Asset order is positional and is
    ``engine.scenario.schema.Assumptions.asset_class_names`` order** — the
    same convention ``assumptions.correlation`` already follows, where
    nothing in the matrix names a class and a reordered ``asset_classes``
    block reinterprets every entry without changing a number.
    ``annual_means[i]`` and ``annual_covariance[i][j]`` are read as that
    class, and ``real_returns[:, i, :]`` on the result is that class's
    monthly draws. ``asset_class_names`` is the single place that order is
    read from; nothing here re-derives or re-states it.

    Converting an annual assumption to a monthly one is a modelling decision,
    not arithmetic, and this function states which convention it uses and
    holds to it. The requirement it satisfies:
    ``E[prod(1 + r_month) over 12 months] == 1 + mu`` — twelve monthly draws
    compounded together must reproduce the specified annual arithmetic mean,
    not merely the annual mean divided by twelve, which understates
    compounding. That equality is what the moment matching in
    :func:`engine.mc.moments.monthly_log_moments` is for, and it is the
    reason the compounding test in ``tests/mc/test_returns.py`` exists.

    The conversion is :func:`engine.mc.moments.monthly_log_moments`: the
    annual arithmetic means and the covariance of annual simple returns go
    in, the mean and covariance of *monthly* log-returns come out. Monthly
    log-returns are drawn iid multivariate normal from that distribution,
    and the simple returns handed back are ``exp(x) - 1``.

    The random stream, in order: :class:`numpy.random.Generator` is built once
    from ``seed`` with ``numpy.random.default_rng``, the return draws are
    taken from it first, and the mortality uniforms are taken from it second.
    That order is part of the contract, not an implementation detail — it is
    what keeps the stream fixed under design decision 2: a caller that adds a
    third draw between these two, or reorders them, changes what every
    existing seed produces.

    **The axis order is the trap in this function.**
    ``rng.multivariate_normal(mean, cov, size=(n_months, n_paths))`` returns
    ``(n_months, n_paths, n_assets)``, and the contract on
    :attr:`RandomDraws.real_returns` is ``(n_months, n_assets, n_paths)``. The
    asset axis is moved explicitly with :func:`numpy.moveaxis` rather than
    relied on to already be in the right place: a transposed result has a
    plausible shape whenever ``n_assets == n_paths``, and when it does not, it
    fails somewhere far downstream with a broadcasting error that names
    neither this function nor the axis.

    Mortality uniforms: shape ``(n_persons, n_paths)``, drawn with
    ``rng.random()``, which returns values in ``[0, 1)`` — so an exact zero is
    possible, not hypothetical, and
    :func:`engine.core.mortality.death_month_index` refuses it. Any exact zero
    drawn is replaced with ``numpy.nextafter(0.0, 1.0)``, the smallest
    positive double, rather than left to chance: relying on ``rng.random()``
    never happening to draw exactly ``0.0`` across every path of every run
    this engine will ever be asked to do is not a guarantee, and the
    replacement is cheap and exact where it applies and a no-op everywhere
    else.

    Validation happens before either draw. ``filterwarnings = ["error"]`` is
    set for the test suite, and ``rng.multivariate_normal`` raises a
    ``RuntimeWarning`` — a test error, under that setting — on a covariance it
    considers non-positive-semi-definite. Checking ``s_month`` ourselves
    first, with :func:`engine.mc.moments.monthly_log_moments`, means our own
    ``ValueError``, naming the matrix, fires before numpy's warning ever has
    the chance to; ``check_valid`` is deliberately not passed to
    ``rng.multivariate_normal``, so that failure mode is not inherited along
    with numpy's own wording.

    Note that a scenario built through ``engine.scenario.schema.Scenario``
    has already had its correlation matrix validated by
    ``Assumptions._check_correlation`` by the time it reaches here — these
    guards are for direct construction of a :class:`RandomDraws` and for
    tests, not for the scenario path. They stay regardless: do not delete
    them as redundant with the schema's.

    Args:
        seed: Fixed seed. The same seed must reproduce identical draws.
        n_months: Number of months to draw. See the module docstring for
            where this count comes from — never a scenario field.
        n_paths: Number of Monte Carlo paths.
        annual_means: Expected **real annual** return per asset class,
            ``(n_assets,)``, as bare fractions, in
            ``Assumptions.asset_class_names`` order.
        annual_covariance: Covariance matrix of real **annual** simple
            returns, ``(n_assets, n_assets)``, same order.
        n_persons: Number of persons in the household.

    Returns:
        A frozen :class:`RandomDraws` with monthly draws.

    Raises:
        ValueError: Listed in the order the checks actually run, since a
            caller fixing one failure at a time meets them in this order and
            not the order they might be listed. If ``n_months``, ``n_paths``,
            or ``n_persons`` is below one. If ``annual_means`` or
            ``annual_covariance`` contains a NaN or an infinite entry. If
            ``1 + annual_means`` is not strictly positive everywhere. If
            ``annual_means`` is not 1-D, is empty, or ``annual_covariance``
            is not square, its size does not match ``annual_means``, it is
            not symmetric, or it is not positive semi-definite. If the
            moment-matching log argument is not strictly positive for some
            ``i, j``. If the moment-matched monthly log-covariance is not
            positive semi-definite — **this can happen even when
            ``annual_covariance`` itself passed every check above**, because
            the moment-matching transform does not preserve positive
            semi-definiteness; see :mod:`engine.mc.moments`.
    """
    if n_months < 1:
        raise ValueError(f"generate: n_months must be at least 1, got {n_months}.")
    if n_paths < 1:
        raise ValueError(f"generate: n_paths must be at least 1, got {n_paths}.")
    if n_persons < 1:
        raise ValueError(f"generate: n_persons must be at least 1, got {n_persons}.")

    annual_means = np.asarray(annual_means, dtype=np.float64)
    annual_covariance = np.asarray(annual_covariance, dtype=np.float64)
    _check_finite("annual_means", annual_means)
    _check_finite("annual_covariance", annual_covariance)
    _check_growth_positive(annual_means)
    _check_covariance(annual_means, annual_covariance)

    m_month, s_month = monthly_log_moments(annual_means, annual_covariance)

    rng = np.random.default_rng(seed)

    # rng.multivariate_normal(..., size=(n_months, n_paths)) returns
    # (n_months, n_paths, n_assets); move the asset axis (2) to position 1 to
    # reach the (n_months, n_assets, n_paths) contract. Explicit and
    # commented on purpose: see the axis-order note in this function's
    # docstring.
    monthly_log_returns = rng.multivariate_normal(m_month, s_month, size=(n_months, n_paths))
    real_returns = np.exp(monthly_log_returns) - 1.0
    real_returns = np.moveaxis(real_returns, 2, 1)

    mortality = rng.random(size=(n_persons, n_paths))
    mortality = np.where(mortality == 0.0, np.nextafter(0.0, 1.0), mortality)

    return RandomDraws(
        seed=seed,
        real_returns=real_returns,
        mortality=mortality,
        n_months=n_months,
        n_paths=n_paths,
    )


def deterministic(
    n_months: int,
    annual_means: NDArray[np.float64],
    n_persons: int,
) -> RandomDraws:
    """Draws with zero volatility: one path, every month at the mean.

    The bridge between the deterministic single-path check and Monte Carlo.
    Running the simulator with these must reproduce the hand-checked
    spreadsheet exactly; that equality is the test that says the Monte Carlo
    wrapper introduced no error of its own.

    Every month gets the monthly return that compounds to ``annual_means``
    over twelve months — ``(1 + mu) ** (1 / 12) - 1`` per asset, not
    ``mu / 12``, which understates compounding the same way it would in
    :func:`generate`.

    Mortality uniforms are ``numpy.nextafter(0.0, 1.0)``, the smallest
    positive double, shape ``(n_persons, 1)`` — never ``0.0`` and never
    ``1.0``. Both endpoints are excluded for reasons specific to this
    function's purpose, not merely because
    :func:`engine.core.mortality.death_month_index` raises on either:
    ``deterministic`` exists to remove *dispersion*, not mortality, so the
    single path must survive as long as the life table allows and die only at
    the very last index of its survival curve — the latest death the curve
    admits. ``u == 1.0`` would instead give the *earliest* possible death,
    not the latest: ``curve[1] < 1`` for any positive hazard, so
    ``death_month_index`` would return month 1 for it, the opposite of what a
    zero-dispersion draw should mean. Using the smallest positive double
    instead of exactly ``0.0`` costs nothing: ``death_month_index`` requires
    every ``u`` strictly inside the open interval ``(0, 1)`` and raises on
    either endpoint, and "nobody dies" is unreachable by construction anyway,
    since every path dies by the table's terminal age
    (``docs/limitations.md`` L10).

    ``RandomDraws.seed`` is set to :data:`DETERMINISTIC_SEED`: no draw is made
    here, so there is no stream for a seed to identify, but the field is not
    optional and a plausible-looking seed would misrepresent that.

    Args:
        n_months: Number of months to draw. See the module docstring for
            where this count comes from — never a scenario field.
        annual_means: Expected real annual return per asset class,
            ``(n_assets,)``, in ``Assumptions.asset_class_names`` order.
        n_persons: Number of persons in the household.

    Returns:
        A :class:`RandomDraws` with ``n_paths == 1`` and no dispersion.

    Raises:
        ValueError: Listed in the order the checks actually run. If
            ``n_months`` or ``n_persons`` is below one. If ``annual_means``
            contains a NaN or an infinite entry. If ``annual_means`` is not
            1-D -- a bare scalar such as ``0.05`` passed in place of
            ``np.array([0.05])`` would otherwise reach
            ``annual_means.shape[0]`` below and raise ``IndexError: tuple
            index out of range``, naming neither this function nor the
            field. If ``annual_means`` is empty. If ``1 + annual_means`` is
            not strictly positive everywhere — this last one is not
            hypothetical: without it, ``annual_means = -5`` (a
            percent-for-fraction typo on -5%) produces an all-NaN
            ``real_returns`` with only a printed
            ``invalid value encountered in power``, and this function is the
            one the hand-checked spreadsheet in issue 21 is compared against,
            so a silent all-NaN array there fails that comparison with
            nothing saying why.
    """
    if n_months < 1:
        raise ValueError(f"deterministic: n_months must be at least 1, got {n_months}.")
    if n_persons < 1:
        raise ValueError(f"deterministic: n_persons must be at least 1, got {n_persons}.")

    annual_means = np.asarray(annual_means, dtype=np.float64)
    _check_finite("annual_means", annual_means)
    if annual_means.ndim != 1:
        raise ValueError(
            f"annual_means: expected a 1-D array of per-asset-class means, "
            f"got {annual_means.ndim} dimension(s), shape "
            f"{annual_means.shape}. A bare scalar (e.g. 0.05) is not the "
            f"same as a one-element array (np.array([0.05]))."
        )
    n_assets = annual_means.shape[0]
    if n_assets < 1:
        raise ValueError(
            "annual_means: no asset classes given (empty array), so there is "
            "nothing for deterministic to draw a return for."
        )
    _check_growth_positive(annual_means)

    monthly_return = (1.0 + annual_means) ** (1.0 / MONTHS_PER_YEAR) - 1.0

    # A fresh, owned array (not a broadcast view onto monthly_return): every
    # month-asset-path entry for a given asset is the same number, by
    # construction, and there is nothing else it could share memory with.
    ones = np.ones((n_months, n_assets, 1), dtype=np.float64)
    real_returns = ones * monthly_return[None, :, None]

    mortality = np.full((n_persons, 1), np.nextafter(0.0, 1.0), dtype=np.float64)

    return RandomDraws(
        seed=DETERMINISTIC_SEED,
        real_returns=real_returns,
        mortality=mortality,
        n_months=n_months,
        n_paths=1,
    )

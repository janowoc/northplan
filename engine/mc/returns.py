# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Return and mortality draw generation.

Generated once per scenario, from an explicit seed, and reused across every policy
evaluation. Draws are monthly, the first axis of every array; scenario assumptions
are annual and converted to monthly exactly once, in
:func:`engine.mc.moments.monthly_log_moments`, called by :func:`generate`.

There is no ``n_months`` derived here: the simulation runs every path to the second
death with no separate horizon (``docs/limitations.md`` L10), so the caller derives
the month count from the household's ages and the life table
(:func:`engine.core.mortality.months_to_terminal`, applied by
``engine/core/build.py``) and hands it in.

Attainability of the moments is checked in :mod:`engine.mc.moments`, called both
here and by ``engine.scenario.schema.Assumptions`` when a scenario loads; this
module's own guards run first and protect direct calls and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from engine.core.timeline import MONTHS_PER_YEAR
from engine.mc.moments import monthly_log_moments

#: Tolerance for how far a covariance matrix entry may sit from its mirror
#: and still count as symmetric. A deliberate duplicate of
#: ``engine.scenario.schema.TOLERANCE``, not an import — this guard exists
#: for direct construction of :class:`RandomDraws` and for tests.
_COVARIANCE_SYMMETRY_TOLERANCE: Final[float] = 1e-9

#: How negative the smallest eigenvalue of the covariance matrix may be and
#: still count as positive semi-definite. Looser than
#: :data:`_COVARIANCE_SYMMETRY_TOLERANCE` since it is an eigenvalue, not a
#: subtraction. Duplicated from ``engine.scenario.schema.PSD_TOLERANCE``.
_COVARIANCE_PSD_TOLERANCE: Final[float] = 1e-8

#: The seed recorded on a :class:`RandomDraws` built by :func:`deterministic`.
#: No draw is ever made from it — :func:`deterministic` never touches
#: ``numpy.random.default_rng`` — so this marks "no draw was made", not a
#: real stream.
DETERMINISTIC_SEED: Final[int] = 0


@dataclass(frozen=True, slots=True)
class RandomDraws:
    """The fixed random inputs to a scenario. Generated once, reused forever.

    Handed unchanged to every policy the optimizer evaluates (common random numbers).
    Both arrays are read-only, marked in ``__post_init__`` since every array is fresh
    from :func:`generate` or :func:`deterministic`, never a slice a caller held
    writeable.

    Attributes:
        seed: The seed these were generated from; :data:`DETERMINISTIC_SEED` for
            :func:`deterministic`.
        real_returns: Real returns per month, ``(n_months, n_assets, n_paths)``.
        mortality: One uniform draw per person per path, ``(n_persons, n_paths)``,
            each in ``(0, 1)``, inverted into a death month by
            :func:`engine.core.mortality.death_month_index`.
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

    Checks the same four things :func:`generate`'s docstring promises: not square,
    size not matching ``annual_means``, not symmetric, not positive semi-definite —
    each with its own message. Symmetry and PSD checks mirror
    ``engine.scenario.schema.Assumptions._check_correlation``, deliberately
    duplicated rather than imported (no diagonal-is-one check here, since this is a
    covariance, not a correlation matrix).

    Also refuses an ``annual_means`` that is not 1-D or is empty, before the checks
    above run, so the failure names this function rather than a bare ``IndexError``.
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

    NaN and +/-inf both pass every other guard in this module silently (e.g.
    ``nan <= 0.0`` is ``False``), so this check must run first.

    Args:
        name: Parameter name to report in the message, e.g. ``"annual_means"``.
        array: Array to check, any shape, including 0-d.

    Raises:
        ValueError: If any entry is not finite, naming ``name``, the index, and value.
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
    """Raise ``ValueError`` unless ``1 + annual_means`` is strictly positive everywhere.

    At or below -1, ``log(1 + mu)`` is undefined — the same rule
    ``engine.scenario.schema.Assumptions.inflation`` enforces via ``Field(gt=-1.0)``,
    duplicated here to guard direct calls to :func:`generate` and :func:`deterministic`.

    Args:
        annual_means: Expected real annual return per asset class.

    Raises:
        ValueError: If ``1 + annual_means[i] <= 0`` for any ``i``, naming the index.
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

    Asset order is positional (``Assumptions.asset_class_names``). Log-returns come from
    :func:`engine.mc.moments.monthly_log_moments`, which moment-matches the monthly
    lognormal draw to the annual mean and covariance given. The asset axis is moved with
    :func:`numpy.moveaxis` to match :attr:`RandomDraws.real_returns`, and an exact ``0.0``
    mortality uniform becomes the smallest positive double, since
    :func:`engine.core.mortality.death_month_index` refuses either endpoint.

    The random stream is drawn in a fixed order — real-return draws first, mortality
    uniforms second — and that order is part of the contract: every recorded seed and
    every characterization snapshot depends on it.

    Args:
        seed: Fixed seed; the same seed reproduces identical draws.
        n_months: Number of months to draw. See the module docstring.
        n_paths: Number of Monte Carlo paths.
        annual_means: Expected real annual return per asset class, as bare fractions
            (``0.05``, not ``5``), ``(n_assets,)``.
        annual_covariance: Covariance of real annual simple returns, ``(n_assets,
            n_assets)``, same order.
        n_persons: Number of persons in the household.

    Returns:
        A frozen :class:`RandomDraws` with monthly draws.

    Raises:
        ValueError: If any count is below one; if the inputs are not finite; if
            ``1 + annual_means`` is not strictly positive everywhere; if
            ``annual_means`` is not 1-D, is empty, or ``annual_covariance`` is not
            square, mismatched, not symmetric, or not PSD; if the moment-matching
            log argument is not strictly positive for some asset pair; or if the
            moment-matched covariance fails PSD despite that.
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

    The bridge between the deterministic single-path check and Monte Carlo: running
    the simulator with these must reproduce the hand-checked spreadsheet exactly.
    Each month's return is ``(1 + mu) ** (1 / 12) - 1`` per asset — the value that
    compounds to ``annual_means`` over twelve months, not ``mu / 12``.

    Mortality uniforms are ``numpy.nextafter(0.0, 1.0)``, shape ``(n_persons, 1)``:
    never ``0.0`` or ``1.0``, so the single path survives to the life table's latest
    death rather than its earliest (``u == 1.0`` would give the earliest, since
    ``curve[1] < 1`` for any positive hazard).

    Args:
        n_months: Number of months to draw. See the module docstring.
        annual_means: Expected real annual return per asset class, ``(n_assets,)``.
        n_persons: Number of persons in the household.

    Returns:
        A :class:`RandomDraws` with ``n_paths == 1`` and no dispersion.

    Raises:
        ValueError: If ``n_months`` or ``n_persons`` is below one; if
            ``annual_means`` is not finite, not 1-D, empty, or ``1 + annual_means``
            is not strictly positive everywhere.
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

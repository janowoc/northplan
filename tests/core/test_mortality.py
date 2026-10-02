# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Survival curves and the per-path death draw: ``engine.core.mortality``.

The table used throughout is synthetic: a constant annual ``q`` at every age,
one constant per sex, with the terminal row at ``1.0`` that the engine's own
convention requires
(L10). A real table falls to a trough in early childhood and then rises; a
flat table has no such shape and could never be mistaken for a transcribed
one. It is built here, once, and written to a ``mortality.yaml`` under
``tmp_path`` so the module under test is exercised through the same loader
path (``engine.params.loader.load_year``) production code uses, rather than a
hand-built ``ParamSet`` that could drift from what the loader actually
produces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.core.mortality import (
    death_month_index,
    monthly_hazard,
    months_to_terminal,
    survival_curve,
)
from engine.core.timeline import MONTHS_PER_YEAR
from engine.params.loader import ParamSet, load_year

START_YEAR = 2026

#: Constant annual death probability for the ``f`` synthetic table. Not a
#: real q(x): chosen only so that the geometric-distribution test below has a
#: mean of 1/monthly_hazard(0.3) = 34.1 months, which sits far inside the
#: curve relative to the terminal age, so truncation at the terminal age
#: cannot bias the sample mean by more than a small fraction of the test's
#: tolerance.
SYNTHETIC_Q_F = 0.3

#: Constant annual death probability for the ``m`` synthetic table.
#: Deliberately different from SYNTHETIC_Q_F, for the same reason the real
#: template's two-repdigit convention makes ``f`` and ``m`` visibly different
#: values: a table that used the same number for both sexes could not catch
#: code that read the wrong one. Not chosen for any distributional property
#: of its own — nothing here runs a geometric-distribution test against it.
SYNTHETIC_Q_M = 0.6

#: Terminal age for the synthetic table. Chosen relative to SYNTHETIC_Q_F so
#: that ``(1 - monthly_hazard(SYNTHETIC_Q_F)) ** (TERMINAL_AGE * 12)`` — the
#: probability of surviving to the terminal age at all — is astronomically
#: small (~2e-8), which is what makes the truncation bias in
#: TestGeometricDistribution negligible rather than merely small.
TERMINAL_AGE = 50


def _write_synthetic_table(
    root: Path,
    terminal_age: int = TERMINAL_AGE,
    q_f: float = SYNTHETIC_Q_F,
    q_m: float = SYNTHETIC_Q_M,
) -> None:
    """Write a synthetic ``mortality.yaml`` — flat q per sex, terminal row at 1.0.

    ``f`` and ``m`` are flat at two different constants, not the same one:
    the real template's two-repdigit convention exists to catch a sex
    mix-up, and a synthetic fixture that gave both sexes the same table would
    have exactly the blind spot that convention exists to close.
    """
    year_dir = root / str(START_YEAR)
    year_dir.mkdir(parents=True, exist_ok=True)
    f_rows = "\n".join(f"    {age}: {q_f}" for age in range(terminal_age))
    m_rows = "\n".join(f"    {age}: {q_m}" for age in range(terminal_age))
    text = f"""\
# SYNTHETIC TEST FIXTURE — flat q(x) per sex, not a real mortality curve.
terminal_age_years: {terminal_age}
q_x:
  f:
{f_rows}
    {terminal_age}: 1.0
  m:
{m_rows}
    {terminal_age}: 1.0
"""
    (year_dir / "mortality.yaml").write_text(text, encoding="utf-8")


@pytest.fixture
def table(tmp_path: Path) -> ParamSet:
    _write_synthetic_table(tmp_path)
    return load_year(START_YEAR, tmp_path)["mortality"]


class TestMonthlyHazard:
    def test_compounds_back_to_the_annual_probability(self) -> None:
        """The only content of monthly_hazard; would catch a stray 1/12 -> /12."""
        h = monthly_hazard(SYNTHETIC_Q_F)
        assert (1 - h) ** MONTHS_PER_YEAR == pytest.approx(1 - SYNTHETIC_Q_F)

    def test_zero_hazard_is_zero(self) -> None:
        assert monthly_hazard(0.0) == pytest.approx(0.0)

    def test_certain_death_is_certain_every_month(self) -> None:
        assert monthly_hazard(1.0) == pytest.approx(1.0)

    def test_broadcasts_over_an_array(self) -> None:
        result = monthly_hazard(np.array([0.0, SYNTHETIC_Q_F, 1.0]))
        assert result.shape == (3,)
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)


class TestSurvivalCurveShape:
    def test_curve_zero_is_one(self, table: ParamSet) -> None:
        curve = survival_curve(START_YEAR, 1, "f", START_YEAR, table)
        assert curve[0] == 1.0

    def test_final_entry_is_exactly_zero(self, table: ParamSet) -> None:
        curve = survival_curve(START_YEAR, 1, "f", START_YEAR, table)
        assert curve[-1] == 0.0

    def test_curve_length_is_terminal_age_in_months_plus_two(self, table: ParamSet) -> None:
        """Person born at the very start of the run: M = TERMINAL_AGE * 12."""
        curve = survival_curve(START_YEAR, 1, "f", START_YEAR, table)
        assert len(curve) == TERMINAL_AGE * MONTHS_PER_YEAR + 2

    def test_curve_is_non_writeable(self, table: ParamSet) -> None:
        """Computed once per person and handed to every path; must not be mutable."""
        curve = survival_curve(START_YEAR, 1, "f", START_YEAR, table)
        assert not curve.flags.writeable

    def test_already_past_terminal_age_dies_in_month_zero(self, table: ParamSet) -> None:
        """No q row exists past the terminal age; the curve is a bare [0.0]."""
        curve = survival_curve(START_YEAR - (TERMINAL_AGE + 1), 1, "f", START_YEAR, table)
        assert curve.tolist() == [0.0]
        assert death_month_index(np.array([0.5]), curve).tolist() == [0]

    def test_exactly_at_terminal_age_at_start_gets_one_alive_month(self, table: ParamSet) -> None:
        """The boundary the case above cannot distinguish: q=1 is still applied once."""
        curve = survival_curve(START_YEAR - TERMINAL_AGE, 1, "f", START_YEAR, table)
        assert curve.tolist() == [1.0, 0.0]
        assert death_month_index(np.array([0.5]), curve).tolist() == [1]


class TestSexIsRespected:
    """Would catch ``sex`` hardcoded to ``"f"``.

    With ``f`` and ``m`` on different constants, a curve built for ``sex="m"``
    that actually read the ``f`` row would produce ``curve_f``'s numbers
    under a ``sex="m"`` label — exactly what these two tests are shaped to
    catch, since the first compares the two curves and the second checks the
    ``m`` curve against a value computed from ``SYNTHETIC_Q_M`` alone, never
    calling ``survival_curve`` or ``monthly_hazard`` to get it.
    """

    def test_f_and_m_curves_differ(self, table: ParamSet) -> None:
        curve_f = survival_curve(START_YEAR, 1, "f", START_YEAR, table)
        curve_m = survival_curve(START_YEAR, 1, "m", START_YEAR, table)
        assert curve_f[1] != curve_m[1]

    def test_m_curve_first_entry_matches_the_m_hazard_by_hand(self, table: ParamSet) -> None:
        """curve[1] = 1.0 * (1 - monthly_hazard(q)) = (1 - q) ** (1/12).

        Written out arithmetically from SYNTHETIC_Q_M, the literal the fixture
        was built from — not derived by calling monthly_hazard or
        survival_curve, so a bug in either could not make this test agree
        with it by construction.
        """
        expected = (1 - SYNTHETIC_Q_M) ** (1 / MONTHS_PER_YEAR)
        curve_m = survival_curve(START_YEAR, 1, "m", START_YEAR, table)
        assert curve_m[1] == pytest.approx(expected)


class TestBirthMonthIsRespected:
    """Would catch ``year - birth_year`` in place of
    ``age_in_years``, which agrees with the real arithmetic only when
    ``birth_month == 1`` — the one value every other test in this file uses.
    """

    def test_curve_length_for_a_person_born_mid_year(self, table: ParamSet) -> None:
        """Born June of the year before start_year: age 0 at month index 0.

        Arithmetic, by hand, not from the function under test:
        - Birth is June of ``START_YEAR - 1``; month index 0 is January of
          ``START_YEAR``, seven whole months later (Jun->Jul->Aug->Sep->Oct->
          Nov->Dec->Jan), so age in months at index ``i`` is ``7 + i``.
        - The terminal age in months is reached when ``7 + i == 50 * 12 ==
          600``, i.e. at ``i = 593``. That is ``M`` in survival_curve's own
          terms.
        - Curve length is ``M + 2 == 595``.
        """
        curve = survival_curve(START_YEAR - 1, 6, "f", START_YEAR, table)
        assert len(curve) == 595


class TestDeathMonthIndexOrdering:
    def test_u_near_zero_dies_later_than_u_near_one(self, table: ParamSet) -> None:
        curve = survival_curve(START_YEAR, 1, "f", START_YEAR, table)
        late = death_month_index(np.array([1e-9]), curve)[0]
        early = death_month_index(np.array([1 - 1e-9]), curve)[0]
        assert late > early


class TestDeathMonthIndexMatchesTheCurve:
    """``P(D > i) == curve[i]`` — the identity that makes death_month_index correct.

    Replaces a prior test asserting ``deaths.max() <= len(curve) - 1``, which
    holds for *any* implementation given the ``u in (0, 1)`` guard and
    ``curve[-1] == 0``: ``searchsorted`` never returns less than 1, so the
    result is at most ``len(curve) - 1`` no matter what the curve contains.
    Its second assertion, ``curve[-1] == 0.0``, duplicates
    ``TestSurvivalCurveShape.test_final_entry_is_exactly_zero`` outright.
    Neither line can fail against a broken inversion.

    This instead checks, at each of several indexes including one deep in the
    tail, that the empirical fraction of draws with ``death_month_index > i``
    matches ``curve[i]`` — the property tying ``death_month_index`` back to
    the curve it was built from, rather than a bound every implementation
    satisfies trivially.
    """

    N_DRAWS = 200_000
    TOLERANCE_IN_SE = 8
    SEED = 20260910
    #: A handful spread across the curve. The tail index is 200, not
    #: something deeper: at index 300 ``curve[i]`` is 1.3e-4, and 8 binomial
    #: standard errors of a proportion that small is 2.1e-4 — wider than the
    #: proportion itself, so the lower bound falls below zero and an
    #: implementation returning *no* survivors past month 300 would pass. At
    #: 200 the expected count is 525 against a window of 183, which
    #: constrains the tail from both sides.
    #:
    #: The one-month shift this class exists to catch is caught at the low
    #: indexes, not in the tail: at index 10 the shift is 2.2e-2 against a
    #: 7.8e-3 tolerance, while in the tail the shift shrinks with curve[i]
    #: faster than the tolerance does. Both kinds of index are here on
    #: purpose.
    INDEXES = (10, 34, 100, 200)

    def test_empirical_survival_matches_the_curve(self, table: ParamSet) -> None:
        curve = survival_curve(START_YEAR, 1, "f", START_YEAR, table)
        rng = np.random.default_rng(self.SEED)
        u = rng.uniform(size=self.N_DRAWS)
        deaths = death_month_index(u, curve)

        for index in self.INDEXES:
            p = float(curve[index])
            empirical = float(np.count_nonzero(deaths > index)) / self.N_DRAWS
            # Binomial standard error of a sample proportion; tolerance is
            # TOLERANCE_IN_SE multiples of it, the same derivation
            # TestGeometricDistribution's mean tolerance uses.
            standard_error = (p * (1 - p) / self.N_DRAWS) ** 0.5
            tolerance = self.TOLERANCE_IN_SE * standard_error
            assert empirical == pytest.approx(p, abs=tolerance), (
                f"index {index}: empirical P(D > {index}) = {empirical} does not "
                f"match curve[{index}] = {p} within {tolerance}"
            )


class TestDeathMonthIndexRejectsTheEndpoints:
    def test_rejects_u_equal_zero(self, table: ParamSet) -> None:
        """At u == 0 no index satisfies curve[i] < u even though the curve ends
        at 0; the natural implementation returns an out-of-range index that
        looks like an ordinary answer instead of failing."""
        curve = survival_curve(START_YEAR, 1, "f", START_YEAR, table)
        with pytest.raises(ValueError):
            death_month_index(np.array([0.0]), curve)

    def test_rejects_u_equal_one(self, table: ParamSet) -> None:
        """At u == 1 every person with any hazard at all dies in month 1."""
        curve = survival_curve(START_YEAR, 1, "f", START_YEAR, table)
        with pytest.raises(ValueError):
            death_month_index(np.array([1.0]), curve)

    def test_names_the_offending_count_and_an_example(self, table: ParamSet) -> None:
        curve = survival_curve(START_YEAR, 1, "f", START_YEAR, table)
        with pytest.raises(ValueError, match="2 of 3"):
            death_month_index(np.array([0.5, 0.0, 1.0]), curve)


class TestGeometricDistribution:
    """With a constant annual q, the death month is geometric.

    ``D`` has ``P(D = k) = (1 - h) ** (k - 1) * h`` for ``k >= 1`` and mean
    ``1 / h``, where ``h`` is the monthly hazard. ``h`` and ``expected_mean``
    are written out arithmetically from ``SYNTHETIC_Q_F`` below, independently
    of :func:`monthly_hazard` — calling that function here would make both
    sides of the comparison move together under the same bug (mutating
    ``monthly_hazard`` to ``q_annual / MONTHS_PER_YEAR`` once left this test
    passing on its own, because ``expected_mean`` was computed with the very
    function ``survival_curve`` is built from).

    With ``N_DRAWS`` draws from a fixed-seed generator the standard error of
    the sample mean is about ``(1 / h) / sqrt(N_DRAWS)``; the tolerance below
    is ``TOLERANCE_IN_SE`` multiples of that, chosen to comfortably clear two
    sources of noise: ordinary sampling error, and the truncation bias from
    the terminal age (negligible here — see ``TERMINAL_AGE``'s docstring —
    but not exactly zero). The seed is fixed so the comparison cannot flake.
    """

    N_DRAWS = 200_000
    TOLERANCE_IN_SE = 8
    SEED = 20260909

    def test_sample_mean_matches_the_closed_form(self, table: ParamSet) -> None:
        curve = survival_curve(START_YEAR, 1, "f", START_YEAR, table)

        # Independent of monthly_hazard(): 1 - (1 - q) ** (1/12), spelled out.
        h = 1 - (1 - SYNTHETIC_Q_F) ** (1 / MONTHS_PER_YEAR)
        expected_mean = 1 / h

        rng = np.random.default_rng(self.SEED)
        u = rng.uniform(size=self.N_DRAWS)
        deaths = death_month_index(u, curve)

        standard_error = expected_mean / np.sqrt(self.N_DRAWS)
        tolerance = self.TOLERANCE_IN_SE * standard_error

        assert deaths.mean() == pytest.approx(expected_mean, abs=tolerance)


class TestMonthsToTerminal:
    def test_matches_the_survival_curve_length(self, table: ParamSet) -> None:
        assert months_to_terminal(START_YEAR, 1, "f", START_YEAR, table) == len(
            survival_curve(START_YEAR, 1, "f", START_YEAR, table)
        )

    def test_the_last_month_of_the_curve_is_reachable_by_death_month_index(
        self, table: ParamSet
    ) -> None:
        """Pins the bound at the end that matters, rather than restating the function body.

        A ``u`` an infinitesimal step above 0 dies as late as ``death_month_index`` ever
        allows; that must land exactly one before ``months_to_terminal``'s count, which is
        the one bare fact ``test_matches_the_survival_curve_length`` above cannot show, since
        it never calls ``death_month_index`` at all.
        """
        curve = survival_curve(START_YEAR, 1, "f", START_YEAR, table)
        months = months_to_terminal(START_YEAR, 1, "f", START_YEAR, table)
        u = np.array([np.nextafter(0.0, 1.0)])
        assert death_month_index(u, curve)[0] == months - 1

    def test_grows_as_birth_year_rises(self, table: ParamSet) -> None:
        """A younger person has more months to run than an older one."""
        older = months_to_terminal(START_YEAR - 40, 1, "f", START_YEAR, table)
        younger = months_to_terminal(START_YEAR - 10, 1, "f", START_YEAR, table)
        assert younger > older

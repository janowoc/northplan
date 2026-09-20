# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The erosion factors and the real-terms parameter view.

Every parameter file in this module is synthetic and says so in its own header.
The numbers are round, wrong, and chosen to make arithmetic checkable by hand —
a basic personal amount of 1000 and an inflation rate of 409500 percent are not
mistakes to be corrected, they are the reason the expected values below can be
written as exact fractions instead of floating-point noise. Nothing here is a
tax value and nothing here should ever be copied into ``params/``.

The rate is the trick worth explaining. At ``inflation_rate = 4095`` the price
level multiplies by ``4096 = 2 ** 12`` each year, so one month of erosion is
exactly one halving: ``(1 + 4095) ** (-1 / 12) == 0.5``. A three-month cycle is
then the mean of ``1``, ``1/2`` and ``1/4``, which is ``7/12`` on paper and to
the last bit in binary floating point. A realistic two-percent rate would make
every expected value here a transcription of whatever the implementation
happened to print, which tests nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.core.indexation import (
    RealParamSet,
    RoutedParameterError,
    UnroutedParameterError,
    erosion_factor,
    nominal_carry_factor,
    real_year,
    schedule,
    unindexed_factor,
)
from engine.params.loader import (
    MalformedParamFileError,
    MissingParameterError,
    ParamError,
    load_year,
)

#: A year that doubles the price level every month, so a month of erosion is a
#: halving and every expected value below is an exact binary fraction.
HALVING = 2**12 - 1

SYNTHETIC = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  yearly:
    adjustment_months: [1]
    applies_to:
      - credits.basic_amount_annual
      - brackets.edges_annual
  quarterly:
    adjustment_months: [1, 4, 7, 10]
    applies_to:
      - pension.bands.*.maximum_monthly
  unindexed:
    adjustment_months: []
    applies_to:
      - credits.frozen_amount_annual

credits:
  basic_amount_annual: 1000
  frozen_amount_annual: 2000
  rate: 0.5
brackets:
  edges_annual: [4000, 8000]
pension:
  bands:
    - from_age_months: 780
      maximum_monthly: 100
    - from_age_months: 900
      maximum_monthly: 200
filing_month: 4
"""

#: A second synthetic fixture, used only for the one shape ``SYNTHETIC`` above
#: does not exercise: a list-valued amount on the unindexed schedule. Not a
#: tax parameter, same as ``SYNTHETIC``.
UNINDEXED_TABLE = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  unindexed:
    adjustment_months: []
    applies_to:
      - frozen.edges_annual

frozen:
  edges_annual: [4000, 8000]
"""


def _write(root: Path, name: str, text: str) -> Path:
    year_dir = root / "2026"
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    _write(tmp_path, "federal", SYNTHETIC)
    return tmp_path


@pytest.fixture
def real(synthetic_root: Path) -> RealParamSet:
    """The synthetic file at the halving rate."""
    return real_year(load_year(2026, synthetic_root), HALVING).federal


# --- erosion_factor ---------------------------------------------------------


@pytest.mark.parametrize("adjustments", [1, 2, 3, 4, 6, 12])
def test_zero_inflation_erodes_nothing(adjustments: int) -> None:
    """With no inflation there is nothing to erode, at any cadence."""
    assert erosion_factor(0, adjustments) == 1


@pytest.mark.parametrize("rate", [0, 0.5, 3, HALVING])
def test_monthly_adjustment_erodes_nothing(rate: float) -> None:
    """An amount adjusted every month never spends a month losing ground.

    ``P == 1``, so the mean is over the single month in which the amount is
    worth exactly its published value. This is the boundary the loop has to get
    right; an off-by-one in the range would make it a two-month mean.
    """
    assert erosion_factor(rate, 12) == 1


@pytest.mark.parametrize("rate", [0, 0.5, 3, HALVING])
def test_annual_adjustment_spans_twelve_months(rate: float) -> None:
    """``P == 12`` is the widest cycle, and the most eroded."""
    expected = sum((1 + rate) ** (-month / 12) for month in range(12)) / 12
    assert erosion_factor(rate, 1) == expected


def test_quarterly_factor_at_the_halving_rate() -> None:
    """Hand-worked: a quarter is three months, so the mean is (1 + 1/2 + 1/4) / 3.

    Exact in binary floating point, which is why the fixture rate is what it is.
    """
    assert erosion_factor(HALVING, 4) == (1 + 0.5 + 0.25) / 3


def test_semiannual_factor_at_the_halving_rate() -> None:
    """Six months of halving: the mean of 1, 1/2, 1/4, 1/8, 1/16, 1/32."""
    expected = sum(0.5**month for month in range(6)) / 6
    assert erosion_factor(HALVING, 2) == expected


def test_the_factor_matches_the_closed_form_geometric_mean() -> None:
    """An independent derivation, not a rerun of the implementation's loop.

    The per-month factors are a geometric series in ``q = (1 + r) ** (-1/12)``,
    so their mean is ``(1 - q ** P) / (P * (1 - q))``. Agreeing with the sum the
    module computes is evidence about the arithmetic rather than about the code
    having been copied into the test.
    """
    rate = 0.375
    for adjustments in (1, 2, 3, 4, 6):
        period = 12 // adjustments
        q = (1 + rate) ** (-1 / 12)
        closed_form = (1 - q**period) / (period * (1 - q))
        assert erosion_factor(rate, adjustments) == pytest.approx(closed_form, rel=1e-12)


def test_a_longer_cycle_erodes_more() -> None:
    """Monotone in the cycle length, at a positive rate."""
    factors = [erosion_factor(0.375, adjustments) for adjustments in (12, 6, 4, 3, 2, 1)]
    assert factors == sorted(factors, reverse=True)
    assert factors[0] == 1
    assert factors[-1] < 1


@pytest.mark.parametrize("adjustments", [0, -1, -12, 5, 7, 8, 9, 10, 11, 13, 24])
def test_a_cadence_that_does_not_divide_the_year_is_refused(adjustments: int) -> None:
    """A cycle has a whole number of months or it is not describable here."""
    with pytest.raises(ValueError, match="positive divisor"):
        erosion_factor(0.02, adjustments)


@pytest.mark.parametrize("rate", [-1, -1.5, -2])
def test_a_rate_at_or_below_minus_one_is_refused(rate: float) -> None:
    """A negative base to a fractional power is a complex number, not an error.

    Without the guard the "factor" would be complex, propagate silently through
    a multiplication, and surface much later as something unreadable.
    """
    with pytest.raises(ValueError, match="greater than -1"):
        erosion_factor(rate, 4)
    with pytest.raises(ValueError, match="greater than -1"):
        unindexed_factor(rate, 1)


# --- unindexed_factor -------------------------------------------------------


def test_an_unindexed_amount_is_whole_in_the_first_month() -> None:
    """Month zero is January of the start year, where real and nominal agree."""
    assert unindexed_factor(HALVING, 0) == 1


@pytest.mark.parametrize(("month", "expected"), [(1, 0.5), (2, 0.25), (12, 2**-12)])
def test_an_unindexed_amount_halves_every_month_at_the_halving_rate(
    month: int, expected: float
) -> None:
    assert unindexed_factor(HALVING, month) == expected


def test_an_unindexed_amount_decays_without_limit() -> None:
    """The property that distinguishes it from an indexed amount.

    An indexed amount's loss is bounded by one cycle; this one keeps falling,
    which over a thirty-year retirement is the larger effect by far.
    """
    factors = [unindexed_factor(0.375, month) for month in range(0, 361, 12)]
    assert factors == sorted(factors, reverse=True)
    # Thirty years of 37.5% inflation: 1.375 ** -30, four ten-thousandths of
    # where it started. The bound says "without limit" without pinning a digit.
    assert factors[-1] == pytest.approx(1.375**-30)
    assert factors[-1] < 1e-3


def test_a_month_before_the_start_of_the_run_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        unindexed_factor(0.02, -1)


# --- nominal_carry_factor ----------------------------------------------------


class TestNominalCarryFactor:
    """One year of :func:`unindexed_factor`, for a state balance rather than a parameter."""

    @pytest.mark.parametrize("rate", [0.0, 0.02, HALVING, -0.5])
    def test_agrees_with_unindexed_factor_at_twelve_months(self, rate: float) -> None:
        """Delegation, asserted directly: the two must agree by construction."""
        assert nominal_carry_factor(rate) == unindexed_factor(rate, 12)

    def test_zero_inflation_carries_the_balance_unchanged(self) -> None:
        assert nominal_carry_factor(0.0) == 1.0

    def test_a_positive_rate_erodes_the_balance(self) -> None:
        assert nominal_carry_factor(0.02) < 1.0

    def test_a_negative_rate_inflates_the_balance(self) -> None:
        assert nominal_carry_factor(-0.5) > 1.0

    @pytest.mark.parametrize("rate", [-1.0, -1.5, -2])
    def test_a_rate_at_or_below_minus_one_is_refused(self, rate: float) -> None:
        with pytest.raises(ValueError, match="greater than -1"):
            nominal_carry_factor(rate)

    def test_the_closed_form_at_two_percent(self) -> None:
        """An independent check, so the test does not merely restate the delegation."""
        assert nominal_carry_factor(0.02) == pytest.approx(1 / 1.02)


# --- schedule ---------------------------------------------------------------


def test_schedule_returns_the_count_of_months_and_the_paths(synthetic_root: Path) -> None:
    """The count, not the months: which month of the quarter changes no result."""
    params = load_year(2026, synthetic_root).federal
    assert schedule("yearly", params) == (
        1,
        ("credits.basic_amount_annual", "brackets.edges_annual"),
    )
    assert schedule("quarterly", params)[0] == 4
    assert schedule("unindexed", params) == (0, ("credits.frozen_amount_annual",))


def test_an_absent_schedule_stops_the_run(synthetic_root: Path) -> None:
    """Not "annually by default": an unverified rule is unknown, and that is loud."""
    params = load_year(2026, synthetic_root).federal
    with pytest.raises(MissingParameterError, match="quarterlyy"):
        schedule("quarterlyy", params)


def test_a_schedule_missing_its_paths_stops_the_run(tmp_path: Path) -> None:
    _write(tmp_path, "federal", "indexation:\n  yearly:\n    adjustment_months: [1]\n")
    params = load_year(2026, tmp_path).federal
    with pytest.raises(MissingParameterError, match="applies_to"):
        schedule("yearly", params)


def test_a_path_that_is_not_a_string_stops_the_run(tmp_path: Path) -> None:
    """A YAML list of numbers where paths were meant is a malformed file, not a route."""
    _write(
        tmp_path,
        "federal",
        "indexation:\n  yearly:\n    adjustment_months: [1]\n    applies_to: [3]\n",
    )
    params = load_year(2026, tmp_path).federal
    with pytest.raises(MalformedParamFileError, match="expected a dotted path"):
        schedule("yearly", params)


# --- RealParamSet: reading an amount ----------------------------------------


def test_an_indexed_amount_is_its_raw_value_times_its_schedules_factor(
    real: RealParamSet,
) -> None:
    assert real.amount("credits.basic_amount_annual", 0) == 1000 * erosion_factor(HALVING, 1)


def test_zero_inflation_returns_the_published_amount(synthetic_root: Path) -> None:
    """The success criterion in the issue: at zero inflation the view is a pass-through."""
    federal = real_year(load_year(2026, synthetic_root), 0).federal
    assert federal.amount("credits.basic_amount_annual", 0) == 1000
    assert federal.amount("credits.frozen_amount_annual", 600) == 2000


@pytest.mark.parametrize("month", [0, 1, 7, 120, 999])
def test_month_index_is_ignored_for_an_indexed_amount(real: RealParamSet, month: int) -> None:
    """The whole simplification, asserted directly: the factor is a constant."""
    assert real.amount("credits.basic_amount_annual", month) == real.amount(
        "credits.basic_amount_annual", 0
    )


@pytest.mark.parametrize(("month", "expected"), [(0, 2000), (1, 1000), (2, 500), (12, 2000 / 4096)])
def test_month_index_is_honoured_for_an_unindexed_amount(
    real: RealParamSet, month: int, expected: float
) -> None:
    """And here it must not be, which is the difference the two factors exist for."""
    assert real.amount("credits.frozen_amount_annual", month) == expected


def test_a_list_valued_amount_comes_back_deflated_in_table_order(real: RealParamSet) -> None:
    factor = erosion_factor(HALVING, 1)
    assert real.amounts("brackets.edges_annual", 0) == (4000 * factor, 8000 * factor)


def test_a_wildcard_names_one_amount_per_element(real: RealParamSet) -> None:
    """``*`` expansion: one float per band, in table order, on the band's schedule."""
    factor = erosion_factor(HALVING, 4)
    assert real.amounts("pension.bands.*.maximum_monthly", 0) == (100 * factor, 200 * factor)


def test_a_wildcard_path_is_not_a_single_amount(real: RealParamSet) -> None:
    with pytest.raises(ValueError, match="names every element"):
        real.amount("pension.bands.*.maximum_monthly", 0)


def test_a_wildcard_that_matches_nothing_stops_the_run(tmp_path: Path) -> None:
    """An empty table is a pattern that has stopped fitting its file, not zero amounts."""
    _write(
        tmp_path,
        "federal",
        "indexation:\n"
        "  yearly:\n"
        "    adjustment_months: [1]\n"
        "    applies_to:\n"
        "      - pension.bands.*.maximum_monthly\n"
        "pension:\n"
        "  bands: []\n",
    )
    federal = real_year(load_year(2026, tmp_path), 0).federal
    with pytest.raises(MissingParameterError, match="matches nothing"):
        federal.amounts("pension.bands.*.maximum_monthly", 0)


# --- RealParamSet: the two errors -------------------------------------------


@pytest.mark.parametrize("accessor", ["get", "number", "numbers", "sequence", "has"])
def test_a_routed_amount_cannot_be_read_undeflated(real: RealParamSet, accessor: str) -> None:
    """The negative guarantee. Forgetting to deflate has to be impossible, not discouraged."""
    with pytest.raises(RoutedParameterError, match="undeflated"):
        getattr(real, accessor)("credits.basic_amount_annual")


def test_a_concrete_element_of_a_routed_table_is_also_refused(real: RealParamSet) -> None:
    """``bands.0.maximum_monthly`` is the same dollar as ``bands.*.maximum_monthly``.

    Matching only the pattern as written would leave the obvious way round it
    open to anyone who had already resolved an index.
    """
    with pytest.raises(RoutedParameterError):
        real.number("pension.bands.0.maximum_monthly")


def test_an_unrouted_path_still_reads_raw(real: RealParamSet) -> None:
    """Only dollar amounts are routed; a month is not a dollar and passes through."""
    assert real.number("filing_month") == 4
    assert real.number("credits.rate") == 0.5
    assert real.has("credits.rate")


def test_an_amount_on_no_schedule_stops_the_run(real: RealParamSet) -> None:
    """There is no default cadence. An unrecorded rule is unknown, and it is loud."""
    with pytest.raises(UnroutedParameterError, match="no indexation schedule"):
        real.amount("credits.rate", 0)


def test_both_errors_are_parameter_errors() -> None:
    """So a caller that means "any parameter problem" catches them with ``ParamError``."""
    assert issubclass(RoutedParameterError, ParamError)
    assert issubclass(UnroutedParameterError, ParamError)


def test_a_path_routed_by_two_schedules_stops_the_run(tmp_path: Path) -> None:
    """The structural test forbids it; picking one silently here would hide it."""
    _write(
        tmp_path,
        "federal",
        "indexation:\n"
        "  yearly:\n"
        "    adjustment_months: [1]\n"
        "    applies_to:\n"
        "      - credits.basic_amount_annual\n"
        "  quarterly:\n"
        "    adjustment_months: [1, 4, 7, 10]\n"
        "    applies_to:\n"
        "      - credits.basic_amount_annual\n"
        "credits:\n"
        "  basic_amount_annual: 1000\n",
    )
    year = real_year(load_year(2026, tmp_path), 0)
    with pytest.raises(MalformedParamFileError, match="routed by both"):
        _ = year.federal


def test_a_file_with_no_indexation_block_routes_nothing(tmp_path: Path) -> None:
    """``mortality.yaml`` holds no dollars and declares no schedules; reads pass through."""
    _write(tmp_path, "mortality", "terminal_age_years: 110\n")
    mortality = real_year(load_year(2026, tmp_path), 0.02)["mortality"]
    assert mortality.number("terminal_age_years") == 110
    with pytest.raises(UnroutedParameterError):
        mortality.amount("terminal_age_years", 0)


@pytest.mark.parametrize("rate", [-1, -3])
def test_the_view_refuses_an_impossible_inflation_rate(synthetic_root: Path, rate: float) -> None:
    """Caught when the view is built, not on the first amount read from it."""
    params = load_year(2026, synthetic_root)
    with pytest.raises(ValueError, match="greater than -1"):
        real_year(params, rate)


# --- RealParamYear ----------------------------------------------------------


def test_the_year_mirrors_the_loaders_accessors(synthetic_root: Path) -> None:
    """Switching a call site to the real view is a change of object, not a rewrite."""
    year = real_year(load_year(2026, synthetic_root), HALVING)
    assert year.year == 2026
    assert year.names() == ("federal",)
    assert "federal" in year
    assert "cpp" not in year
    assert isinstance(year.federal, RealParamSet)
    assert year["federal"] is year.federal
    assert year.federal.name == "federal"
    assert year.federal.year == 2026


def test_a_province_is_reached_by_code(tmp_path: Path) -> None:
    _write(tmp_path, "ab", SYNTHETIC)
    year = real_year(load_year(2026, tmp_path), 0)
    assert year.province("AB").amount("credits.basic_amount_annual", 0) == 1000
    assert year.jurisdiction("ab").amount("credits.basic_amount_annual", 0) == 1000


def test_the_raw_year_is_still_reachable_for_the_published_figure(
    synthetic_root: Path,
) -> None:
    """Tests and diagnostics legitimately want the number as published."""
    year = real_year(load_year(2026, synthetic_root), HALVING)
    assert year.raw.federal.number("credits.basic_amount_annual") == 1000
    assert year.federal.raw.number("credits.basic_amount_annual") == 1000


# --- RealParamSet: annual_amount / annual_amounts ---------------------------


@pytest.mark.parametrize("bad_index", [-12, -1, 1, 11, 13, 23])
@pytest.mark.parametrize("path", ["credits.basic_amount_annual", "credits.frozen_amount_annual"])
def test_annual_amount_refuses_an_index_that_is_not_january(
    real: RealParamSet, path: str, bad_index: int
) -> None:
    """The whole point: a caller cannot pass December, or any other month, by habit."""
    with pytest.raises(ValueError, match="January"):
        real.annual_amount(path, bad_index)


@pytest.mark.parametrize("bad_index", [-12, -1, 1, 11, 13, 23])
@pytest.mark.parametrize("path", ["brackets.edges_annual", "pension.bands.*.maximum_monthly"])
def test_annual_amounts_refuses_an_index_that_is_not_january(
    real: RealParamSet, path: str, bad_index: int
) -> None:
    with pytest.raises(ValueError, match="January"):
        real.annual_amounts(path, bad_index)


@pytest.mark.parametrize("good_index", [0, 12, 24])
def test_annual_amount_accepts_a_multiple_of_twelve(real: RealParamSet, good_index: int) -> None:
    real.annual_amount("credits.basic_amount_annual", good_index)  # must not raise
    real.annual_amount("credits.frozen_amount_annual", good_index)  # must not raise


@pytest.mark.parametrize("january", [0, 12, 120])
def test_zero_inflation_makes_annual_amount_match_amount(
    synthetic_root: Path, january: int
) -> None:
    """At zero inflation there is no averaging or decay to distinguish the two."""
    federal = real_year(load_year(2026, synthetic_root), 0).federal
    assert federal.annual_amount("credits.basic_amount_annual", january) == federal.amount(
        "credits.basic_amount_annual", january
    )
    assert federal.annual_amount("credits.frozen_amount_annual", january) == federal.amount(
        "credits.frozen_amount_annual", january
    )


@pytest.mark.parametrize("january", [0, 12, 120])
@pytest.mark.parametrize("month", [0, 1, 7, 11, 120, 999])
def test_annual_amount_of_an_indexed_path_matches_amount_at_any_month(
    real: RealParamSet, month: int, january: int
) -> None:
    """An indexed amount's factor does not depend on the month, in either accessor.

    Parametrized over ``january`` as well as ``month``: an ``_annual_factor``
    whose indexed branch wrongly multiplied by ``unindexed_factor(rate,
    january)`` would still pass at ``january == 0``, where that factor is 1.
    """
    assert real.annual_amount("credits.basic_amount_annual", january) == real.amount(
        "credits.basic_amount_annual", month
    )


def test_annual_amount_of_an_unindexed_path_decays_to_january_then_averages(
    real: RealParamSet,
) -> None:
    """The two-step rule, spelled out: decay to January of the tax year, then
    apply the same within-year averaging an annually indexed amount gets."""
    expected = 2000 * unindexed_factor(HALVING, 12) * erosion_factor(HALVING, 1)
    assert real.annual_amount("credits.frozen_amount_annual", 12) == expected


@pytest.mark.parametrize("january", [0, 12, 120])
def test_annual_amounts_on_a_list_valued_path_returns_the_table_in_order(
    real: RealParamSet, january: int
) -> None:
    """Parametrized over ``january`` for the same reason as the ``annual_amount`` test above."""
    factor = erosion_factor(HALVING, 1)
    assert real.annual_amounts("brackets.edges_annual", january) == (4000 * factor, 8000 * factor)


@pytest.mark.parametrize("january", [0, 12, 120])
def test_annual_amounts_on_a_wildcard_path_returns_the_table_in_order(
    real: RealParamSet, january: int
) -> None:
    factor = erosion_factor(HALVING, 4)
    assert real.annual_amounts("pension.bands.*.maximum_monthly", january) == (
        100 * factor,
        200 * factor,
    )


def test_a_wildcard_path_is_not_a_single_annual_amount(real: RealParamSet) -> None:
    with pytest.raises(ValueError, match="names every element"):
        real.annual_amount("pension.bands.*.maximum_monthly", 0)


def test_an_unrouted_path_still_stops_the_run_via_annual_amount(real: RealParamSet) -> None:
    with pytest.raises(UnroutedParameterError, match="no indexation schedule"):
        real.annual_amount("credits.rate", 0)


def test_annual_amounts_on_an_unindexed_list_valued_path_decays_then_averages(
    tmp_path: Path,
) -> None:
    """The list shape on the unindexed schedule, which ``SYNTHETIC`` does not exercise.

    An ``annual_amounts`` that fell back to plain ``amounts(path, j)`` after
    its guard would pass every other test in this file but get this one
    wrong: the unindexed rule needs the extra within-year averaging factor on
    top of the decay to January, not the decay alone.
    """
    _write(tmp_path, "federal", UNINDEXED_TABLE)
    federal = real_year(load_year(2026, tmp_path), HALVING).federal
    expected = tuple(
        value * unindexed_factor(HALVING, 12) * erosion_factor(HALVING, 1) for value in (4000, 8000)
    )
    assert federal.annual_amounts("frozen.edges_annual", 12) == expected

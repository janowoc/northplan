# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``engine.benefits.oas``.

Structural tests against the real 2026 files at zero inflation; every
threshold, age, and rate is read from the loaded params, never typed in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.benefits.oas import (
    deferral_factor,
    gross_pension_annual,
    gross_pension_monthly,
)
from engine.core.indexation import erosion_factor, real_year, schedule
from engine.core.state import BenefitState
from engine.core.timeline import MONTHS_PER_YEAR
from engine.params.loader import MalformedParamFileError, load_year

JANUARY = 0
MONTH = 0
N_PATHS = 3


@pytest.fixture
def oas():
    return real_year(load_year(2026), 0.0).oas


def _benefit_elected(start_age_months):
    return BenefitState(
        start_age_months=start_age_months,
        in_pay_monthly=None,
        contributory_history=None,
        monthly_amount=np.zeros(N_PATHS, dtype=np.float64),
    )


def _benefit_in_pay(amount):
    arr = np.full(N_PATHS, amount, dtype=np.float64)
    return BenefitState(
        start_age_months=None,
        in_pay_monthly=arr,
        contributory_history=None,
        monthly_amount=np.zeros(N_PATHS, dtype=np.float64),
    )


def _band_maxima(oas, month_index: int) -> tuple[float, float]:
    """The two published band maxima, in real dollars, by explicit index."""
    maxima = oas.amounts("pension.age_bands.*.maximum_monthly", month_index)
    return maxima[0], maxima[1]


def _second_band_from_age(oas) -> int:
    bands = oas.sequence("pension.age_bands")
    return int(bands[1]["from_age_months"])


# --- deferral_factor ---------------------------------------------------------


def test_deferral_factor_is_one_at_earliest(oas) -> None:
    earliest = oas.number("start_age.earliest_months")
    assert deferral_factor(earliest, oas) == pytest.approx(1.0)


def test_deferral_factor_rises_per_month(oas) -> None:
    earliest = oas.number("start_age.earliest_months")
    increment = oas.number("deferral.increment_rate_per_month")
    assert deferral_factor(earliest + 1, oas) == pytest.approx(1 + increment)


def test_deferral_factor_flat_past_maximum(oas) -> None:
    earliest = oas.number("start_age.earliest_months")
    maximum_months = oas.number("deferral.maximum_months")
    at_max = deferral_factor(earliest + maximum_months, oas)
    past_max = deferral_factor(earliest + maximum_months + 12, oas)
    assert at_max == pytest.approx(past_max)


# --- band step-up -------------------------------------------------------------


def test_band_steps_up_exactly_at_second_band_threshold(oas) -> None:
    threshold = _second_band_from_age(oas)
    earliest = int(oas.number("start_age.earliest_months"))
    age_at_open = earliest  # elected at the earliest age, already in payment by threshold
    benefit = _benefit_elected(earliest)
    just_below = gross_pension_monthly(benefit, threshold - 1, threshold - 1 - age_at_open, oas)
    at_threshold = gross_pension_monthly(benefit, threshold, threshold - age_at_open, oas)
    assert np.all(just_below < at_threshold)


# --- gross_pension_monthly: elected -------------------------------------------


def test_elected_zero_through_the_start_month_paid_the_month_after(oas) -> None:
    earliest = int(oas.number("start_age.earliest_months"))
    benefit = _benefit_elected(earliest)
    age_at_open = earliest - 6
    before = gross_pension_monthly(benefit, age_at_open + 5, 5, oas)
    at_start = gross_pension_monthly(benefit, age_at_open + 6, 6, oas)
    after_start = gross_pension_monthly(benefit, age_at_open + 7, 7, oas)
    assert np.all(before == 0.0)
    assert np.all(at_start == 0.0)
    band0, _ = _band_maxima(oas, 7)
    expected = band0 * deferral_factor(earliest, oas)
    assert after_start == pytest.approx(expected)


def test_elected_already_past_starts_at_month_zero_with_deferral_for_the_month_before_opening(
    oas,
) -> None:
    earliest = int(oas.number("start_age.earliest_months"))
    elected = earliest
    age_at_open = earliest + 24  # already past the election when the run opens
    benefit = _benefit_elected(elected)
    result = gross_pension_monthly(benefit, age_at_open, 0, oas)
    band0, _ = _band_maxima(oas, 0)
    expected = band0 * deferral_factor(age_at_open - 1, oas)
    assert result == pytest.approx(expected)


def test_elected_at_whole_year_age_uses_deferral_factor_for_the_month_before_opening(oas) -> None:
    """Elected at k years, opened at k years 11 months: paid from month 0.

    The deferral factor is the one for the month before the opening, the same
    at month 5. This case matters because
    ``engine.scenario.start_ages.check_start_ages`` accepts it.
    """
    earliest = oas.number("start_age.earliest_months")
    assert earliest % MONTHS_PER_YEAR == 0
    k = int(earliest) // MONTHS_PER_YEAR
    age_at_open = k * MONTHS_PER_YEAR + (MONTHS_PER_YEAR - 1)
    assert age_at_open + 5 < _second_band_from_age(oas)  # band 0 throughout

    benefit = _benefit_elected(k * MONTHS_PER_YEAR)

    band0_at_zero, _ = _band_maxima(oas, 0)
    at_month_zero = gross_pension_monthly(benefit, age_at_open, 0, oas)
    expected_zero = band0_at_zero * deferral_factor(age_at_open - 1, oas)
    assert at_month_zero == pytest.approx(expected_zero)

    band0_at_five, _ = _band_maxima(oas, 5)
    at_month_five = gross_pension_monthly(benefit, age_at_open + 5, 5, oas)
    expected_five = band0_at_five * deferral_factor(age_at_open - 1, oas)
    assert at_month_five == pytest.approx(expected_five)


def test_elected_above_latest_is_clipped_and_first_paid_the_month_after_latest(oas) -> None:
    """An election above the latest start age is clipped to it.

    A person at the latest age at opening is first paid in month 1, at the latest factor.
    """
    latest = oas.number("start_age.latest_months")
    assert latest % MONTHS_PER_YEAR == 0
    latest_int = int(latest)
    assert latest_int + 1 < _second_band_from_age(oas)  # band 0

    elected = latest_int + MONTHS_PER_YEAR  # unchecked; the clip must still bound it
    benefit = _benefit_elected(elected)
    band0, _ = _band_maxima(oas, 1)
    expected = band0 * deferral_factor(latest_int, oas)
    assert expected > 0.0  # a zero result must not pass this test by accident

    at_open = gross_pension_monthly(benefit, latest_int, 0, oas)
    assert np.all(at_open == 0.0)
    result = gross_pension_monthly(benefit, latest_int + 1, 1, oas)
    assert result == pytest.approx(expected)


def test_elected_at_latest_opened_past_latest_is_paid_from_month_zero_at_the_maximum_deferral(
    oas,
) -> None:
    latest = int(oas.number("start_age.latest_months"))
    assert latest % MONTHS_PER_YEAR == 0
    age_at_open = latest + 5
    # The person's age in whole years equals the latest start age, so the
    # election is accepted at load.
    assert age_at_open // MONTHS_PER_YEAR == latest // MONTHS_PER_YEAR
    assert latest + 5 < _second_band_from_age(oas)  # band 0
    benefit = _benefit_elected(latest)
    band0, _ = _band_maxima(oas, 0)
    expected = band0 * deferral_factor(latest, oas)
    assert expected > 0.0  # a zero result must not pass this test by accident
    result = gross_pension_monthly(benefit, age_at_open, 0, oas)
    assert result == pytest.approx(expected)


def test_elected_at_latest_first_paid_the_month_after_with_maximum_deferral(oas) -> None:
    latest = int(oas.number("start_age.latest_months"))
    increment = oas.number("deferral.increment_rate_per_month")
    maximum_months = oas.number("deferral.maximum_months")
    assert latest + 1 < _second_band_from_age(oas)  # band 0
    assert deferral_factor(latest, oas) == pytest.approx(1 + increment * maximum_months)
    benefit = _benefit_elected(latest)
    age_at_open = latest - 6
    at_start = gross_pension_monthly(benefit, age_at_open + 6, 6, oas)
    after_start = gross_pension_monthly(benefit, age_at_open + 7, 7, oas)
    assert np.all(at_start == 0.0)
    band0, _ = _band_maxima(oas, 7)
    assert after_start == pytest.approx(band0 * deferral_factor(latest, oas))


def test_elected_reached_in_the_opening_month_is_first_paid_in_month_one(oas) -> None:
    earliest = int(oas.number("start_age.earliest_months"))
    benefit = _benefit_elected(earliest)
    age_at_open = earliest
    at_open = gross_pension_monthly(benefit, age_at_open, 0, oas)
    month_one = gross_pension_monthly(benefit, age_at_open + 1, 1, oas)
    assert np.all(at_open == 0.0)
    band0, _ = _band_maxima(oas, 1)
    assert month_one == pytest.approx(band0 * deferral_factor(earliest, oas))


def test_elected_one_month_past_at_opening_is_paid_from_month_zero_at_the_elected_deferral(
    oas,
) -> None:
    earliest = int(oas.number("start_age.earliest_months"))
    benefit = _benefit_elected(earliest)
    age_at_open = earliest + 1
    result = gross_pension_monthly(benefit, age_at_open, 0, oas)
    band0, _ = _band_maxima(oas, 0)
    # The first payment for an election at earliest falls in this month anyway,
    # so nothing is forgone and the factor is the elected one.
    assert result == pytest.approx(band0 * deferral_factor(earliest, oas))


def test_elected_four_months_past_at_opening_uses_the_deferral_for_three_months_past(oas) -> None:
    earliest = int(oas.number("start_age.earliest_months"))
    benefit = _benefit_elected(earliest)
    age_at_open = earliest + 4
    result = gross_pension_monthly(benefit, age_at_open, 0, oas)
    band0, _ = _band_maxima(oas, 0)
    assert result == pytest.approx(band0 * deferral_factor(earliest + 3, oas))


def test_elected_below_earliest_is_clipped_and_first_paid_the_month_after_earliest(oas) -> None:
    earliest = int(oas.number("start_age.earliest_months"))
    benefit = _benefit_elected(earliest - 60)  # unchecked, below the window; the clip must bound it
    age_at_open = earliest - 30
    at_open = gross_pension_monthly(benefit, age_at_open, 0, oas)
    at_earliest = gross_pension_monthly(benefit, age_at_open + 30, 30, oas)
    after_earliest = gross_pension_monthly(benefit, age_at_open + 31, 31, oas)
    assert np.all(at_open == 0.0)
    assert np.all(at_earliest == 0.0)
    band0, _ = _band_maxima(oas, 31)
    assert after_earliest == pytest.approx(band0 * deferral_factor(earliest, oas))


def test_gross_pension_monthly_raises_when_neither_source_set(oas) -> None:
    earliest = int(oas.number("start_age.earliest_months"))
    benefit = BenefitState(
        start_age_months=None,
        in_pay_monthly=None,
        contributory_history=None,
        monthly_amount=np.zeros(N_PATHS, dtype=np.float64),
    )
    with pytest.raises(ValueError):
        gross_pension_monthly(benefit, earliest, 0, oas)


# --- gross_pension_monthly: in pay ---------------------------------------------


def test_in_pay_unchanged_below_threshold(oas) -> None:
    threshold = _second_band_from_age(oas)
    benefit = _benefit_in_pay(500.0)
    age_at_open = threshold - 100
    result = gross_pension_monthly(benefit, age_at_open + 1, 1, oas)
    assert np.all(result == pytest.approx(500.0))


def test_in_pay_at_threshold_month_equals_ratio_of_maxima(oas) -> None:
    threshold = _second_band_from_age(oas)
    age_at_open = threshold - 12
    benefit = _benefit_in_pay(500.0)
    month_index = 12
    result = gross_pension_monthly(benefit, age_at_open + month_index, month_index, oas)
    band0, band1 = _band_maxima(oas, month_index)
    assert np.all(result == pytest.approx(500.0 * band1 / band0))


def test_in_pay_already_past_threshold_at_open_has_ratio_one(oas) -> None:
    threshold = _second_band_from_age(oas)
    age_at_open = threshold + 12
    benefit = _benefit_in_pay(500.0)
    result = gross_pension_monthly(benefit, age_at_open + 6, 6, oas)
    assert np.all(result == pytest.approx(500.0))


# --- gross_pension_annual ------------------------------------------------------


def test_gross_pension_annual_at_earliest_in_first_band(oas) -> None:
    earliest = int(oas.number("start_age.earliest_months"))
    band0, _ = _band_maxima(oas, JANUARY)
    assert gross_pension_annual(earliest, earliest, JANUARY, oas) == pytest.approx(0.0)
    result = gross_pension_annual(earliest + 1, earliest, JANUARY, oas)
    assert result == pytest.approx(12 * band0)


def test_gross_pension_annual_at_maximum_deferral(oas) -> None:
    latest = int(oas.number("start_age.latest_months"))
    maximum_months = oas.number("deferral.maximum_months")
    increment = oas.number("deferral.increment_rate_per_month")
    band0, _ = _band_maxima(oas, JANUARY)
    at_latest = gross_pension_annual(latest, latest, JANUARY, oas)
    result = gross_pension_annual(latest + 1, latest, JANUARY, oas)
    expected = 12 * band0 * (1 + increment * maximum_months)
    assert at_latest == pytest.approx(0.0)
    assert result == pytest.approx(expected)


def test_gross_pension_annual_zero_before_start(oas) -> None:
    earliest = int(oas.number("start_age.earliest_months"))
    result = gross_pension_annual(earliest - 1, earliest, JANUARY, oas)
    assert result == pytest.approx(0.0)


def test_gross_pension_annual_clips_start_age_below_earliest(oas) -> None:
    earliest = int(oas.number("start_age.earliest_months"))
    start_age_months = earliest - 60  # below the statutory window
    # A current age at or above the unclipped (too-early) start, but still
    # below earliest, must not be treated as started: without the clip in
    # gross_pension_annual, this would incorrectly pay out.
    not_yet = gross_pension_annual(start_age_months + 6, start_age_months, JANUARY, oas)
    assert not_yet == pytest.approx(0.0)

    band0, _ = _band_maxima(oas, JANUARY)
    at_earliest = gross_pension_annual(earliest, start_age_months, JANUARY, oas)
    assert at_earliest == pytest.approx(0.0)
    after_earliest = gross_pension_annual(earliest + 1, start_age_months, JANUARY, oas)
    assert after_earliest == pytest.approx(12 * band0)


# --- real view: erosion -------------------------------------------------------


def test_elected_band_maximum_is_eroded_at_positive_inflation(oas) -> None:
    earliest = int(oas.number("start_age.earliest_months"))
    oas_real = real_year(load_year(2026), 0.02).oas
    raw_band0 = oas_real.raw.sequence("pension.age_bands")[0]["maximum_monthly"]
    k, _ = schedule("pension", oas_real.raw)
    benefit = _benefit_elected(earliest)
    result = gross_pension_monthly(benefit, earliest + 1, 1, oas_real)
    expected = raw_band0 * erosion_factor(0.02, k) * deferral_factor(earliest, oas_real)
    assert result == pytest.approx(expected)


def test_in_pay_identical_at_zero_and_positive_inflation_below_threshold() -> None:
    oas_zero = real_year(load_year(2026), 0.0).oas
    oas_positive = real_year(load_year(2026), 0.02).oas
    threshold = _second_band_from_age(oas_zero)
    age_at_open = threshold - 100
    benefit = _benefit_in_pay(500.0)
    at_zero = gross_pension_monthly(benefit, age_at_open + 1, 1, oas_zero)
    at_positive = gross_pension_monthly(benefit, age_at_open + 1, 1, oas_positive)
    assert at_zero == pytest.approx(at_positive)


def test_in_pay_identical_at_zero_and_positive_inflation_at_threshold() -> None:
    oas_zero = real_year(load_year(2026), 0.0).oas
    oas_positive = real_year(load_year(2026), 0.02).oas
    threshold = _second_band_from_age(oas_zero)
    age_at_open = threshold - 12
    benefit = _benefit_in_pay(500.0)
    at_zero = gross_pension_monthly(benefit, age_at_open + 12, 12, oas_zero)
    at_positive = gross_pension_monthly(benefit, age_at_open + 12, 12, oas_positive)
    assert at_zero == pytest.approx(at_positive)


# --- malformed band table ------------------------------------------------------

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
#: The third band has no maximum_monthly, so the wildcard expansion of
#: maxima has fewer elements than the raw list of band starting ages.
SYNTHETIC_OAS_MISMATCHED_BAND_COUNT = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  pension:
    adjustment_months: [1, 4, 7, 10]
    applies_to:
      - pension.age_bands.*.maximum_monthly

pension:
  age_bands:
    - from_age_months: 780
      maximum_monthly: 100
    - from_age_months: 900
      maximum_monthly: 200
    - from_age_months: 1000
      note: "missing maximum_monthly on purpose"
"""

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
#: The second band has no from_age_months at all.
SYNTHETIC_OAS_MISSING_FROM_AGE = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  pension:
    adjustment_months: [1, 4, 7, 10]
    applies_to:
      - pension.age_bands.*.maximum_monthly

pension:
  age_bands:
    - from_age_months: 780
      maximum_monthly: 100
    - maximum_monthly: 200
"""

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
#: The second band is a bare number, not a mapping at all.
SYNTHETIC_OAS_BAND_NOT_A_MAPPING = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  pension:
    adjustment_months: [1, 4, 7, 10]
    applies_to:
      - pension.age_bands.*.maximum_monthly

pension:
  age_bands:
    - from_age_months: 780
      maximum_monthly: 100
    - 999
"""

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
#: Band starting ages given out of order.
SYNTHETIC_OAS_NON_ASCENDING_BANDS = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  pension:
    adjustment_months: [1, 4, 7, 10]
    applies_to:
      - pension.age_bands.*.maximum_monthly

pension:
  age_bands:
    - from_age_months: 900
      maximum_monthly: 200
    - from_age_months: 780
      maximum_monthly: 100
"""


def _write(root: Path, name: str, text: str) -> Path:
    year_dir = root / "2026"
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def synth_oas_mismatched(tmp_path: Path):
    _write(tmp_path, "oas", SYNTHETIC_OAS_MISMATCHED_BAND_COUNT)
    return real_year(load_year(2026, tmp_path), 0.0).oas


@pytest.fixture
def synth_oas_non_ascending(tmp_path: Path):
    _write(tmp_path, "oas", SYNTHETIC_OAS_NON_ASCENDING_BANDS)
    return real_year(load_year(2026, tmp_path), 0.0).oas


@pytest.fixture
def synth_oas_missing_from_age(tmp_path: Path):
    _write(tmp_path, "oas", SYNTHETIC_OAS_MISSING_FROM_AGE)
    return real_year(load_year(2026, tmp_path), 0.0).oas


@pytest.fixture
def synth_oas_band_not_a_mapping(tmp_path: Path):
    _write(tmp_path, "oas", SYNTHETIC_OAS_BAND_NOT_A_MAPPING)
    return real_year(load_year(2026, tmp_path), 0.0).oas


def test_band_max_raises_on_mismatched_band_count(synth_oas_mismatched) -> None:
    from engine.benefits.oas import _band_max

    with pytest.raises(MalformedParamFileError, match="maxima"):
        _band_max(800, JANUARY, synth_oas_mismatched)


def test_band_max_raises_on_band_missing_from_age_months(synth_oas_missing_from_age) -> None:
    from engine.benefits.oas import _band_max

    with pytest.raises(MalformedParamFileError, match="from_age_months"):
        _band_max(800, JANUARY, synth_oas_missing_from_age)


def test_band_max_raises_on_band_that_is_not_a_mapping(synth_oas_band_not_a_mapping) -> None:
    from engine.benefits.oas import _band_max

    with pytest.raises(MalformedParamFileError, match="expected a mapping") as exc_info:
        _band_max(800, JANUARY, synth_oas_band_not_a_mapping)
    assert "pension.age_bands.1" in str(exc_info.value)


def test_band_max_raises_on_non_ascending_band_ages(synth_oas_non_ascending) -> None:
    from engine.benefits.oas import _band_max

    with pytest.raises(MalformedParamFileError, match="ascend"):
        _band_max(800, JANUARY, synth_oas_non_ascending)

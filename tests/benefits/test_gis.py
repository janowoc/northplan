# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``engine.benefits.gis``.

Structural tests against the real 2026 files at zero inflation, plus a
synthetic ``modelled: true`` file to exercise the all-false branch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.benefits.gis import band_threshold_annual, in_band, is_modelled
from engine.core.indexation import erosion_factor, real_year, schedule
from engine.params.loader import load_year

JANUARY = 0


@pytest.fixture
def oas():
    return real_year(load_year(2026), 0.0).oas


def test_is_modelled_false_for_real_2026(oas) -> None:
    assert is_modelled(oas) is False


def test_in_band_flips_exactly_at_single_threshold(oas) -> None:
    threshold = band_threshold_annual(False, JANUARY, oas)
    at = in_band(threshold, 0.0, False, JANUARY, oas)
    above = in_band(threshold + 1.0, 0.0, False, JANUARY, oas)
    assert bool(at)
    assert not bool(above)


def test_in_band_flips_exactly_at_couple_threshold(oas) -> None:
    threshold = band_threshold_annual(True, JANUARY, oas)
    at = in_band(threshold, 0.0, True, JANUARY, oas)
    above = in_band(threshold + 1.0, 0.0, True, JANUARY, oas)
    assert bool(at)
    assert not bool(above)


def test_in_band_subtracts_oas(oas) -> None:
    threshold = band_threshold_annual(False, JANUARY, oas)
    net_income = threshold + 500.0
    # Net income alone is above the threshold, but net minus OAS is at it.
    above_on_net_income_alone = in_band(net_income, 0.0, False, JANUARY, oas)
    in_band_after_oas_subtracted = in_band(net_income, 500.0, False, JANUARY, oas)
    assert not bool(above_on_net_income_alone)
    assert bool(in_band_after_oas_subtracted)


# --- real view: erosion --------------------------------------------------


def test_band_threshold_annual_is_eroded_at_positive_inflation() -> None:
    oas_zero = real_year(load_year(2026), 0.0).oas
    oas_positive = real_year(load_year(2026), 0.02).oas
    raw_single = oas_zero.raw.number("gis.band_thresholds.single_testable_income_annual")
    raw_couple = oas_zero.raw.number("gis.band_thresholds.couple_combined_testable_income_annual")
    k, _ = schedule("gis", oas_positive.raw)

    single = band_threshold_annual(False, JANUARY, oas_positive)
    couple = band_threshold_annual(True, JANUARY, oas_positive)
    assert single == pytest.approx(raw_single * erosion_factor(0.02, k))
    assert couple == pytest.approx(raw_couple * erosion_factor(0.02, k))


# --- synthetic: modelled = true -----------------------------------------------

SYNTHETIC_OAS_MODELLED = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  gis:
    adjustment_months: [1, 4, 7, 10]
    applies_to:
      - gis.band_thresholds.single_testable_income_annual
      - gis.band_thresholds.couple_combined_testable_income_annual

gis:
  modelled: true
  band_thresholds:
    single_testable_income_annual: 1000
    couple_combined_testable_income_annual: 2000
"""


def _write(root: Path, name: str, text: str) -> Path:
    year_dir = root / "2026"
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def synth_oas_modelled(tmp_path: Path):
    _write(tmp_path, "oas", SYNTHETIC_OAS_MODELLED)
    return real_year(load_year(2026, tmp_path), 0.0).oas


def test_in_band_all_false_when_gis_modelled(synth_oas_modelled) -> None:
    assert is_modelled(synth_oas_modelled) is True
    result = in_band(
        np.array([0.0, -1000.0]),
        np.array([0.0, 0.0]),
        np.array([False, True]),
        JANUARY,
        synth_oas_modelled,
    )
    assert not np.any(result)
    assert result.shape == (2,)

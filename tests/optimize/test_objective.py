# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``engine.optimize.objective`` against synthetic :class:`SimulationResult` values.

Every number here is made up to exercise arithmetic; none is a real value.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from engine.mc.simulate import SimulationResult
from engine.optimize.objective import (
    OBJECTIVE_NAMES,
    certainty_equivalent_estate,
    gis_exposure,
    median_estate_after_tax,
    select_objective,
    success_probability,
)


def make_result(
    *,
    estate: list[float] | None = None,
    depleted: list[list[bool]] | None = None,
    gis: list[list[int]] | None = None,
    living: list[list[int]] | None = None,
) -> SimulationResult:
    """A synthetic result; per-year arrays are ``(n_years, n_paths)``."""
    estate_arr = np.asarray(estate if estate is not None else [1.0, 2.0, 3.0], dtype=np.float64)
    n_paths = estate_arr.shape[0]
    depleted_arr = np.asarray(
        depleted if depleted is not None else [[False] * n_paths] * 2, dtype=np.bool_
    )
    n_years = depleted_arr.shape[0]
    gis_arr = np.asarray(gis if gis is not None else np.zeros((n_years, n_paths)), dtype=np.int64)
    living_arr = np.asarray(
        living if living is not None else np.ones((n_years, n_paths)), dtype=np.int64
    )
    zeros = np.zeros((n_years, n_paths))
    return SimulationResult(
        years=np.arange(2026, 2026 + n_years, dtype=np.int64),
        net_worth=zeros,
        after_tax_net_worth=zeros,
        spending_achieved=zeros,
        tax_assessed=zeros,
        depleted=depleted_arr,
        estate_after_tax=estate_arr,
        death_year=np.zeros((1, n_paths), dtype=np.int64),
        gis_band_count=gis_arr,
        living_count=living_arr,
        seed=0,
    )


def ce(estate: list[float], g: float | None, shift: float | None) -> float:
    return certainty_equivalent_estate(make_result(estate=estate), g, shift)


# --- median, success, exposure ----------------------------------------------


def test_median_over_paths_with_an_odd_count() -> None:
    assert median_estate_after_tax(make_result(estate=[5.0, 1.0, 100.0])) == 5.0


def test_median_over_paths_with_an_even_count() -> None:
    assert median_estate_after_tax(make_result(estate=[1.0, 2.0, 10.0, 100.0])) == 6.0


def test_a_path_depleted_in_only_a_middle_year_counts_as_depleted() -> None:
    result = make_result(
        estate=[1.0, 1.0],
        depleted=[[False, False], [True, False], [False, False]],
    )
    assert success_probability(result) == 0.5


def test_success_probability_is_one_when_no_path_depletes() -> None:
    assert success_probability(make_result()) == 1.0


def test_success_probability_is_the_exact_fraction_of_a_mixture() -> None:
    result = make_result(
        estate=[1.0, 1.0, 1.0, 1.0],
        depleted=[[False, True, False, False], [False, True, True, False]],
    )
    assert success_probability(result) == 0.5


def test_gis_exposure_is_pooled_not_a_mean_of_path_ratios() -> None:
    # Path 0: 1 of 1 person-years in the band; path 1: 0 of 3.
    result = make_result(
        estate=[1.0, 1.0],
        depleted=[[False, False]] * 3,
        gis=[[1, 0], [0, 0], [0, 0]],
        living=[[1, 1], [0, 1], [0, 1]],
    )
    assert gis_exposure(result) == pytest.approx(1.0 / 4.0)
    assert gis_exposure(result) != pytest.approx((1.0 + 0.0) / 2.0)


def test_gis_exposure_is_zero_with_no_living_person_years() -> None:
    result = make_result(
        estate=[1.0, 1.0],
        depleted=[[False, False]] * 2,
        gis=np.zeros((2, 2), dtype=np.int64).tolist(),
        living=np.zeros((2, 2), dtype=np.int64).tolist(),
    )
    assert gis_exposure(result) == 0.0


# --- certainty equivalent ---------------------------------------------------

ESTATES = [0.0, 10_000.0, 250_000.0, 1_000_000.0]
SHIFT = 5_000.0


def test_ce_at_zero_risk_aversion_is_the_mean() -> None:
    assert ce(ESTATES, 0.0, SHIFT) == pytest.approx(float(np.mean(ESTATES)))


def test_ce_at_one_is_the_geometric_mean_of_the_shifted_estate_less_the_shift() -> None:
    expected = math.exp(float(np.mean(np.log(np.asarray(ESTATES) + SHIFT)))) - SHIFT
    assert ce(ESTATES, 1.0, SHIFT) == pytest.approx(expected, rel=1e-12)


def test_ce_is_continuous_through_one() -> None:
    at_one = ce(ESTATES, 1.0, SHIFT)
    below = ce(ESTATES, 1.0 - 1e-6, SHIFT)
    above = ce(ESTATES, 1.0 + 1e-6, SHIFT)
    assert below >= at_one >= above
    assert below == pytest.approx(at_one, rel=1e-4)
    assert above == pytest.approx(at_one, rel=1e-4)


def test_ce_is_non_increasing_in_risk_aversion_through_one() -> None:
    gammas = [1 - 1e-6, 1 - 1e-12, 1 - 1e-15, 1.0, 1 + 1e-15, 1 + 1e-12, 1 + 1e-6]
    values = [ce(ESTATES, g, SHIFT) for g in gammas]
    at_one = ce(ESTATES, 1.0, SHIFT)

    for earlier, later in pairwise(values):
        assert earlier >= later or earlier == pytest.approx(later, rel=1e-12)
    for value in values:
        assert value == pytest.approx(at_one, rel=1e-5)


def test_ce_at_a_large_risk_aversion_with_a_zero_estate_is_finite() -> None:
    value = ce(ESTATES, 500.0, SHIFT)
    assert math.isfinite(value)
    assert value >= -SHIFT


def test_ce_separates_two_results_that_differ_in_one_positive_path() -> None:
    low = ce([0.0, 100_000.0, 200_000.0], 2.0, SHIFT)
    high = ce([0.0, 100_000.0, 300_000.0], 2.0, SHIFT)
    assert high > low


def test_ce_matches_the_closed_form_when_no_estate_is_zero() -> None:
    estates = [20_000.0, 90_000.0, 400_000.0]
    g = 2.0
    x = np.asarray(estates) + SHIFT
    expected = float(np.mean(x ** (1.0 - g))) ** (1.0 / (1.0 - g)) - SHIFT
    assert ce(estates, g, SHIFT) == pytest.approx(expected, rel=1e-12)


def test_ce_without_a_risk_aversion_raises_naming_the_field_and_objective() -> None:
    with pytest.raises(ValueError, match="risk_aversion") as caught:
        ce(ESTATES, None, SHIFT)
    assert "certainty_equivalent_estate" in str(caught.value)
    assert "estate_utility_shift" not in str(caught.value)


def test_ce_without_a_shift_raises_naming_the_field_and_objective() -> None:
    with pytest.raises(ValueError, match="estate_utility_shift") as caught:
        ce(ESTATES, 2.0, None)
    assert "certainty_equivalent_estate" in str(caught.value)
    assert "risk_aversion" not in str(caught.value)


def test_ce_without_either_preference_names_both() -> None:
    with pytest.raises(ValueError) as caught:
        ce(ESTATES, None, None)
    assert "Scenario.risk_aversion" in str(caught.value)
    assert "Scenario.estate_utility_shift" in str(caught.value)


# --- select_objective -------------------------------------------------------


def test_each_selectable_objective_matches_its_function() -> None:
    result = make_result(
        estate=[0.0, 10.0, 300.0, 5000.0],
        depleted=[[False, True, False, False], [False, True, False, False]],
    )
    expected = {
        "median_estate_after_tax": median_estate_after_tax(result),
        "certainty_equivalent_estate": certainty_equivalent_estate(result, 2.0, 100.0),
        "success_probability": success_probability(result),
    }
    assert set(expected) == set(OBJECTIVE_NAMES)
    for name, value in expected.items():
        objective = select_objective(name, risk_aversion=2.0, estate_utility_shift=100.0)
        assert callable(objective)
        assert objective(result) == value


def test_an_unknown_objective_name_raises_listing_the_names() -> None:
    with pytest.raises(ValueError, match="nonsense") as caught:
        select_objective("nonsense", risk_aversion=2.0, estate_utility_shift=1.0)
    for name in OBJECTIVE_NAMES:
        assert name in str(caught.value)


def test_gis_exposure_is_not_a_selectable_objective() -> None:
    with pytest.raises(ValueError, match="gis_exposure"):
        select_objective("gis_exposure", risk_aversion=2.0, estate_utility_shift=1.0)


@pytest.mark.parametrize(
    ("risk_aversion", "shift", "field"),
    [
        (None, 1.0, "risk_aversion"),
        (2.0, None, "estate_utility_shift"),
        (None, None, "risk_aversion"),
    ],
)
def test_selecting_the_certainty_equivalent_without_a_preference_raises_at_selection(
    risk_aversion: float | None, shift: float | None, field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        select_objective(
            "certainty_equivalent_estate",
            risk_aversion=risk_aversion,
            estate_utility_shift=shift,
        )

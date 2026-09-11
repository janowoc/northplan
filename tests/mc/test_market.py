# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``engine.mc.market.MarketInputs``, built directly from synthetic arrays.

``tests/core/test_build.py`` exercises the builder from a real scenario;
this module exercises ``MarketInputs`` itself, in isolation, against inputs
constructed by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.mc.market import DEFAULT_KIND, MarketInputs

NAMES = ("equity", "bonds")
MEANS = np.array([0.05, 0.01])
COV = np.array([[0.16**2, 0.0016], [0.0016, 0.05**2]])
INTEREST = np.array([0.0, 0.03])
DIVIDENDS = np.array([0.02, 0.0])
GAINS = np.array([0.005, 0.0])
DEFAULT_WEIGHTS = np.array([0.6, 0.4])
RESP_WEIGHTS = np.array([0.3, 0.7])
INVESTABLE_KINDS = frozenset({"rrsp", "rrif", "lira", "lif", "tfsa", "taxable", "resp"})


def _build(**overrides: object) -> MarketInputs:
    kwargs = dict(
        asset_class_names=NAMES,
        annual_means=MEANS,
        annual_covariance=COV,
        interest_yields=INTEREST,
        dividend_yields=DIVIDENDS,
        distributed_gains_yields=GAINS,
        weights_by_kind={DEFAULT_KIND: DEFAULT_WEIGHTS, "resp": RESP_WEIGHTS},
        investable_kinds=INVESTABLE_KINDS,
    )
    kwargs.update(overrides)
    return MarketInputs(**kwargs)


_FIELD_CONSTANTS = {
    "annual_means": MEANS,
    "annual_covariance": COV,
    "interest_yields": INTEREST,
    "dividend_yields": DIVIDENDS,
    "distributed_gains_yields": GAINS,
}


class TestConstructionCopies:
    @pytest.mark.parametrize("field_name", list(_FIELD_CONSTANTS))
    def test_caller_array_stays_writeable_and_independent(self, field_name: str) -> None:
        constant = _FIELD_CONSTANTS[field_name]
        caller = constant.copy()
        market = _build(**{field_name: caller})

        caller[(0,) * caller.ndim] = 999.0

        assert np.array_equal(getattr(market, field_name), constant)
        assert caller.flags.writeable

    def test_a_caller_weight_vector_is_copied(self) -> None:
        caller = RESP_WEIGHTS.copy()
        market = _build(weights_by_kind={DEFAULT_KIND: DEFAULT_WEIGHTS, "resp": caller})

        caller[0] = 999.0

        assert np.array_equal(market.weights("resp"), RESP_WEIGHTS)
        assert caller.flags.writeable


class TestReadOnly:
    @pytest.mark.parametrize(
        "field_name",
        [
            "annual_means",
            "annual_covariance",
            "interest_yields",
            "dividend_yields",
            "distributed_gains_yields",
        ],
    )
    def test_array_field_is_read_only(self, field_name: str) -> None:
        market = _build()
        array = getattr(market, field_name)
        with pytest.raises(ValueError):
            array[(0,) * array.ndim] = 0.0

    def test_weight_vectors_are_read_only(self) -> None:
        market = _build()
        with pytest.raises(ValueError):
            market.weights_by_kind[DEFAULT_KIND][0] = 0.0

    def test_weights_by_kind_rejects_item_assignment(self) -> None:
        market = _build()
        with pytest.raises(TypeError):
            market.weights_by_kind["default"] = np.array([1.0, 0.0])


class TestWeights:
    def test_explicit_entry_is_returned(self) -> None:
        market = _build()
        np.testing.assert_array_equal(market.weights("resp"), RESP_WEIGHTS)

    def test_investable_kind_with_no_entry_falls_back_to_default(self) -> None:
        market = _build()
        np.testing.assert_array_equal(market.weights("rrsp"), DEFAULT_WEIGHTS)

    def test_non_investable_kind_raises_and_names_it(self) -> None:
        market = _build()
        with pytest.raises(ValueError, match="'cash'"):
            market.weights("cash")


class TestWeightedYields:
    def test_equals_the_dot_products(self) -> None:
        market = _build()
        expected = (
            float(RESP_WEIGHTS @ INTEREST),
            float(RESP_WEIGHTS @ DIVIDENDS),
            float(RESP_WEIGHTS @ GAINS),
        )
        assert market.weighted_yields("resp") == pytest.approx(expected)


class TestRefusals:
    def test_empty_names(self) -> None:
        with pytest.raises(ValueError, match="asset_class_names"):
            _build(
                asset_class_names=(),
                annual_means=np.array([]),
                annual_covariance=np.zeros((0, 0)),
                interest_yields=np.array([]),
                dividend_yields=np.array([]),
                distributed_gains_yields=np.array([]),
                weights_by_kind={DEFAULT_KIND: np.array([])},
            )

    def test_annual_means_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="annual_means"):
            _build(annual_means=np.array([0.05]))

    def test_annual_covariance_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="annual_covariance"):
            _build(annual_covariance=np.eye(3))

    def test_a_yields_array_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="interest_yields"):
            _build(interest_yields=np.array([0.0]))

    def test_a_weight_vector_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="weights_by_kind\\['resp'\\]"):
            _build(weights_by_kind={DEFAULT_KIND: DEFAULT_WEIGHTS, "resp": np.array([1.0])})

    def test_missing_default(self) -> None:
        with pytest.raises(ValueError, match="'default'"):
            _build(weights_by_kind={"resp": RESP_WEIGHTS})

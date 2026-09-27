# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``engine.policy.build``: ``build_policy``, ``CompositePolicy``, and ``expand_grid``."""

from __future__ import annotations

import pydantic
import pytest

from engine.core.build import build_initial_state
from engine.params.loader import load_year
from engine.policy.build import CompositePolicy, build_policy, expand_grid
from engine.scenario.schema import resolve_policy_path
from engine.scenario.start_ages import StartAgeNotAllowedError, check_start_ages


class TestElectionsRoundTrip:
    def test_the_example(self, scenario):
        policy = build_policy(scenario.policies[0], scenario.household)
        state = build_initial_state(scenario, n_paths=1)
        assert policy.elections() == state.elections

    def test_the_couple(self, couple_scenario):
        policy = build_policy(couple_scenario.policies[0], couple_scenario.household)
        state = build_initial_state(couple_scenario, n_paths=1)
        assert policy.elections() == state.elections

    def test_withdrawal_order_matches_the_spec(self, scenario):
        policy = build_policy(scenario.policies[0], scenario.household)
        assert policy.withdrawal_order() == scenario.policies[0].withdrawal.order

    def test_every_free_parameter_resolves_to_its_own_value(self, scenario):
        policy = build_policy(scenario.policies[0], scenario.household)
        for path, value in policy.free_parameters().items():
            assert resolve_policy_path(scenario.policies[0], path) == value

    def test_free_parameters_exact_key_set_the_example(self, scenario):
        """The example's own ceiling is null, so it is excluded entirely -- unlike the
        couple, below, whose ceiling is a real bracket index.
        """
        policy = build_policy(scenario.policies[0], scenario.household)
        assert set(policy.free_parameters()) == {
            "contribution.weights.rrsp",
            "contribution.weights.tfsa",
            "contribution.weights.taxable",
            "contribution.weights.resp",
            "elections.cpp_start_age_years.a",
            "elections.oas_start_age_years.a",
            "elections.rrif_conversion.age_years",
            "elections.rrif_conversion.fraction",
        }

    def test_free_parameters_exact_key_set_the_couple(self, couple_scenario):
        """Both persons are already in pay (no CPP/OAS election keys), the contribution
        weights name only ``tfsa``, and -- unlike the example -- the ceiling is a real bracket
        index (1), so ``withdrawal.taxable_ceiling_bracket`` is included.
        """
        policy = build_policy(couple_scenario.policies[0], couple_scenario.household)
        assert set(policy.free_parameters()) == {
            "contribution.weights.tfsa",
            "elections.rrif_conversion.age_years",
            "elections.rrif_conversion.fraction",
            "withdrawal.taxable_ceiling_bracket",
        }


class TestExpandGrid:
    def test_the_example_grid(self, scenario):
        expanded = expand_grid(scenario)

        assert expanded.grid == {}
        expected_names = [
            "taxable-first[elections.cpp_start_age_years.a=60]",
            "taxable-first[elections.cpp_start_age_years.a=65]",
            "taxable-first[elections.cpp_start_age_years.a=70]",
        ]
        assert [p.name for p in expanded.policies] == expected_names
        assert [p.elections.cpp_start_age_years["a"] for p in expanded.policies] == [60, 65, 70]

        # expand_grid touches only the policies and the grid -- everything else about the
        # scenario is untouched.
        assert expanded.household == scenario.household
        assert expanded.assumptions == scenario.assumptions
        assert expanded.spending == scenario.spending

        params = load_year(expanded.start_year)
        check_start_ages(expanded, params)
        for policy_spec in expanded.policies:
            built = build_policy(policy_spec, expanded.household)
            assert isinstance(built, CompositePolicy)

    def test_a_grid_value_outside_the_cpp_window_raises_on_check_start_ages(self, scenario):
        widened = scenario.model_copy(
            update={"grid": {"elections.cpp_start_age_years.a": (1, 65, 70)}}
        )
        expanded = expand_grid(widened)
        params = load_year(expanded.start_year)
        with pytest.raises(StartAgeNotAllowedError):
            check_start_ages(expanded, params)

    def test_a_grid_over_one_weight_raises_the_schema_sum_error(self, scenario):
        widened = scenario.model_copy(update={"grid": {"contribution.weights.rrsp": (0.9, 0.95)}})
        with pytest.raises(pydantic.ValidationError, match="sum to"):
            expand_grid(widened)

    def test_empty_grid_returns_an_equal_scenario(self, scenario):
        no_grid = scenario.model_copy(update={"grid": {}})
        assert expand_grid(no_grid) == no_grid

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the refusal of a ceiling bracket the parameter year has no federal edge for."""

from __future__ import annotations

import dataclasses
import types
from pathlib import Path

import pytest

from engine.params.loader import MissingParameterError, ParamYear, load_year
from engine.scenario import (
    CeilingBracketNotAllowedError,
    Scenario,
    ScenarioError,
    check_ceiling_bracket,
    load_scenario,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "scenarios" / "example.yaml"
EDGES = len(load_year(2026).federal.sequence("brackets.edges_annual"))


def with_ceilings(*ceilings: int | None) -> Scenario:
    """The example with one policy per ceiling; the second is named "second"."""
    scenario = load_scenario(EXAMPLE)
    first = scenario.policies[0]
    policies = []
    for i, ceiling in enumerate(ceilings):
        withdrawal = first.withdrawal.model_copy(update={"taxable_ceiling_bracket": ceiling})
        name = first.name if i == 0 else "second"
        policies.append(first.model_copy(update={"name": name, "withdrawal": withdrawal}))
    return scenario.model_copy(update={"policies": tuple(policies)})


def with_federal_brackets(brackets: object) -> ParamYear:
    """2026 with the federal ``brackets`` replaced."""
    loaded = load_year(2026)
    federal = loaded.federal
    values = {**federal.values, "brackets": brackets}
    sets = {
        **loaded.sets,
        "federal": dataclasses.replace(federal, values=types.MappingProxyType(values)),
    }
    return dataclasses.replace(loaded, sets=types.MappingProxyType(sets))


def refusal(scenario: Scenario, params: ParamYear) -> CeilingBracketNotAllowedError:
    with pytest.raises(CeilingBracketNotAllowedError) as excinfo:
        check_ceiling_bracket(scenario, params)
    return excinfo.value


@pytest.mark.parametrize("index", [0, EDGES - 1])
def test_an_index_below_the_edge_count_is_accepted(index: int) -> None:
    assert EDGES >= 2
    assert check_ceiling_bracket(with_ceilings(index), load_year(2026)) is None


def test_no_ceiling_is_accepted() -> None:
    scenario = load_scenario(EXAMPLE)
    assert scenario.policies[0].withdrawal.taxable_ceiling_bracket is None
    assert check_ceiling_bracket(scenario, load_year(2026)) is None


@pytest.mark.parametrize("index", [EDGES, EDGES + 1, 10**400], ids=["edges", "edges+1", "10**400"])
def test_an_index_at_or_past_the_edge_count_is_refused(index: int) -> None:
    params = load_year(2026)
    name = load_scenario(EXAMPLE).policies[0].name

    error = refusal(with_ceilings(index), params)

    assert str(error) == (
        f"policies[{name!r}].withdrawal.taxable_ceiling_bracket: no federal "
        f"bracket edge has this index. brackets.edges_annual in {params.federal.source} "
        f"has {EDGES} edge(s), counted from 0, so the index must be below {EDGES}."
    )
    if index == 10**400:
        assert "0" * 100 not in str(error)


def test_the_edge_count_comes_from_the_parameter_year() -> None:
    """A synthetic one-edge table: the count is read from the year, not remembered."""
    params = with_federal_brackets(
        types.MappingProxyType({"edges_annual": (50000.0,), "rates": (0.1, 0.2)})
    )

    assert check_ceiling_bracket(with_ceilings(0), params) is None
    error = refusal(with_ceilings(1), params)
    assert str(error).endswith("has 1 edge(s), counted from 0, so the index must be below 1.")


def test_a_later_policy_is_checked() -> None:
    error = refusal(with_ceilings(EDGES - 1, EDGES), load_year(2026))
    assert str(error).startswith("policies['second'].")


def test_the_first_refused_policy_is_named() -> None:
    scenario = with_ceilings(EDGES, EDGES)
    error = refusal(scenario, load_year(2026))
    assert str(error).startswith(f"policies[{scenario.policies[0].name!r}].")


def test_a_parameter_year_for_another_year_is_refused() -> None:
    params = dataclasses.replace(load_year(2026), year=2027)

    with pytest.raises(ValueError, match="check_ceiling_bracket: params is for tax year 2027") as e:
        check_ceiling_bracket(with_ceilings(EDGES), params)

    assert type(e.value) is ValueError


def test_a_missing_edge_table_passes_through() -> None:
    params = with_federal_brackets(types.MappingProxyType({"rates": (0.1, 0.2)}))

    with pytest.raises(MissingParameterError):
        check_ceiling_bracket(load_scenario(EXAMPLE), params)


def test_it_is_a_scenario_error() -> None:
    assert issubclass(CeilingBracketNotAllowedError, ScenarioError)

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared fixtures and small builders for ``tests/policy``.

Two committed scenarios cover the two shapes ``engine.policy`` needs: ``example.yaml`` (one
person, an RRSP, a TFSA, a taxable account, one RESP beneficiary -- an accumulator) and
``late_life_couple.yaml`` (two persons, both already drawing RRIF/LIF income -- a
decumulator). ``two_person_scenario`` builds a third shape neither committed file has: two
persons who both hold an RRSP/TFSA/taxable account and two RESP beneficiaries, so the
contribution component's equal-split and multi-beneficiary behaviour has something to split.

This package carries no ``__init__.py``, so a test module
cannot import a sibling helper with ``from .conftest import ...``: pytest's rootless
collection would import ``conftest.py`` a second time under a different name, exactly the
failure ``tests/core/__init__.py`` documents and exists to prevent. Every helper below is
therefore exposed as a fixture -- most as a small factory fixture that hands back a callable,
so a test can still call it more than once with different arguments, or with a ``tmp_path``
only it has.

``make_context`` builds a :class:`~engine.core.context.MonthContext` whose boilerplate fields
(inflows, forced withdrawals, and so on) a unit test never reads, filled with correctly-shaped
zeros, so a test can set only the few fields (``cash_after_flows``, the month position) that
``OrderedWithdrawal.transfers``/``SplitContribution.transfers`` read.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from engine.core.build import (
    build_deterministic_draws,
    build_initial_state,
    build_market_inputs,
    draw_deaths,
)
from engine.core.context import AccountAmounts, ByKind, MonthContext, PersonInflows
from engine.core.indexation import real_year
from engine.core.state import HouseholdState, PersonState, updated
from engine.params.loader import load_year
from engine.scenario import load_scenario
from engine.scenario.schema import Scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "scenarios" / "example.yaml"
COUPLE = REPO_ROOT / "scenarios" / "late_life_couple.yaml"


def _example_values() -> dict[str, Any]:
    values = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    assert isinstance(values, dict)
    return values


def _couple_values() -> dict[str, Any]:
    values = yaml.safe_load(COUPLE.read_text(encoding="utf-8"))
    assert isinstance(values, dict)
    return values


def _build_from_values(
    values: dict[str, Any], tmp_path: Path, n_paths: int = 1
) -> tuple[Scenario, HouseholdState]:
    path = tmp_path / "policy_case.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    scenario = load_scenario(path)
    state = build_initial_state(scenario, n_paths=n_paths)
    return scenario, state


@pytest.fixture
def example_values() -> Callable[[], dict[str, Any]]:
    """A factory for a fresh, mutable dict of ``example.yaml`` each call.

    Mirrors ``tests/scenario/test_schema.py``'s helper of the same name, for a test here that
    needs a custom scenario built by dict mutation rather than a committed file as-is.
    """
    return _example_values


@pytest.fixture
def couple_values() -> Callable[[], dict[str, Any]]:
    """A factory for a fresh, mutable dict of ``late_life_couple.yaml`` each call."""
    return _couple_values


@pytest.fixture
def build_from_values(
    tmp_path: Path,
) -> Callable[..., tuple[Scenario, HouseholdState]]:
    """A factory: write a scenario dict out, load it, and build its opening state.

    ``factory(values, n_paths=1)`` returns ``(scenario, state)``. ``state`` has ``alive`` true
    and ``death_month_index`` at the not-yet-drawn sentinel on every path --
    ``build_initial_state`` alone, no ``draw_deaths`` -- enough for a unit test of one
    component that never reads ``death_month_index``.
    """

    def factory(values: dict[str, Any], n_paths: int = 1) -> tuple[Scenario, HouseholdState]:
        return _build_from_values(values, tmp_path, n_paths)

    return factory


@pytest.fixture
def scenario() -> Scenario:
    return load_scenario(EXAMPLE)


@pytest.fixture
def mortality(scenario):
    return load_year(scenario.start_year)["mortality"]


@pytest.fixture
def market(scenario):
    return build_market_inputs(scenario.assumptions)


@pytest.fixture
def real_params(scenario):
    return real_year(load_year(scenario.start_year), scenario.assumptions.inflation)


@pytest.fixture
def draws(scenario, market, mortality):
    return build_deterministic_draws(scenario, market, mortality)


@pytest.fixture
def opening_state(scenario, draws, mortality):
    state = build_initial_state(scenario, n_paths=draws.n_paths)
    return draw_deaths(state, draws, mortality)


@pytest.fixture
def couple_scenario() -> Scenario:
    return load_scenario(COUPLE)


@pytest.fixture
def couple_mortality(couple_scenario):
    return load_year(couple_scenario.start_year)["mortality"]


@pytest.fixture
def couple_market(couple_scenario):
    return build_market_inputs(couple_scenario.assumptions)


@pytest.fixture
def couple_real_params(couple_scenario):
    return real_year(load_year(couple_scenario.start_year), couple_scenario.assumptions.inflation)


@pytest.fixture
def couple_draws(couple_scenario, couple_market, couple_mortality):
    return build_deterministic_draws(couple_scenario, couple_market, couple_mortality)


@pytest.fixture
def couple_opening_state(couple_scenario, couple_draws, couple_mortality):
    state = build_initial_state(couple_scenario, n_paths=couple_draws.n_paths)
    return draw_deaths(state, couple_draws, couple_mortality)


def _two_person_scenario(scenario: Scenario) -> Scenario:
    """``scenario`` (the example) with a second person and a second beneficiary added.

    Person ``b`` is a copy of person ``a`` with both CPP and OAS already in pay (so no
    election is needed for them) and the same account balances/room. Beneficiary ``child2``
    is a copy of ``child1``, still subscribed to person ``a``. Neither committed scenario has
    two persons who both hold an RRSP/TFSA/taxable account, or two RESP beneficiaries, which
    the contribution component's equal-split behaviour needs to exercise. Built by dict
    mutation on ``scenario.model_dump()``, mirroring ``tests/scenario/test_schema.py``'s
    ``clone_person``, and revalidated through :meth:`Scenario.model_validate`.
    """
    values = scenario.model_dump(warnings=False)

    person_b = copy.deepcopy(values["household"]["persons"][0])
    person_b["id"] = "b"
    person_b["cpp"] = {"in_pay_monthly": 500.0}
    person_b["oas"] = {"in_pay_monthly": 500.0}
    person_b["employment"] = []
    person_b["db_pensions"] = []
    values["household"]["persons"] = [*values["household"]["persons"], person_b]

    beneficiary2 = copy.deepcopy(values["household"]["beneficiaries"][0])
    beneficiary2["id"] = "child2"
    values["household"]["beneficiaries"] = [*values["household"]["beneficiaries"], beneficiary2]

    return Scenario.model_validate(values)


@pytest.fixture
def two_person_scenario(scenario: Scenario) -> Scenario:
    return _two_person_scenario(scenario)


@pytest.fixture
def two_person_state(two_person_scenario: Scenario) -> HouseholdState:
    """Opening state for ``two_person_scenario``, one path, no deaths drawn (both alive)."""
    return build_initial_state(two_person_scenario, n_paths=1)


def _make_context(
    *,
    n_paths: int,
    n_persons: int,
    n_beneficiaries: int = 0,
    year: int,
    month: int,
    month_index: int,
    cash_after_flows: np.ndarray,
) -> MonthContext:
    zeros = np.zeros(n_paths, dtype=np.float64)
    zero_account = AccountAmounts(
        rrsp=zeros, rrif=zeros, lira=zeros, lif=zeros, tfsa=zeros, taxable=zeros
    )
    zero_inflow = PersonInflows(
        employment=zeros,
        cpp_contributions=zeros,
        ei_premiums=zeros,
        cpp=zeros,
        oas=zeros,
        db_pension=zeros,
        cpp_survivor=zeros,
        db_pension_survivor=zeros,
    )
    zero_bykind = ByKind(rrsp=zeros, rrif=zeros, lif=zeros, tfsa=zeros, taxable=zeros)
    return MonthContext(
        month_index=month_index,
        year=year,
        month=month,
        cash_opening=zeros,
        rolled_out=tuple(zero_account for _ in range(n_persons)),
        rolled_acb=tuple(zeros for _ in range(n_persons)),
        terminal_assessment=zeros,
        cash_to_estate=zeros,
        inflows=tuple(zero_inflow for _ in range(n_persons)),
        education_draws=tuple(zeros for _ in range(n_beneficiaries)),
        payroll_withholding=tuple(zeros for _ in range(n_persons)),
        spending=zeros,
        education_costs=tuple(zeros for _ in range(n_beneficiaries)),
        tax_settlement=tuple(zeros for _ in range(n_persons)),
        forced_withdrawals=tuple(zero_bykind for _ in range(n_persons)),
        forced_withholding=tuple(zeros for _ in range(n_persons)),
        cash_after_flows=np.asarray(cash_after_flows, dtype=np.float64),
    )


@pytest.fixture
def make_context() -> Callable[..., MonthContext]:
    """A factory for a :class:`MonthContext` with every boilerplate field zeroed, for a unit
    test that reads only ``cash_after_flows`` and the month position.
    """
    return _make_context


def _set_person(state: HouseholdState, index: int, **changes: object) -> HouseholdState:
    persons = list(state.persons)
    persons[index] = updated(persons[index], **changes)
    return updated(state, persons=tuple(persons))


@pytest.fixture
def set_person() -> Callable[..., HouseholdState]:
    """A factory: ``set_person(state, index, **changes)`` replaces ``persons[index]`` with
    ``updated(persons[index], **changes)``.
    """
    return _set_person


def _set_persons(state: HouseholdState, **person_changes: dict[str, object]) -> HouseholdState:
    """Apply :func:`_set_person`-style changes to more than one person index at once.

    ``person_changes`` maps a person index, as a string (``"0"``, ``"1"``), to the ``changes``
    mapping for that person -- a plain function parameter cannot be keyed by an ``int``.
    """
    new_state = state
    for index_str, changes in person_changes.items():
        new_state = _set_person(new_state, int(index_str), **changes)
    return new_state


@pytest.fixture
def set_persons() -> Callable[..., HouseholdState]:
    return _set_persons


def _replace_person_field(person: PersonState, field_path: tuple[str, ...], value: object):
    """Return ``person`` with the nested field named by ``field_path`` replaced by ``value``.

    ``field_path`` is e.g. ``("rrsp", "room")``: walks every segment but the last with
    ``getattr``, and rebuilds each level with :func:`~engine.core.state.updated`.
    """
    if len(field_path) == 1:
        return updated(person, **{field_path[0]: value})
    head, *rest = field_path
    child = getattr(person, head)
    return updated(person, **{head: _replace_person_field(child, tuple(rest), value)})


@pytest.fixture
def replace_person_field() -> Callable[..., PersonState]:
    return _replace_person_field

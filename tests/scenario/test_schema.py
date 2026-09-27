# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Every validation rule in the scenario schema, one failing file each.

Each case starts from the committed example, changes exactly one thing, writes
the result out as YAML, and loads it. Two consequences worth stating:

- The fixtures cannot drift away from the real format. A case is the example
  plus a diff, so a change to the schema that the example keeps up with is a
  change every fixture here keeps up with too.
- A rejection is asserted against a substring naming the *field*, not against a
  whole message. The wording of an error is meant to be improved; which field
  it points at is the contract.

The accepted cases at the end matter as much as the rejected ones. Three of
them are things the schema is deliberately not in the business of checking — a
loss position, a class that distributes more than it earns, and a benefit start
age outside the statutory window — and a well-meant tightening that broke any
of them would be caught here rather than by a household whose true position the
model refuses to accept.
"""

from __future__ import annotations

import collections.abc
import copy
import types
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import pydantic
import pytest
import yaml
from pydantic.fields import FieldInfo

from engine.scenario import (
    Assumptions,
    InvalidScenarioError,
    PolicySpec,
    Scenario,
    load_scenario,
    resolve_policy_path,
)
from engine.scenario.schema import Finite, FrozenMapping

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "scenarios" / "example.yaml"

#: A change to make to the parsed example before writing it back out.
Mutation = Callable[[dict[str, Any]], None]


def example_values() -> dict[str, Any]:
    """The example, parsed, as a fresh mutable structure each call."""
    values = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    assert isinstance(values, dict)
    return values


def _matched_bonds_vol() -> float:
    """A bonds vol whose ``vol / (1 + real_mean)`` ratio equals equity's.

    Computed from the example's own figures rather than a literal: at
    correlation +1, this is the boundary the schema must accept, and a
    hand-typed number would drift silently if the example's own vol or
    real_mean ever changed.
    """
    classes = example_values()["assumptions"]["asset_classes"]
    equity, bonds = classes["equity"], classes["bonds"]
    return equity["vol"] * (1.0 + bonds["real_mean"]) / (1.0 + equity["real_mean"])


def _walk(values: dict[str, Any], path: str) -> tuple[Any, Any]:
    """Return the container holding ``path``'s last segment, and that segment.

    Digits index into lists, so ``household.persons.0.birth_month`` reaches a
    person. Kept out of the schema deliberately: this is test scaffolding for
    poking at YAML, not an addressing convention the engine has.
    """
    segments = path.split(".")
    current: Any = values
    for segment in segments[:-1]:
        current = current[int(segment)] if segment.isdigit() else current[segment]
    last = segments[-1]
    return current, int(last) if last.isdigit() else last


def sets(path: str, value: Any) -> Mutation:
    """A mutation that writes ``value`` at ``path``."""

    def mutate(values: dict[str, Any]) -> None:
        container, key = _walk(values, path)
        container[key] = value

    return mutate


def deletes(path: str) -> Mutation:
    """A mutation that removes ``path``."""

    def mutate(values: dict[str, Any]) -> None:
        container, key = _walk(values, path)
        del container[key]

    return mutate


def clone_person(person_id: str, **changes: Any) -> Mutation:
    """A mutation that adds a copy of the first person under a new id."""

    def mutate(values: dict[str, Any]) -> None:
        persons = values["household"]["persons"]
        clone = copy.deepcopy(persons[0])
        clone["id"] = person_id
        clone.update(changes)
        persons.append(clone)

    return mutate


def write_scenario(tmp_path: Path, values: dict[str, Any], name: str = "case") -> Path:
    """Write ``values`` as a YAML scenario file and return its path."""
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def load_mutated(tmp_path: Path, mutations: Iterable[Mutation]) -> Scenario:
    """Apply ``mutations`` to the example, write it out, and load it."""
    values = example_values()
    for mutate in mutations:
        mutate(values)
    return load_scenario(write_scenario(tmp_path, values))


def rejected(case_id: str, expected: str, *mutations: Mutation) -> Any:
    """One rejection case: what to change, and the field the message must name."""
    return pytest.param(mutations, expected, id=case_id)


def accepted(case_id: str, *mutations: Mutation) -> Any:
    """One acceptance case: a change the schema must not object to."""
    return pytest.param(mutations, id=case_id)


REJECTIONS = [
    # --- Identity and household shape ---------------------------------------
    rejected(
        "two-persons-share-an-id",
        "household:",
        clone_person("a"),
    ),
    rejected(
        "beneficiary-takes-a-person-s-id",
        "household:",
        sets("household.beneficiaries.0.id", "a"),
    ),
    rejected(
        "three-persons",
        "persons",
        clone_person("b"),
        clone_person("c"),
    ),
    rejected(
        "no-persons",
        "persons",
        sets("household.persons", []),
    ),
    rejected(
        "birth-month-out-of-range",
        "birth_month",
        sets("household.persons.0.birth_month", 13),
    ),
    rejected(
        "birth-month-zero",
        "birth_month",
        sets("household.persons.0.birth_month", 0),
    ),
    rejected(
        "unknown-sex",
        "sex",
        sets("household.persons.0.sex", "x"),
    ),
    rejected(
        "province-is-not-a-code",
        "province",
        sets("household.province", "alberta"),
    ),
    rejected(
        "a-person-born-after-the-run-opens",
        "household.persons",
        sets("household.persons.0.birth_year", 2026),
        sets("household.persons.0.birth_month", 3),
    ),
    rejected(
        "a-beneficiary-born-after-the-run-opens",
        "household.beneficiaries",
        sets("household.beneficiaries.0.birth_year", 2026),
        sets("household.beneficiaries.0.birth_month", 3),
    ),
    # --- CPP entitlement ----------------------------------------------------
    rejected(
        "cpp-given-both-ways",
        "cpp:",
        sets("household.persons.0.cpp", {"contributory_history": 0.85, "in_pay_monthly": 1200}),
    ),
    rejected(
        "cpp-given-neither-way",
        "cpp:",
        sets("household.persons.0.cpp", {}),
    ),
    rejected(
        "contributory-history-above-one",
        "contributory_history",
        sets("household.persons.0.cpp", {"contributory_history": 1.5}),
    ),
    # --- Prior year net income -----------------------------------------------
    rejected(
        "prior-year-net-income-absent",
        "prior_year_net_income",
        deletes("household.persons.0.prior_year_net_income"),
    ),
    rejected(
        "negative-prior-year-net-income",
        "prior_year_net_income",
        sets("household.persons.0.prior_year_net_income", -1),
    ),
    rejected(
        "net-income-two-years-prior-absent",
        "net_income_two_years_prior",
        deletes("household.persons.0.net_income_two_years_prior"),
    ),
    rejected(
        "negative-net-income-two-years-prior",
        "net_income_two_years_prior",
        sets("household.persons.0.net_income_two_years_prior", -1),
    ),
    # --- Employment ---------------------------------------------------------
    rejected(
        "employment-bands-overlap",
        "employment:",
        sets(
            "household.persons.0.employment",
            [
                {"from_year": 2026, "to_year": 2031, "annual": 95000},
                {"from_year": 2031, "to_year": 2035, "annual": 50000},
            ],
        ),
    ),
    rejected(
        "employment-band-ends-before-it-starts",
        "employment:",
        sets("household.persons.0.employment.0.to_year", 2020),
    ),
    rejected(
        # ge=0.0 alone refuses nan but accepts inf, so this is the case that
        # fails if the flag is dropped from Money.
        "money-infinite",
        "employment.0.annual",
        sets("household.persons.0.employment.0.annual", float("inf")),
    ),
    # --- Accounts -----------------------------------------------------------
    rejected(
        "negative-balance",
        "balance",
        sets("household.persons.0.accounts.rrsp.balance", -1),
    ),
    rejected(
        "negative-room",
        "room",
        sets("household.persons.0.accounts.tfsa.room", -1),
    ),
    rejected(
        "negative-acb",
        "acb",
        sets("household.persons.0.accounts.taxable.acb", -1),
    ),
    rejected(
        "locked-in-balance-without-a-jurisdiction",
        "jurisdiction",
        deletes("household.persons.0.accounts.lira.jurisdiction"),
    ),
    rejected(
        "lif-balance-without-a-jurisdiction",
        "jurisdiction",
        sets("household.persons.0.accounts.lif", {"balance": 1000}),
    ),
    rejected(
        # The example's lira is registered in "ab".
        "lira-and-lif-in-different-jurisdictions",
        "jurisdiction",
        sets("household.persons.0.accounts.lif", {"balance": 1000, "jurisdiction": "on"}),
    ),
    rejected(
        "lira-and-lif-disagree-even-with-a-zero-balance",
        "jurisdiction",
        sets("household.persons.0.accounts.lif", {"balance": 0, "jurisdiction": "on"}),
    ),
    # --- Defined-benefit pensions -------------------------------------------
    rejected(
        "pension-indexation-is-not-one-of-the-two",
        "indexation",
        sets("household.persons.0.db_pensions.0.indexation", "partial"),
    ),
    rejected(
        "bridge-with-no-end",
        "bridge_to_age_years",
        sets("household.persons.0.db_pensions.0.bridge_annual", 8000),
        deletes("household.persons.0.db_pensions.0.bridge_to_age_years"),
    ),
    rejected(
        # The example person's own figures trip this the moment the bridge
        # is non-zero: born 1966-03 with bridge_to_age_years 65 ends the
        # bridge in 2031-03, and the pension itself starts 2031-04.
        "bridge-ends-before-the-pension-starts",
        "bridge",
        sets("household.persons.0.db_pensions.0.bridge_annual", 8000),
    ),
    rejected(
        # Fraction's bounds refuse nan (ge) and inf (le) without the flag, so
        # this case cannot tell whether the flag is present; it exists so
        # that each alias has a case.
        "fraction-nan",
        "db_pensions.0.survivor_share",
        sets("household.persons.0.db_pensions.0.survivor_share", float("nan")),
    ),
    rejected(
        "two-pensions-with-one-name",
        "db_pensions",
        sets(
            "household.persons.0.db_pensions",
            [
                {
                    "name": "employer",
                    "start_year": 2031,
                    "start_month": 4,
                    "annual": 30000,
                    "indexation": "none",
                },
                {
                    "name": "employer",
                    "start_year": 2036,
                    "start_month": 1,
                    "annual": 10000,
                    "indexation": "full",
                },
            ],
        ),
    ),
    # --- RESP and education -------------------------------------------------
    rejected(
        "subscriber-is-not-in-the-household",
        "resp:",
        sets("household.beneficiaries.0.resp.subscriber", "nobody"),
    ),
    rejected(
        "negative-grant-room",
        "grant_room_carried",
        sets("household.beneficiaries.0.resp.grant_room_carried", -100),
    ),
    rejected(
        "education-lasts-no-months",
        "months",
        sets("household.beneficiaries.0.education.months", 0),
    ),
    rejected(
        "education-starts-in-month-thirteen",
        "start_month",
        sets("household.beneficiaries.0.education.start_month", 13),
    ),
    # --- Spending -----------------------------------------------------------
    rejected(
        "spending-bands-out-of-order",
        "spending.schedule",
        sets(
            "spending.schedule",
            [{"from_year": 2032, "annual": 65000}, {"from_year": 2026, "annual": 80000}],
        ),
    ),
    rejected(
        "spending-bands-repeat-a-year",
        "spending.schedule",
        sets(
            "spending.schedule",
            [{"from_year": 2026, "annual": 80000}, {"from_year": 2026, "annual": 65000}],
        ),
    ),
    rejected(
        "spending-starts-after-the-run-does",
        "spending.schedule",
        sets("spending.schedule.0.from_year", 2027),
    ),
    rejected(
        "survivor-share-of-zero",
        "survivor_share",
        sets("spending.survivor_share", 0),
    ),
    rejected(
        "survivor-share-above-one",
        "survivor_share",
        sets("spending.survivor_share", 1.2),
    ),
    # --- Asset classes ------------------------------------------------------
    rejected(
        "negative-volatility",
        "vol",
        sets("assumptions.asset_classes.equity.vol", -0.16),
    ),
    rejected(
        "negative-yield",
        "dividend_yield",
        sets("assumptions.asset_classes.equity.dividend_yield", -0.02),
    ),
    rejected(
        # ge=0.0 accepts inf and no later validator reads a yield, so the
        # flag is the only guard and this is the case that fails if it is
        # dropped.
        "yield-infinite",
        "interest_yield",
        sets("assumptions.asset_classes.bonds.interest_yield", float("inf")),
    ),
    rejected(
        "no-asset-classes",
        "asset_classes",
        sets("assumptions.asset_classes", {}),
    ),
    rejected(
        "inflation-at-minus-one",
        "inflation",
        sets("assumptions.inflation", -1.0),
    ),
    rejected(
        "inflation-infinite",
        "assumptions.inflation",
        sets("assumptions.inflation", float("inf")),
    ),
    # --- Correlation --------------------------------------------------------
    rejected(
        "correlation-not-square",
        "correlation",
        sets("assumptions.correlation", [[1.0, 0.1], [0.1]]),
    ),
    rejected(
        "correlation-wrong-size-for-the-classes",
        "correlation",
        sets("assumptions.correlation", [[1.0]]),
    ),
    rejected(
        "correlation-not-symmetric",
        "correlation",
        sets("assumptions.correlation", [[1.0, 0.1], [0.9, 1.0]]),
    ),
    rejected(
        "correlation-diagonal-is-not-one",
        "correlation",
        sets("assumptions.correlation", [[0.9, 0.1], [0.1, 1.0]]),
    ),
    rejected(
        "correlation-not-positive-semi-definite",
        "correlation",
        sets("assumptions.correlation", [[1.0, 1.5], [1.5, 1.0]]),
    ),
    rejected(
        # The matrix is symmetric, so a refusal blaming asymmetry would be
        # the wrong diagnosis; allow_inf_nan=False on the element type
        # catches it before any model validator runs.
        "correlation-entry-infinite",
        "assumptions.correlation.0.1",
        sets("assumptions.correlation", [[1.0, float("inf")], [float("inf"), 1.0]]),
    ),
    rejected(
        "correlation-diagonal-nan",
        "assumptions.correlation.0.0",
        sets("assumptions.correlation", [[float("nan"), 0.1], [0.1, 1.0]]),
    ),
    # --- Attainability of the return assumptions -----------------------------
    rejected(
        "correlation-minus-one-at-five-percent-vol",
        "asset classes 'equity' and 'bonds': no lognormal distribution",
        sets("assumptions.asset_classes.equity.vol", 0.05),
        sets("assumptions.correlation", [[1.0, -1.0], [-1.0, 1.0]]),
    ),
    rejected(
        "correlation-plus-one-with-unequal-vol-to-growth-ratios",
        "asset classes 'equity' and 'bonds': no lognormal distribution",
        sets("assumptions.correlation", [[1.0, 1.0], [1.0, 1.0]]),
    ),
    rejected(
        "a-negative-correlation-too-strong-for-the-log-argument",
        "asset classes 'equity' and 'bonds': the moment-matching log argument",
        sets("assumptions.asset_classes.equity.vol", 1.2),
        sets("assumptions.asset_classes.bonds.vol", 1.2),
        sets("assumptions.correlation", [[1.0, -1.0], [-1.0, 1.0]]),
    ),
    rejected(
        "a-real-mean-at-minus-one",
        "asset class 'equity': real_mean",
        sets("assumptions.asset_classes.equity.real_mean", -1.0),
    ),
    rejected(
        "a-real-mean-that-is-not-a-number",
        "asset_classes.equity.real_mean",
        sets("assumptions.asset_classes.equity.real_mean", float("nan")),
    ),
    # --- Allocations --------------------------------------------------------
    rejected(
        "allocation-names-an-unknown-class",
        "allocations",
        sets("assumptions.allocations.default", {"equity": 0.6, "gold": 0.4}),
    ),
    rejected(
        "allocation-does-not-sum-to-one",
        "allocations",
        sets("assumptions.allocations.default", {"equity": 0.6, "bonds": 0.3}),
    ),
    rejected(
        "no-default-allocation",
        "allocations",
        deletes("assumptions.allocations.default"),
    ),
    rejected(
        "allocation-for-an-account-that-holds-nothing",
        "allocations",
        sets("assumptions.allocations.cash", {"equity": 0.6, "bonds": 0.4}),
    ),
    rejected(
        "allocation-with-a-short-position",
        "allocations",
        sets("assumptions.allocations.default", {"equity": 1.4, "bonds": -0.4}),
    ),
    rejected(
        # Pins allow_inf_nan=False on the element type: nan is refused at its
        # own location, which the sum check's message
        # ("assumptions.allocations['default']: weights sum to ...") never
        # contains.
        "allocation-weight-nan",
        "assumptions.allocations.default.equity",
        sets("assumptions.allocations.default", {"equity": float("nan"), "bonds": 0.4}),
    ),
    # --- Contribution rule --------------------------------------------------
    rejected(
        "contribution-weights-do-not-sum-to-one",
        "contribution.weights",
        sets("policies.0.contribution.weights", {"rrsp": 0.5, "tfsa": 0.3, "taxable": 0.1}),
    ),
    rejected(
        "contribution-weight-is-negative",
        "contribution.weights",
        sets("policies.0.contribution.weights", {"rrsp": -0.5, "tfsa": 0.3, "taxable": 1.2}),
    ),
    rejected(
        "contribution-to-an-account-that-takes-none",
        "contribution.weights",
        sets("policies.0.contribution.weights", {"rrsp": 0.5, "tfsa": 0.3, "rrif": 0.2}),
    ),
    rejected(
        "spill-order-names-an-unknown-account",
        "contribution.spill_order",
        sets("policies.0.contribution.spill_order", ["tfsa", "rrif", "taxable"]),
    ),
    rejected(
        "spill-order-repeats-an-account",
        "contribution.spill_order",
        sets("policies.0.contribution.spill_order", ["tfsa", "tfsa", "taxable"]),
    ),
    rejected(
        # Pins allow_inf_nan=False on the element type: nan is refused at its
        # own location, which the sum check's message
        # ("contribution.weights: sum to ...") never contains.
        "contribution-weight-nan",
        "contribution.weights.rrsp",
        sets(
            "policies.0.contribution.weights",
            {"rrsp": float("nan"), "tfsa": 0.3, "taxable": 0.2, "resp": 0.0},
        ),
    ),
    # --- Withdrawal rule ----------------------------------------------------
    rejected(
        "withdrawal-order-names-an-unknown-account",
        "withdrawal.order",
        sets("policies.0.withdrawal.order", ["taxable", "resp", "tfsa"]),
    ),
    rejected(
        "withdrawal-order-repeats-an-account",
        "withdrawal.order",
        sets("policies.0.withdrawal.order", ["taxable", "rrif", "taxable"]),
    ),
    rejected(
        "withdrawal-order-names-a-lira",
        "withdrawal.order",
        sets("policies.0.withdrawal.order", ["taxable", "lira", "tfsa"]),
    ),
    rejected(
        "negative-bracket-index",
        "taxable_ceiling_bracket",
        sets("policies.0.withdrawal.taxable_ceiling_bracket", -1),
    ),
    # --- Elections ----------------------------------------------------------
    rejected(
        "rrif-conversion-fraction-above-one",
        "fraction",
        sets("policies.0.elections.rrif_conversion.fraction", 1.5),
    ),
    rejected(
        "election-for-someone-not-in-the-household",
        "cpp_start_age_years",
        sets("policies.0.elections.cpp_start_age_years", {"b": 65}),
    ),
    rejected(
        "no-election-for-a-person-who-has-not-started-cpp",
        "cpp_start_age_years",
        clone_person("b"),
        sets("policies.0.elections.oas_start_age_years", {"a": 65, "b": 65}),
    ),
    rejected(
        "no-election-for-a-person-who-has-not-started-oas",
        "oas_start_age_years",
        sets("policies.0.elections.oas_start_age_years", {}),
    ),
    rejected(
        "two-policies-with-one-name",
        "policies:",
        sets("policies", [example_values()["policies"][0]] * 2),
    ),
    rejected(
        "no-policies",
        "policies",
        sets("policies", []),
    ),
    # --- Grid ---------------------------------------------------------------
    rejected(
        "grid-key-names-nobody",
        "grid[",
        sets("grid", {"elections.cpp_start_age_years.b": [60, 65]}),
    ),
    rejected(
        "grid-key-names-no-field",
        "grid[",
        sets("grid", {"withdrawal.ceiling": [1, 2]}),
    ),
    rejected(
        "grid-key-names-a-block-rather-than-a-number",
        "grid[",
        sets("grid", {"elections.rrif_conversion": [1, 2]}),
    ),
    rejected(
        "grid-key-names-a-flag",
        "grid[",
        sets("grid", {"withdrawal.fill_pension_credit": [0, 1]}),
    ),
    rejected(
        # With the ceiling left null (the example's own default), walking `.0` off it
        # would already raise -- "cannot be walked into any further" fires on any
        # terminal leaf, None included. Restoring the ceiling to a number first keeps
        # this case testing what its name says: walking past a resolved *number*, not
        # merely landing on the same branch by accident on a null one.
        "grid-key-walks-into-a-number",
        "grid[",
        sets("policies.0.withdrawal.taxable_ceiling_bracket", 1),
        sets("grid", {"withdrawal.taxable_ceiling_bracket.0": [1, 2]}),
    ),
    rejected(
        "grid-over-a-null-ceiling-is-not-a-number",
        "grid['withdrawal.taxable_ceiling_bracket']",
        sets("grid", {"withdrawal.taxable_ceiling_bracket": [1, 2]}),
    ),
    rejected(
        "grid-with-nothing-to-try",
        "grid[",
        sets("grid", {"elections.cpp_start_age_years.a": []}),
    ),
    rejected(
        # Replace the whole mapping: the grid key contains dots, which
        # sets() would split.
        "grid-value-infinite",
        "grid.`elections.cpp_start_age_years.a`",
        sets("grid", {"elections.cpp_start_age_years.a": [60, float("inf"), 70]}),
    ),
    # --- Unknown keys -------------------------------------------------------
    rejected(
        "unknown-top-level-key",
        "horizon_years",
        sets("horizon_years", 30),
    ),
    rejected(
        "misspelled-nested-key",
        "survivorship_share",
        deletes("spending.survivor_share"),
        sets("spending.survivorship_share", 0.75),
    ),
    rejected(
        "unknown-account-kind",
        "crypto",
        sets("household.persons.0.accounts.crypto", {"balance": 1000}),
    ),
    # --- Risk aversion --------------------------------------------------------
    rejected(
        "risk-aversion-negative",
        "risk_aversion",
        sets("risk_aversion", -1.0),
    ),
    rejected(
        # The one of the three that fails if allow_inf_nan=False is dropped:
        # ge=0.0 alone already refuses nan, but accepts inf.
        "risk-aversion-infinite",
        "risk_aversion",
        sets("risk_aversion", float("inf")),
    ),
    rejected(
        "risk-aversion-nan",
        "risk_aversion",
        sets("risk_aversion", float("nan")),
    ),
]


ACCEPTANCES = [
    accepted(
        # The boundary of the rule the rejection above tests. The person
        # reaches 65 in 2031-03; starting the pension in that same month
        # makes a one-month bridge, which is legal. Written as its own case
        # because the rejection alone cannot tell `<` from `<=` — the
        # example's own figures are strictly before, so both comparisons
        # reject it and a widened rule would go unnoticed.
        "a-bridge-ending-the-month-the-pension-starts",
        sets("household.persons.0.db_pensions.0.bridge_annual", 8000),
        sets("household.persons.0.db_pensions.0.start_month", 3),
    ),
    accepted(
        "a-taxable-holding-standing-at-a-loss",
        sets("household.persons.0.accounts.taxable.acb", 200000),
    ),
    accepted(
        "a-class-that-distributes-more-than-it-earns",
        sets(
            "assumptions.asset_classes.bonds",
            {
                "real_mean": 0.01,
                "vol": 0.05,
                "interest_yield": 0.04,
                "dividend_yield": 0.0,
                "distributed_gains_yield": 0.0,
            },
        ),
    ),
    accepted(
        "a-benefit-start-age-outside-the-statutory-window",
        sets("policies.0.elections.cpp_start_age_years", {"a": 55}),
        sets("grid", {}),
    ),
    accepted(
        "a-riskless-class",
        sets("assumptions.asset_classes.bonds.vol", 0.0),
    ),
    accepted(
        "correlation-plus-one-with-equal-vol-to-growth-ratios",
        sets("assumptions.asset_classes.bonds.vol", _matched_bonds_vol()),
        sets("assumptions.correlation", [[1.0, 1.0], [1.0, 1.0]]),
    ),
    accepted(
        "deflation",
        sets("assumptions.inflation", -0.01),
    ),
    accepted(
        "a-household-with-no-locked-in-money",
        deletes("household.persons.0.accounts.lira"),
    ),
    accepted(
        "a-prior-year-net-income-of-zero",
        sets("household.persons.0.prior_year_net_income", 0),
    ),
    accepted(
        "a-net-income-two-years-prior-of-zero",
        sets("household.persons.0.net_income_two_years_prior", 0),
    ),
    accepted(
        "a-person-already-receiving-oas-needs-no-start-age",
        sets("household.persons.0.oas", {"in_pay_monthly": 800}),
        sets("policies.0.elections.oas_start_age_years", {}),
    ),
    accepted(
        "a-withdrawal-order-naming-a-lif",
        sets("policies.0.withdrawal.order", ["lif", "taxable"]),
    ),
    accepted(
        "a-lira-and-a-lif-in-one-jurisdiction",
        sets("household.persons.0.accounts.lif", {"balance": 50000, "jurisdiction": "ab"}),
    ),
    accepted(
        "a-person-already-converted-to-a-lif",
        deletes("household.persons.0.accounts.lira"),
        sets("household.persons.0.accounts.lif", {"balance": 80000, "jurisdiction": "ab"}),
    ),
    accepted(
        "an-allocation-for-a-lira",
        sets("assumptions.allocations.lira", {"equity": 0.5, "bonds": 0.5}),
    ),
    accepted(
        "an-allocation-for-a-lif",
        sets("assumptions.allocations.lif", {"equity": 0.5, "bonds": 0.5}),
    ),
    accepted(
        "a-person-with-no-accounts-at-all",
        deletes("household.persons.0.accounts"),
    ),
    accepted(
        "no-beneficiaries",
        deletes("household.beneficiaries"),
    ),
    accepted(
        "no-grid",
        deletes("grid"),
    ),
    accepted(
        "no-employment-and-no-pension",
        deletes("household.persons.0.employment"),
        deletes("household.persons.0.db_pensions"),
    ),
    accepted(
        "a-second-person-already-receiving-cpp-needs-no-start-age",
        clone_person("b", cpp={"in_pay_monthly": 1200}),
        sets("policies.0.elections.oas_start_age_years", {"a": 65, "b": 65}),
    ),
    accepted(
        "a-single-life-pension",
        sets("household.persons.0.db_pensions.0.survivor_share", 0.0),
    ),
    accepted(
        # The boundary of the rule the two rejections above test. A birth on
        # 1 January of start_year is age zero months at month index 0, which
        # is legal, and the rejection cases alone cannot tell `<=` from `<`:
        # their birth month (March) is already past the boundary, so a
        # narrower rule that rejected January too would go unnoticed without
        # this case.
        "a-person-born-in-january-of-the-start-year",
        sets("household.persons.0.birth_year", 2026),
        sets("household.persons.0.birth_month", 1),
    ),
    accepted(
        # Zero is risk neutrality, a legal preference, and the only case that
        # can tell ge=0.0 from gt=0.0.
        "risk-aversion-zero",
        sets("risk_aversion", 0.0),
    ),
    accepted(
        # The example's own ceiling, unmutated -- an explicit case so a future
        # change to the example does not quietly stop covering it.
        "a-null-taxable-ceiling-bracket",
        sets("policies.0.withdrawal.taxable_ceiling_bracket", None),
    ),
]


def test_the_unmutated_example_loads_through_this_harness(tmp_path: Path) -> None:
    """The parse-mutate-dump-load round trip is sound.

    Without this, a harness that produced an unloadable file would make every
    rejection below pass for the wrong reason, and the whole suite would be
    asserting that a broken YAML writer is broken.
    """
    scenario = load_mutated(tmp_path, ())

    assert scenario.name == "example-household"
    assert scenario.policies[0].name == "taxable-first"


@pytest.mark.parametrize(("mutations", "expected"), REJECTIONS)
def test_an_invalid_scenario_is_rejected_and_the_message_names_the_field(
    tmp_path: Path, mutations: tuple[Mutation, ...], expected: str
) -> None:
    """One rule, one broken file, one message pointing at the field to fix."""
    with pytest.raises(InvalidScenarioError) as excinfo:
        load_mutated(tmp_path, mutations)

    message = str(excinfo.value)
    assert expected in message, (
        f"The rejection must name {expected!r} so the reader knows what to change. Got:\n{message}"
    )


@pytest.mark.parametrize("mutations", ACCEPTANCES)
def test_a_valid_scenario_is_accepted(tmp_path: Path, mutations: tuple[Mutation, ...]) -> None:
    """Things the schema must not object to, including three it must never check."""
    assert isinstance(load_mutated(tmp_path, mutations), Scenario)


def test_a_broken_person_is_reported_once_and_not_as_an_empty_household(
    tmp_path: Path,
) -> None:
    """One mistake, one message.

    ``min_length`` on the persons tuple is applied after item validation, so a
    bad month inside a person used to be reported twice: once truthfully, and
    once as "should have at least 1 item, not 0" — which reads as though the
    file listed nobody, and sends the reader to the wrong part of it. The count
    checks are model validators for this reason, and this is what says so.
    """
    with pytest.raises(InvalidScenarioError) as excinfo:
        load_mutated(tmp_path, (sets("household.persons.0.birth_month", 13),))

    message = str(excinfo.value)
    assert "1 validation error" in message, message
    assert "at least 1 item" not in message, message


def test_absent_prior_year_net_income_names_the_person_by_id(tmp_path: Path) -> None:
    """Pydantic's own "Field required" only names ``household.persons.0``.

    A missing key is refused by our own before-validator instead, precisely
    so the message can point at the person's ``id`` rather than a positional
    index.
    """
    with pytest.raises(InvalidScenarioError) as excinfo:
        load_mutated(tmp_path, (deletes("household.persons.0.prior_year_net_income"),))

    message = str(excinfo.value)
    assert "prior_year_net_income" in message, message
    assert "person 'a'" in message, message
    assert "Field required" not in message, message


def test_absent_net_income_two_years_prior_names_the_person_by_id(tmp_path: Path) -> None:
    """Mirrors :func:`test_absent_prior_year_net_income_names_the_person_by_id` for the
    sibling field; the two keys share one before-validator (brief-49 s4(b))."""
    with pytest.raises(InvalidScenarioError) as excinfo:
        load_mutated(tmp_path, (deletes("household.persons.0.net_income_two_years_prior"),))

    message = str(excinfo.value)
    assert "net_income_two_years_prior" in message, message
    assert "person 'a'" in message, message
    assert "Field required" not in message, message


def test_absent_both_net_income_fields_names_both_in_one_message(tmp_path: Path) -> None:
    """One before-validator covers both keys (brief-49 s4(b)) precisely so a scenario
    missing both gets one message naming both, not just the first pydantic would report."""
    with pytest.raises(InvalidScenarioError) as excinfo:
        load_mutated(
            tmp_path,
            (
                deletes("household.persons.0.prior_year_net_income"),
                deletes("household.persons.0.net_income_two_years_prior"),
            ),
        )

    message = str(excinfo.value)
    assert "1 validation error" in message, message
    assert "prior_year_net_income" in message, message
    assert "net_income_two_years_prior" in message, message


def test_absent_prior_year_net_income_and_absent_id_names_neither(tmp_path: Path) -> None:
    """With no ``id`` either, the message says so rather than printing ``None``."""
    with pytest.raises(InvalidScenarioError) as excinfo:
        load_mutated(
            tmp_path,
            (
                deletes("household.persons.0.id"),
                deletes("household.persons.0.prior_year_net_income"),
            ),
        )

    message = str(excinfo.value)
    assert "person (no id)" in message, message
    assert "person None" not in message, message


def test_every_rejection_has_its_own_case_id() -> None:
    """No rule is silently covered twice while another is not covered at all.

    Duplicated ids also make a failure report ambiguous about which case broke.
    """
    ids = [case.id for case in REJECTIONS] + [case.id for case in ACCEPTANCES]

    assert len(ids) == len(set(ids)), sorted({name for name in ids if ids.count(name) > 1})


# --- Every float in the schema is finite ------------------------------------


def _finite_metadata(metadata: Iterable[Any]) -> bool:
    """Whether any item of ``metadata`` pins ``allow_inf_nan=False``."""
    return any(getattr(item, "allow_inf_nan", None) is False for item in metadata)


def _expand_metadata(raw: Iterable[Any]) -> list[Any]:
    """Flatten ``raw``, pulling a nested ``FieldInfo``'s own metadata out.

    A ``Field(...)`` used inside a nested ``Annotated`` (an element of a tuple
    or a mapping value, rather than the field's own top-level type) shows up
    as a bare ``FieldInfo`` object among the ``Annotated`` metadata, with the
    actual constraints one level further in, on ``FieldInfo.metadata``.
    """
    expanded: list[Any] = []
    for item in raw:
        if isinstance(item, FieldInfo):
            expanded.extend(item.metadata)
        else:
            expanded.append(item)
    return expanded


#: Leaf annotations the walk accepts without descending further, because none
#: of them can ever be a bare float: a field of one of these types is never a
#: site :func:`_bare_float_sites` needs to flag.
_SAFE_LEAVES = (str, int, bool, type(None))

#: Generic origins the walk treats as transparent containers whose type
#: arguments it descends into. Any other origin — a generic pydantic
#: dataclass, a generic ``TypedDict``, a parameterised ``type X[T] = ...``
#: alias, or anything else this schema does not use — is reported as
#: unverifiable instead: such a generic resolves its own fields (or its
#: ``__value__``) by a route this walk does not retrace, so treating its type
#: arguments as if they were the whole story would silently miss a float
#: living inside it.
_CONTAINER_ORIGINS = frozenset(
    {
        tuple,
        list,
        set,
        frozenset,
        dict,
        collections.abc.Mapping,
        collections.abc.MutableMapping,
        collections.abc.Sequence,
        collections.abc.Set,
        Union,
        types.UnionType,
    }
)


def _bare_float_sites(root: type[pydantic.BaseModel]) -> list[str]:
    """Every ``Model.field`` reachable from ``root`` that is a bare, non-finite
    float, or that the walk cannot rule out being one.

    Walks ``field.annotation`` together with ``field.metadata`` (where
    pydantic moves an outer ``Annotated``'s metadata), descending through a
    nested ``Annotated`` (collecting its own metadata, via
    :func:`_expand_metadata`), the type arguments — keys and values alike,
    ``Ellipsis`` and ``NoneType`` skipped — of a generic whose origin is in
    :data:`_CONTAINER_ORIGINS`, and nested pydantic models, each visited
    once. Does not descend into ``Literal`` values. Metadata from an outer
    container never carries down to its elements: each recursive call starts
    from either the fresh metadata of its own ``Annotated`` layer or none at
    all.

    Fails closed: a leaf that is not ``float`` and not one of
    :data:`_SAFE_LEAVES`, and a parameterised generic whose origin is not in
    :data:`_CONTAINER_ORIGINS`, are both reported — an unresolved ``TypeVar``,
    ``Any``, a bare ``dict`` or ``tuple``, a dataclass, a forward reference,
    or anything else the walk has no rule for — marked ``(cannot check:
    ...)`` so it reads as "unverifiable" rather than as a confirmed
    non-finite float. A caller that wants only the confirmed floats filters
    on the absence of ``"cannot check"``.

    Returns each site once, sorted.
    """
    sites: set[str] = set()
    visited_models: set[type[pydantic.BaseModel]] = set()

    def visit_model(model: type[pydantic.BaseModel]) -> None:
        if model in visited_models:
            return
        visited_models.add(model)
        for field_name, field in model.model_fields.items():
            walk(model.__name__, field_name, field.annotation, list(field.metadata))

    def walk(model_name: str, field_name: str, annotation: Any, metadata: list[Any]) -> None:
        site = f"{model_name}.{field_name}"

        if hasattr(annotation, "__metadata__"):
            walk(
                model_name,
                field_name,
                annotation.__origin__,
                _expand_metadata(annotation.__metadata__),
            )
            return

        if annotation is float:
            if not _finite_metadata(metadata):
                sites.add(site)
            return

        if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
            visit_model(annotation)
            return

        origin = get_origin(annotation)
        if origin is Literal:
            return

        args = get_args(annotation)
        if origin in _CONTAINER_ORIGINS and args:
            for arg in args:
                if arg is Ellipsis or arg is type(None):
                    continue
                walk(model_name, field_name, arg, [])
            return

        if annotation in _SAFE_LEAVES:
            return

        sites.add(f"{site} (cannot check: {annotation!r})")

    visit_model(root)
    return sorted(sites)


def test_every_float_in_the_scenario_schema_is_finite() -> None:
    """No field anywhere under :class:`Scenario` accepts a bare, non-finite float,
    and the walk can account for every leaf it visits.

    Every float either uses the ``Finite`` alias or carries
    ``allow_inf_nan=False`` alongside its own bounds — including the elements
    of a correlation row, an allocation, and a contribution-weight mapping,
    not just the tuple or mapping that holds them. If this ever lists a site
    marked "cannot check", the walk has met a leaf type or a generic origin
    the schema does not use today; extend :data:`_SAFE_LEAVES`,
    :data:`_CONTAINER_ORIGINS` or the walk, don't loosen this assertion.
    """
    sites = _bare_float_sites(Scenario)

    assert not sites, (
        "sites that accept a bare, non-finite float, or that the walk "
        f"cannot verify: {', '.join(sites)}"
    )


class _NestedSyntheticModel(pydantic.BaseModel):
    """Reached only through :class:`_SyntheticModel`'s ``nested`` field.

    Exists so the control test can tell a walker that visits only the top
    level from one that actually recurses into a nested model.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    inner: float


#: A parameterised ``type X[T] = ...`` alias, used as ``_SyntheticAlias[str]``
#: below. Its origin (``_SyntheticAlias`` itself) is not in
#: :data:`_CONTAINER_ORIGINS`, so the walk must not expand its ``__value__``
#: and report the field as unverifiable instead — the same shape a generic
#: pydantic dataclass or a generic ``TypedDict`` would present.
type _SyntheticAlias[T] = dict[T, float]


def test_the_walk_helper_finds_every_bare_float_in_a_synthetic_model() -> None:
    """A control model, so a walker that under-visits or fails open cannot pass.

    Obviously synthetic, never a real scenario type. Covers a nested model, a
    bare tuple, ``float | None``, and a mixed-type tuple — each a shape a
    walker that only checks the top level or only ``float``/``Annotated``/
    ``BaseModel`` would miss — a pinned ``FrozenMapping[Finite]`` that must
    *not* be reported, an unparameterized ``FrozenMapping``, an ``Any`` field,
    and a parameterised type alias that the walk cannot verify and must
    report as such, and an ``Annotated`` tuple whose own
    ``allow_inf_nan=False`` must not be mistaken for its element's: the pin
    belongs to the tuple, not the float inside it, so the element is still a
    bare float that must be reported.
    """

    class _SyntheticModel(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

        bare: float
        weights: FrozenMapping[float]
        rate: Finite
        nested: _NestedSyntheticModel
        tuple_of_floats: tuple[float, ...]
        optional_float: float | None
        mixed_tuple: tuple[int | float, ...]
        pinned_mapping: FrozenMapping[Finite]
        unbound_mapping: FrozenMapping
        anything: Any
        alias_field: _SyntheticAlias[str]
        outer_pinned: Annotated[tuple[float, ...], pydantic.Field(allow_inf_nan=False)]

    sites = _bare_float_sites(_SyntheticModel)
    confirmed = [site for site in sites if "cannot check" not in site]
    unverifiable = [site for site in sites if "cannot check" in site]

    assert confirmed == [
        "_NestedSyntheticModel.inner",
        "_SyntheticModel.bare",
        "_SyntheticModel.mixed_tuple",
        "_SyntheticModel.optional_float",
        "_SyntheticModel.outer_pinned",
        "_SyntheticModel.tuple_of_floats",
        "_SyntheticModel.weights",
    ]
    assert {site.split(" ", 1)[0] for site in unverifiable} == {
        "_SyntheticModel.alias_field",
        "_SyntheticModel.anything",
        "_SyntheticModel.unbound_mapping",
    }


# --- Risk aversion ------------------------------------------------------


def test_the_example_carries_its_risk_aversion(tmp_path: Path) -> None:
    """The expected value is read out of the file, not written here.

    Pinning the example's own coefficient in this test would make it a second
    place the preference is written down, and would fail on a change to the
    example this test has no opinion about.
    """
    expected = example_values()["risk_aversion"]
    assert expected is not None

    scenario = load_mutated(tmp_path, ())

    assert scenario.risk_aversion == expected


def test_a_scenario_with_no_risk_aversion_loads_as_none(tmp_path: Path) -> None:
    """Not just another ACCEPTANCES entry: those only assert that a file
    loads, so an absent preference could silently acquire a numeric default
    and nothing in the acceptance list would notice."""
    scenario = load_mutated(tmp_path, (deletes("risk_aversion"),))

    assert scenario.risk_aversion is None


# --- Attainability checked at load time, both entry points -----------------


def test_load_scenario_names_both_classes_and_not_a_bare_index(tmp_path: Path) -> None:
    """The -1-at-5%-vol case's message names both classes, in the wording a
    reader of the file understands, not the index-only wording
    ``engine.mc.returns.generate`` uses for direct construction."""
    with pytest.raises(InvalidScenarioError) as excinfo:
        load_mutated(
            tmp_path,
            (
                sets("assumptions.asset_classes.equity.vol", 0.05),
                sets("assumptions.correlation", [[1.0, -1.0], [-1.0, 1.0]]),
            ),
        )

    message = str(excinfo.value)
    assert "'equity'" in message
    assert "'bonds'" in message
    assert "annual_covariance[" not in message


def test_model_validate_refuses_the_same_case() -> None:
    """``Assumptions.model_validate`` and ``Scenario.model_validate`` are
    covered too, not only the ``load_scenario`` file path."""
    values = example_values()
    values["assumptions"]["asset_classes"]["equity"]["vol"] = 0.05
    values["assumptions"]["correlation"] = [[1.0, -1.0], [-1.0, 1.0]]

    with pytest.raises(pydantic.ValidationError) as assumptions_excinfo:
        Assumptions.model_validate(values["assumptions"])
    assert "asset classes 'equity' and 'bonds'" in str(assumptions_excinfo.value)

    with pytest.raises(pydantic.ValidationError) as scenario_excinfo:
        Scenario.model_validate(values)
    assert "asset classes 'equity' and 'bonds'" in str(scenario_excinfo.value)


# --- The addressing the grid uses ------------------------------------------


@pytest.fixture
def policy() -> PolicySpec:
    """The example's only policy."""
    return load_scenario(EXAMPLE).policies[0]


def test_a_path_reaches_a_field_and_a_mapping_key(policy: PolicySpec) -> None:
    """Model fields and mapping keys are both walked, in one path."""
    assert resolve_policy_path(policy, "elections.rrif_conversion.fraction") == 0.05
    assert resolve_policy_path(policy, "elections.cpp_start_age_years.a") == 65.0
    assert resolve_policy_path(policy, "contribution.weights.rrsp") == 0.5


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        pytest.param("elections.cpp_start_age_yrs.a", "cpp_start_age_yrs", id="misspelled-field"),
        pytest.param("elections.cpp_start_age_years.b", "'b'", id="unknown-key"),
        pytest.param("name.first", "'first'", id="walks-into-a-string"),
        pytest.param("withdrawal.fill_pension_credit", "bool", id="a-flag-is-not-a-number"),
        pytest.param("elections", "ElectionsSpec", id="a-block-is-not-a-number"),
    ],
)
def test_a_path_that_names_no_number_is_an_error(
    policy: PolicySpec, path: str, expected: str
) -> None:
    """Every way of missing says what was missed.

    A flag is called out separately from a misspelling because a grid over
    ``fill_pension_credit`` is a plausible thing to write and a silent
    ``True``/``False`` search expressed as ones and zeros is not what the
    author would have meant.
    """
    with pytest.raises(ValueError, match=expected):
        resolve_policy_path(policy, path)


def test_the_grid_in_the_example_resolves(policy: PolicySpec) -> None:
    """The committed example's own grid key is not a dead path."""
    scenario = load_scenario(EXAMPLE)

    for path in scenario.grid:
        assert resolve_policy_path(policy, path) == 65.0

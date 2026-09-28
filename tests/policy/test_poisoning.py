# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guard 2: ``CompositePolicy.decide`` reads only what a policy is allowed to see.

One household, two persons each holding an RRSP, a RRIF, an AB LIF, a TFSA, and a taxable
account, one eligible RESP beneficiary, five paths -- only ``context.cash_after_flows``
differs path to path, engineered so that, together, the five paths exercise: the bracket
fill; a discretionary RRSP draw crossing a withholding band; a LIF capped at its remaining
maximum; a clean weighted contribution placement; and a contribution spill. A control run on
the clean state confirms each of those five actually produced a non-zero transfer before any
poisoning happens -- a NaN result would otherwise pass every equality check vacuously.

Every reachable ``float64`` array in ``state`` and ``context`` is then replaced by NaN, except
the read-set the docstring of :func:`_exempt_state_paths`/:func:`_exempt_context_paths`
states -- and ``decide`` must return exactly the same transfers. A separate run poisons only
``death_month_index``, to the not-yet-drawn sentinel, on every (alive) path, and checks the
same equality.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from engine.core.build import build_draws, build_initial_state, build_market_inputs, draw_deaths
from engine.core.indexation import real_year
from engine.core.state import DEATH_NOT_DRAWN, updated
from engine.core.step import open_year
from engine.params.loader import load_year
from engine.policy.base import Decision
from engine.policy.build import build_policy

N_PATHS = 5
#: path index -> context.cash_after_flows, engineered for the five branches the module
#: docstring lists, in order.
_CASH_AFTER_FLOWS = (-3_000.0, -50_000.0, -5_000_000.0, 3_000.0, 80_000.0)


def _build_scenario(example_values, build_from_values):
    values = example_values()
    person_a = values["household"]["persons"][0]
    person_a["cpp"] = {"in_pay_monthly": 1000.0}
    person_a["oas"] = {"in_pay_monthly": 700.0}
    person_a.pop("employment", None)
    person_a.pop("db_pensions", None)
    # SYNTHETIC balances, sized only to make the five branches reachable at the cash levels
    # above: the RRIF/RRSP dwarf a year's bracket fill, the LIF's own maximum is comfortably
    # below its balance, and the TFSA's committed room (7000, from the example) is smaller
    # than an equal share of the large surplus path.
    person_a["accounts"] = {
        "cash": {"balance": 0},
        "rrsp": {"balance": 300000, "room": 20000},
        "rrif": {"balance": 150000},
        # A LIRA with a real jurisdiction, so its balance is a plain nonzero float the
        # poisoning sweep can carry a NaN in -- LiraState's own invariant (L3) refuses a
        # nonzero/NaN balance paired with jurisdiction "", which an un-held LIRA would be.
        "lira": {"balance": 25000, "jurisdiction": "ab"},
        "lif": {"balance": 60000, "jurisdiction": "ab"},
        "tfsa": {"balance": 90000, "room": 7000},
        "taxable": {"balance": 150000, "acb": 110000},
    }
    person_b = copy.deepcopy(person_a)
    person_b["id"] = "b"
    values["household"]["persons"].append(person_b)
    values["policies"][0]["contribution"] = {
        "weights": {"rrsp": 0.2, "tfsa": 0.3, "taxable": 0.3, "resp": 0.2},
        "spill_order": ["taxable"],
    }
    values["policies"][0]["withdrawal"] = {
        "order": ["rrsp", "rrif", "lif", "tfsa", "taxable"],
        "taxable_ceiling_bracket": 1,
        "fill_pension_credit": True,
    }
    values["policies"][0]["elections"]["cpp_start_age_years"] = {}
    values["policies"][0]["elections"]["oas_start_age_years"] = {}
    del values["grid"]

    return build_from_values(values, n_paths=N_PATHS)


@pytest.fixture
def rig(example_values, build_from_values):
    """``(state, context, real_params, policy)`` for the five-path household above.

    ``state`` has gone through one ``open_year`` (RRIF/LIF minimum and maximum fixed) and a
    real ``draw_deaths`` (every ``death_month_index`` a genuine, finite future fact, not the
    build-time sentinel) -- both persons alive on every path, asserted below.
    """
    scenario, _built_state = _build_scenario(example_values, build_from_values)
    market = build_market_inputs(scenario.assumptions)
    mortality = load_year(scenario.start_year)["mortality"]
    real_params = real_year(load_year(scenario.start_year), scenario.assumptions.inflation)
    draws = build_draws(scenario, market, n_paths=N_PATHS, mortality=mortality)

    state = build_initial_state(scenario, n_paths=N_PATHS)
    state = draw_deaths(state, draws, mortality)
    for person in state.persons:
        assert np.all(person.alive), "every path must stay alive for this test to mean anything"
        assert np.all(person.death_month_index != DEATH_NOT_DRAWN)
    state = open_year(state, real_params)

    context = _make_context(state)
    policy = build_policy(scenario.policies[0], scenario.household)
    return state, context, real_params, policy


def _make_context(state):
    from engine.core.context import AccountAmounts, ByKind, MonthContext, PersonInflows
    from engine.core.state import Assessment

    zeros = np.zeros(N_PATHS, dtype=np.float64)
    n_persons = len(state.persons)
    n_beneficiaries = len(state.beneficiaries)
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

    def _zero_assessment() -> Assessment:
        return Assessment(
            federal=np.zeros(N_PATHS, dtype=np.float64),
            provincial=np.zeros(N_PATHS, dtype=np.float64),
            oas_repayment=np.zeros(N_PATHS, dtype=np.float64),
            aip_penalty=np.zeros(N_PATHS, dtype=np.float64),
            total=np.zeros(N_PATHS, dtype=np.float64),
            net_income=np.zeros(N_PATHS, dtype=np.float64),
            net_income_after_repayment=np.zeros(N_PATHS, dtype=np.float64),
            taxable_income=np.zeros(N_PATHS, dtype=np.float64),
            transfer_in=np.zeros(N_PATHS, dtype=np.float64),
            transfer_out=np.zeros(N_PATHS, dtype=np.float64),
        )

    return MonthContext(
        month_index=0,
        year=2026,
        month=1,
        cash_opening=zeros,
        rolled_out=tuple(zero_account for _ in range(n_persons)),
        rolled_acb=tuple(zeros for _ in range(n_persons)),
        terminal_assessment=zeros,
        terminal_assessments=tuple(_zero_assessment() for _ in range(n_persons)),
        cash_to_estate=zeros,
        inflows=tuple(zero_inflow for _ in range(n_persons)),
        education_draws=tuple(zeros for _ in range(n_beneficiaries)),
        payroll_withholding=tuple(zeros for _ in range(n_persons)),
        spending=zeros,
        education_costs=tuple(zeros for _ in range(n_beneficiaries)),
        tax_settlement=tuple(zeros for _ in range(n_persons)),
        forced_withdrawals=tuple(zero_bykind for _ in range(n_persons)),
        forced_withholding=tuple(zeros for _ in range(n_persons)),
        cash_after_flows=np.array(_CASH_AFTER_FLOWS, dtype=np.float64),
    )


def test_the_clean_state_reaches_all_five_branches(rig):
    state, context, real_params, policy = rig
    decision = policy.decide(state, context, real_params)

    def amounts(from_kind=None, to_kind=None, person_index=None):
        return [
            t
            for t in decision.transfers
            if (from_kind is None or t.from_kind == from_kind)
            and (to_kind is None or t.to_kind == to_kind)
            and (person_index is None or t.person_index == person_index)
        ]

    # Path 0: the fill.
    rrif0 = amounts(from_kind="rrif", person_index=0)[0]
    assert rrif0.amount[0] > 0

    # Path 1: a discretionary RRSP draw comfortably past the top withholding band.
    top_edge = real_params.rrif.amounts("withholding.edges_each", 0)[-1]
    rrsp0 = amounts(from_kind="rrsp", person_index=0)[0]
    assert rrsp0.amount[1] > top_edge

    # The RESP weight (0.2) is always on, so the clean run must actually place something
    # in it -- otherwise the poisoning sweep below would pass on this branch vacuously.
    resp_transfers = amounts(to_kind="resp")
    assert resp_transfers, "the clean run must reach the RESP for the poisoning sweep to test it"
    assert np.any(resp_transfers[0].amount > 0)

    # Path 2: the LIF capped at its remaining maximum.
    lif0 = amounts(from_kind="lif", person_index=0)[0]
    np.testing.assert_allclose(lif0.amount[2], state.persons[0].lif.annual_maximum[2])

    # Path 3: a clean weighted placement (no cap binds).
    tfsa_contrib0 = amounts(to_kind="tfsa", person_index=0)[0]
    assert 0 < tfsa_contrib0.amount[3] < state.persons[0].tfsa.room[3]

    # Path 4: the TFSA caps at room and the excess spills to taxable.
    tfsa_contrib1 = amounts(to_kind="tfsa", person_index=1)[0]
    np.testing.assert_allclose(tfsa_contrib1.amount[4], state.persons[1].tfsa.room[4])
    taxable_contrib0 = amounts(to_kind="taxable", person_index=0)[0]
    assert taxable_contrib0.amount[4] > state.persons[0].tfsa.room[4]


# --- The read-set: exempt from poisoning. -----------------------------------------------

#: Dotted field paths (tuple element names dropped) exempt from the NaN sweep on ``state``.
_STATE_EXEMPT = frozenset(
    {
        ("persons", "alive"),
        ("persons", "rrsp", "balance"),
        ("persons", "rrsp", "room"),
        ("persons", "rrif", "balance"),
        ("persons", "rrif", "annual_minimum"),
        ("persons", "rrif", "withdrawn_ytd"),
        ("persons", "lif", "balance"),
        ("persons", "lif", "annual_minimum"),
        ("persons", "lif", "annual_maximum"),
        ("persons", "lif", "withdrawn_ytd"),
        ("persons", "tfsa", "balance"),
        ("persons", "tfsa", "room"),
        ("persons", "taxable", "balance"),
        ("persons", "income", "employment"),
        ("persons", "income", "cpp"),
        ("persons", "income", "oas"),
        ("persons", "income", "db_pension"),
        ("persons", "income", "rrsp_withdrawals"),
        ("persons", "income", "rrif_lif_withdrawals"),
        ("persons", "income", "interest"),
        ("persons", "income", "eligible_dividends"),
        ("persons", "income", "capital_gains"),
        ("persons", "income", "resp_accumulated_income"),
        ("persons", "income", "rrsp_deductions"),
        ("persons", "income", "cpp_enhanced_contributions"),
        ("beneficiaries", "resp", "contributions_lifetime"),
        ("beneficiaries", "resp", "wound_up"),
    }
)

#: Nothing on ``context`` is read except ``cash_after_flows``.
_CONTEXT_EXEMPT = frozenset({("cash_after_flows",)})


def _poison(obj, exempt: frozenset[tuple[str, ...]], path: tuple[str, ...] = ()):
    """A copy of ``obj`` with every reachable ``float64`` array replaced by NaN, except a
    field whose dotted path (tuple/list index dropped) is in ``exempt``.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        changes = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            changes[field.name] = _poison(value, exempt, (*path, field.name))
        return dataclasses.replace(obj, **changes)
    if isinstance(obj, tuple):
        return tuple(_poison(item, exempt, path) for item in obj)
    if isinstance(obj, np.ndarray) and obj.dtype == np.float64:
        if path in exempt:
            return obj
        return np.full_like(obj, np.nan)
    return obj


def _assert_same_decision(clean: Decision, poisoned: Decision) -> None:
    assert len(clean.transfers) == len(poisoned.transfers)
    for a, b in zip(clean.transfers, poisoned.transfers, strict=True):
        assert a.person_index == b.person_index
        assert a.from_kind == b.from_kind
        assert a.to_kind == b.to_kind
        assert np.array_equal(a.amount, b.amount), (a, b)


def test_poisoning_every_unread_float_array_leaves_the_decision_unchanged(rig):
    state, context, real_params, policy = rig
    clean = policy.decide(state, context, real_params)

    poisoned_state = _poison(state, _STATE_EXEMPT)
    poisoned_context = _poison(context, _CONTEXT_EXEMPT)
    poisoned = policy.decide(poisoned_state, poisoned_context, real_params)

    _assert_same_decision(clean, poisoned)


def test_poisoning_death_month_index_to_the_sentinel_leaves_the_decision_unchanged(rig):
    state, context, real_params, policy = rig
    clean = policy.decide(state, context, real_params)

    new_persons = []
    for person in state.persons:
        sentinel = np.where(person.alive, DEATH_NOT_DRAWN, person.death_month_index)
        new_persons.append(updated(person, death_month_index=sentinel.astype(np.int64)))
    poisoned_state = updated(state, persons=tuple(new_persons))

    poisoned = policy.decide(poisoned_state, context, real_params)

    _assert_same_decision(clean, poisoned)

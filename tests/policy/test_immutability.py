# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``CompositePolicy``, ``SplitContribution``, and ``OrderedWithdrawal`` are frozen, and
``decide`` leaves every field on all three equal to before.
"""

from __future__ import annotations

import copy
import dataclasses
import types

import numpy as np
import pytest

from engine.core.build import build_initial_state
from engine.policy.build import build_policy
from engine.policy.contribution import SplitContribution
from engine.policy.withdrawal import OrderedWithdrawal


@pytest.fixture
def deepcopy_supports_mappingproxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registers, for this test only, the one deep-copy rule ``copy.deepcopy`` is missing for
    ``types.MappingProxyType`` (``SplitContribution.weights``'s type -- see its
    ``__post_init__``): it is not picklable, which is what ``deepcopy`` falls back to for a
    type it does not otherwise recognise. Copies the underlying mapping and wraps it again,
    exactly as ``deepcopy`` already does for any read-only container type stdlib does know
    about.
    """
    monkeypatch.setitem(
        copy._deepcopy_dispatch,
        types.MappingProxyType,
        lambda x, memo: types.MappingProxyType(copy.deepcopy(dict(x), memo)),
    )


@pytest.mark.parametrize(
    "instance",
    [
        SplitContribution(weights={"rrsp": 1.0}, spill_order=("tfsa",)),
        OrderedWithdrawal(order=("rrsp",), taxable_ceiling_bracket=1),
    ],
    ids=["SplitContribution", "OrderedWithdrawal"],
)
def test_components_are_frozen(instance) -> None:
    field_name = next(iter(dataclasses.fields(instance))).name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field_name, getattr(instance, field_name))


def test_composite_policy_is_frozen(scenario) -> None:
    policy = build_policy(scenario.policies[0], scenario.household)
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.spec = policy.spec  # type: ignore[misc]


def test_decide_leaves_every_field_equal_to_before(
    couple_scenario, couple_real_params, make_context, deepcopy_supports_mappingproxy
) -> None:
    policy = build_policy(couple_scenario.policies[0], couple_scenario.household)
    # A real (deep) copy taken before `decide` runs, not a shallow `dataclasses.replace`
    # that would still share every nested object with `policy` -- a mutation `decide` made
    # in place on a shared object would otherwise go unnoticed.
    before = copy.deepcopy(policy)

    state = build_initial_state(couple_scenario, n_paths=1)
    context = make_context(
        n_paths=1,
        n_persons=2,
        year=2026,
        month=1,
        month_index=0,
        cash_after_flows=np.array([-1000.0]),
    )
    policy.decide(state, context, couple_real_params)

    for field in dataclasses.fields(policy):
        current = getattr(policy, field.name)
        expected = getattr(before, field.name)
        if isinstance(current, np.ndarray):
            assert np.array_equal(current, expected), field.name
        else:
            assert current == expected, field.name

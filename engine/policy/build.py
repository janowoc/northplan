# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Turning a :class:`~engine.scenario.schema.PolicySpec` into a runnable policy, and a grid
of one policy's values into a scenario's worth of named policies.

:class:`CompositePolicy` is the one :class:`~engine.policy.base.Policy` implementation the
engine ships: a :class:`~engine.policy.withdrawal.OrderedWithdrawal` and a
:class:`~engine.policy.contribution.SplitContribution`, called in that order every month --
the withdrawal's net proceeds top up ``context.cash_after_flows`` into the contribution's
budget, so a household drawing down and topping up a TFSA in the same month is not a second
loop over time, only two calls inside one.

:func:`expand_grid` is not a Cartesian-product search over months or paths -- a search is
tens to a hundred *policies*, evaluated with common random numbers, never a per-step choice
(decision 18, ``docs/limitations.md`` L44). It is the one place a :class:`Scenario` gets
built twice: once as written, to find the grid, and once again, expanded, so that every
schema validator -- the weight sum, the CPP/OAS election checks, the grid resolution check
itself -- runs again on what the optimizer will actually evaluate.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel

from engine.core.build import build_elections
from engine.core.context import MonthContext
from engine.core.indexation import RealParamYear
from engine.core.state import Elections, HouseholdState
from engine.policy.base import Decision
from engine.policy.contribution import SplitContribution
from engine.policy.withdrawal import OrderedWithdrawal
from engine.scenario import Household, PolicySpec, Scenario

__all__ = ["CompositePolicy", "build_policy", "expand_grid"]


@dataclass(frozen=True, slots=True)
class CompositePolicy:
    """The one :class:`~engine.policy.base.Policy` this engine ships: a withdrawal component
    and a contribution component, called in that order every month.

    Nothing on this object changes between months: ``decide`` reads only its three arguments
    and these constant fields.

    Attributes:
        contribution: Where money goes.
        withdrawal: Where money comes from.
        chosen_elections: The elections this policy reports back, named to avoid shadowing
            the :meth:`elections` method the :class:`~engine.policy.base.Policy` protocol
            requires.
        spec: The scenario's own description of this policy, the source
            :meth:`free_parameters` walks.
    """

    contribution: SplitContribution
    withdrawal: OrderedWithdrawal
    chosen_elections: Elections
    spec: PolicySpec

    def decide(
        self,
        state: HouseholdState,
        context: MonthContext,
        real_params: RealParamYear,
    ) -> Decision:
        """Withdraw, then contribute the proceeds plus whatever cash is left over.

        The withdrawal's net proceeds are added to ``context.cash_after_flows`` to form the
        contribution's budget, floored at zero -- a deficit month leaves nothing to
        contribute, it does not borrow from the withdrawal.
        """
        withdrawal_transfers, net = self.withdrawal.transfers(state, context, real_params)
        budget = np.clip(context.cash_after_flows + net, 0, None)
        contribution_transfers = self.contribution.transfers(state, context, real_params, budget)
        return Decision(transfers=withdrawal_transfers + contribution_transfers)

    def elections(self) -> Elections:
        """The elections this policy was built with."""
        return self.chosen_elections

    def withdrawal_order(self) -> tuple[str, ...]:
        """The withdrawal component's account order."""
        return self.withdrawal.order

    def free_parameters(self) -> dict[str, float]:
        """Every numeric leaf of ``spec``, keyed by the dotted path
        :func:`~engine.scenario.schema.resolve_policy_path` accepts.

        A bool or ``None`` leaf (``fill_pension_credit``, a null
        ``taxable_ceiling_bracket``) is not a number the optimizer can search, so neither is
        collected here -- consistent with what a grid key is allowed to name.
        """
        return _numeric_leaves(self.spec)


def build_policy(spec: PolicySpec, household: Household) -> CompositePolicy:
    """Build the one runnable policy a :class:`~engine.scenario.schema.PolicySpec` describes.

    Args:
        spec: The policy to build.
        household: Supplies the persons :func:`~engine.core.build.build_elections` reads.

    Returns:
        A :class:`CompositePolicy` ready for :func:`engine.mc.simulate.run`.
    """
    contribution = SplitContribution(
        weights=spec.contribution.weights, spill_order=spec.contribution.spill_order
    )
    withdrawal = OrderedWithdrawal(
        order=spec.withdrawal.order, taxable_ceiling_bracket=spec.withdrawal.taxable_ceiling_bracket
    )
    elections = build_elections(household, spec)
    return CompositePolicy(
        contribution=contribution, withdrawal=withdrawal, chosen_elections=elections, spec=spec
    )


def _numeric_leaves(model: BaseModel) -> dict[str, float]:
    """Every ``int``/``float`` leaf reachable from ``model``, keyed by dotted path.

    Walks the same shapes :func:`~engine.scenario.schema.resolve_policy_path` walks --
    pydantic model fields and mapping keys -- collecting every leaf that is a number and not
    a ``bool`` or ``None``. A ``str`` or a ``tuple`` of strings (an order, a spill order) is
    not a mapping, a model, or a number, and is silently skipped, same as a path into one
    would be refused by ``resolve_policy_path``.
    """
    leaves: dict[str, float] = {}

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, BaseModel):
            for name in type(value).model_fields:
                child_prefix = f"{prefix}.{name}" if prefix else name
                walk(getattr(value, name), child_prefix)
        elif isinstance(value, Mapping):
            for key, sub in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                walk(sub, child_prefix)
        elif isinstance(value, bool) or value is None:
            return
        elif isinstance(value, (int, float)):
            leaves[prefix] = float(value)

    walk(model, "")
    return leaves


def _to_plain(value: Any) -> Any:
    """Recursively convert every ``Mapping`` to a ``dict`` and every ``tuple`` to a ``list``.

    Args:
        value: A tree of the shapes ``model_dump()`` produces -- ``dict``, ``list``,
            ``MappingProxyType``, ``tuple``, and scalars.

    Returns:
        The same tree, with every ``Mapping`` and ``tuple`` replaced by a plain ``dict`` or
        ``list``, both handled by ``copy.deepcopy`` and by index/key assignment.
    """
    if isinstance(value, Mapping):
        return {key: _to_plain(sub) for key, sub in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _set_dotted(container: Any, path: str, value: Any) -> None:
    """Set the value at dotted ``path`` in a nested ``dict``/``list`` tree, in place.

    ``container`` is the shape :func:`_to_plain` produces from ``PolicySpec.model_dump()``: a
    model becomes a ``dict`` keyed by field name, a mapping becomes a ``dict`` too. ``path``
    is already known to resolve on this exact policy -- ``Scenario._check_grid_resolves``
    walked it with :func:`~engine.scenario.schema.resolve_policy_path` at load time -- so
    every segment but the last is a ``dict`` key lookup.
    """
    segments = path.split(".")
    cursor = container
    for segment in segments[:-1]:
        cursor = cursor[segment]
    cursor[segments[-1]] = value


def expand_grid(scenario: Scenario) -> Scenario:
    """Expand ``scenario.grid`` into one named policy per combination, grid emptied.

    For each policy in file order, for each combination of the grid's values -- Cartesian
    product, grid keys in mapping order, values in list order -- substitutes the values at
    their dotted paths and names the result ``f"{name}[{path}={value}, ...]"`` (keys in grid
    order, values printed as written: ``60``, not ``60.0``). Returns
    ``Scenario.model_validate(...)`` of the scenario with those policies and an empty grid,
    so every schema validator runs again on what the optimizer will actually evaluate. The
    statutory guards (:func:`~engine.scenario.start_ages.check_start_ages`,
    :func:`~engine.scenario.lifespan.check_lifespan`) are not schema validators and do not run
    here; :func:`engine.mc.prepare.prepare_run` runs them on the expansion.

    An empty grid returns a scenario equal to the input, unchanged.

    Args:
        scenario: The scenario as written, possibly carrying a grid.

    Returns:
        The expanded scenario, with ``grid`` empty.
    """
    if not scenario.grid:
        return scenario

    keys = tuple(scenario.grid.keys())
    value_lists = [scenario.grid[key] for key in keys]

    scenario_dict = _to_plain(scenario.model_dump(warnings=False))
    new_policy_dicts: list[dict[str, Any]] = []
    for policy, policy_dict in zip(scenario.policies, scenario_dict["policies"], strict=True):
        for combo in itertools.product(*value_lists):
            candidate = copy.deepcopy(policy_dict)
            labels = []
            for path, value in zip(keys, combo, strict=True):
                _set_dotted(candidate, path, value)
                labels.append(f"{path}={value}")
            candidate["name"] = f"{policy.name}[{', '.join(labels)}]"
            new_policy_dicts.append(candidate)

    scenario_dict["policies"] = new_policy_dicts
    scenario_dict["grid"] = {}
    return Scenario.model_validate(scenario_dict)

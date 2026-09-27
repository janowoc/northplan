# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The accumulation component: split a contribution budget across accounts, by fixed weights.

:class:`SplitContribution` is not a :class:`~engine.policy.base.Policy` on its own -- it has
no ``elections``, no ``withdrawal_order()``, and no ``free_parameters()``. It is the
contribution half a :class:`~engine.policy.build.CompositePolicy` combines with
:class:`~engine.policy.withdrawal.OrderedWithdrawal`.

Weights are the optimizer's free variables. Each kind's share of the month's budget is split
equally across the persons or beneficiaries eligible for it; whatever a kind's share cannot
place -- no room, no eligible beneficiary, nobody alive -- pours into ``spill_order``, kind by
kind, from the top. What the pool cannot place there stays in cash.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from engine.core.context import MonthContext
from engine.core.indexation import RealParamYear
from engine.core.state import BeneficiaryState, HouseholdState
from engine.core.timeline import age_at_end_of_year
from engine.policy.base import Transfer

#: Account kinds one person holds, split equally across the persons alive on a path.
_PERSON_KINDS = ("rrsp", "tfsa", "taxable")


@dataclass(frozen=True, slots=True)
class SplitContribution:
    """Split a monthly contribution budget across accounts by fixed weights.

    Attributes:
        weights: Share of the budget directed to each account kind
            (``engine.scenario.CONTRIBUTION_KINDS``); a kind not named has weight zero.
        spill_order: Where the pool of unplaceable contribution goes, tried in order, each
            kind at most once.
    """

    weights: Mapping[str, float]
    spill_order: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))

    def transfers(
        self,
        state: HouseholdState,
        context: MonthContext,
        real_params: RealParamYear,
        budget: NDArray[np.float64],
    ) -> tuple[Transfer, ...]:
        """This month's contribution transfers.

        Args:
            state: State after phase 6 of the current month.
            context: This month's context; only the month position is read.
            real_params: Parameters for the current tax year.
            budget: Amount available to contribute this month, ``(n_paths,)``, floored at
                zero -- ``engine.policy.build.CompositePolicy`` computes it from the
                withdrawal's net proceeds and ``context.cash_after_flows``.

        Returns:
            One transfer per (person, kind) and per eligible beneficiary whose weighted
            placement plus spill is non-zero on some path.
        """
        persons = state.persons
        n_persons = len(persons)
        beneficiaries = state.beneficiaries
        n_paths = np.asarray(budget, dtype=np.float64).shape[0]
        zeros = np.zeros(n_paths, dtype=np.float64)
        budget_arr = np.asarray(budget, dtype=np.float64)

        january_month_index = context.month_index - (context.month - 1)

        alive = [person.alive for person in persons]
        eligible_beneficiary = [
            _eligible(beneficiary, context, real_params) for beneficiary in beneficiaries
        ]

        alloc: dict[str, list[NDArray[np.float64]]] = {
            kind: [zeros.copy() for _ in persons] for kind in _PERSON_KINDS
        }
        resp_alloc = [zeros.copy() for _ in beneficiaries]

        pool = zeros.copy()

        # --- Weighted placement, per kind. ---------------------------------
        for kind in _PERSON_KINDS:
            weight = self.weights.get(kind, 0.0)
            share = budget_arr * weight
            alive_count = sum((mask.astype(np.float64) for mask in alive), zeros.copy())
            safe_count = np.where(alive_count > 0, alive_count, 1.0)
            each = np.where(alive_count > 0, share / safe_count, 0.0)
            placed_total = zeros.copy()
            for i, person in enumerate(persons):
                wanted = np.where(alive[i], each, 0.0)
                if kind == "taxable":
                    placed = wanted
                else:
                    room = getattr(person, kind).room
                    placed = np.minimum(wanted, np.clip(room, 0, None))
                alloc[kind][i] = alloc[kind][i] + placed
                placed_total = placed_total + placed
            pool = pool + (share - placed_total)

        resp_weight = self.weights.get("resp", 0.0)
        resp_share = budget_arr * resp_weight
        n_eligible = sum((mask.astype(np.float64) for mask in eligible_beneficiary), zeros.copy())
        safe_n_eligible = np.where(n_eligible > 0, n_eligible, 1.0)
        each_resp = np.where(n_eligible > 0, resp_share / safe_n_eligible, 0.0)
        resp_placed_total = zeros.copy()
        lifetime_cap = real_params.resp.annual_amount(
            "contributions.maximum_lifetime", january_month_index
        )
        lifetime_caps = [lifetime_cap for _ in beneficiaries]
        for b, beneficiary in enumerate(beneficiaries):
            wanted = np.where(eligible_beneficiary[b], each_resp, 0.0)
            cap_left = np.clip(lifetime_caps[b] - beneficiary.resp.contributions_lifetime, 0, None)
            placed = np.minimum(wanted, cap_left)
            resp_alloc[b] = resp_alloc[b] + placed
            resp_placed_total = resp_placed_total + placed
        pool = pool + (resp_share - resp_placed_total)

        # --- Spill: pour the pool through spill_order, from the top. -------
        room_left = {
            kind: [
                (
                    np.full(n_paths, np.inf)
                    if kind == "taxable"
                    else np.clip(getattr(person, kind).room - alloc[kind][i], 0, None)
                )
                for i, person in enumerate(persons)
            ]
            for kind in _PERSON_KINDS
        }
        resp_room_left = [
            np.where(
                eligible_beneficiary[b],
                np.clip(
                    lifetime_caps[b] - beneficiary.resp.contributions_lifetime - resp_alloc[b],
                    0,
                    None,
                ),
                0.0,
            )
            for b, beneficiary in enumerate(beneficiaries)
        ]

        for kind in self.spill_order:
            if kind in _PERSON_KINDS:
                for i, _person in enumerate(persons):
                    take = np.where(alive[i], np.minimum(pool, room_left[kind][i]), 0.0)
                    alloc[kind][i] = alloc[kind][i] + take
                    room_left[kind][i] = room_left[kind][i] - take
                    pool = pool - take
            elif kind == "resp":
                # Fill in rounds rather than one offer: a beneficiary capped by its own
                # room_left must not keep the part it refused out of circulation -- the
                # room it did not take is re-offered, equally, to whichever beneficiaries
                # still have room left. At most one round per beneficiary is ever needed,
                # since each round either exhausts the pool or removes at least one
                # beneficiary from contention for good.
                for _round in range(len(beneficiaries)):
                    active = [
                        eligible_beneficiary[b] & (resp_room_left[b] > 0)
                        for b in range(len(beneficiaries))
                    ]
                    n_active = sum((mask.astype(np.float64) for mask in active), zeros.copy())
                    if not np.any(n_active > 0):
                        break
                    safe_n_active = np.where(n_active > 0, n_active, 1.0)
                    capacity = zeros.copy()
                    for b, room in enumerate(resp_room_left):
                        capacity = capacity + np.where(active[b], room, 0.0)
                    offer_total = np.minimum(pool, capacity)
                    each_offer = np.where(n_active > 0, offer_total / safe_n_active, 0.0)
                    consumed = zeros.copy()
                    for b, _beneficiary in enumerate(beneficiaries):
                        wanted = np.where(active[b], each_offer, 0.0)
                        placed = np.minimum(wanted, resp_room_left[b])
                        resp_alloc[b] = resp_alloc[b] + placed
                        resp_room_left[b] = resp_room_left[b] - placed
                        consumed = consumed + placed
                    pool = pool - consumed

        transfers_out: list[Transfer] = []
        for kind in _PERSON_KINDS:
            for i in range(n_persons):
                amount = alloc[kind][i]
                if np.any(amount != 0):
                    transfers_out.append(
                        Transfer(person_index=i, from_kind="cash", to_kind=kind, amount=amount)
                    )
        for b in range(len(beneficiaries)):
            amount = resp_alloc[b]
            if np.any(amount != 0):
                transfers_out.append(
                    Transfer(person_index=b, from_kind="cash", to_kind="resp", amount=amount)
                )

        return tuple(transfers_out)


def _eligible(
    beneficiary: BeneficiaryState, context: MonthContext, real_params: RealParamYear
) -> NDArray[np.bool_]:
    """Whether ``beneficiary`` may receive a contribution this month, ``(n_paths,)``.

    All three of: age at or below ``real_params.resp.number("grant.cessation_age_years")``
    at the end of ``context.year``; ``context.month_index`` still inside the education
    window (before ``education_start_month_index + education_months``); and not wound up,
    per path.
    """
    resp_state = beneficiary.resp
    cessation_age = real_params.resp.number("grant.cessation_age_years")
    age_end = age_at_end_of_year(beneficiary.birth_year, beneficiary.birth_month, context.year)
    within_age = age_end <= cessation_age
    within_window = context.month_index < (
        resp_state.education_start_month_index + resp_state.education_months
    )
    if within_age and within_window:
        return ~resp_state.wound_up
    return np.zeros_like(resp_state.wound_up)

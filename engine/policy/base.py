# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The policy interface and the information a policy is allowed to see."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from engine.core.context import MonthContext
from engine.core.indexation import RealParamYear
from engine.core.state import Elections, HouseholdState


@dataclass(frozen=True, slots=True)
class Transfer:
    """One movement of money between household cash and one other account, this month.

    Exactly one of :attr:`from_kind`/:attr:`to_kind` is ``"cash"`` — the other names
    the account the money moves to or from. :attr:`person_index` names whose account
    that is, indexing ``HouseholdState.persons``, **except** when the non-cash side is
    ``"resp"``: an RESP is a beneficiary's account, not a person's, so
    ``person_index`` is then an index into ``HouseholdState.beneficiaries`` instead.

    Attributes:
        person_index: Index into ``HouseholdState.persons``, or into
            ``HouseholdState.beneficiaries`` when the non-cash side is ``"resp"``.
        from_kind: Where the money comes from: ``"cash"``, or an account kind in
            ``engine.scenario.WITHDRAWAL_KINDS``.
        to_kind: Where the money goes: ``"cash"``, or an account kind in
            ``engine.scenario.CONTRIBUTION_KINDS``.
        amount: Requested amount this month, real dollars, ``(n_paths,)``. The step
            applies mandatory minimums, room, and the cash available on top of this;
            it is not guaranteed to move in full.
    """

    person_index: int
    from_kind: str
    to_kind: str
    amount: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Decision:
    """What a policy decided for one **month**, for all paths.

    Attributes:
        transfers: This month's transfers, in the order the step should apply
            them — withdrawals before contributions regardless of the order they
            appear here (:func:`engine.core.step.advance_month`'s phase 8), but the
            relative order *within* each of those two groups is exactly the order
            given. Mandatory minimums and the cash floor are the step's, not the
            policy's: this is only what the policy chose on top of them.
    """

    transfers: tuple[Transfer, ...]


class Policy(Protocol):
    """A decision rule the optimizer can evaluate.

    Implementations are pure and stateless: everything they need arrives in
    ``decide``. Anything a policy needs to remember from an earlier month is
    already in ``HouseholdState``: year-to-date income, room consumed,
    minimums still outstanding.
    """

    def decide(
        self,
        state: HouseholdState,
        context: MonthContext,
        real_params: RealParamYear,
    ) -> Decision:
        """Choose this month's transfers.

        ``state`` is the state as it stands after phase 6 of this month
        (January's ``open_year`` applied; this month's receipts, withholding,
        outflows and forced withdrawals already in cash and in the year-to-date
        ledger), so ``state.cash.balance`` equals ``context.cash_after_flows``;
        year to date is knowable, the year's total is not.

        ``state.persons[i].death_month_index`` is a future fact from month zero
        onward. Only ``alive``, at the current month, is knowable to a policy. No
        policy may read ``death_month_index``.

        Args:
            state: State after phase 6 of the current month, carrying ``year`` and
                ``month``.
            context: This month's inflows and required outflows, already applied.
            real_params: Parameters for the current tax year, in the scenario's
                real-dollar view.

        Returns:
            The month's :class:`Decision`.
        """
        ...

    def elections(self) -> Elections:
        """The dated choices this policy makes.

        Read once, by :func:`engine.mc.simulate.run`, only to assert it equals
        ``state.elections`` — ``engine.core.build.build_initial_state`` is the only
        writer of :class:`~engine.core.state.Elections`, so a policy handed a state
        it did not build the elections for is a caller bug, caught here rather than
        producing a silently wrong run.
        """
        ...

    def withdrawal_order(self) -> tuple[str, ...]:
        """Account kinds, from ``engine.scenario.WITHDRAWAL_KINDS``, each at most once.

        The order phase 9's cash floor draws non-RESP accounts in, kind by kind
        across every person before moving to the next kind. Constant for the run.
        """
        ...

    def free_parameters(self) -> dict[str, float]:
        """The numbers the optimizer is searching over, by name.

        Returns:
            Parameter name to current value. The optimizer varies these and
            rebuilds the policy; it never mutates a policy in place.
        """
        ...

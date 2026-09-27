# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The decumulation component: withdrawal order and a bracket-edge fill.

:class:`OrderedWithdrawal` is not a :class:`~engine.policy.base.Policy` on its own -- it has
no ``elections``, no ``withdrawal_order()``, and no ``free_parameters()``. It is the
withdrawal half a :class:`~engine.policy.build.CompositePolicy` combines with
:class:`~engine.policy.contribution.SplitContribution`. Benefit start ages and the RRIF
conversion election live on ``engine.core.state.Elections``, not here.

**The bracket-filling trap.** "Withdraw up to the top of a bracket" is an annual instruction;
filling to the ceiling every month withdraws roughly twelve times the intended amount. Nor is
year to date alone enough: once it alone reaches the edge, the income the rest of the year
still brings pushes the year's total past it. The room is therefore measured against
year-to-date net income *plus a projection of the rest of the year*, and what remains under
the ceiling then spread evenly over the months left -- never against the month alone and
never against year to date alone -- which is why :func:`fill_to_bracket` takes both the year
to date and a projection of what is still to come (L44).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.core.context import MonthContext
from engine.core.indexation import RealParamYear
from engine.core.state import HouseholdState, PersonState
from engine.policy.base import Transfer
from engine.tax import federal as federal_mod
from engine.tax.brackets import room_below_edge
from engine.tax.withholding import registered_withholding

#: Together, the two tuples below are every kind :meth:`OrderedWithdrawal.transfers` may draw
#: -- ``engine.scenario.WITHDRAWAL_KINDS``, split by whether withholding applies, and restated
#: here rather than imported so this module never has to import ``engine.scenario`` for one
#: constant. The step validates every transfer against the real ``WITHDRAWAL_KINDS`` regardless.
#: Registered accounts: withholding on the gross (``rrsp``) or on the excess above the
#: withholding-free floor ``m`` (``rrif``, ``lif``).
_REGISTERED_KINDS = ("rrsp", "rrif", "lif")
#: Untaxed at withdrawal: no withholding, so ``net == gross``.
_UNTAXED_KINDS = ("tfsa", "taxable")


@dataclass(frozen=True, slots=True)
class OrderedWithdrawal:
    """Draw accounts in a fixed order, up to a taxable-income ceiling for the year.

    Two things happen every month, independently:

    - **The bracket fill** (skipped entirely when ``taxable_ceiling_bracket`` is ``None``):
      draws RRIF then RRSP, every month, whatever the household's cash need, toward a federal
      bracket edge measured on year-to-date net income plus a projection of the rest of the
      year (no pension split, no OAS repayment -- neither is knowable before December),
      spreading what remains under the edge evenly over the months left in the year.
    - **The need**: whatever of this month's cash shortfall the fill did not already cover is
      met by walking ``order`` account by account, each drawn up to what :func:`_gross_for_net`
      says is required to raise the shortfall in net cash, capped at what is left in the
      account (and, for a LIF, at what is left of the year's maximum).

    A kind omitted from ``order`` is never drawn for the need (the cash floor, not this
    class, reaches it); RRIF and RRSP are the one exception, reached by the fill regardless
    of ``order``.

    Attributes:
        order: Account types in withdrawal priority order, from
            ``engine.scenario.WITHDRAWAL_KINDS``, each at most once.
        taxable_ceiling_bracket: Index into the federal bracket edges to fill year-to-date net
            income up to, every month, or ``None`` to turn the fill off entirely.
    """

    order: tuple[str, ...]
    taxable_ceiling_bracket: int | None

    def transfers(
        self,
        state: HouseholdState,
        context: MonthContext,
        real_params: RealParamYear,
    ) -> tuple[tuple[Transfer, ...], NDArray[np.float64]]:
        """This month's withdrawal transfers, and the net cash they raise.

        Args:
            state: State after phase 6 of the current month.
            context: This month's context; only ``cash_after_flows`` and the month position
                are read.
            real_params: Parameters for the current tax year.

        Returns:
            ``(transfers, net)``: the withdrawal transfers to emit, in kind-major,
            person-minor order (RRIF then RRSP first when the fill is on, then ``order`` with
            those two removed), and the total net cash they raise, ``(n_paths,)`` -- gross
            less the registered withholding phase 8 will apply to each.
        """
        persons = state.persons
        n_persons = len(persons)
        n_paths = context.cash_after_flows.shape[0]
        zeros = np.zeros(n_paths, dtype=np.float64)

        january_month_index = context.month_index - (context.month - 1)
        months_remaining = 13 - context.month

        rrif_params = real_params.rrif
        edges = rrif_params.amounts("withholding.edges_each", context.month_index)
        rates = rrif_params.numbers("withholding.rates")

        def m_of(kind: str, person) -> NDArray[np.float64]:
            if kind == "rrsp":
                return zeros
            account = getattr(person, kind)
            return np.clip(account.annual_minimum - account.withdrawn_ytd, 0, None)

        def net_of(gross: ArrayLike, m: ArrayLike) -> NDArray[np.float64]:
            gross_arr = np.asarray(gross, dtype=np.float64)
            above = np.clip(gross_arr - np.asarray(m, dtype=np.float64), 0, None)
            return gross_arr - registered_withholding(above, rrif_params, context.month_index)

        m_by_person = [
            {kind: m_of(kind, person) for kind in _REGISTERED_KINDS} for person in persons
        ]
        g0: list[dict[str, NDArray[np.float64]]] = [
            {kind: zeros.copy() for kind in (*_REGISTERED_KINDS, *_UNTAXED_KINDS)} for _ in persons
        ]

        # Step W1: the bracket fill, every month, whatever the need.
        if self.taxable_ceiling_bracket is not None:
            bracket_edges = real_params.federal.annual_amounts(
                "brackets.edges_annual", january_month_index
            )
            for i, person in enumerate(persons):
                ytd = federal_mod.net_income(person.income, real_params.federal, 0.0, 0.0)
                projected = _projected_income_rest_of_year(ytd, person, context.month)
                available = person.rrif.balance + person.rrsp.balance
                fill = fill_to_bracket(
                    ytd,
                    bracket_edges,
                    self.taxable_ceiling_bracket,
                    available,
                    months_remaining,
                    projected_income_rest_of_year=projected,
                )
                rrif_fill = np.minimum(fill, person.rrif.balance)
                g0[i]["rrif"] = rrif_fill
                g0[i]["rrsp"] = fill - rrif_fill

        # Step W2: the need.
        need = np.clip(-context.cash_after_flows, 0, None)
        w1_net_total = zeros.copy()
        for i in range(n_persons):
            w1_net_total = w1_net_total + net_of(g0[i]["rrif"], m_by_person[i]["rrif"])
            w1_net_total = w1_net_total + net_of(g0[i]["rrsp"], m_by_person[i]["rrsp"])
        remaining = np.clip(need - w1_net_total, 0, None)

        g1 = [dict(person_g0) for person_g0 in g0]

        for kind in self.order:
            for i, person in enumerate(persons):
                g0_ik = g0[i][kind]
                if kind in _UNTAXED_KINDS:
                    account = getattr(person, kind)
                    avail = np.clip(account.balance - g0_ik, 0, None)
                    take = np.minimum(avail, remaining)
                    g1_ik = g0_ik + take
                    delta_net = take
                else:
                    m_ik = m_by_person[i][kind]
                    account = getattr(person, kind)
                    if kind == "lif":
                        max_remaining = np.clip(
                            person.lif.annual_maximum - person.lif.withdrawn_ytd, 0, None
                        )
                        avail = np.clip(np.minimum(account.balance, max_remaining) - g0_ik, 0, None)
                    else:
                        avail = np.clip(account.balance - g0_ik, 0, None)
                    net_g0 = net_of(g0_ik, m_ik)
                    target = net_g0 + remaining
                    cap = g0_ik + avail
                    candidate = np.minimum(cap, _gross_for_net(target, g0_ik, m_ik, edges, rates))
                    net_candidate = net_of(candidate, m_ik)
                    g1_ik = np.where(net_candidate < net_g0, g0_ik, candidate)
                    delta_net = net_of(g1_ik, m_ik) - net_g0
                remaining = np.clip(remaining - delta_net, 0, None)
                g1[i][kind] = g1_ik

        emitted: set[tuple[int, str]] = set()
        transfers_out: list[Transfer] = []

        def emit(kind: str, i: int) -> None:
            key = (i, kind)
            if key in emitted:
                return
            emitted.add(key)
            amount = g1[i][kind]
            if np.any(amount != 0):
                transfers_out.append(
                    Transfer(person_index=i, from_kind=kind, to_kind="cash", amount=amount)
                )

        kind_order: list[str] = []
        if self.taxable_ceiling_bracket is not None:
            kind_order.extend(("rrif", "rrsp"))
        kind_order.extend(
            kind
            for kind in self.order
            if not (self.taxable_ceiling_bracket is not None and kind in ("rrif", "rrsp"))
        )

        for kind in kind_order:
            for i in range(n_persons):
                emit(kind, i)

        total_net = zeros.copy()
        for i in range(n_persons):
            for kind in _REGISTERED_KINDS:
                total_net = total_net + net_of(g1[i][kind], m_by_person[i][kind])
            for kind in _UNTAXED_KINDS:
                total_net = total_net + g1[i][kind]

        return tuple(transfers_out), total_net


def fill_to_bracket(
    taxable_income_ytd: ArrayLike,
    bracket_edges: tuple[float, ...],
    ceiling_index: int,
    available: ArrayLike,
    months_remaining_in_year: int,
    *,
    projected_income_rest_of_year: ArrayLike,
) -> NDArray[np.float64]:
    """Amount to withdraw **this month** toward filling the year to a bracket edge.

    The room below the edge is divided by the months remaining, so twelve calls over a year
    fill it once rather than twelve times.

    Args:
        taxable_income_ytd: Net income recognised so far this calendar year, before this
            withdrawal, ``(n_paths,)`` -- year to date, not a projection of the year's total.
        bracket_edges: Current year's bracket edges, from ``params/``.
        ceiling_index: Which edge to fill to.
        available: Balance available to withdraw this month, ``(n_paths,)``.
        months_remaining_in_year: Months left including this one, from twelve in January
            down to one in December.
        projected_income_rest_of_year: Non-negative income expected to be recognised in the
            months after this one, before any further fill, ``(n_paths,)``. Added to
            ``taxable_income_ytd`` before the room below the edge is measured.

    Returns:
        This month's withdrawal amount, non-negative and capped by ``available``. Zero where
        year-to-date income plus the projection already reaches or exceeds the ceiling.

    Raises:
        IndexError: If ``ceiling_index`` is not a valid, non-negative index into
            ``bracket_edges``.
    """
    projected_total = np.asarray(taxable_income_ytd, dtype=np.float64) + np.asarray(
        projected_income_rest_of_year, dtype=np.float64
    )
    room = room_below_edge(projected_total, bracket_edges, ceiling_index)
    monthly = room / months_remaining_in_year
    return np.asarray(
        np.clip(np.minimum(monthly, np.asarray(available, dtype=np.float64)), 0, None),
        dtype=np.float64,
    )


def _projected_income_rest_of_year(
    net_income_ytd: NDArray[np.float64], person: PersonState, month: int
) -> NDArray[np.float64]:
    """Project this person's net income for the months of the year after ``month``.

    Args:
        net_income_ytd: This person's year-to-date net income, ``(n_paths,)`` -- the same
            figure passed as ``fill_to_bracket``'s ``taxable_income_ytd``.
        person: This person's state after phase 6.
        month: The current calendar month, 1 to 12. In December the run-rate term is zero.

    Returns:
        The run-rate projection plus the LIF minimum not yet drawn, capped at the LIF
        balance, non-negative, ``(n_paths,)``.
    """
    income = person.income
    base = np.clip(
        np.asarray(net_income_ytd, dtype=np.float64)
        - income.rrsp_withdrawals
        - income.rrif_lif_withdrawals,
        0,
        None,
    )
    run_rate = base / month
    projected = run_rate * (12 - month)
    lif_remaining_minimum = np.clip(person.lif.annual_minimum - person.lif.withdrawn_ytd, 0, None)
    projected = projected + np.minimum(lif_remaining_minimum, person.lif.balance)
    return np.asarray(projected, dtype=np.float64)


def _gross_for_net(
    target: ArrayLike,
    g0: ArrayLike,
    m: ArrayLike,
    edges: tuple[float, ...],
    rates: tuple[float, ...],
) -> NDArray[np.float64]:
    """The smallest gross ``G >= g0`` with ``net(G) >= target``, on one withholding table.

    ``net(G) = m + (G - m) * (1 - rate)``, where ``rate`` is the rate of the band
    ``G - m`` itself falls in; the first self-consistent band, ascending, is taken. Below
    ``m`` there is no withholding, so the answer is simply ``max(target, g0)``.

    Args:
        target: Desired net proceeds, ``(n_paths,)``.
        g0: Gross already committed on this account this month; the floor on the answer,
            ``(n_paths,)``.
        m: This account's withholding-free floor this month -- zero for an RRSP, or
            ``max(0, annual_minimum - withdrawn_ytd)`` for a RRIF/LIF, read before this
            month's own withdrawal.
        edges: The withholding table's band edges (``real_params.rrif.amounts(
            "withholding.edges_each", month_index)``).
        rates: The withholding table's band rates, one more than ``edges``.

    Returns:
        The smallest qualifying gross, ``(n_paths,)``, vectorised over paths.

    Raises:
        ValueError: If ``rates`` is not non-decreasing, naming the table -- the
            first-valid-band search below is correct only then.
    """
    edges_arr = np.asarray(edges, dtype=np.float64)
    rates_arr = np.asarray(rates, dtype=np.float64)
    if rates_arr.size > 1 and np.any(np.diff(rates_arr) < 0):
        raise ValueError(
            f"withholding.rates must be non-decreasing for _gross_for_net's first-valid-band "
            f"search to be correct, got {rates_arr!r}."
        )

    target_arr = np.asarray(target, dtype=np.float64)
    g0_arr = np.asarray(g0, dtype=np.float64)
    m_arr = np.asarray(m, dtype=np.float64)

    below_m = np.maximum(target_arr, g0_arr) <= m_arr
    simple = np.maximum(target_arr, g0_arr)

    x0 = np.clip(g0_arr - m_arr, 0, None)
    t = target_arr - m_arr

    lowers = np.concatenate(([0.0], edges_arr))
    uppers = np.concatenate((edges_arr, [np.inf]))

    resolved = np.zeros_like(target_arr)
    found = np.zeros(target_arr.shape, dtype=bool)
    for lower, upper, rate in zip(lowers, uppers, rates_arr, strict=True):
        candidate = np.maximum(t / (1 - rate), x0)
        valid = (~found) & (candidate > lower) & (candidate <= upper)
        resolved = np.where(valid, candidate, resolved)
        found = found | valid

    from_bands = m_arr + resolved
    return np.asarray(np.where(below_m, simple, from_bands), dtype=np.float64)

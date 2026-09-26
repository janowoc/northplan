# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What a policy is shown, and what one month's step remembers of itself.

:class:`MonthContext` is the read-only snapshot :func:`engine.core.step.advance_month`
hands a :class:`~engine.policy.base.Policy` at phase 7: the opening cash balance, this
month's inflows and required outflows already applied, and the cash balance that
results. A policy reads it and ``state`` together and returns a
:class:`~engine.policy.base.Decision`; it may not reach behind either for a figure the
step has not yet computed.

:class:`MonthRecord` is built once phase 11 (December) has run, after growth and any
conversion, because :attr:`MonthRecord.balances_close` and :attr:`MonthRecord.resp_close`
need the balances the *next* month opens with. :attr:`MonthRecord.cash_close` keeps its own
meaning regardless (after phase 9; cash neither grows nor changes in phases 10-11). The
context plus everything the policy's transfers and the cash floor did. Nothing outside
``engine.core.step`` constructs either class;
``engine.mc.simulate.run`` only ever reads one back, via :meth:`MonthRecord.at_path`, for a
single traced path.

Every class here is frozen and ``slots=True``. Every array is ``(n_paths,)``, float64 unless
noted otherwise, and read-only, built with a fresh copy so that nothing a caller still holds
writeable can change one of these from underneath it. Per-person and per-beneficiary parts are
tuples, indexed exactly like ``HouseholdState.persons`` / ``HouseholdState.beneficiaries``.

Two identities every :class:`MonthRecord` satisfies, to the cent, on every path
(``tests/core/test_step.py`` checks both against a live run):

- ``cash_after_flows == cash_opening - cash_to_estate + sum(inflows.to_cash) +
  sum(education_draws) - sum(payroll_withholding) - spending - sum(education_costs)
  - sum(tax_settlement) + sum(forced_withdrawals) - sum(forced_withholding)``
- ``cash_close == cash_after_flows + sum(withdrawals) - sum(withdrawal_withholding)
  - sum(contributions) - sum(resp_contributions) + sum(wind_up_to_cash)
  + sum(floor_withdrawals) + depletion_deficit``

where ``sum(...)`` over a tuple of :class:`ByKind` means the total of all five kinds,
summed again over every person or beneficiary, and ``inflows.to_cash`` includes the CPP and
DB-pension survivor streams (:attr:`PersonInflows.cpp_survivor`,
:attr:`PersonInflows.db_pension_survivor`).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.core.state import freeze

__all__ = [
    "AccountAmounts",
    "ByKind",
    "MonthContext",
    "MonthRecord",
    "PersonInflows",
]


def _arr(value: ArrayLike) -> NDArray[np.float64]:
    """A fresh, read-only ``float64`` copy of ``value``."""
    return freeze(np.array(value, dtype=np.float64))


def _bool_arr(value: ArrayLike) -> NDArray[np.bool_]:
    """A fresh, read-only ``bool`` copy of ``value``."""
    return freeze(np.array(value, dtype=np.bool_))


def _freeze_fields(obj: object, *names: str) -> None:
    """Replace each named field of ``obj`` with a fresh, frozen copy of itself."""
    for name in names:
        object.__setattr__(obj, name, _arr(getattr(obj, name)))


def _freeze_bool_fields(obj: object, *names: str) -> None:
    """Replace each named field of ``obj`` with a fresh, frozen bool copy of itself."""
    for name in names:
        object.__setattr__(obj, name, _bool_arr(getattr(obj, name)))


def _freeze_tuple_of_arrays(obj: object, name: str) -> None:
    """Replace tuple field ``name`` with a tuple of fresh, frozen copies."""
    object.__setattr__(obj, name, tuple(_arr(item) for item in getattr(obj, name)))


def _freeze_tuple_of_bool_arrays(obj: object, name: str) -> None:
    """Replace tuple field ``name`` with a tuple of fresh, frozen bool copies."""
    object.__setattr__(obj, name, tuple(_bool_arr(item) for item in getattr(obj, name)))


@dataclasses.dataclass(frozen=True, slots=True)
class AccountAmounts:
    """One person's amount across the six accounts a spousal rollover can touch.

    Used for :attr:`MonthContext.rolled_out` (the deceased's pre-roll balances, per
    account) and :attr:`MonthRecord.balances_close` (closing balances, per person).

    Attributes:
        rrsp, rrif, lira, lif, tfsa, taxable: Amount for that account, ``(n_paths,)``.
    """

    rrsp: NDArray[np.float64]
    rrif: NDArray[np.float64]
    lira: NDArray[np.float64]
    lif: NDArray[np.float64]
    tfsa: NDArray[np.float64]
    taxable: NDArray[np.float64]

    def __post_init__(self) -> None:
        _freeze_fields(self, "rrsp", "rrif", "lira", "lif", "tfsa", "taxable")


@dataclasses.dataclass(frozen=True, slots=True)
class PersonInflows:
    """One person's phase-3 inflows to household cash, for one month.

    Attributes:
        employment: Gross employment income this month.
        cpp_contributions: This month's CPP contributions (base tier plus enhanced),
            deducted from employment before it reaches cash.
        ei_premiums: This month's EI premium, deducted from employment.
        cpp: This person's **own** CPP paid this month (excludes the survivor
            increment).
        oas: Gross OAS paid this month, before any repayment (assessed at the
            December close, not deducted here).
        db_pension: Every defined-benefit pension this person receives in their own
            right, including any bridge, summed (excludes a survivor share).
        cpp_survivor: The CPP survivor increment this person receives this month,
            zero unless the other person in a two-person household has died and this
            person is alive (``docs/limitations.md`` L19, L40).
        db_pension_survivor: The sum, over the other person's defined-benefit
            pensions, of the survivor share this person receives this month, zero on
            the same condition as ``cpp_survivor`` (L40, L59).
    """

    employment: NDArray[np.float64]
    cpp_contributions: NDArray[np.float64]
    ei_premiums: NDArray[np.float64]
    cpp: NDArray[np.float64]
    oas: NDArray[np.float64]
    db_pension: NDArray[np.float64]
    cpp_survivor: NDArray[np.float64]
    db_pension_survivor: NDArray[np.float64]

    def __post_init__(self) -> None:
        _freeze_fields(
            self,
            "employment",
            "cpp_contributions",
            "ei_premiums",
            "cpp",
            "oas",
            "db_pension",
            "cpp_survivor",
            "db_pension_survivor",
        )

    @property
    def to_cash(self) -> NDArray[np.float64]:
        """Net amount reaching household cash: ``employment - cpp_contributions -
        ei_premiums + cpp + oas + db_pension + cpp_survivor + db_pension_survivor``."""
        return (
            self.employment
            - self.cpp_contributions
            - self.ei_premiums
            + self.cpp
            + self.oas
            + self.db_pension
            + self.cpp_survivor
            + self.db_pension_survivor
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ByKind:
    """One person's amounts by registered/taxable account kind.

    Carries all five :data:`~engine.scenario.WITHDRAWAL_KINDS`, even where a
    particular use only ever fills a subset (forced withdrawals fill only ``rrif``
    and ``lif``; contributions fill only ``rrsp``, ``tfsa``, and ``taxable``) — the
    other fields hold zero rather than being omitted, so every consumer sums the same
    five fields without first asking which use built it.

    Attributes:
        rrsp, rrif, lif, tfsa, taxable: Amount for that account kind, ``(n_paths,)``.
    """

    rrsp: NDArray[np.float64]
    rrif: NDArray[np.float64]
    lif: NDArray[np.float64]
    tfsa: NDArray[np.float64]
    taxable: NDArray[np.float64]

    def __post_init__(self) -> None:
        _freeze_fields(self, "rrsp", "rrif", "lif", "tfsa", "taxable")

    def total(self) -> NDArray[np.float64]:
        """Sum across every kind, ``(n_paths,)``."""
        return self.rrsp + self.rrif + self.lif + self.tfsa + self.taxable


@dataclasses.dataclass(frozen=True, slots=True)
class MonthContext:
    """The read-only snapshot a :class:`~engine.policy.base.Policy` is shown at phase 7.

    ``state`` handed alongside this is the state after phase 6: January's
    :func:`~engine.core.step.open_year` already applied, this month's receipts,
    withholding, outflows, and forced withdrawals already in cash and in the
    year-to-date ledger. ``state.cash.balance`` therefore equals
    :attr:`cash_after_flows`.

    Attributes:
        month_index, year, month: This month's position.
        cash_opening: Household cash before phase 1 of this month.
        rolled_out: Phase 2 spousal-rollover source, per person: the deceased's
            pre-roll balance in each of the six accounts, where that person died this
            month with a surviving spouse, zero elsewhere (including for every person
            who does not die this month). See
            ``engine.core.step.resolve_deaths``.
        rolled_acb: Phase 2 spousal-rollover source, per person: the deceased's
            pre-roll taxable ACB, on the same condition as ``rolled_out``.
        terminal_assessment: Phase 2 terminal-return tax, household total, non-zero
            only in the month of the second death.
        cash_to_estate: Phase 2 cash paid to the estate, non-zero only in the month
            of the second death, when it equals that month's opening cash.
        inflows: Phase 3 income, per person.
        education_draws: Phase 3 RESP education draw to cash, per beneficiary.
        payroll_withholding: Phase 4 withholding remitted, per person.
        spending: This month's spending actually paid (phase 5), ``(n_paths,)`` --
            ``state.spending_monthly`` scaled by the survivor share or by zero on a
            finished path; see ``engine.core.step.advance_month``.
        education_costs: Phase 5 education cost paid, per beneficiary.
        tax_settlement: Phase 5 filing-month settlement, per person: the prior year's balance
            owing paid from cash; negative is a refund deposited. Zero outside the filing
            month.
        forced_withdrawals: Phase 6 forced RRIF/LIF withdrawals, gross, per person;
            only ``rrif`` and ``lif`` are ever non-zero.
        forced_withholding: Phase 6 registered withholding on the excess above the
            minimum, per person.
        cash_after_flows: Household cash after phase 6.
    """

    month_index: int
    year: int
    month: int
    cash_opening: NDArray[np.float64]
    rolled_out: tuple[AccountAmounts, ...]
    rolled_acb: tuple[NDArray[np.float64], ...]
    terminal_assessment: NDArray[np.float64]
    cash_to_estate: NDArray[np.float64]
    inflows: tuple[PersonInflows, ...]
    education_draws: tuple[NDArray[np.float64], ...]
    payroll_withholding: tuple[NDArray[np.float64], ...]
    spending: NDArray[np.float64]
    education_costs: tuple[NDArray[np.float64], ...]
    tax_settlement: tuple[NDArray[np.float64], ...]
    forced_withdrawals: tuple[ByKind, ...]
    forced_withholding: tuple[NDArray[np.float64], ...]
    cash_after_flows: NDArray[np.float64]

    def __post_init__(self) -> None:
        _freeze_fields(
            self,
            "cash_opening",
            "terminal_assessment",
            "cash_to_estate",
            "spending",
            "cash_after_flows",
        )
        _freeze_tuple_of_arrays(self, "rolled_acb")
        _freeze_tuple_of_arrays(self, "education_draws")
        _freeze_tuple_of_arrays(self, "payroll_withholding")
        _freeze_tuple_of_arrays(self, "education_costs")
        _freeze_tuple_of_arrays(self, "tax_settlement")
        _freeze_tuple_of_arrays(self, "forced_withholding")
        object.__setattr__(self, "rolled_out", tuple(self.rolled_out))
        object.__setattr__(self, "inflows", tuple(self.inflows))
        object.__setattr__(self, "forced_withdrawals", tuple(self.forced_withdrawals))


@dataclasses.dataclass(frozen=True, slots=True)
class MonthRecord:
    """Everything one month's step did, built after phase 11 (December), not before growth.

    Attributes:
        context: The :class:`MonthContext` phase 7 was shown.
        withdrawals: Phase 8 transfers to cash, gross, effective (after room and cash
            caps), per person.
        withdrawal_withholding: Phase 8 registered withholding on those withdrawals,
            per person.
        contributions: Phase 8 transfers from cash, effective, per person; only
            ``rrsp``, ``tfsa``, and ``taxable`` are ever non-zero.
        resp_contributions: Phase 8 RESP contributions, effective, per beneficiary.
        wind_up_to_cash: Phase 8 RESP wind-up proceeds to cash, per beneficiary:
            ``to_cash_tax_free + accumulated`` (excludes ``grants_repaid``, which
            leaves the household).
        floor_withdrawals: Phase 9 cash-floor withdrawals, gross, no withholding, per
            person.
        depletion_deficit: What remained negative after the phase 9 floor, forgiven,
            floored at zero.
        spending_cut: ``min(depletion_deficit, spending)`` taken off
            ``spending_achieved_ytd``.
        cash_close: Household cash after phase 9 -- unchanged by phases 10-11, since
            cash neither grows nor is touched by ``close_year``.
        balances_close: Per person, the balance in each of the six accounts that the
            *next* month opens with: after phase 10 growth and, in December, after
            phase 11's RRSP-to-RRIF and LIRA-to-LIF conversions.
        resp_close: Per beneficiary, ``contributions + grants + income`` at the same
            point as ``balances_close``.
        alive: Per person, whether alive after phase 2 -- exactly
            ``death_month_index > month_index``.
        depleted: Whether the household is depleted, after phase 9,
            ``(n_paths,)`` bool.
        finished: Whether every person is not alive after phase 2, ``(n_paths,)``
            bool -- see ``engine.core.step.resolve_deaths``.
        estate_after_tax: The state's ``estate_after_tax`` after phase 2,
            ``(n_paths,)`` -- ``NaN`` until the month of the second death, finite
            from it on.
    """

    context: MonthContext
    withdrawals: tuple[ByKind, ...]
    withdrawal_withholding: tuple[NDArray[np.float64], ...]
    contributions: tuple[ByKind, ...]
    resp_contributions: tuple[NDArray[np.float64], ...]
    wind_up_to_cash: tuple[NDArray[np.float64], ...]
    floor_withdrawals: tuple[ByKind, ...]
    depletion_deficit: NDArray[np.float64]
    spending_cut: NDArray[np.float64]
    cash_close: NDArray[np.float64]
    balances_close: tuple[AccountAmounts, ...]
    resp_close: tuple[NDArray[np.float64], ...]
    alive: tuple[NDArray[np.bool_], ...]
    depleted: NDArray[np.bool_]
    finished: NDArray[np.bool_]
    estate_after_tax: NDArray[np.float64]

    def __post_init__(self) -> None:
        _freeze_fields(self, "depletion_deficit", "spending_cut", "cash_close")
        _freeze_fields(self, "estate_after_tax")
        _freeze_bool_fields(self, "depleted", "finished")
        _freeze_tuple_of_arrays(self, "withdrawal_withholding")
        _freeze_tuple_of_arrays(self, "resp_contributions")
        _freeze_tuple_of_arrays(self, "wind_up_to_cash")
        _freeze_tuple_of_arrays(self, "resp_close")
        _freeze_tuple_of_bool_arrays(self, "alive")
        object.__setattr__(self, "withdrawals", tuple(self.withdrawals))
        object.__setattr__(self, "contributions", tuple(self.contributions))
        object.__setattr__(self, "floor_withdrawals", tuple(self.floor_withdrawals))
        object.__setattr__(self, "balances_close", tuple(self.balances_close))

    def at_path(self, k: int) -> MonthRecord:
        """This record with every array sliced to path ``k``, copied and frozen.

        Args:
            k: Path index, ``0 <= k < n_paths``.

        Returns:
            An equivalent :class:`MonthRecord` whose every array has shape ``(1,)``.
        """
        return _slice_at_path(self, k)


def _slice_at_path(value: Any, k: int) -> Any:
    """Recursively slice every array reached from ``value`` to ``[k : k + 1]``.

    Copies each slice before freezing it, so the result shares no memory with
    ``value``. Recurses into dataclass fields and tuple elements; leaves every other
    value (an ``int``, for instance) untouched.
    """
    if isinstance(value, np.ndarray):
        return freeze(value[k : k + 1].copy())
    if isinstance(value, tuple):
        return tuple(_slice_at_path(item, k) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {
            field.name: _slice_at_path(getattr(value, field.name), k)
            for field in dataclasses.fields(value)
        }
        return dataclasses.replace(value, **changes)
    return value

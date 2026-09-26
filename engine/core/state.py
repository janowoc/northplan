# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Household state carried from one simulated month to the next.

Every class here is a frozen, ``slots``-only dataclass; the monthly step returns a new state
rather than mutating the old one. Every per-path field is a NumPy array of shape ``(n_paths,)``,
real dollars unless noted otherwise; per-person and per-beneficiary state are tuples of these
classes.

**Month indexes may be negative**, counting from January of the scenario's start year: a
negative one means the event predates the run, except :attr:`PersonState.death_month_index`,
guarded instead by :data:`DEATH_NOT_DRAWN`.

Whether a dollar-typed field is fixed in real terms or in nominal terms is classified field by
field in ``tests/core/test_state_nominal_or_real.py``; a field classified nominal there is eroded
once a year, in January, by its owning account module's ``erode_nominal``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

#: Sentinel for a death that has not been drawn yet: the largest
#: representable ``int64``. Never do arithmetic on this value —
#: ``death_month_index - month_index`` silently overflows ``int64`` for a
#: path that has not died. Compare against it directly, or read ``alive``.
DEATH_NOT_DRAWN: Final[int] = int(np.iinfo(np.int64).max)


def freeze(array: NDArray) -> NDArray:
    """Mark ``array`` non-writeable in place and return it.

    Idempotent. Refuses a view onto a buffer still writeable elsewhere (walks the ``.base`` chain).

    Args:
        array: A NumPy array, of any dtype.

    Returns:
        The same array object, non-writeable.

    Raises:
        AssertionError: If ``array`` is not one-dimensional.
        ValueError: If ``array`` is a view onto a buffer still writeable elsewhere; pass a copy.
    """
    assert array.ndim == 1, (
        f"expected a one-dimensional array, got shape {array.shape}. Every "
        "per-path field here is a flat (n_paths,) array; per-person and "
        "per-beneficiary state is a tuple of objects, not an extra axis."
    )
    base = array.base
    while base is not None:
        if base.flags.writeable:
            raise ValueError(
                "this array is a view onto a buffer that is still writeable "
                "through some other reference. Marking the view read-only "
                "would not protect it: the buffer it shares memory with can "
                "still be mutated, and the mutation would show up in this "
                "state with no error. Pass a copy instead, e.g. array.copy()."
            )
        base = base.base
    array.flags.writeable = False
    return array


def updated[T](obj: T, **changes: object) -> T:
    """Return a copy of ``obj`` with ``changes`` applied, frozen like the original; re-runs
    ``__post_init__`` and re-freezes, like ``dataclasses.replace``.

    Args:
        obj: One of the frozen dataclasses defined in this module.
        changes: Field name to new value, as ``dataclasses.replace`` takes them.

    Returns:
        A new, frozen instance of ``type(obj)``.
    """
    return dataclasses.replace(obj, **changes)


def _freeze_fields(obj: object, *names: str) -> None:
    """Freeze the named array fields of ``obj`` in place, via :func:`freeze`."""
    for name in names:
        object.__setattr__(obj, name, freeze(getattr(obj, name)))


def _iter_arrays(value: object) -> Iterator[NDArray]:
    """Yield every NumPy array reachable from ``value``, at any depth.

    Recurses into dataclass fields and tuple elements. Used by
    :meth:`HouseholdState.__post_init__` to check every array against ``n_paths``.
    """
    if isinstance(value, np.ndarray):
        yield value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _iter_arrays(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            yield from _iter_arrays(item)


@dataclass(frozen=True, slots=True)
class CashState:
    """The hub account. Every inflow and outflow passes through it.

    Attributes:
        balance: Real dollars, ``(n_paths,)``.
    """

    balance: NDArray[np.float64]

    def __post_init__(self) -> None:
        _freeze_fields(self, "balance")


@dataclass(frozen=True, slots=True)
class RrspState:
    """One person's RRSP, before conversion to a RRIF.

    Attributes:
        balance: Real dollars, ``(n_paths,)``.
        room: Unused contribution room.
        contributed_ytd: Contributions made so far this year; reset in January.
        converted_fraction_applied: Whether an RRSP-to-RRIF conversion, elected or statutory,
            has fired for this person.
    """

    balance: NDArray[np.float64]
    room: NDArray[np.float64]
    contributed_ytd: NDArray[np.float64]
    converted_fraction_applied: bool

    def __post_init__(self) -> None:
        _freeze_fields(self, "balance", "room", "contributed_ytd")


@dataclass(frozen=True, slots=True)
class RrifState:
    """One person's RRIF, whether converted already or opening empty.

    Attributes:
        balance: Real dollars, ``(n_paths,)``.
        annual_minimum: This year's statutory minimum, fixed in January and never decremented;
            what remains to be withdrawn is ``annual_minimum - withdrawn_ytd``.
        withdrawn_ytd: Withdrawn so far this year; reset in January.
        opened_year: The calendar year opened, or ``None`` if the opening balance is zero. If
            already open at scenario start, set to ``scenario.start_year - 1``, not the start year.
    """

    balance: NDArray[np.float64]
    annual_minimum: NDArray[np.float64]
    withdrawn_ytd: NDArray[np.float64]
    opened_year: int | None

    def __post_init__(self) -> None:
        _freeze_fields(self, "balance", "annual_minimum", "withdrawn_ytd")
        if self.opened_year is None and not np.all(self.balance == 0.0):
            raise ValueError(
                "opened_year is None (nothing has been opened) but balance "
                "is nonzero on at least one path. opened_year is None only "
                "when there is no RRIF to have opened at all."
            )


@dataclass(frozen=True, slots=True)
class LiraState:
    """A LIRA, before conversion to a LIF. Takes no withdrawals.

    Attributes:
        balance: Real dollars, ``(n_paths,)``.
        jurisdiction: Two-letter pension-jurisdiction code, e.g. ``"ab"``; empty only while balance
            is zero (L3); carried onto the LIF on conversion (L47).
    """

    balance: NDArray[np.float64]
    jurisdiction: str

    def __post_init__(self) -> None:
        _freeze_fields(self, "balance")
        if self.jurisdiction == "" and not np.all(self.balance == 0.0):
            raise ValueError(
                "jurisdiction is '' (this LIRA names no jurisdiction of its "
                "own) but balance is nonzero on at least one path. An empty "
                "jurisdiction is valid only while the balance is zero; "
                "falling back to the province of residence is exactly the "
                "mistake docs/limitations.md L3 exists to prevent."
            )


@dataclass(frozen=True, slots=True)
class LifState:
    """A LIF, for a person already converted (``docs/limitations.md`` L3: governed by the
    jurisdiction of the originating pension). Has a minimum and, in most jurisdictions, a maximum.

    Attributes:
        balance: Real dollars, ``(n_paths,)``.
        jurisdiction: Two-letter pension-jurisdiction code; empty only while balance is zero (L3);
            carried from the LIRA on conversion (L47).
        annual_minimum: This year's RRIF-equivalent minimum, fixed in January and never
            decremented; what remains to be withdrawn is ``annual_minimum - withdrawn_ytd``.
        annual_maximum: This year's jurisdiction-specific maximum (where imposed), fixed in
            January the same way.
        withdrawn_ytd: Amount withdrawn so far this year; fixed/reset in January.
        opened_year: Mirrors :attr:`RrifState.opened_year` exactly.
    """

    balance: NDArray[np.float64]
    jurisdiction: str
    annual_minimum: NDArray[np.float64]
    annual_maximum: NDArray[np.float64]
    withdrawn_ytd: NDArray[np.float64]
    opened_year: int | None

    def __post_init__(self) -> None:
        _freeze_fields(self, "balance", "annual_minimum", "annual_maximum", "withdrawn_ytd")
        if self.jurisdiction == "" and not np.all(self.balance == 0.0):
            raise ValueError(
                "jurisdiction is '' (this LIF names no jurisdiction of its "
                "own) but balance is nonzero on at least one path. An empty "
                "jurisdiction is valid only while the balance is zero; "
                "falling back to the province of residence is exactly the "
                "mistake docs/limitations.md L3 exists to prevent."
            )
        if self.opened_year is None and not np.all(self.balance == 0.0):
            raise ValueError(
                "opened_year is None (nothing has been opened) but balance "
                "is nonzero on at least one path. opened_year is None only "
                "when there is no LIF to have opened at all."
            )


@dataclass(frozen=True, slots=True)
class TfsaState:
    """One person's TFSA.

    Attributes:
        balance: Real dollars, ``(n_paths,)``.
        room: Unused contribution room, including room restored from earlier withdrawals.
        withdrawn_this_year: Withdrawn so far this year; restored to room the *following* January.
    """

    balance: NDArray[np.float64]
    room: NDArray[np.float64]
    withdrawn_this_year: NDArray[np.float64]

    def __post_init__(self) -> None:
        _freeze_fields(self, "balance", "room", "withdrawn_this_year")


@dataclass(frozen=True, slots=True)
class TaxableState:
    """One person's non-registered holding.

    Attributes:
        balance: Market value.
        acb: Adjusted cost base; may exceed ``balance`` (a loss is ordinary, not an error).
    """

    balance: NDArray[np.float64]
    acb: NDArray[np.float64]

    def __post_init__(self) -> None:
        _freeze_fields(self, "balance", "acb")


@dataclass(frozen=True, slots=True)
class EmploymentBand:
    """One band of employment income, resolved to a monthly figure.

    Attributes:
        from_month_index, to_month_index: Inclusive start/end month index; may be
            negative, per the module docstring's convention. A band lying entirely
            before the run is still carried rather than dropped.
        monthly_amount: Real dollars for one month while the band is in force,
            ``(n_paths,)`` — the scenario's ``annual`` divided by twelve.
    """

    from_month_index: int
    to_month_index: int
    monthly_amount: NDArray[np.float64]

    def __post_init__(self) -> None:
        _freeze_fields(self, "monthly_amount")


@dataclass(frozen=True, slots=True)
class BenefitState:
    """One person's standing on a public pension: CPP or OAS.

    The same shape serves both: an amount either starting on an elected age or already flowing.

    Attributes:
        start_age_months: Age in months elected to start, or ``None`` if already in pay.
        in_pay_monthly: Gross monthly amount received, before withholding, ``None`` if not started.
        contributory_history: Fraction of the maximum CPP pension earned, ``[0, 1]``; always
            ``None`` for OAS. At most one of this and ``in_pay_monthly`` may be set.
        monthly_amount: The amount actually in pay this month; zero until started.
    """

    start_age_months: int | None
    in_pay_monthly: NDArray[np.float64] | None
    contributory_history: float | None
    monthly_amount: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.contributory_history is not None and self.in_pay_monthly is not None:
            raise ValueError(
                "contributory_history and in_pay_monthly are both set; at "
                "most one way of knowing this benefit's amount is allowed. "
                "engine.core.build enforces exactly one for CPP, mirroring "
                "engine.scenario.schema.CppEntitlement; the monthly step "
                "can produce this violation too, e.g. updated(cpp, "
                "in_pay_monthly=arr) on a BenefitState that still carries "
                "contributory_history, which the schema cannot see."
            )
        if self.in_pay_monthly is not None:
            _freeze_fields(self, "in_pay_monthly")
        _freeze_fields(self, "monthly_amount")


@dataclass(frozen=True, slots=True)
class PensionState:
    """One defined-benefit pension one person receives or will receive.

    Attributes:
        name, start_month_index: Label (unique per person) and month index the pension begins.
        monthly_amount: The base pension; a real-dollar constant once in pay if fully indexed
            (on no indexation schedule, so no erosion factor applies), decaying under
            ``engine.core.indexation.unindexed_factor`` otherwise.
        indexed, bridge_monthly, bridge_end_month_index: Whether the pension moves with CPI
            (``False`` means fixed in nominal terms); the bridge paid on top, and the last
            month it is paid, inclusive. The end index may be negative, like any month index.
            A ``None`` end index means no bridge and requires ``bridge_monthly`` zero on every
            path; a zeroed bridge with the end index left in place is allowed.
        survivor_share: Fraction the survivor keeps after death, ``[0, 1]``. Not per-path.
    """

    name: str
    monthly_amount: NDArray[np.float64]
    start_month_index: int
    indexed: bool
    bridge_monthly: NDArray[np.float64]
    bridge_end_month_index: int | None
    survivor_share: float

    def __post_init__(self) -> None:
        _freeze_fields(self, "monthly_amount", "bridge_monthly")
        if self.bridge_end_month_index is None and not np.all(self.bridge_monthly == 0.0):
            raise ValueError(
                f"pension {self.name!r}: bridge_end_month_index is None (nothing "
                "ends the bridge) but bridge_monthly is nonzero on at least one "
                "path. A bridge with no end month is never paid; the end month is "
                "the last month paid, inclusive, and a nonzero bridge requires one."
            )


@dataclass(frozen=True, slots=True)
class IncomeLedger:
    """One person's income components, accumulated year to date; reset in January, consumed
    once in December.

    Attributes:
        employment, cpp, db_pension, rrsp_withdrawals, rrif_lif_withdrawals, interest:
            Income by component and source.
        eligible_dividends: Eligible dividends from taxable holdings, before gross-up.
        capital_gains: Net capital gains from taxable holdings, realized on disposition or
            distributed by the holding without a sale, before the inclusion rate. Signed:
            a realized loss reduces it and the year's total may be negative (L17).
        oas, resp_accumulated_income: Gross OAS received (repayment assessed in
            December), and accumulated-income payments from an RESP wind-up.
        rrsp_deductions, cpp_enhanced_contributions, cpp_base_contributions, ei_premiums:
            This year's amounts (the first two are deductions, the last two credits).
        remitted: Tax already withheld and remitted this year.
    """

    employment: NDArray[np.float64]
    cpp: NDArray[np.float64]
    oas: NDArray[np.float64]
    db_pension: NDArray[np.float64]
    rrsp_withdrawals: NDArray[np.float64]
    rrif_lif_withdrawals: NDArray[np.float64]
    interest: NDArray[np.float64]
    eligible_dividends: NDArray[np.float64]
    capital_gains: NDArray[np.float64]
    resp_accumulated_income: NDArray[np.float64]
    rrsp_deductions: NDArray[np.float64]
    cpp_base_contributions: NDArray[np.float64]
    cpp_enhanced_contributions: NDArray[np.float64]
    ei_premiums: NDArray[np.float64]
    remitted: NDArray[np.float64]

    def __post_init__(self) -> None:
        _freeze_fields(self, *(field.name for field in dataclasses.fields(self)))


@dataclass(frozen=True, slots=True)
class PersonState:
    """One member of the household, at the boundary between two months.

    Attributes:
        person_id, sex, birth_year, birth_month: Identifier; ``"f"``/``"m"`` (selects the life
            table); and birth date (month ``1..12``).
        alive, death_month_index: Survival flag (monotonic) and month index of death, int64 or
            :data:`DEATH_NOT_DRAWN`.
        rrsp, rrif, lira, lif, tfsa, taxable, cpp, oas: Standing on each account or benefit.
        employment, pensions: Bands and defined-benefit pensions, in scenario order, possibly empty.
        income, balance_owing: Year-to-date income components, and the prior year's assessed tax
            unpaid (paid in the filing month).
        prior_year_net_income, net_income_two_years_prior: Net income after the
            social benefits repayment (line 23600), per person, ``(n_paths,)``:
            ``prior_year_net_income`` for the calendar year before the
            scenario's start year, ``net_income_two_years_prior`` for the year
            before that. Their one consumer is the RESP enhanced-grant rate
            (``grant.enhanced.income_year_offset``), which reads the **older**
            of the two, ``net_income_two_years_prior``, because the reach-back
            is two years; summed across every person in the household,
            including one who has died, whose last-written figure keeps
            counting (L33); not read by the OAS repayment (assessed on the
            current year) nor by the GIS band indicator (a different income
            basis). At each December close the pair shifts — the older takes
            the value of the newer, the newer takes this year's line 23600 —
            for every person alive at any point in the calendar year, and
            freezes thereafter. A person who has died keeps the two figures
            written at the close of the last year they were alive in; the
            older, the one the RESP reads, is their net income for the year
            before that. Both open as real figures: a scenario states them as filed, and
            ``engine.core.build.build_initial_state`` restates the opening pair to real
            dollars (``engine.core.indexation.as_filed_to_real_factor``) before the state ever
            carries them. Each is then rewritten in turn at every December close that follows
            (``engine.core.step.close_year``, item 5).
    """

    person_id: str
    sex: str
    birth_year: int
    birth_month: int
    alive: NDArray[np.bool_]
    death_month_index: NDArray[np.int64]
    rrsp: RrspState
    rrif: RrifState
    lira: LiraState
    lif: LifState
    tfsa: TfsaState
    taxable: TaxableState
    cpp: BenefitState
    oas: BenefitState
    employment: tuple[EmploymentBand, ...]
    pensions: tuple[PensionState, ...]
    income: IncomeLedger
    balance_owing: NDArray[np.float64]
    prior_year_net_income: NDArray[np.float64]
    net_income_two_years_prior: NDArray[np.float64]

    def __post_init__(self) -> None:
        _freeze_fields(self, "alive", "death_month_index", "balance_owing")
        _freeze_fields(self, "prior_year_net_income", "net_income_two_years_prior")
        if np.any(~self.alive & (self.death_month_index == DEATH_NOT_DRAWN)):
            raise ValueError(
                "at least one path has alive=False but death_month_index is "
                "still DEATH_NOT_DRAWN; dead with no recorded death month is "
                "a contradiction."
            )
        # This is only half of the invariant. The other half — alive is
        # exactly (death_month_index > month_index) — needs a month_index,
        # which this class does not carry. The semantics are already fixed:
        # engine.core.mortality.death_month_index is the first month the
        # person is not alive, so a death drawn for month k means dead at
        # the open of month k. #36 tests it after advance_month's step 2,
        # with month_index in hand. Do not add that half here "for now".

    def age_months(self, year: int, month: int) -> int:
        """Age in whole months at the start of ``(year, month)``.

        Delegates to ``engine.core.timeline.age_in_months``; computed fresh from the birth
        date on every call, never cached.
        """
        from engine.core import timeline

        return timeline.age_in_months(self.birth_year, self.birth_month, year, month)


@dataclass(frozen=True, slots=True)
class RespState:
    """The RESP standing to one beneficiary, split into its three buckets: contributions come
    out tax-free, grants/income taxable to the student, grants clawed back if wound up without one.

    Attributes:
        contributions, grants, income: The three buckets. ``income`` may be negative: growth and
            losses both accrue here, so that ``contributions`` and ``grants`` keep the nominal
            amounts the wind-up acts on.
        contributions_lifetime, grants_lifetime, grant_room: Totals ever received (for the
            lifetime ceilings), and unused grant room, in grant dollars — an amount of grant, not
            of contribution.
        grant_received_ytd, contributed_ytd, subscriber_index: Basic grant received so far this
            year (reset in January) — the additional tier is paid over and above the annual
            maximum and is bounded by the lifetime cap instead, so it does not count here;
            contributions so far this year; and index into ``HouseholdState.persons`` of the
            subscriber.
        education_start_month_index, education_months, education_monthly_cost: Enrolment's start
            month index, length in months, and monthly cost (not per-path).
        wound_up: Whether the plan has been wound up, ``(n_paths,)``.
    """

    contributions: NDArray[np.float64]
    grants: NDArray[np.float64]
    income: NDArray[np.float64]
    contributions_lifetime: NDArray[np.float64]
    grants_lifetime: NDArray[np.float64]
    grant_room: NDArray[np.float64]
    grant_received_ytd: NDArray[np.float64]
    contributed_ytd: NDArray[np.float64]
    subscriber_index: int
    education_start_month_index: int
    education_months: int
    education_monthly_cost: float
    wound_up: NDArray[np.bool_]

    def __post_init__(self) -> None:
        _freeze_fields(
            self,
            "contributions",
            "grants",
            "income",
            "contributions_lifetime",
            "grants_lifetime",
            "grant_room",
            "grant_received_ytd",
            "contributed_ytd",
            "wound_up",
        )


@dataclass(frozen=True, slots=True)
class BeneficiaryState:
    """An RESP beneficiary, tracked individually and never pooled.

    Attributes:
        beneficiary_id, birth_year, birth_month: Identifier and birth date (month ``1..12``).
        resp: The plan standing to them.
    """

    beneficiary_id: str
    birth_year: int
    birth_month: int
    resp: RespState


@dataclass(frozen=True, slots=True)
class Elections:
    """The dated choices a policy makes, set once from the policy by the simulator.

    Attributes:
        cpp_start_age_months, oas_start_age_months: Age in months each starts, indexed like
            ``HouseholdState.persons``; ``None`` if already in pay.
        rrif_conversion_age_years, rrif_conversion_fraction: Age and share converted, ``[0, 1]``;
            household-wide, not per person.
        fill_pension_credit: Whether to draw enough eligible pension income to use the credit.
    """

    cpp_start_age_months: tuple[int | None, ...]
    oas_start_age_months: tuple[int | None, ...]
    rrif_conversion_age_years: int
    rrif_conversion_fraction: float
    fill_pension_credit: bool


@dataclass(frozen=True, slots=True)
class SpendingLevel:
    """One step of the household spending schedule, resolved to a monthly figure.

    Attributes:
        from_year: First calendar year this level applies in.
        monthly_level: Real dollars for one month at full share — the scenario's
            ``annual`` divided by twelve; a bare ``float`` (household-wide), unlike
            the per-path ``monthly_amount`` on :class:`EmploymentBand`/
            :class:`PensionState`.
    """

    from_year: int
    monthly_level: float


def select_spending_level(schedule: tuple[SpendingLevel, ...], year: int) -> float:
    """The monthly figure in force for ``year``: the latest level starting at or before it.

    Raises:
        ValueError: If ``schedule`` is empty, or no level's ``from_year`` is at or before ``year``.
    """
    if not schedule:
        raise ValueError("spending_schedule is empty; there is no level to select.")
    candidates = [level for level in schedule if level.from_year <= year]
    if not candidates:
        earliest = min(level.from_year for level in schedule)
        raise ValueError(
            f"no level in spending_schedule has from_year <= {year}; the earliest is {earliest}."
        )
    return max(candidates, key=lambda level: level.from_year).monthly_level


@dataclass(frozen=True, slots=True)
class YearRecord:
    """One calendar year's summary, appended to :attr:`HouseholdState.history`.

    Attributes:
        year, net_worth, spending: Calendar year, net worth at 31 December, and
            spending over it.
        after_tax_net_worth: The liquidation value as if every person in the household died on
            31 December of this year, with no spousal rollover: this year's ledger plus every
            person's registered balances and the deemed gain on their taxable holding, each
            assessed alone (``engine.core.step._deemed_single_assessments``, with
            ``died_in_year=True`` for everyone). Comes out ``0`` on a finished path.
        tax_assessed: Tax assessed *for* this year at the December close, not paid
            in cash during it.
        net_income: Line 23600 (``engine.tax.combined.Assessment.net_income_after_repayment``)
            per person, in ``HouseholdState.persons`` order. Among pension splits that give
            equal household tax, the division between persons is arbitrary — the argmin in
            ``engine.tax.combined.household_assessment`` picks on rounding noise (#50) — so
            only the household sum of this tuple is meaningful, never one person's share read
            alone.
        gis_band: ``engine.core.step.close_year`` item 6's living-pensioner-in-band indicator,
            per person, in the same order.
        depleted: Whether the household ran out of money during the year.
    """

    year: int
    net_worth: NDArray[np.float64]
    after_tax_net_worth: NDArray[np.float64]
    spending: NDArray[np.float64]
    tax_assessed: NDArray[np.float64]
    net_income: tuple[NDArray[np.float64], ...]
    gis_band: tuple[NDArray[np.bool_], ...]
    depleted: NDArray[np.bool_]

    def __post_init__(self) -> None:
        _freeze_fields(self, "net_worth", "after_tax_net_worth", "spending", "tax_assessed")
        _freeze_fields(self, "depleted")
        object.__setattr__(self, "net_income", tuple(freeze(arr) for arr in self.net_income))
        object.__setattr__(self, "gis_band", tuple(freeze(arr) for arr in self.gis_band))


@dataclass(frozen=True, slots=True)
class HouseholdState:
    """The full state of the household at the boundary between two months.

    Attributes:
        year, month, month_index, n_paths, province: Calendar position (month
            ``1..12``, months elapsed since January of the start year), path count,
            and two-letter tax-province code.
        persons, beneficiaries, cash: Adults and RESP beneficiaries, in scenario
            order (persons indexed by position downstream), and the one household
            cash account, not per person (``docs/limitations.md`` L38; L50).
        elections: The dated choices in force for this run.
        spending_schedule, spending_monthly, spending_survivor_share: Full
            schedule; derived spending for one month at full share (must equal
            ``select_spending_level(...)``, recomputed at the year roll); and the
            survivor's continuing share after the first death. The per-path figure
            actually paid a month -- ``spending_monthly`` scaled by
            ``spending_survivor_share`` after the first death, or by zero on a
            finished household -- is derived fresh in ``engine.core.step.advance_month``
            (``docs/limitations.md`` L40) and never stored on this state.
        spending_achieved_ytd, depleted, estate_after_tax, history: What has
            actually been spent this year; whether out of money (monotonic);
            estate value after tax (NaN until the second death); and the
            year-by-year record, append-only.
    """

    year: int
    month: int
    month_index: int
    n_paths: int
    province: str
    persons: tuple[PersonState, ...]
    beneficiaries: tuple[BeneficiaryState, ...]
    cash: CashState
    elections: Elections
    spending_schedule: tuple[SpendingLevel, ...]
    spending_monthly: float
    spending_survivor_share: float
    spending_achieved_ytd: NDArray[np.float64]
    depleted: NDArray[np.bool_]
    estate_after_tax: NDArray[np.float64]
    history: tuple[YearRecord, ...]

    def __post_init__(self) -> None:
        _freeze_fields(self, "spending_achieved_ytd", "depleted", "estate_after_tax")

        expected_spending_monthly = select_spending_level(self.spending_schedule, self.year)
        if self.spending_monthly != expected_spending_monthly:
            raise ValueError(
                f"spending_monthly ({self.spending_monthly!r}) does not match "
                f"the level spending_schedule selects for {self.year} "
                f"({expected_spending_monthly!r}). spending_monthly is "
                "derived, never set independently of spending_schedule and "
                "year — see its docstring."
            )

        for array in _iter_arrays(self):
            assert array.shape == (self.n_paths,), (
                f"array of shape {array.shape} does not match "
                f"n_paths={self.n_paths}; every per-path field in the state "
                "tree must be exactly (n_paths,)."
            )

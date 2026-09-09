# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Household state carried from one simulated month to the next.

Everything here is a frozen, ``slots``-only dataclass. The monthly step
returns a new state rather than mutating the old one, which keeps a path's
history inspectable and makes it impossible for a policy to write into state
it should only read.

Shape convention: scalar fields (a person's birth year, a pension's name) are
plain Python values, shared across every path. Every field that varies by
path is a NumPy array of shape ``(n_paths,)`` — never ``(n_persons,
n_paths)`` or anything higher-rank. Per-person and per-beneficiary state are
instead **tuples** of these classes, in scenario order, so a person's whole
position is one object rather than one row spread across a dozen arrays.
There is no ``dict`` field anywhere in this module and no untyped catch-all:
every collection a caller might reach for a mapping is a tuple addressed by
position, and every value has a concrete type.

Two invariants are enforced structurally rather than left to convention:

- **Read-only arrays.** :func:`freeze` is the one place that marks a NumPy
  array non-writeable, called once per array field from the owning class's
  ``__post_init__``. Freezing happens *in place* and returns the same array,
  deliberately: a caller that kept a reference to the array it constructed
  the state from must not retain a writeable handle on it once that array is
  state.
- **Shape agreement.** A leaf class such as :class:`CashState` cannot check
  that its array is ``(n_paths,)`` because it has no ``n_paths`` to check
  against — only :class:`HouseholdState` does. Its ``__post_init__`` walks
  the whole tree once, so a mismatch anywhere below it — a pension array
  built with the wrong path count, say — fails at construction rather than
  three months into a run.

:func:`updated` is the one way to produce a changed copy of any class here.
It wraps :func:`dataclasses.replace`, which re-runs ``__post_init__`` on the
copy, so the re-freeze happens for free and does not need its own code path.

Because the timestep is a month and the tax year is a year, :class:`IncomeLedger`
exists to bridge the two: year-to-date income accumulates over twelve monthly
steps before the December close assesses it once, and what that assessment
still owes sits on :attr:`PersonState.balance_owing` until the filing month.

**Month indexes may be negative.** A month index counts months from January
of the scenario's start year, and a negative one is not an error: it means
the event it describes predates the run and was already under way when the
run opened — a pension already in payment, a bridge that already ended, a
subscriber already partway through a programme, an employment band that
started years before the household became a scenario. Every field typed as a
month index inherits this without restating it, for example
:attr:`PensionState.start_month_index`, :attr:`PensionState
.bridge_end_month_index`, :attr:`RespState.education_start_month_index`, and
:attr:`EmploymentBand.from_month_index` / :attr:`EmploymentBand
.to_month_index`. :attr:`PersonState.death_month_index` is the one exception:
it is guarded by its own sentinel, :data:`DEATH_NOT_DRAWN` — the maximum
``int64`` — rather than by a negative value, because a death cannot precede
the run that will draw it, and because a negative index is this module's
convention for a different thing entirely (see :data:`DEATH_NOT_DRAWN`'s own
comment).

Field lists here are deliberately minimal and grow as the modules that need
them land, in the order the README's build order gives.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

#: Sentinel for a death that has not been drawn yet.
#:
#: The largest representable ``int64``, not a small or negative number.
#: Decision B makes a negative month index this module's *ordinary* meaning
#: for "already under way when the run opened", so a small sentinel like
#: ``-1`` would no longer read as nonsense — it would read as a real, if
#: implausible, December-before-the-run death. What used to justify a small
#: sentinel — "a huge one reads as alive for the whole run" — is retired by
#: :attr:`PersonState.alive`, which answers "is this person dead" on its
#: own; this sentinel only has to avoid colliding with a real month index,
#: and a value no calendar this run's month indexes could ever reach does
#: that without needing to look small or look negative.
#:
#: Never do arithmetic on this value. ``death_month_index - month_index``
#: silently overflows ``int64`` for a path that has not died — NumPy wraps
#: rather than raising. Compare against :data:`DEATH_NOT_DRAWN` directly, or
#: read ``alive``, which is what it exists for.
DEATH_NOT_DRAWN: Final[int] = int(np.iinfo(np.int64).max)


def freeze(array: NDArray) -> NDArray:
    """Mark ``array`` non-writeable in place and return it.

    The one routine in this module that flips ``flags.writeable``. Every
    dataclass below calls this, once per array field, from its own
    ``__post_init__`` — never anywhere else — so there is exactly one place
    that decides an array has become state and stops being scratch space.

    Idempotent: freezing an already-frozen array is a no-op rather than an
    error, because :func:`dataclasses.replace` re-runs ``__post_init__`` on
    every field of a copy, including the ones that were not named in the
    change and were already read-only.

    Setting ``array.flags.writeable = False`` protects exactly one array
    object. It does nothing to ``array.base``: a column slice of a bigger
    buffer (``big[0, :]``) shares memory with ``big``, and freezing the
    slice leaves ``big`` itself writeable, so ``big[0, 0] = 1e9`` mutates the
    "frozen" state with no error at all. ``engine.mc.simulate`` produces
    ``real_returns`` shaped ``(n_assets, n_paths)`` and hands slices of it
    downstream, so this is not a hypothetical: it is the shape of the next
    caller. This walks the whole ``.base`` chain — a view of a view of a
    view is possible (``a[0:2][0:1]``) and one link is not enough — and
    refuses a writeable one rather than silently failing to protect it.

    **This guard is one-directional, and there is no cheap fix.** It catches
    a view handed to state *as* the array: ``freeze`` sees ``array.base`` and
    can refuse it. It cannot catch a view taken *of* an array that was
    already handed to state whole: ``big = np.zeros(10); col = big[0:4];
    CashState(balance=big)`` freezes ``big`` itself (``big.base`` is
    ``None``, so nothing here objects), but ``col`` was created before that
    call and stays writeable — NumPy does not retroactively mark existing
    views read-only when their base becomes read-only, only views taken
    afterward inherit it. ``col[0] = 999`` then mutates the "frozen" state
    with no error, the same failure this function exists to prevent, from
    the opposite direction. Say it plainly: this protects against handing
    state a view of a live buffer, not against handing state a buffer you
    kept a view of. There is nothing in ``array`` at the point ``freeze``
    sees it that distinguishes "nobody else holds a reference" from "a
    reference was taken and is still live" — both are one array with
    ``base is None`` — so no check here can close this side.

    Args:
        array: A NumPy array, of any dtype.

    Returns:
        The same array object, non-writeable.

    Raises:
        AssertionError: If ``array`` is not one-dimensional. Every per-path
            field here is a flat ``(n_paths,)`` array; a caller building a
            ``(1, n_paths)`` or ``(n_persons, n_paths)`` array has the wrong
            shape convention for this module, not merely an oversized one.
        ValueError: If ``array`` is a view onto a buffer that is still
            writeable through some other reference. Freezing the view cannot
            fix this; the caller must pass a copy (``array.copy()``) instead.
            This over-rejects as well as under-protects: a view of a
            freshly built, otherwise-unreferenced temporary —
            ``np.zeros((1, n)).reshape(n)``, ``np.squeeze(...)``,
            ``np.broadcast_to(...)`` — carries a ``.base`` too and is
            refused exactly like a view anyone could still reach, even
            though nothing else can reach this one. The remedy is the same
            ``.copy()``.
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
    """Return a copy of ``obj`` with ``changes`` applied, frozen like the original.

    A thin wrapper: ``dataclasses.replace`` builds the copy, and building it
    re-runs the class's own ``__post_init__`` on every field — the changed
    ones and the untouched ones alike — which is what re-freezes the arrays
    and, for :class:`HouseholdState`, re-checks that everything below it
    still agrees with ``n_paths``. Nothing here duplicates that logic; it
    relies on every class in this module doing its own freezing correctly.

    Args:
        obj: One of the frozen dataclasses defined in this module.
        changes: Field name to new value, exactly as ``dataclasses.replace``
            takes them.

    Returns:
        A new, frozen instance of ``type(obj)``. "New" describes the
        instance, not necessarily every array reachable from it: a field
        ``changes`` did not name keeps the same array object the original
        holds, shared between the two. That sharing is safe rather than a
        leak, because both instances hold it read-only — it is not a copy
        one of them could go on to mutate out from under the other.
    """
    return dataclasses.replace(obj, **changes)


def _freeze_fields(obj: object, *names: str) -> None:
    """Freeze the named array fields of ``obj`` in place, via :func:`freeze`.

    Loop plumbing only — the read-only marking still happens nowhere but
    :func:`freeze`. Every ``__post_init__`` below calls this once, naming
    exactly its array fields, so a field a class forgets to list here is a
    field that stays writeable, which is the failure mode the walker test
    exists to catch.
    """
    for name in names:
        object.__setattr__(obj, name, freeze(getattr(obj, name)))


def _iter_arrays(value: object) -> Iterator[NDArray]:
    """Yield every NumPy array reachable from ``value``, at any depth.

    Recurses into dataclass fields and into tuple elements; stops at anything
    else, which in this module means a plain scalar (``int``, ``float``,
    ``str``, ``bool``, or ``None``) or a leaf array itself. Used only by
    :meth:`HouseholdState.__post_init__` to check every array in the tree
    against ``n_paths`` in one pass, without a second, hand-maintained list of
    "everywhere an array might be".
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
        room: Unused contribution room, ``(n_paths,)``.
        contributed_ytd: Contributions made so far this calendar year,
            ``(n_paths,)``. Reset in January.
        converted_fraction_applied: Whether the scenario's RRIF-conversion
            election has already moved money out of this account. A person
            partially converts once; this flag is what stops a second
            conversion from firing on the same path.
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
        annual_minimum: The statutory minimum still to be withdrawn this
            calendar year, fixed in January from the 1 January balance and
            drawn down over the months that follow, ``(n_paths,)``.
        withdrawn_ytd: Withdrawn so far this calendar year, ``(n_paths,)``.
            Reset in January.
        opened_year: The calendar year the plan was opened, or ``None`` when
            the opening balance is zero — there is nothing to have opened,
            enforced in ``__post_init__``. The only question ever asked of
            this field is "was it opened in the current year", which decides
            whether the first year's minimum is exempt (a plan opened
            partway through a year owes no minimum until the following
            January). For a plan with a nonzero opening balance, the exact
            year it was opened is not knowable from a scenario, so this is
            set to ``scenario.start_year - 1`` — the year *before* the run,
            never the start year itself. Setting it to the start year would
            wrongly exempt the first simulated year from the minimum, which
            flatters the plan.
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
class LockedInState:
    """A locked-in account: a LIRA, and the LIF it becomes.

    Governed by the pension legislation of the jurisdiction the originating
    pension was registered in — not by where the household lives now. See
    ``docs/limitations.md`` L3.

    Attributes:
        balance: Real dollars, ``(n_paths,)``.
        is_lif: Whether the account has converted from a LIRA to a LIF. A
            LIRA takes no withdrawals at all; a LIF has both a minimum and,
            in most jurisdictions, a maximum.
        jurisdiction: Two-letter code for the pension jurisdiction of
            registration, e.g. ``"ab"``. Empty string, never a fallback to
            the province of residence, when there is no locked-in money —
            see ``docs/limitations.md`` L3 and ``engine/accounts/lira.py``,
            both of which turn on that substitution being impossible.
        annual_minimum: This year's RRIF-equivalent minimum, fixed in
            January, ``(n_paths,)``.
        annual_maximum: This year's jurisdiction-specific maximum
            withdrawal, fixed in January, ``(n_paths,)``. Only meaningful
            once ``is_lif`` is true and the jurisdiction imposes one.
        withdrawn_ytd: Withdrawn so far this calendar year, ``(n_paths,)``.
    """

    balance: NDArray[np.float64]
    is_lif: bool
    jurisdiction: str
    annual_minimum: NDArray[np.float64]
    annual_maximum: NDArray[np.float64]
    withdrawn_ytd: NDArray[np.float64]

    def __post_init__(self) -> None:
        _freeze_fields(self, "balance", "annual_minimum", "annual_maximum", "withdrawn_ytd")
        if self.jurisdiction == "" and not np.all(self.balance == 0.0):
            raise ValueError(
                "jurisdiction is '' (no locked-in account registered) but "
                "balance is nonzero on at least one path. An empty "
                "jurisdiction is only ever valid for a household with no "
                "locked-in money at all — falling back to the province of "
                "residence is exactly the mistake docs/limitations.md L3 "
                "exists to prevent."
            )


@dataclass(frozen=True, slots=True)
class TfsaState:
    """One person's TFSA.

    Attributes:
        balance: Real dollars, ``(n_paths,)``.
        room: Unused contribution room, including room restored from earlier
            withdrawals, ``(n_paths,)``.
        withdrawn_this_year: Withdrawn so far this calendar year,
            ``(n_paths,)``. Restored to room the following January, not this
            one — a TFSA withdrawal does not free up room until the year
            after it happens.
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
        balance: Market value, real dollars, ``(n_paths,)``.
        acb: Adjusted cost base, same dollars, ``(n_paths,)``. May exceed
            ``balance``: a holding standing at a loss is an ordinary
            position, not an error.
    """

    balance: NDArray[np.float64]
    acb: NDArray[np.float64]

    def __post_init__(self) -> None:
        _freeze_fields(self, "balance", "acb")


@dataclass(frozen=True, slots=True)
class EmploymentBand:
    """One band of employment income, resolved to a monthly figure.

    Attributes:
        from_month_index: Month index the band starts in, inclusive —
            January of the scenario's ``from_year``. May be negative; see
            the module docstring's convention on month indexes. A band
            lying entirely before the run (``to_month_index`` also
            negative) is still carried rather than dropped: discarding
            scenario data silently is worse than carrying a band no month
            of the run will ever match.
        to_month_index: Month index the band ends in, inclusive — December
            of the scenario's ``to_year``, matching that field's own
            inclusive convention.
        monthly_amount: Real dollars a month while the band is in force,
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

    The same shape serves both, because both are "a monthly amount that
    either starts on an elected age or is already flowing", and nothing else
    about the two differs at the level of state.

    Attributes:
        start_age_months: Age in whole months the benefit is elected to
            start at, or ``None`` when there is no election left to make —
            the only case today is a person whose CPP is already in pay,
            recorded through ``in_pay_monthly`` instead.
        in_pay_monthly: The known cheque for a benefit already being
            received, ``(n_paths,)``, or ``None`` when the benefit has not
            started and will be computed from ``start_age_months`` once it
            does.
        contributory_history: Fraction of the maximum CPP pension earned,
            ``[0, 1]``, the other input ``engine.benefits.cpp
            .pension_monthly`` needs alongside ``start_age_months``. Always
            ``None`` for OAS, which has no such input. At most one of this
            and ``in_pay_monthly`` is set — enforced in ``__post_init__``,
            since this class serves OAS too, where both are correctly
            ``None`` and "exactly one" would be false. For CPP exactly one
            *is* set, which ``engine.core.build`` enforces (mirroring
            ``engine.scenario.schema.CppEntitlement``'s own rule): a person
            already receiving CPP has no earnings fraction left to apply a
            start-age adjustment to, so the cheque is the only figure left
            to carry.
        monthly_amount: The amount actually in pay this month, ``(n_paths,)``.
            Zero until the benefit starts; computed by the benefit modules,
            not by this class.
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
        name: Label, unique within the person, used to tell two pensions
            apart in output.
        monthly_amount: The base pension, real dollars a month,
            ``(n_paths,)``. Constant once in pay for a fully indexed
            pension (the indexation cost is carried separately, in
            ``engine.core.indexation``); decaying in real terms for one
            that is not.
        start_month_index: Month index, counted from January of the
            scenario's start year, the pension begins.
        indexed: Whether the pension moves with CPI. ``False`` means fixed
            in nominal terms and losing real value every month it is in
            pay.
        bridge_monthly: A bridge benefit paid on top through
            ``bridge_end_month_index``, real dollars a month,
            ``(n_paths,)``. Zero when there is no bridge.
        bridge_end_month_index: The *last* month the bridge is paid,
            inclusive, or ``None`` when there is no bridge at all. May be
            negative, per the module's month-index convention, when the
            bridge already ended before the run opened. A bridge whose end
            falls before the pension starts would never pay a cent, and is
            refused by ``engine.scenario.schema.Person``, so this field is
            never built for one — refused at ``load_scenario``, not here,
            because it is a statement about a well-formed scenario rather
            than about a state. A bridge ending in the *same* month the
            pension starts is legal and pays for that one month.
        survivor_share: Fraction of the pension the survivor continues to
            receive after the member's death, ``[0, 1]``. Not per-path: a
            product term fixed by the pension's own rules, not something
            that varies by simulated draw.
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


@dataclass(frozen=True, slots=True)
class IncomeLedger:
    """One person's income components, accumulated year to date.

    Not a month's income: these are running totals for the current calendar
    year, added to by each monthly step and consumed once by the December
    assessment. January resets them. A field read mid-year is the income
    *so far*, which is what a policy is entitled to know — never the year's
    total, which is not knowable until December.

    Kept as components rather than one total because the tax treatment
    differs: only some of it is eligible for pension income splitting, only
    some counts toward the OAS recovery tax, dividends and capital gains
    enter taxable income at their own inclusion rates, and CPP and EI
    premiums feed a tax credit rather than income at all.

    Every field is real dollars, ``(n_paths,)``.

    Attributes:
        employment: Salary and wages.
        cpp: CPP retirement pension received.
        oas: Gross OAS received, before the recovery tax withheld from it.
        db_pension: Defined-benefit pension income, including any bridge.
        rrsp_withdrawals: Withdrawals from an RRSP that has not converted.
        rrif_lif_withdrawals: Withdrawals from a RRIF or a LIF.
        interest: Interest income from taxable holdings.
        eligible_dividends: Eligible dividends from taxable holdings, before
            gross-up.
        capital_gains: Realized capital gains from taxable holdings, before
            the inclusion rate.
        resp_accumulated_income: Accumulated income payments received from
            an RESP wind-up.
        rrsp_deductions: RRSP contributions made this year, an above-the-line
            deduction rather than income.
        cpp_base_contributions: Base CPP contributions made this year, which
            feed a credit.
        cpp_enhanced_contributions: Enhanced CPP contributions made this
            year, which feed a deduction rather than a credit.
        ei_premiums: EI premiums paid this year, which feed a credit.
        remitted: Tax already withheld and remitted this year, reducing the
            balance the December assessment leaves owing.
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

    A household is a list of these from day one, even when only one person
    is modelled, because pension splitting, survivor benefits, the RRIF
    spousal rollover, OAS ceasing at first death, and two mortality
    timelines all require the second slot to exist.

    Attributes:
        person_id: Stable identifier, unique within the household.
        sex: ``"f"`` or ``"m"``, selecting the life table. A mortality
            input, not a demographic statement.
        birth_year: Calendar year of birth.
        birth_month: Month of birth, ``1..12``.
        alive: Per-path survival flag, ``(n_paths,)``. Once false it stays
            false.
        death_month_index: Month index the person died in, ``(n_paths,)``,
            dtype ``int64``, or :data:`DEATH_NOT_DRAWN` on a path where
            death has not yet happened (or has not yet been drawn at all).
        cash: This person's share of... no — cash is household-level in
            spirit but, like every account here, tracked per person because
            two people's accounts do not merge while both are alive.
        rrsp: RRSP standing.
        rrif: RRIF standing.
        locked_in: LIRA/LIF standing.
        tfsa: TFSA standing.
        taxable: Non-registered holding.
        cpp: CPP standing.
        oas: OAS standing.
        employment: Bands of employment income, in scenario order, possibly
            empty. Carried in full even where a band lies wholly before or
            after the run; see :class:`EmploymentBand`.
        pensions: Defined-benefit pensions, in scenario order, possibly
            empty.
        income: Year-to-date income components.
        balance_owing: Assessed tax for the *prior* year not yet paid,
            ``(n_paths,)``. Created by the December close, discharged in the
            filing month, zero in between only if withholding happened to
            be exact.
        prior_year_net_income: Net income for the calendar year that governs
            the OAS recovery tax (and, eventually, other income-tested
            benefits) for the benefit months currently in force,
            ``(n_paths,)``. See ``docs/limitations.md`` L46 for what this is
            at the opening of a run.
    """

    person_id: str
    sex: str
    birth_year: int
    birth_month: int
    alive: NDArray[np.bool_]
    death_month_index: NDArray[np.int64]
    cash: CashState
    rrsp: RrspState
    rrif: RrifState
    locked_in: LockedInState
    tfsa: TfsaState
    taxable: TaxableState
    cpp: BenefitState
    oas: BenefitState
    employment: tuple[EmploymentBand, ...]
    pensions: tuple[PensionState, ...]
    income: IncomeLedger
    balance_owing: NDArray[np.float64]
    prior_year_net_income: NDArray[np.float64]

    def __post_init__(self) -> None:
        _freeze_fields(self, "alive", "death_month_index", "balance_owing")
        _freeze_fields(self, "prior_year_net_income")
        if np.any(~self.alive & (self.death_month_index == DEATH_NOT_DRAWN)):
            raise ValueError(
                "at least one path has alive=False but death_month_index is "
                "still DEATH_NOT_DRAWN; dead with no recorded death month is "
                "a contradiction."
            )
        # This is only half of the invariant. The other half — alive is
        # exactly (death_month_index > month_index) — needs a month_index,
        # which this class does not carry, and needs a decision nobody has
        # made yet: a state is the *opening* position for its month, and
        # deaths for that month are resolved inside advance_month's step 2,
        # so whether a death drawn for month k means dead at the open of
        # month k is issue 19's call, not this constructor's. Do not add
        # that half here "for now" — it belongs wherever issue 19 settles
        # the question, with month_index in hand to check it against.

    def age_months(self, year: int, month: int) -> int:
        """Age in whole months at the start of ``(year, month)``.

        Delegates to ``engine.core.timeline.age_in_months``. Age is derived
        from the birth date on every call and never stored, so the two
        cannot drift apart.
        """
        from engine.core import timeline

        return timeline.age_in_months(self.birth_year, self.birth_month, year, month)


@dataclass(frozen=True, slots=True)
class RespState:
    """The RESP standing to one beneficiary, split into its three buckets.

    The split is not bookkeeping: a withdrawal is taxed by which bucket it
    comes out of. Contributions come out tax-free, grants and accumulated
    income are taxable to the student, and grants are clawed back if the
    plan winds up without one.

    Attributes:
        contributions: Contributions bucket, real dollars, ``(n_paths,)``.
        grants: Grant bucket, real dollars, ``(n_paths,)``.
        income: Accumulated-income bucket, real dollars, ``(n_paths,)``.
        contributions_lifetime: Total ever contributed, for the lifetime
            contribution ceiling, ``(n_paths,)``. Equal to ``contributions``
            at the opening of a run — nothing has been withdrawn yet — and
            diverges from it once a withdrawal draws the bucket down without
            reducing the lifetime figure the ceiling is checked against.
        grants_lifetime: Total grant ever received, for the lifetime grant
            maximum, ``(n_paths,)``. Equal to ``grants`` at the opening for
            the same reason.
        grant_room: Unused grant-eligible contribution room carried into the
            run, ``(n_paths,)``.
        grant_received_ytd: Grant received so far this calendar year,
            ``(n_paths,)``. Reset in January.
        contributed_ytd: Contributed so far this calendar year,
            ``(n_paths,)``. Reset in January.
        subscriber_index: Index into ``HouseholdState.persons`` of the
            subscriber who owns the plan.
        education_start_month_index: Month index enrolment begins.
        education_months: Length of the programme, in months.
        education_monthly_cost: Real dollars a month while enrolled, spent
            whether or not the RESP covers it. Not per-path: a household
            input, not a simulated quantity.
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
    """An RESP beneficiary.

    Tracked individually and never pooled: grant room, the lifetime
    contribution limit, and the withdrawal window are all per-beneficiary
    and do not aggregate across children.

    Attributes:
        beneficiary_id: Stable identifier, unique within the household.
        birth_year: Calendar year of birth.
        birth_month: Month of birth, ``1..12``.
        resp: The plan standing to them.
    """

    beneficiary_id: str
    birth_year: int
    birth_month: int
    resp: RespState


@dataclass(frozen=True, slots=True)
class Elections:
    """The dated choices a policy makes, set once from the policy by the simulator.

    Named to match ``engine.scenario.schema.ElectionsSpec``, which is the same
    choices *by person id*, as written in a scenario file. This is the
    runtime form, addressed *by person index* instead — the position in
    ``HouseholdState.persons`` — because state carries no name-to-index
    mapping and every other per-person field here is already positional.

    Attributes:
        cpp_start_age_months: Age in whole months each person starts CPP at,
            indexed the same way as ``HouseholdState.persons``. An entry is
            ``None`` exactly when that person's CPP has no election left to
            make because it is already in pay — see
            :attr:`BenefitState.start_age_months`, which this is copied from
            per person. Typed ``tuple[int | None, ...]`` rather than the
            issue's plain ``tuple[int, ...]`` for that reason: there is no
            election age to put in a slot that has nothing to elect.
        oas_start_age_months: Age in whole months each person starts OAS at,
            indexed the same way. OAS has no "already in pay" input in the
            scenario schema, so every entry is populated in practice; the
            type stays ``int | None`` to match ``cpp_start_age_months`` and
            because nothing about the field name promises otherwise.
        rrif_conversion_age_years: Age the RRSP-to-RRIF conversion happens
            at. Household-wide, not per person: one policy names one age.
        rrif_conversion_fraction: Share of the RRSP converted, ``[0, 1]``.
            Household-wide for the same reason.
        fill_pension_credit: Whether to draw enough eligible pension income
            to use the pension income credit. Household-wide: a single
            withdrawal-order decision, not one made per person.
    """

    cpp_start_age_months: tuple[int | None, ...]
    oas_start_age_months: tuple[int | None, ...]
    rrif_conversion_age_years: int
    rrif_conversion_fraction: float
    fill_pension_credit: bool


@dataclass(frozen=True, slots=True)
class SpendingLevel:
    """One step of the household spending schedule, resolved to a monthly figure.

    Named ``SpendingLevel`` and deliberately not ``SpendingBand``:
    ``engine.scenario.schema.SpendingBand`` already owns that name and states
    an *annual* figure, and ``tests/core/test_build.py`` imports both side by
    side to compare them — the two must not collide.

    Attributes:
        from_year: First calendar year this level applies in.
        monthly_level: Real dollars for one month at full household share —
            the scenario's ``annual`` divided by twelve. Named
            ``monthly_level`` rather than ``monthly_amount`` because that
            name is already taken by the *per-path* kind, on
            :class:`EmploymentBand` and :class:`PensionState`. Those stay
            arrays — a salary or a pension becomes per-path the moment
            deaths start landing in different months on different paths —
            and this is a household-wide figure with nothing to broadcast
            over, deliberately a bare ``float`` rather than a same-named
            array that would happen to broadcast against them.
    """

    from_year: int
    monthly_level: float


def select_spending_level(schedule: tuple[SpendingLevel, ...], year: int) -> float:
    """The monthly figure in force for ``year``: the latest level starting at or before it.

    One selection rule, read from both ends of its own honour system:
    ``engine.core.build`` calls this to compute ``HouseholdState
    .spending_monthly`` in the first place, and :meth:`HouseholdState
    .__post_init__` calls it again to check that field was not set out of
    step with ``spending_schedule`` and ``year`` — by a January rollover, for
    instance, that updated ``year`` without recomputing ``spending_monthly``
    to match. Both sides computing the *same* selection, rather than one
    computing it and the other guessing, is what makes that check exact
    rather than approximate.

    Raises:
        ValueError: If ``schedule`` is empty, or if no level's ``from_year``
            is at or before ``year``. A schedule built by ``engine.core
            .build`` cannot hit the second case — the scenario schema
            requires the first band to cover the start year, and ``year``
            only advances from there — but this function makes no
            assumption about where its ``schedule`` came from.
    """
    if not schedule:
        raise ValueError("spending_schedule is empty; there is no level to select.")
    candidates = [level for level in schedule if level.from_year <= year]
    if not candidates:
        earliest = min(level.from_year for level in schedule)
        raise ValueError(
            f"no level in spending_schedule has from_year <= {year}; the "
            f"earliest is {earliest}."
        )
    return max(candidates, key=lambda level: level.from_year).monthly_level


@dataclass(frozen=True, slots=True)
class YearRecord:
    """One calendar year's summary, appended to :attr:`HouseholdState.history`.

    Not enumerated by the issue that introduced this class; the fields are
    exactly the per-path arrays that ``api/schemas.py::YearRow`` takes its
    percentiles from, because that is the only thing downstream that reads a
    year of history. Assembled once, at the December close.

    Attributes:
        year: The calendar year this record summarizes.
        net_worth: Household net worth at 31 December, ``(n_paths,)``.
        spending: Household spending over the year, ``(n_paths,)``.
        tax_assessed: Combined household tax assessed *for* this year at the
            December close — not what was paid in cash during it, which is
            mostly the prior year's balance settled in the filing month —
            ``(n_paths,)``.
        depleted: Whether the household ran out of money during the year,
            ``(n_paths,)``.
    """

    year: int
    net_worth: NDArray[np.float64]
    spending: NDArray[np.float64]
    tax_assessed: NDArray[np.float64]
    depleted: NDArray[np.bool_]

    def __post_init__(self) -> None:
        _freeze_fields(self, "net_worth", "spending", "tax_assessed", "depleted")


@dataclass(frozen=True, slots=True)
class HouseholdState:
    """The full state of the household at the boundary between two months.

    Returned by the monthly step and fed straight back into it. A policy
    function receives this and may read all of it — everything in here is
    knowable at that simulated moment. It may not read anything else, and in
    particular may not read this year's total income, which is not known
    until December.

    Attributes:
        year: The calendar year this state is the opening position for.
        month: The month this state is the opening position for, ``1..12``.
            January is 1.
        month_index: Months elapsed since January of the scenario's start
            year. Zero at the opening position.
        n_paths: Number of Monte Carlo paths every array in this tree must
            agree on. Checked, not merely declared: see
            :meth:`__post_init__`.
        province: Two-letter code for the province of residence, selecting
            the provincial parameter file for income tax. Not the
            jurisdiction a locked-in account is governed by; see
            :class:`LockedInState`.
        persons: One or two adults, in scenario order. Every per-person
            tuple downstream is indexed by position here.
        beneficiaries: RESP beneficiaries, in scenario order, possibly none.
        elections: The dated choices in force for this run.
        spending_schedule: The full household spending schedule, in scenario
            order. Carried so a January recomputation of
            ``spending_monthly`` never needs to go back to the scenario it
            was built from.
        spending_monthly: Household spending for one month at full share,
            real dollars, for the current ``year``/``month``. **Derived, not
            independent, and checked:** it must equal
            ``select_spending_level(spending_schedule, year)``, enforced in
            ``__post_init__`` rather than left on the honour system — unlike
            the other invariants here, a violation of this one is a wrong
            dollar amount, not a crash, and the wrong dollar amount is a
            plausible one: ``updated(state, year=state.year + 1)`` at a
            December rollover, without a matching ``spending_monthly=``,
            would otherwise carry the outgoing year's level silently into
            the next. The builder computes the opening value with the same
            function this checks against, so the comparison is exact. A
            call site wanting a different month's figure reads
            ``spending_schedule`` directly rather than setting this field
            out of step with ``year``. Not per-path: a schedule input, not a
            simulated quantity.

            **Recomputed by the year roll**, which is step 12 of
            ``engine.core.step.advance_month`` and not ``open_year``: the
            year changes as December's state is built, one call before
            January's phases run, so a December step that rolled ``year``
            without this could not construct its own return value.
        spending_survivor_share: Fraction of ``spending_monthly`` the
            survivor continues from the month after the first death.
        spending_achieved_ytd: What the household has actually spent so far
            this calendar year, ``(n_paths,)``. May fall short of the target
            on a path that has run out of money.
        depleted: Whether the household has run out of money, ``(n_paths,)``.
            Once true it stays true.
        estate_after_tax: Value of the estate after tax, ``(n_paths,)``. NaN,
            never zero, until the second death: zero is a real estate value
            and would be indistinguishable from an estate that is genuinely
            worth nothing, whereas NaN can only mean "not yet settled".
        history: Appended once a year, at the December close. Append-only.
    """

    year: int
    month: int
    month_index: int
    n_paths: int
    province: str
    persons: tuple[PersonState, ...]
    beneficiaries: tuple[BeneficiaryState, ...]
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

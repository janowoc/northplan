# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The scenario: a household, its assumptions, and the policies to evaluate.

A scenario is the *input* to a run, and everything in it is a statement by the
person writing the file. Nothing here is a tax rule — those live in ``params/``
and are read through :mod:`engine.params.loader` — and nothing here is engine
state. :mod:`engine.core.build` turns one of these into the opening
:class:`~engine.core.state.HouseholdState`; this module only says what a
well-formed scenario is and refuses one that is not.

Every model is ``frozen``, forbids unknown keys, and every mapping is a
read-only view: a scenario is loaded once and shared across every Monte Carlo
path and every policy the optimizer evaluates, so a mutation or a silently
ignored misspelled key would corrupt a whole run without a visible error.

**Dollars are real dollars of January of ``start_year``**, the same convention
the engine holds everywhere, except a person's ``prior_year_net_income``, taken
as filed, not restated.

**Amounts are what the household has, not what the rules allow.** Contribution
room and prior-year net income are scenario inputs because they depend on a
filing history the model does not have. The schema checks that a number is
non-negative and self-consistent; it never checks a number against a statutory
limit, since those live in ``params/`` for a particular year.

What this module deliberately does not validate:

- **Statutory ranges for elections**, e.g. ``cpp_start_age_years: 55`` loads:
  the legal window is a parameter belonging to a tax year, and checking it
  here would put a copy of it in a ``.py`` file.
  ``engine/scenario/start_ages.py::check_start_ages`` checks them, against
  the parameter year, once a scenario is paired with one.
- **That ``params/<start_year>/`` exists** — a scenario is a document; whether
  a parameter year has been transcribed is the loader's business.
- **That ``acb <= balance``** — a taxable holding at a loss is an ordinary
  position, not a typo. Only ``acb >= 0`` is required.
- **That an asset class pays out no more than it earns** —
  ``interest_yield + dividend_yield + distributed_gains_yield`` may exceed
  ``real_mean``; a class that distributes more than it returns is losing
  value, which portfolios do.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal, Self, TypeVar

import numpy as np
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from engine.mc.moments import covariance_from_correlation, monthly_log_moments

__all__ = [
    "CONTRIBUTION_KINDS",
    "DEFAULT_ALLOCATION",
    "INVESTABLE_KINDS",
    "PSD_TOLERANCE",
    "TOLERANCE",
    "WITHDRAWAL_KINDS",
    "Accounts",
    "AssetClass",
    "Assumptions",
    "Beneficiary",
    "CashAccount",
    "ContributionRule",
    "CppEntitlement",
    "DbPension",
    "Education",
    "ElectionsSpec",
    "Employment",
    "Household",
    "LifAccount",
    "LiraAccount",
    "OasEntitlement",
    "Person",
    "PolicySpec",
    "Resp",
    "RrifAccount",
    "RrifConversion",
    "RrspAccount",
    "Scenario",
    "Spending",
    "SpendingBand",
    "TaxableAccount",
    "TfsaAccount",
    "WithdrawalRule",
    "resolve_policy_path",
]

#: Account kinds new money may be contributed to. ``cash`` is absent: it is the
#: hub every flow passes through, not a destination. ``rrif``, ``lira`` and
#: ``lif`` cannot receive a contribution at all.
CONTRIBUTION_KINDS: Final[frozenset[str]] = frozenset({"rrsp", "tfsa", "taxable", "resp"})

#: Account kinds a withdrawal order may name. ``resp`` is absent: drawn down by
#: the education window, not the withdrawal order. ``cash`` is absent since
#: spending comes out of it by construction. ``lira`` is absent: it takes no
#: withdrawals directly, only after becoming a LIF.
WITHDRAWAL_KINDS: Final[frozenset[str]] = frozenset({"rrsp", "rrif", "lif", "tfsa", "taxable"})

#: Account kinds that hold investments and therefore carry an asset allocation.
#:
#: Cash is excluded because it pays zero real return (limitations.md L38); an
#: allocation for it would be a number with nothing to multiply.
INVESTABLE_KINDS: Final[frozenset[str]] = frozenset(
    {"rrsp", "rrif", "lira", "lif", "tfsa", "taxable", "resp"}
)

#: The allocation every account without one of its own uses.
DEFAULT_ALLOCATION: Final[str] = "default"

#: Tolerance for sums and symmetry that a human writes out by hand.
#:
#: ``0.6 + 0.4`` is not 1.0 in binary floating point, and demanding exactness
#: would reject the example scenario in this repository.
TOLERANCE: Final[float] = 1e-9

#: How negative the smallest eigenvalue of the correlation matrix may be.
#: Looser than :data:`TOLERANCE` since it is an eigenvalue, not a sum: exact
#: positive semi-definiteness routinely produces an eigenvalue a few units in
#: the last place below zero.
PSD_TOLERANCE: Final[float] = 1e-8

_V = TypeVar("_V")


def _freeze[V](mapping: Mapping[str, V]) -> Mapping[str, V]:
    """Return a read-only view of ``mapping``.

    Pydantic's ``frozen`` protects the *fields* of a model, not the contents of
    a ``dict`` one of them holds. Without this, ``scenario.assumptions
    .allocations["default"]["equity"] = 1.0`` would succeed and change the
    portfolio under every path in flight.
    """
    return MappingProxyType(dict(mapping))


#: A mapping that cannot be written to after validation. Use as
#: ``FrozenMapping[float]``; nests as ``FrozenMapping[FrozenMapping[float]]``.
FrozenMapping = Annotated[Mapping[str, _V], AfterValidator(_freeze)]

#: A dollar amount that cannot be negative. Real dollars of January of the
#: scenario's start year, like every amount below except
#: ``Person.prior_year_net_income``.
Money = Annotated[float, Field(ge=0.0)]

#: A calendar month, January is 1.
Month = Annotated[int, Field(ge=1, le=12)]

#: A bare fraction in ``[0, 1]``, never a percentage.
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]

#: A two-letter jurisdiction code, e.g. ``"ab"``. Lower case, so that the
#: string can be handed to ``ParamYear.province`` and reach a file name
#: unchanged. Which codes actually have a parameter file is not checked here;
#: the loader stops on the ones that do not (limitations.md L1, L3).
JurisdictionCode = Annotated[str, Field(pattern=r"^[a-z]{2}$")]


class _Base(BaseModel):
    """Shared configuration: immutable, and no unknown keys anywhere."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Employment(_Base):
    """One band of employment income for one person.

    ``from_year`` and ``to_year`` are both **inclusive**: a band of 2026 to
    2031 pays in six calendar years. Stated because the alternative reading
    costs a year of salary and CPP contributions and produces a perfectly
    plausible answer.

    Attributes:
        from_year: First calendar year the income is earned in.
        to_year: Last calendar year the income is earned in, inclusive.
        annual: Real dollars per year, before tax.
    """

    from_year: int
    to_year: int
    annual: Money

    @model_validator(mode="after")
    def _check_years_ordered(self) -> Self:
        if self.to_year < self.from_year:
            raise ValueError(
                f"employment: to_year ({self.to_year}) is before from_year "
                f"({self.from_year}); the band would pay in no year at all."
            )
        return self


class CppEntitlement(_Base):
    """What the household knows about a person's CPP, one way or the other.

    Exactly one of these is given, because they are answers to two different
    questions and there is no scenario in which both are known:

    - ``contributory_history`` — the person has not started CPP, and this is
      the fraction of the maximum they are on course for. The engine turns it
      into an amount using the start age the policy elects.
    - ``in_pay_monthly`` — the person is already receiving CPP, and this is
      the gross monthly amount. The start-age election no longer applies to
      them.

    Attributes:
        contributory_history: Fraction of the maximum pension earned, ``[0,
            1]``. ``0.85`` is a career with some low-earning years.
        in_pay_monthly: Real dollars a month, already being received: the
            gross amount, before any income-tax withholding.
    """

    contributory_history: Fraction | None = None
    in_pay_monthly: Money | None = None

    @model_validator(mode="after")
    def _check_exactly_one(self) -> Self:
        given = [
            name
            for name, value in (
                ("contributory_history", self.contributory_history),
                ("in_pay_monthly", self.in_pay_monthly),
            )
            if value is not None
        ]
        if len(given) != 1:
            raise ValueError(
                "cpp: give exactly one of contributory_history (the person has "
                "not started CPP) or in_pay_monthly (they have), not "
                + (f"both ({', '.join(given)})" if given else "neither")
                + "."
            )
        return self


class DbPension(_Base):
    """A defined-benefit pension one person will receive or is receiving.

    Attributes:
        name: Label for the pension, unique within the person; used to tell
            two pensions apart in output.
        start_year: Calendar year the pension begins.
        start_month: Month within that year it begins, 1..12.
        annual: Real dollars a year at the start date.
        indexation: ``"full"`` if the pension moves with CPI (constant in real
            terms); ``"none"`` if fixed in nominal terms and eroding. Required,
            no default, since it is the largest lever on a DB pension's value.
            In YAML, ``none`` must be the *string* ``"none"`` — PyYAML reads a
            bare ``null``, ``~``, or empty value as nothing instead.
        bridge_annual: Real dollars a year of a bridge benefit paid on top,
            ending at ``bridge_to_age_years``. Zero if there is none.
        bridge_to_age_years: Age the bridge stops at, typically the age the
            public pension it bridges to begins.
        survivor_share: Fraction of the pension the survivor continues to
            receive after the member's death, ``[0, 1]`` (zero allowed, unlike
            :class:`Spending`'s ``(0, 1]``).
    """

    name: str
    start_year: int
    start_month: Month
    annual: Money
    indexation: Literal["none", "full"]
    bridge_annual: Money = 0.0
    bridge_to_age_years: int | None = None
    survivor_share: Fraction = 0.0

    @model_validator(mode="after")
    def _check_bridge_ends(self) -> Self:
        if self.bridge_annual > 0.0 and self.bridge_to_age_years is None:
            raise ValueError(
                f"db_pensions[{self.name!r}]: bridge_annual is "
                f"{self.bridge_annual} but bridge_to_age_years is absent, so "
                "the bridge would be paid for life. Give the age it stops at."
            )
        return self


class CashAccount(_Base):
    """Cash one person holds on 1 January of the start year.

    A per-person input: the builder sums every person's balance into the one
    household cash account, which every inflow and outflow passes through
    (limitations.md L38).

    Attributes:
        balance: Real dollars on 1 January of the start year.
    """

    balance: Money = 0.0


class RrspAccount(_Base):
    """An RRSP, before conversion to a RRIF.

    Attributes:
        balance: Real dollars on 1 January of the start year.
        room: The unused room available on 1 January of the start year after
            that year's grant, as a notice of assessment states it. A
            scenario input rather than a computed figure: it depends on a
            filing history the model does not have.
    """

    balance: Money = 0.0
    room: Money = 0.0


class RrifAccount(_Base):
    """A RRIF, whether converted already or opening empty.

    No ``room`` field: a RRIF takes no contributions.

    Attributes:
        balance: Real dollars on 1 January of the start year.
    """

    balance: Money = 0.0


class TfsaAccount(_Base):
    """A TFSA.

    Attributes:
        balance: Real dollars on 1 January of the start year.
        room: The unused room available on 1 January of the start year after
            that year's grant, including room restored from withdrawals in
            earlier years.
    """

    balance: Money = 0.0
    room: Money = 0.0


class TaxableAccount(_Base):
    """A non-registered account.

    ``acb`` is not required to be at or below ``balance``. A holding standing
    at a loss is an ordinary position, and rejecting it would reject a true
    statement about the household.

    Attributes:
        balance: Market value in real dollars on 1 January of the start year.
        acb: Adjusted cost base in the same dollars.
    """

    balance: Money = 0.0
    acb: Money = 0.0


def _check_jurisdiction_required_when_funded(
    kind: str, balance: float, jurisdiction: str | None
) -> None:
    """Shared by :class:`LiraAccount` and :class:`LifAccount`: same rule, same shape of message.

    ``kind`` is the field name the message begins with (``"lira"`` or
    ``"lif"``), so the two accounts' messages stay specific to their own kind
    rather than reading as a generic "locked-in" complaint.
    """
    if balance > 0.0 and jurisdiction is None:
        raise ValueError(
            f"{kind}: jurisdiction is required when balance is above zero "
            f"(got {balance}). A locked-in account is governed by the "
            "pension law it was registered under, which the province of "
            "residence does not imply."
        )


class LiraAccount(_Base):
    """A LIRA, before conversion to a LIF.

    Attributes:
        balance: Real dollars on 1 January of the start year.
        jurisdiction: The pension jurisdiction the account is **registered**
            in, which is not necessarily where the household lives. An Alberta
            resident may hold an Ontario-registered LIRA: once it becomes a
            LIF they draw it under Ontario's maximum table, and they file
            Alberta income tax. Required as soon as there is a balance,
            because there is no safe fallback — defaulting to the province of
            residence is right until it is silently wrong, and no test on a
            household that never moved would catch it (limitations.md L3).
    """

    balance: Money = 0.0
    jurisdiction: JurisdictionCode | None = None

    @model_validator(mode="after")
    def _check_jurisdiction_present(self) -> Self:
        _check_jurisdiction_required_when_funded("lira", self.balance, self.jurisdiction)
        return self


class LifAccount(_Base):
    """A LIF, for a person already converted.

    Attributes:
        balance: Real dollars on 1 January of the start year.
        jurisdiction: The pension jurisdiction the account is **registered**
            in, which is not necessarily where the household lives. An Alberta
            resident may hold an Ontario-registered LIF: they draw it under
            Ontario's maximum table and file Alberta income tax. Required as
            soon as there is a balance, because there is no safe fallback —
            defaulting to the province of residence is right until it is
            silently wrong, and no test on a household that never moved would
            catch it (limitations.md L3).
    """

    balance: Money = 0.0
    jurisdiction: JurisdictionCode | None = None

    @model_validator(mode="after")
    def _check_jurisdiction_present(self) -> Self:
        _check_jurisdiction_required_when_funded("lif", self.balance, self.jurisdiction)
        return self


class Accounts(_Base):
    """One person's accounts. Every kind is optional and opens empty.

    An absent account and one written as ``{balance: 0}`` mean the same thing,
    so a person with no locked-in money simply omits ``lira`` and ``lif``.

    Attributes:
        cash: Cash held, summed into the household's one cash account.
        rrsp: Registered retirement savings.
        rrif: Registered retirement income fund.
        tfsa: Tax-free savings account.
        taxable: Non-registered holdings.
        lira: Locked-in retirement account before conversion.
        lif: Life income fund, for a person already converted.
    """

    cash: CashAccount = Field(default_factory=CashAccount)
    rrsp: RrspAccount = Field(default_factory=RrspAccount)
    rrif: RrifAccount = Field(default_factory=RrifAccount)
    tfsa: TfsaAccount = Field(default_factory=TfsaAccount)
    taxable: TaxableAccount = Field(default_factory=TaxableAccount)
    lira: LiraAccount = Field(default_factory=LiraAccount)
    lif: LifAccount = Field(default_factory=LifAccount)

    @model_validator(mode="after")
    def _check_lira_and_lif_agree_on_jurisdiction(self) -> Self:
        lira_jurisdiction = self.lira.jurisdiction
        lif_jurisdiction = self.lif.jurisdiction
        if (
            lira_jurisdiction is not None
            and lif_jurisdiction is not None
            and lira_jurisdiction != lif_jurisdiction
        ):
            raise ValueError(
                f"accounts: lira.jurisdiction ({lira_jurisdiction!r}) and "
                f"lif.jurisdiction ({lif_jurisdiction!r}) disagree. A person "
                "carries at most one locked-in jurisdiction, sharing one "
                "jurisdiction between a LIRA and the LIF it becomes "
                "(limitations.md L47)."
            )
        return self


class OasEntitlement(_Base):
    """What the household knows about a person's OAS, when it is already in pay.

    Absent, or given without ``in_pay_monthly``, means the person has not
    started OAS — the common case, where the start-age election in
    :class:`ElectionsSpec` decides when it begins. When ``in_pay_monthly`` is
    given, the person is already receiving OAS, it is the gross monthly
    amount, and the start-age election no longer applies to them.

    Attributes:
        in_pay_monthly: Real dollars a month, already being received: the
            gross amount, before any recovery-tax or income-tax withholding.
    """

    in_pay_monthly: Money | None = None


class Person(_Base):
    """One adult in the household.

    Attributes:
        id: Stable identifier, unique across every person and beneficiary in the
            household. Elections and RESP subscriptions name it.
        birth_year: Calendar year of birth.
        birth_month: Month of birth, 1..12. Required, not optional: on a monthly
            timeline it decides the month a pension may start, and CPP's
            adjustment is defined per month away from 65.
        sex: ``"f"`` or ``"m"``, selecting the life table — a mortality input,
            not a demographic statement.
        employment: Bands of employment income, in any order, which must not
            overlap.
        cpp: What is known about their CPP.
        oas: What is known about their OAS, if it is already in pay.
        db_pensions: Defined-benefit pensions, possibly none.
        accounts: Opening balances.
        prior_year_net_income: Net income (line 23600) for the calendar year
            before ``start_year``, as reported on that year's return (an
            estimate if not yet filed) — not restated in start-year dollars,
            the one amount in a scenario that is not real dollars. Required,
            no default: zero is a legitimate value the author types.
    """

    id: str
    birth_year: int
    birth_month: Month
    sex: Literal["f", "m"]
    employment: tuple[Employment, ...] = ()
    cpp: CppEntitlement
    oas: OasEntitlement = Field(default_factory=OasEntitlement)
    db_pensions: tuple[DbPension, ...] = ()
    accounts: Accounts = Field(default_factory=Accounts)
    prior_year_net_income: Money

    @model_validator(mode="before")
    @classmethod
    def _check_prior_year_net_income_present(cls, data: Any) -> Any:
        """Reject a missing ``prior_year_net_income`` before field validation.

        Runs before field validation so the message can name the person by
        ``id``, which pydantic's own missing-field error does not. When this
        raises, pydantic reports no other error for this person, so their
        other mistakes surface only on the next load.
        """
        if isinstance(data, Mapping) and "prior_year_net_income" not in data:
            who = f"person {data['id']!r}" if "id" in data else "person (no id)"
            raise ValueError(
                f"{who}: prior_year_net_income is required and "
                "has no default; a scenario carries no filing history the "
                "model could otherwise derive it from. Zero is a legitimate "
                "value; write 0 if that is the figure."
            )
        return data

    @model_validator(mode="after")
    def _check_employment_does_not_overlap(self) -> Self:
        bands = sorted(self.employment, key=lambda band: band.from_year)
        for earlier, later in pairwise(bands):
            if later.from_year <= earlier.to_year:
                raise ValueError(
                    f"employment: person {self.id!r} has overlapping bands "
                    f"{earlier.from_year}-{earlier.to_year} and "
                    f"{later.from_year}-{later.to_year}. Both years are "
                    "inclusive, so a band ending in a year and the next one "
                    "starting in it would pay twice."
                )
        return self

    @model_validator(mode="after")
    def _check_pension_names_unique(self) -> Self:
        names = [pension.name for pension in self.db_pensions]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ValueError(
                f"db_pensions: person {self.id!r} has more than one pension "
                f"named {', '.join(repr(name) for name in duplicated)}. Names "
                "are how two pensions are told apart in output."
            )
        return self

    @model_validator(mode="after")
    def _check_bridges_do_not_end_before_they_start(self) -> Self:
        """A bridge that ends before its pension starts would never pay a cent.

        Lives on ``Person`` rather than on ``DbPension`` because the bridge's
        end date is an *age* — it needs this person's ``birth_year`` and
        ``birth_month`` to become a date at all, and ``DbPension`` has
        neither. The comparison is calendar dates, ``(year, month)`` tuples
        compared lexicographically, not engine month-index arithmetic: a
        scenario is a document with no opinion on what a Monte Carlo run's
        month zero is, and this check does not need one either.
        """
        for pension in self.db_pensions:
            if pension.bridge_annual <= 0.0:
                continue
            # bridge_to_age_years is required whenever bridge_annual > 0
            # (_check_bridge_ends, above), so this is never None here.
            assert pension.bridge_to_age_years is not None
            bridge_ends = (self.birth_year + pension.bridge_to_age_years, self.birth_month)
            pension_starts = (pension.start_year, pension.start_month)
            if bridge_ends < pension_starts:
                raise ValueError(
                    f"db_pensions[{pension.name!r}]: person {self.id!r} turns "
                    f"{pension.bridge_to_age_years} in "
                    f"{bridge_ends[0]}-{bridge_ends[1]:02d}, ending the bridge "
                    "before the pension itself starts in "
                    f"{pension_starts[0]}-{pension_starts[1]:02d}; the bridge "
                    "would never pay a cent."
                )
        return self


class Resp(_Base):
    """The RESP standing to one beneficiary, split into its three buckets.

    The split is not bookkeeping: a withdrawal is taxed by which bucket it
    comes out of. Contributions come out tax-free, grants and accumulated
    income are taxable to the student, and grants are clawed back if the plan
    is wound up without one.

    Everything here is **per beneficiary**. Nothing in an RESP aggregates
    across children, and reading one of these as a plan-level figure is the
    error this program is most prone to.

    Attributes:
        subscriber: The ``id`` of the person who owns the plan. Must name a
            person in the household: the accumulated income payment on a
            wind-up is taxed to them, so the wrong id sends a tax bill to the
            wrong return.
        contributions: Contributions made to date, real dollars.
        grants: Grant received to date, real dollars.
        income: Accumulated income to date, real dollars.
        grant_room_carried: The unused grant room available on 1 January of
            the start year after that year's grant.
    """

    subscriber: str
    contributions: Money = 0.0
    grants: Money = 0.0
    income: Money = 0.0
    grant_room_carried: Money = 0.0


class Education(_Base):
    """When a beneficiary is in school and what it costs.

    Attributes:
        start_year: Calendar year enrolment begins.
        start_month: Month enrolment begins, 1..12. September, not January, for
            most programmes, and the RESP withdrawal rules key off the first
            months of enrolment.
        months: Length of the programme in months. At least one.
        annual_cost: Real dollars a year while enrolled. Spent whether or not
            the RESP covers it; a shortfall comes from the household.
    """

    start_year: int
    start_month: Month
    months: Annotated[int, Field(ge=1)]
    annual_cost: Money


class Beneficiary(_Base):
    """A child with an RESP and an education to pay for.

    Both blocks are required. A beneficiary with no plan and no schooling is
    not a thing the model does anything with, and accepting one would silently
    drop them out of every result.

    Attributes:
        id: Stable identifier, unique across every person and beneficiary.
        birth_year: Calendar year of birth; drives grant eligibility.
        birth_month: Month of birth, 1..12.
        resp: The plan standing to them.
        education: The window it is meant to pay for.
    """

    id: str
    birth_year: int
    birth_month: Month
    resp: Resp
    education: Education


class Household(_Base):
    """The people being simulated.

    Attributes:
        province: Two-letter code for the province of **residence**, selecting
            the provincial parameter file for income tax. Not the jurisdiction
            a locked-in account is governed by; see :class:`LiraAccount` and
            :class:`LifAccount`.
        persons: One or two adults, in a stable order. Every per-person array
            downstream is indexed by position here.
        beneficiaries: RESP beneficiaries, possibly none.
    """

    province: JurisdictionCode
    persons: tuple[Person, ...]
    beneficiaries: tuple[Beneficiary, ...] = ()

    @property
    def person_ids(self) -> tuple[str, ...]:
        """Every person's id, in scenario order."""
        return tuple(person.id for person in self.persons)

    @model_validator(mode="after")
    def _check_one_or_two_persons(self) -> Self:
        """One or two adults, and the count is checked here rather than on the field.

        A ``min_length`` on the tuple would be applied *after* item validation,
        so any error inside a person would be reported twice: once truthfully,
        and once as "should have at least 1 item, not 0" — which reads as
        though the file listed nobody. A model validator does not run at all
        when a person failed, leaving the one error that says what to fix.
        """
        if not 1 <= len(self.persons) <= 2:
            raise ValueError(
                f"household.persons: {len(self.persons)} people. The model "
                "carries one or two adults: pension splitting, the survivor "
                "pension, the spousal rollover and two mortality timelines all "
                "need the second slot to exist, and nothing in it knows what a "
                "third person would be."
            )
        return self

    @model_validator(mode="after")
    def _check_ids_unique(self) -> Self:
        ids = [person.id for person in self.persons]
        ids += [beneficiary.id for beneficiary in self.beneficiaries]
        duplicated = sorted({identifier for identifier in ids if ids.count(identifier) > 1})
        if duplicated:
            raise ValueError(
                f"household: id {', '.join(repr(i) for i in duplicated)} is used "
                "more than once. Person and beneficiary ids share one namespace "
                "because an RESP subscriber is named by id, and a beneficiary "
                "sharing a person's id would make that reference ambiguous."
            )
        return self

    @model_validator(mode="after")
    def _check_subscribers_exist(self) -> Self:
        known = set(self.person_ids)
        for beneficiary in self.beneficiaries:
            if beneficiary.resp.subscriber not in known:
                raise ValueError(
                    f"resp: subscriber {beneficiary.resp.subscriber!r} on "
                    f"beneficiary {beneficiary.id!r} is not a person in this "
                    f"household. Known: {', '.join(sorted(known))}."
                )
        return self


class SpendingBand(_Base):
    """One step of the household spending schedule.

    Open-ended: a band runs until the next one begins, and the last runs to the
    second death.

    Attributes:
        from_year: First calendar year this level applies in.
        annual: Real dollars a year for the household.
    """

    from_year: int
    annual: Money


class Spending(_Base):
    """What the household spends, and what changes when one of them dies.

    Attributes:
        schedule: Bands in strictly ascending ``from_year`` order, the first
            beginning at or before the scenario's start year so that no year of
            the run is left without a spending level.
        survivor_share: Fraction of household spending the survivor continues
            from the month after the first death. Strictly above zero — a
            survivor who spends nothing is not a plan, it is a modelling
            mistake — and at most one.
    """

    schedule: tuple[SpendingBand, ...]
    survivor_share: Annotated[float, Field(gt=0.0, le=1.0)]

    @model_validator(mode="after")
    def _check_schedule_is_not_empty(self) -> Self:
        """At least one band, checked here for the reason given on ``persons``."""
        if not self.schedule:
            raise ValueError(
                "spending.schedule: no bands, so the household spends nothing "
                "in any year of the run."
            )
        return self

    @model_validator(mode="after")
    def _check_schedule_ascending(self) -> Self:
        years = [band.from_year for band in self.schedule]
        if years != sorted(set(years)):
            raise ValueError(
                f"spending.schedule: from_year values {years} must be in "
                "strictly ascending order. A band takes effect until the next "
                "one begins, so order is meaning, not presentation, and a "
                "repeated year makes two bands claim the same year."
            )
        return self


class AssetClass(_Base):
    """One asset class: what it returns and how that return is taxed.

    The yields are the *tax character* of the return, not additions to it.
    Price change is the total return less the three yields, so a class with a
    5% real mean and a 2% dividend yield grows 3% in price and distributes 2%.
    Distributions are reinvested and raise the adjusted cost base.

    Attributes:
        real_mean: Expected real annual total return, a bare fraction. May be
            negative; cash-like classes have been for years at a time. Must
            be above -1.
        vol: Annual standard deviation of the real return. Zero is legal and
            means a riskless class.
        interest_yield: Fraction of balance a year distributed as interest,
            fully taxable.
        dividend_yield: Fraction distributed as eligible dividends, grossed up
            and credited.
        distributed_gains_yield: Fraction distributed as realized capital
            gains.
    """

    real_mean: float
    vol: Annotated[float, Field(ge=0.0)]
    interest_yield: Annotated[float, Field(ge=0.0)]
    dividend_yield: Annotated[float, Field(ge=0.0)]
    distributed_gains_yield: Annotated[float, Field(ge=0.0)]


class Assumptions(_Base):
    """The capital-market and inflation assumptions the run is driven by.

    **The correlation matrix is positional and its order is the order
    ``asset_classes`` is written in.** There is nothing in the matrix that
    names a class, so reordering the ``asset_classes`` block reinterprets every
    off-diagonal entry without changing a number. Read the ordering from
    :attr:`asset_class_names` and never from a literal at a call site.

    Attributes:
        inflation: Assumed annual inflation, a bare fraction. Constant for the
            run (limitations.md L6). Used for the indexation erosion factor and
            for the decay of amounts fixed in nominal terms — returns are real,
            so it enters nowhere else. Must be above -1: at or below it,
            ``(1 + inflation)`` raised to a fractional power is a complex
            number rather than an error.
        asset_classes: Named classes, at least one. Declaration order is the
            order of ``correlation``.
        correlation: Correlation matrix of real annual returns, square,
            symmetric, unit diagonal, positive semi-definite, and the same size
            as ``asset_classes``, and realisable, together with each class's
            ``real_mean`` and ``vol``, by a lognormal distribution.
        allocations: Portfolio weights by account kind, plus the required
            ``default`` used by any account without an entry of its own. Each
            allocation names known classes and sums to one.
    """

    inflation: Annotated[float, Field(gt=-1.0)]
    asset_classes: FrozenMapping[AssetClass]
    correlation: tuple[tuple[float, ...], ...]
    allocations: FrozenMapping[FrozenMapping[float]]

    @property
    def asset_class_names(self) -> tuple[str, ...]:
        """Class names in declaration order, which is ``correlation``'s order."""
        return tuple(self.asset_classes)

    @model_validator(mode="after")
    def _check_asset_classes_exist(self) -> Self:
        """At least one class, and first, so the correlation check has a size to want.

        Declared above ``_check_correlation`` on purpose: validators run in
        definition order, and an empty block would otherwise be reported as a
        correlation matrix of the wrong size for nothing.
        """
        if not self.asset_classes:
            raise ValueError(
                "assumptions.asset_classes: none declared, so there is nothing "
                "for an allocation to hold or for a return to be drawn for."
            )
        return self

    @model_validator(mode="after")
    def _check_correlation(self) -> Self:
        size = len(self.asset_classes)
        widths = {len(row) for row in self.correlation}
        if len(self.correlation) != size or widths != {size}:
            raise ValueError(
                f"assumptions.correlation: expected a {size}x{size} matrix for "
                f"asset classes {', '.join(self.asset_class_names)}, got "
                f"{len(self.correlation)} rows of widths {sorted(widths)}."
            )

        matrix = np.array(self.correlation, dtype=float)
        off_diagonal = np.abs(matrix - matrix.T).max()
        if off_diagonal > TOLERANCE:
            raise ValueError(
                f"assumptions.correlation: not symmetric; the largest gap "
                f"between an entry and its mirror is {off_diagonal:g}. A "
                "correlation is a property of a pair, so the two halves are "
                "the same number written twice."
            )

        diagonal_error = np.abs(np.diag(matrix) - 1.0).max()
        if diagonal_error > TOLERANCE:
            raise ValueError(
                f"assumptions.correlation: the diagonal must be all ones, and "
                f"is off by up to {diagonal_error:g}. An entry below one says a "
                "class is imperfectly correlated with itself."
            )

        smallest = float(np.linalg.eigvalsh(matrix).min())
        if smallest < -PSD_TOLERANCE:
            raise ValueError(
                f"assumptions.correlation: not positive semi-definite; its "
                f"smallest eigenvalue is {smallest:g}. No set of random "
                "variables has these correlations, so there is nothing to draw "
                "from — the draw would fail or, worse, be silently altered to "
                "the nearest matrix that works."
            )
        return self

    @model_validator(mode="after")
    def _check_moments_attainable(self) -> Self:
        """Refuse assumptions no lognormal distribution realises.

        Checks ``1 + real_mean <= 0``, a non-positive moment-matching log
        argument, or a moment-matched monthly log-covariance that is not
        positive semi-definite -- naming the asset class(es) involved.
        Nothing is clipped or nudged to the nearest matrix that works.

        The scenario package imports :mod:`engine.mc.moments` and nothing
        else from ``engine.mc``, and :mod:`engine.mc.moments` imports nothing
        from ``engine.scenario``, so the dependency runs one way only and
        there is no cycle; ``tests/test_layering.py`` enforces both.
        """
        names = self.asset_class_names
        classes = [self.asset_classes[name] for name in names]
        means = np.array([c.real_mean for c in classes], dtype=np.float64)
        vols = np.array([c.vol for c in classes], dtype=np.float64)
        covariance = covariance_from_correlation(vols, np.array(self.correlation, dtype=float))
        monthly_log_moments(means, covariance, names)
        return self

    @model_validator(mode="after")
    def _check_allocations(self) -> Self:
        if DEFAULT_ALLOCATION not in self.allocations:
            raise ValueError(
                f"assumptions.allocations: {DEFAULT_ALLOCATION!r} is required. "
                "It is what every account without an allocation of its own "
                "holds, so without it an account would have no portfolio."
            )

        permitted = INVESTABLE_KINDS | {DEFAULT_ALLOCATION}
        unknown = sorted(set(self.allocations) - permitted)
        if unknown:
            raise ValueError(
                f"assumptions.allocations: {', '.join(repr(k) for k in unknown)} "
                f"is not an account that holds investments. Use "
                f"{DEFAULT_ALLOCATION!r} or one of: "
                f"{', '.join(sorted(INVESTABLE_KINDS))}."
            )

        known = set(self.asset_classes)
        for account, weights in self.allocations.items():
            missing = sorted(set(weights) - known)
            if missing:
                raise ValueError(
                    f"assumptions.allocations[{account!r}]: no asset class "
                    f"named {', '.join(repr(name) for name in missing)}. "
                    f"Declared: {', '.join(self.asset_class_names)}."
                )
            total = sum(weights.values())
            if abs(total - 1.0) > TOLERANCE:
                raise ValueError(
                    f"assumptions.allocations[{account!r}]: weights sum to "
                    f"{total!r}, not 1. An account holds all of itself."
                )
            negative = sorted(name for name, weight in weights.items() if weight < 0.0)
            if negative:
                raise ValueError(
                    f"assumptions.allocations[{account!r}]: negative weight on "
                    f"{', '.join(repr(name) for name in negative)}. Shorting is "
                    "not modelled."
                )
        return self


class ContributionRule(_Base):
    """Where a policy puts money it has to invest.

    Attributes:
        weights: Share of each contribution by account kind, non-negative and
            summing to one. A kind may be given zero; leaving it out means the
            same thing and is clearer.
        spill_order: Where a contribution goes when the account its weight
            names has no room left. Tried in order, each kind at most once.
    """

    weights: FrozenMapping[float]
    spill_order: tuple[str, ...]

    @model_validator(mode="after")
    def _check_weights(self) -> Self:
        unknown = sorted(set(self.weights) - CONTRIBUTION_KINDS)
        if unknown:
            raise ValueError(
                f"contribution.weights: {', '.join(repr(k) for k in unknown)} "
                f"cannot receive a contribution. Known: "
                f"{', '.join(sorted(CONTRIBUTION_KINDS))}."
            )
        negative = sorted(kind for kind, weight in self.weights.items() if weight < 0.0)
        if negative:
            raise ValueError(
                f"contribution.weights: negative weight on "
                f"{', '.join(repr(k) for k in negative)}. A negative "
                "contribution is a withdrawal, which the withdrawal rule owns."
            )
        total = sum(self.weights.values())
        if abs(total - 1.0) > TOLERANCE:
            raise ValueError(
                f"contribution.weights: sum to {total!r}, not 1. The weights "
                "split one contribution, so anything else silently invents or "
                "destroys money."
            )
        return self

    @model_validator(mode="after")
    def _check_spill_order(self) -> Self:
        _check_kind_order("contribution.spill_order", self.spill_order, CONTRIBUTION_KINDS)
        return self


class WithdrawalRule(_Base):
    """Where a policy takes money from when cash runs short.

    Attributes:
        order: Account kinds drawn on in order, each at most once. Mandatory
            minimums are applied by the step regardless of this order; it
            governs discretionary withdrawals.
        taxable_ceiling_bracket: Index into the federal bracket table, from
            zero, that a registered withdrawal is filled up to. The bracket
            edges themselves are parameters and are not checked here.
        fill_pension_credit: Whether to draw enough eligible pension income to
            use the pension income credit.
    """

    order: tuple[str, ...]
    taxable_ceiling_bracket: Annotated[int, Field(ge=0)]
    fill_pension_credit: bool

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        _check_kind_order("withdrawal.order", self.order, WITHDRAWAL_KINDS)
        return self


class RrifConversion(_Base):
    """When an RRSP becomes a RRIF, and how much of it.

    Attributes:
        age_years: Age the conversion happens at. Not checked against the
            statutory deadline, which is a parameter of a tax year.
        fraction: Share of the RRSP converted, ``[0, 1]``. A partial conversion
            is how a household buys eligible pension income without committing
            the whole balance to a minimum.
    """

    age_years: int
    fraction: Fraction


class ElectionsSpec(_Base):
    """The dated choices a policy makes, per person where they are per person.

    Named ``ElectionsSpec`` rather than ``Elections`` because the engine has a
    runtime ``Elections`` carrying the same choices *by person index* rather
    than by id. The builder that converts one into the other has both in front
    of it at the same call site, and two classes with one name there is how the
    conversion gets skipped.

    Ages are **not** checked against the statutory windows. Those bounds belong
    to a tax year and live in ``params/``; a copy of them here would be an
    invented parameter. ``engine/scenario/start_ages.py::check_start_ages``
    checks them, against the parameter year.

    Attributes:
        cpp_start_age_years: Age each person starts CPP, keyed by person id.
        oas_start_age_years: Age each person starts OAS, keyed by person id.
        rrif_conversion: The RRSP-to-RRIF conversion, household-wide.
    """

    cpp_start_age_years: FrozenMapping[int]
    oas_start_age_years: FrozenMapping[int]
    rrif_conversion: RrifConversion


class PolicySpec(_Base):
    """One policy to evaluate, as written in the file.

    Named ``PolicySpec`` rather than ``Policy`` because
    :class:`engine.policy.base.Policy` is a *runnable* decision rule with a
    ``decide`` method. This is the description the optimizer builds one from,
    and confusing a description with the thing it describes is a bug that type
    checking would not catch if both were called ``Policy``.

    Attributes:
        name: Label, unique within the scenario. Reported against results.
        contribution: Where money goes.
        withdrawal: Where money comes from.
        elections: The dated choices.
    """

    name: str
    contribution: ContributionRule
    withdrawal: WithdrawalRule
    elections: ElectionsSpec


class Scenario(_Base):
    """One household, its assumptions, and the policies to evaluate.

    Attributes:
        name: Label for the run, used for export filenames.
        start_year: The simulation opens on 1 January of this year, every
            amount but ``Person.prior_year_net_income`` is in that January's
            dollars, and ``params/<start_year>/`` serves the whole run.
        n_paths: Monte Carlo paths.
        seed: Seed for the common random numbers. Fixed so the same scenario
            reproduces exactly, and shared across every policy so the optimizer
            compares policies rather than draws.
        household: The people.
        spending: What they spend.
        assumptions: Inflation and capital markets.
        policies: At least one policy to evaluate.
        grid: Optional expansion of the policies. Each key is a dotted path
            into a policy naming a numeric field, and each value is the list of
            values to try. Every key must resolve on every policy, because the
            grid is expanded against all of them.
    """

    name: str
    start_year: int
    n_paths: Annotated[int, Field(ge=1)]
    seed: int
    household: Household
    spending: Spending
    assumptions: Assumptions
    policies: tuple[PolicySpec, ...]
    grid: FrozenMapping[tuple[int | float, ...]] = Field(
        default_factory=lambda: MappingProxyType({})
    )

    @model_validator(mode="after")
    def _check_policies_exist(self) -> Self:
        """At least one policy, checked here for the reason given on ``persons``."""
        if not self.policies:
            raise ValueError(
                "policies: none given, so the run has no decision rule and "
                "nothing to evaluate."
            )
        return self

    @model_validator(mode="after")
    def _check_spending_covers_the_first_year(self) -> Self:
        first = self.spending.schedule[0].from_year
        if first > self.start_year:
            raise ValueError(
                f"spending.schedule: the first band begins in {first} but the "
                f"run opens in {self.start_year}, leaving "
                f"{first - self.start_year} year(s) with no spending level. A "
                "band runs until the next one starts, so the first has to be "
                "in effect on day one."
            )
        return self

    @model_validator(mode="after")
    def _check_no_one_is_born_after_the_run_opens(self) -> Self:
        """A birth date after the run opens is refused, not silently mishandled.

        The run opens on 1 January of ``start_year`` — month index 0 — and
        every age the engine derives assumes the birth already happened by
        then. ``engine.core.timeline.age_in_months`` promises a non-negative
        result but does not enforce it: a person born in March of
        ``start_year``, say, would be -2 months old at month index 0, and the
        failure would arrive many steps downstream as a missing life-table row
        rather than as a rejected scenario. Refusing it here, where the whole
        document is validated, is where a malformed *scenario* belongs; see
        ``docs/limitations.md`` L48.

        The boundary is 1 January itself: a birth in January of ``start_year``
        is already legal (age zero months at month index 0), so the admissible
        condition is ``(birth_year, birth_month) <= (start_year, 1)``.

        A mid-run birth is a case we would plausibly want later — an RESP
        beneficiary a scenario states as "born in 2030" — so this is a
        deliberate narrowing, recorded as L48, not an oversight left for later.
        """
        boundary = (self.start_year, 1)
        for person in self.household.persons:
            if (person.birth_year, person.birth_month) > boundary:
                raise ValueError(
                    f"household.persons: {person.id!r} is born "
                    f"{person.birth_year}-{person.birth_month:02d}, after the "
                    f"run opens on 1 January {self.start_year}. A birth after "
                    "the run opens is not modelled (docs/limitations.md L48)."
                )
        for beneficiary in self.household.beneficiaries:
            if (beneficiary.birth_year, beneficiary.birth_month) > boundary:
                raise ValueError(
                    f"household.beneficiaries: {beneficiary.id!r} is born "
                    f"{beneficiary.birth_year}-{beneficiary.birth_month:02d}, "
                    f"after the run opens on 1 January {self.start_year}. A "
                    "birth after the run opens is not modelled "
                    "(docs/limitations.md L48)."
                )
        return self

    @model_validator(mode="after")
    def _check_policy_names_unique(self) -> Self:
        names = [policy.name for policy in self.policies]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ValueError(
                f"policies: more than one policy named "
                f"{', '.join(repr(name) for name in duplicated)}. Results are "
                "reported against the name, so two would be indistinguishable."
            )
        return self

    @model_validator(mode="after")
    def _check_elections_name_persons(self) -> Self:
        known = set(self.household.person_ids)
        for policy in self.policies:
            for field_name in ("cpp_start_age_years", "oas_start_age_years"):
                elected: Mapping[str, int] = getattr(policy.elections, field_name)
                unknown = sorted(set(elected) - known)
                if unknown:
                    raise ValueError(
                        f"elections.{field_name}: policy {policy.name!r} makes "
                        f"an election for {', '.join(repr(i) for i in unknown)}, "
                        f"who is not in the household. Known: "
                        f"{', '.join(sorted(known))}."
                    )
        return self

    @model_validator(mode="after")
    def _check_every_person_has_a_start_age(self) -> Self:
        """Every person who has a benefit still to start must have an age for it.

        The builder builds a per-person tuple of start ages from these, so a
        person left out has no age at all rather than a default one — and there
        is no default to fall back on, because the start age *is* the decision
        being modelled.

        A person already receiving CPP or OAS is exempt from that benefit's
        election and only that one: the pension started on a date the
        scenario records as an amount, and an age would be a second,
        contradictory answer.
        """
        for policy in self.policies:
            for field_name, needs_one in (
                (
                    "cpp_start_age_years",
                    [p.id for p in self.household.persons if p.cpp.in_pay_monthly is None],
                ),
                (
                    "oas_start_age_years",
                    [p.id for p in self.household.persons if p.oas.in_pay_monthly is None],
                ),
            ):
                elected: Mapping[str, int] = getattr(policy.elections, field_name)
                missing = [person_id for person_id in needs_one if person_id not in elected]
                if missing:
                    raise ValueError(
                        f"elections.{field_name}: policy {policy.name!r} makes "
                        f"no election for {', '.join(repr(i) for i in missing)}. "
                        "There is no default start age: it is the decision "
                        "being modelled."
                    )
        return self

    @model_validator(mode="after")
    def _check_grid_resolves(self) -> Self:
        for path, values in self.grid.items():
            if not values:
                raise ValueError(
                    f"grid[{path!r}]: no values to try. An empty list expands "
                    "to no policies at all, which would silently drop every "
                    "policy from the search."
                )
            for policy in self.policies:
                try:
                    resolve_policy_path(policy, path)
                except ValueError as exc:
                    raise ValueError(f"grid[{path!r}]: {exc}") from exc
        return self


def resolve_policy_path(policy: PolicySpec, path: str) -> float:
    """Follow a dotted ``path`` into ``policy`` and return the number it names.

    The addressing the grid uses, and the same walk the optimizer needs when it
    substitutes a value. Both model fields and mapping keys are traversed, so
    ``elections.cpp_start_age_years.a`` reaches the election for the person
    with id ``a``.

    Only a number is a valid target. A path stopping on a bool, a string, or a
    whole block is an error rather than something to coerce: a grid over
    ``fill_pension_credit`` would be a search over ``True`` and ``False``
    expressed as numbers, and a grid over ``contribution`` names no single
    quantity at all.

    Args:
        policy: The policy to address into.
        path: Dotted path relative to the policy, e.g. ``"withdrawal.
            taxable_ceiling_bracket"``.

    Returns:
        The value at ``path``, as a float.

    Raises:
        ValueError: If a segment names nothing, if the walk runs into a value
            that cannot be traversed, or if the target is not a number.
    """
    current: Any = policy
    walked: list[str] = []
    for segment in path.split("."):
        where = ".".join(walked) or "the policy"
        if isinstance(current, BaseModel):
            if segment not in type(current).model_fields:
                raise ValueError(
                    f"{where} has no field {segment!r}. Available: "
                    f"{', '.join(sorted(type(current).model_fields))}."
                )
            current = getattr(current, segment)
        elif isinstance(current, Mapping):
            if segment not in current:
                raise ValueError(
                    f"{where} has no key {segment!r}. Available: "
                    f"{', '.join(sorted(map(str, current)))}."
                )
            current = current[segment]
        else:
            raise ValueError(
                f"{where} is a {type(current).__name__} and cannot be walked "
                f"into any further, so {segment!r} names nothing."
            )
        walked.append(segment)

    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ValueError(
            f"{path} is a {type(current).__name__}, and only a number can be "
            "put on a grid."
        )
    return float(current)


def _check_kind_order(field_name: str, order: tuple[str, ...], permitted: frozenset[str]) -> None:
    """Reject an account-kind order that names something unknown or repeats.

    Shared by the contribution spill order and the withdrawal order because the
    two failures are identical and their messages should be too. A repeat is
    called out separately from an unknown kind: they are different mistakes and
    "not a known account" would be a misleading thing to say about ``tfsa``
    written twice.
    """
    unknown = [kind for kind in order if kind not in permitted]
    if unknown:
        raise ValueError(
            f"{field_name}: {', '.join(repr(kind) for kind in unknown)} is not "
            f"an account this order may name. Known: "
            f"{', '.join(sorted(permitted))}."
        )
    repeated = sorted({kind for kind in order if order.count(kind) > 1})
    if repeated:
        raise ValueError(
            f"{field_name}: {', '.join(repr(kind) for kind in repeated)} "
            "appears more than once. The order is tried once through, so a "
            "repeat is either a typo or an account left out."
        )

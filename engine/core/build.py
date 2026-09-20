# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Scenario -> opening :class:`~engine.core.state.HouseholdState` and
:class:`~engine.mc.market.MarketInputs`.

The one place a :class:`~engine.scenario.schema.Scenario` — real dollars of January of its
start year, except each person's prior-year net income, taken as filed — is turned into the
array-valued state the monthly loop steps forward. Everything here is a mapping, not a
decision: no balance is projected, no benefit is computed, no bracket is consulted.

This is the only scenario-to-engine boundary: ``engine.core.step.advance_month`` takes no
scenario, only the state this module produces, so every field the scenario states is carried
onto it, even one no phase reads yet. The opening position is always 1 January of
``scenario.start_year``.

:func:`build_market_inputs` maps a scenario's :class:`~engine.scenario.schema.Assumptions` to
the :class:`~engine.mc.market.MarketInputs` the draws and the step read, in
``asset_class_names`` order; it validates nothing, since attainability was checked at load.

Two things worked out here rather than invented: which policy's elections to use when a
scenario carries more than one (:func:`build_initial_state` requires exactly one, or an
explicit choice); and month arithmetic, via :func:`_month_offset` — permanently distinct from
``engine.core.timeline.month_index``, which indexes the draws' first axis and so rejects a
negative result, where this one places a calendar date that may precede the run's opening. See
``state.py``'s module docstring for that negative-index convention.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from engine.core.state import (
    DEATH_NOT_DRAWN,
    BeneficiaryState,
    BenefitState,
    CashState,
    Elections,
    EmploymentBand,
    HouseholdState,
    IncomeLedger,
    LifState,
    LiraState,
    PensionState,
    PersonState,
    RespState,
    RrifState,
    RrspState,
    SpendingLevel,
    TaxableState,
    TfsaState,
    select_spending_level,
)
from engine.mc.market import MarketInputs
from engine.mc.moments import covariance_from_correlation
from engine.scenario import (
    INVESTABLE_KINDS,
    Assumptions,
    Beneficiary,
    DbPension,
    Employment,
    Person,
    PolicySpec,
    Scenario,
)

__all__ = ["build_initial_state", "build_market_inputs"]


def _month_offset(base_year: int, year: int, month: int) -> int:
    """Signed month offset from January of ``base_year`` to ``(year, month)``.

    Not ``engine.core.timeline.month_index`` in miniature — see the module docstring for why
    the two are permanently different functions. A negative result is an ordinary answer, not
    a signal to reject: several scenario inputs legitimately precede the run's opening (a
    pension already in pay, a bridge that already ended, an employment band begun years ago).

    Args:
        base_year: The simulation's first calendar year.
        year: Calendar year to locate.
        month: Month within that year, ``1..12``.

    Returns:
        Signed month offset, zero in January of ``base_year``, negative for
        a date before it.

    Raises:
        ValueError: If ``month`` is outside ``1..12``.
    """
    if not 1 <= month <= 12:
        raise ValueError(f"month must be in 1..12, got {month!r}.")
    return (year - base_year) * 12 + (month - 1)


def build_initial_state(
    scenario: Scenario,
    n_paths: int,
    policy: PolicySpec | None = None,
) -> HouseholdState:
    """Build the opening :class:`~engine.core.state.HouseholdState` for a scenario.

    Every balance, room, and ledger field is broadcast from the scenario's scalar to
    ``(n_paths,)``; every year-to-date and ``withdrawn_*`` field opens at zero. Nothing here
    reads ``params/`` or computes a benefit amount — those happen once the monthly loop starts.

    Args:
        scenario: The validated scenario to build from.
        n_paths: Monte Carlo paths every array opens with; must be at least one, since zero
            builds silently and fails much later, far from this call.
        policy: Which of the scenario's policies to take elections from. If ``None``, the
            scenario's one policy is used, or this raises if there are several — the CPP/OAS
            start ages, RRIF conversion, and pension-credit fill differ between them.

    Returns:
        The state as of 1 January of ``scenario.start_year``.

    Raises:
        ValueError: If ``n_paths`` is less than one, or ``policy`` is ``None`` with more
            than one policy on the scenario. A bridge ending before its pension starts is
            rejected earlier, by the schema (``engine.scenario.schema.Person``), so it never
            reaches here.
    """
    if n_paths < 1:
        raise ValueError(f"n_paths must be at least 1, got {n_paths!r}.")

    chosen = _select_policy(scenario, policy)
    start_year = scenario.start_year

    persons = tuple(
        _build_person(person, n_paths, start_year, chosen)
        for person in scenario.household.persons
    )
    beneficiaries = tuple(
        _build_beneficiary(beneficiary, n_paths, start_year, scenario.household.persons)
        for beneficiary in scenario.household.beneficiaries
    )

    elections = Elections(
        cpp_start_age_months=tuple(
            _cpp_start_age_months(person, chosen) for person in scenario.household.persons
        ),
        oas_start_age_months=tuple(
            _oas_start_age_months(person, chosen) for person in scenario.household.persons
        ),
        rrif_conversion_age_years=chosen.elections.rrif_conversion.age_years,
        rrif_conversion_fraction=chosen.elections.rrif_conversion.fraction,
        fill_pension_credit=chosen.withdrawal.fill_pension_credit,
    )

    spending_schedule = _build_spending_schedule(scenario)

    household_cash = sum(person.accounts.cash.balance for person in scenario.household.persons)

    return HouseholdState(
        year=start_year,
        month=1,
        month_index=0,
        n_paths=n_paths,
        province=scenario.household.province,
        persons=persons,
        beneficiaries=beneficiaries,
        cash=CashState(balance=_broadcast(household_cash, n_paths)),
        elections=elections,
        spending_schedule=spending_schedule,
        # select_spending_level is also what HouseholdState.__post_init__
        # uses to check spending_monthly stays in step with year; using it
        # here too, rather than a second copy of the selection rule, is what
        # makes that check exact rather than approximate.
        spending_monthly=select_spending_level(spending_schedule, start_year),
        spending_survivor_share=scenario.spending.survivor_share,
        spending_achieved_ytd=_zeros(n_paths),
        depleted=np.zeros(n_paths, dtype=np.bool_),
        estate_after_tax=np.full(n_paths, np.nan, dtype=np.float64),
        history=(),
    )


def build_market_inputs(assumptions: Assumptions) -> MarketInputs:
    """Build a :class:`~engine.mc.market.MarketInputs` from ``assumptions``.

    Validates nothing: attainability of the moments was already checked
    when the scenario loaded (``Assumptions._check_moments_attainable``), and
    ``MarketInputs`` itself checks only shapes.

    Args:
        assumptions: The validated capital-market assumptions to build from.

    Returns:
        A :class:`~engine.mc.market.MarketInputs`, in
        ``assumptions.asset_class_names`` order.
    """
    names = assumptions.asset_class_names
    classes = [assumptions.asset_classes[name] for name in names]

    real_means = np.array([c.real_mean for c in classes], dtype=np.float64)
    vols = np.array([c.vol for c in classes], dtype=np.float64)
    interest_yields = np.array([c.interest_yield for c in classes], dtype=np.float64)
    dividend_yields = np.array([c.dividend_yield for c in classes], dtype=np.float64)
    distributed_gains_yields = np.array(
        [c.distributed_gains_yield for c in classes], dtype=np.float64
    )

    annual_covariance = covariance_from_correlation(
        vols, np.array(assumptions.correlation, dtype=np.float64)
    )

    weights_by_kind = {
        kind: np.array([weights.get(name, 0.0) for name in names], dtype=np.float64)
        for kind, weights in assumptions.allocations.items()
    }

    return MarketInputs(
        asset_class_names=names,
        annual_means=real_means,
        annual_covariance=annual_covariance,
        interest_yields=interest_yields,
        dividend_yields=dividend_yields,
        distributed_gains_yields=distributed_gains_yields,
        weights_by_kind=weights_by_kind,
        investable_kinds=INVESTABLE_KINDS,
    )


def _select_policy(scenario: Scenario, policy: PolicySpec | None) -> PolicySpec:
    """Return ``policy``, or the scenario's only one, or raise."""
    if policy is not None:
        return policy
    if len(scenario.policies) == 1:
        return scenario.policies[0]
    names = ", ".join(repr(p.name) for p in scenario.policies)
    raise ValueError(
        f"scenario {scenario.name!r} has {len(scenario.policies)} policies "
        f"({names}); pass policy= to say which one the opening state's "
        "elections come from."
    )


def _cpp_start_age_months(person: Person, policy: PolicySpec) -> int | None:
    """The election feeding both :class:`BenefitState.start_age_months` and ``Elections``.

    ``None`` when the person's CPP is already in pay: there is no start age
    left to elect for them, only the amount already flowing.
    """
    if person.cpp.in_pay_monthly is not None:
        return None
    return policy.elections.cpp_start_age_years[person.id] * 12


def _oas_start_age_months(person: Person, policy: PolicySpec) -> int | None:
    """The election feeding both :class:`BenefitState.start_age_months` and ``Elections``.

    ``None`` when the person's OAS is already in pay: there is no start age
    left to elect for them, only the amount already flowing. Mirrors
    :func:`_cpp_start_age_months` exactly.
    """
    if person.oas.in_pay_monthly is not None:
        return None
    return policy.elections.oas_start_age_years[person.id] * 12


def _zeros(n_paths: int) -> NDArray[np.float64]:
    return np.zeros(n_paths, dtype=np.float64)


def _broadcast(value: float, n_paths: int) -> NDArray[np.float64]:
    return np.full(n_paths, value, dtype=np.float64)


def _build_person(
    person: Person,
    n_paths: int,
    start_year: int,
    policy: PolicySpec,
) -> PersonState:
    accounts = person.accounts
    lira = accounts.lira
    lif = accounts.lif

    cpp_start_months = _cpp_start_age_months(person, policy)
    cpp_in_pay = (
        None
        if person.cpp.in_pay_monthly is None
        else _broadcast(person.cpp.in_pay_monthly, n_paths)
    )

    oas_start_months = _oas_start_age_months(person, policy)
    oas_in_pay = (
        None
        if person.oas.in_pay_monthly is None
        else _broadcast(person.oas.in_pay_monthly, n_paths)
    )

    employment = tuple(
        _build_employment_band(band, n_paths, start_year) for band in person.employment
    )
    pensions = tuple(
        _build_pension(pension, n_paths, start_year, person) for pension in person.db_pensions
    )

    return PersonState(
        person_id=person.id,
        sex=person.sex,
        birth_year=person.birth_year,
        birth_month=person.birth_month,
        alive=np.ones(n_paths, dtype=np.bool_),
        death_month_index=np.full(n_paths, DEATH_NOT_DRAWN, dtype=np.int64),
        rrsp=RrspState(
            balance=_broadcast(accounts.rrsp.balance, n_paths),
            room=_broadcast(accounts.rrsp.room, n_paths),
            contributed_ytd=_zeros(n_paths),
            converted_fraction_applied=False,
        ),
        rrif=RrifState(
            balance=_broadcast(accounts.rrif.balance, n_paths),
            annual_minimum=_zeros(n_paths),
            withdrawn_ytd=_zeros(n_paths),
            # None when nothing has been opened yet. Otherwise the year
            # before the run: the true opening year is not knowable from a
            # scenario, and start_year would wrongly exempt this year's
            # minimum. See engine.core.state.RrifState.opened_year.
            opened_year=None if accounts.rrif.balance == 0.0 else start_year - 1,
        ),
        lira=LiraState(
            balance=_broadcast(lira.balance, n_paths),
            jurisdiction=lira.jurisdiction or "",
        ),
        lif=LifState(
            balance=_broadcast(lif.balance, n_paths),
            jurisdiction=lif.jurisdiction or "",
            annual_minimum=_zeros(n_paths),
            annual_maximum=_zeros(n_paths),
            withdrawn_ytd=_zeros(n_paths),
            # None when nothing has been opened yet. Otherwise the year
            # before the run: the true opening year is not knowable from a
            # scenario, and start_year would wrongly exempt this year's
            # minimum. See engine.core.state.LifState.opened_year.
            opened_year=None if lif.balance == 0.0 else start_year - 1,
        ),
        tfsa=TfsaState(
            balance=_broadcast(accounts.tfsa.balance, n_paths),
            room=_broadcast(accounts.tfsa.room, n_paths),
            withdrawn_this_year=_zeros(n_paths),
        ),
        taxable=TaxableState(
            balance=_broadcast(accounts.taxable.balance, n_paths),
            acb=_broadcast(accounts.taxable.acb, n_paths),
        ),
        cpp=BenefitState(
            start_age_months=cpp_start_months,
            in_pay_monthly=cpp_in_pay,
            contributory_history=person.cpp.contributory_history,
            monthly_amount=_zeros(n_paths),
        ),
        oas=BenefitState(
            start_age_months=oas_start_months,
            in_pay_monthly=oas_in_pay,
            contributory_history=None,
            monthly_amount=_zeros(n_paths),
        ),
        employment=employment,
        pensions=pensions,
        income=_empty_income_ledger(n_paths),
        balance_owing=_zeros(n_paths),
        prior_year_net_income=_broadcast(person.prior_year_net_income, n_paths),
        net_income_two_years_prior=_broadcast(person.net_income_two_years_prior, n_paths),
    )


def _build_employment_band(band: Employment, n_paths: int, start_year: int) -> EmploymentBand:
    return EmploymentBand(
        from_month_index=_month_offset(start_year, band.from_year, 1),
        to_month_index=_month_offset(start_year, band.to_year, 12),
        monthly_amount=_broadcast(band.annual / 12.0, n_paths),
    )


def _build_pension(
    pension: DbPension,
    n_paths: int,
    start_year: int,
    person: Person,
) -> PensionState:
    start_month_index = _month_offset(start_year, pension.start_year, pension.start_month)

    bridge_end_month_index = None
    if pension.bridge_annual > 0.0:
        # bridge_to_age_years is required whenever bridge_annual > 0 (schema
        # validator), so this is safe. The age-to-month-index conversion stays
        # here rather than moving to engine.core.timeline: _month_offset is
        # deliberately a different, permanent function from
        # engine.core.timeline.month_index, and this is one more caller of it,
        # not a reason to grow a second helper. That the bridge does not end
        # before the pension starts is guaranteed by
        # engine.scenario.schema.Person, so it is not re-checked here.
        target_year = person.birth_year + pension.bridge_to_age_years
        bridge_end_month_index = _month_offset(start_year, target_year, person.birth_month)

    return PensionState(
        name=pension.name,
        monthly_amount=_broadcast(pension.annual / 12.0, n_paths),
        start_month_index=start_month_index,
        indexed=pension.indexation == "full",
        bridge_monthly=_broadcast(pension.bridge_annual / 12.0, n_paths),
        bridge_end_month_index=bridge_end_month_index,
        survivor_share=pension.survivor_share,
    )


def _empty_income_ledger(n_paths: int) -> IncomeLedger:
    return IncomeLedger(
        employment=_zeros(n_paths),
        cpp=_zeros(n_paths),
        oas=_zeros(n_paths),
        db_pension=_zeros(n_paths),
        rrsp_withdrawals=_zeros(n_paths),
        rrif_lif_withdrawals=_zeros(n_paths),
        interest=_zeros(n_paths),
        eligible_dividends=_zeros(n_paths),
        capital_gains=_zeros(n_paths),
        resp_accumulated_income=_zeros(n_paths),
        rrsp_deductions=_zeros(n_paths),
        cpp_base_contributions=_zeros(n_paths),
        cpp_enhanced_contributions=_zeros(n_paths),
        ei_premiums=_zeros(n_paths),
        remitted=_zeros(n_paths),
    )


def _build_beneficiary(
    beneficiary: Beneficiary,
    n_paths: int,
    start_year: int,
    persons: tuple[Person, ...],
) -> BeneficiaryState:
    subscriber_index = next(
        (
            index
            for index, person in enumerate(persons)
            if person.id == beneficiary.resp.subscriber
        ),
        None,
    )
    if subscriber_index is None:
        # The schema guarantees a subscriber names a person in the
        # household (Household._check_subscribers_exist), so this is a
        # guard against that invariant breaking, not a lookup expected to
        # fail in practice.
        raise ValueError(
            f"beneficiary {beneficiary.id!r}: subscriber "
            f"{beneficiary.resp.subscriber!r} is not a person in this "
            "household."
        )

    resp = beneficiary.resp
    education = beneficiary.education

    return BeneficiaryState(
        beneficiary_id=beneficiary.id,
        birth_year=beneficiary.birth_year,
        birth_month=beneficiary.birth_month,
        resp=RespState(
            contributions=_broadcast(resp.contributions, n_paths),
            grants=_broadcast(resp.grants, n_paths),
            income=_broadcast(resp.income, n_paths),
            # What has been put in to date *is* the lifetime figure at the
            # opening of a run; nothing has been withdrawn yet to make the
            # two diverge.
            contributions_lifetime=_broadcast(resp.contributions, n_paths),
            grants_lifetime=_broadcast(resp.grants, n_paths),
            grant_room=_broadcast(resp.grant_room_carried, n_paths),
            grant_received_ytd=_zeros(n_paths),
            contributed_ytd=_zeros(n_paths),
            subscriber_index=subscriber_index,
            education_start_month_index=_month_offset(
                start_year, education.start_year, education.start_month
            ),
            education_months=education.months,
            education_monthly_cost=education.annual_cost / 12.0,
            wound_up=np.zeros(n_paths, dtype=np.bool_),
        ),
    )


def _build_spending_schedule(scenario: Scenario) -> tuple[SpendingLevel, ...]:
    return tuple(
        SpendingLevel(from_year=band.from_year, monthly_level=band.annual / 12.0)
        for band in scenario.spending.schedule
    )

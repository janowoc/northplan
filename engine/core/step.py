# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The monthly step. One month, all paths, one implementation.

``advance_month`` is the only place simulated time passes; Monte Carlo calls it in a loop over
months, and the optimizer calls Monte Carlo.

The annual events are *phases* ``advance_month`` invokes when due: ``open_year`` in January,
``close_year`` in December, ``settle_tax_balance`` in the filing month. Nothing outside this
module may call them directly.

Ordering is a correctness decision, fixed here so no account module can quietly disagree with
it; the order is stated in each function's docstring and every change to it needs a verification
row.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from engine.accounts import base as accounts_base
from engine.accounts import cash as cash_mod
from engine.accounts import lif as lif_mod
from engine.accounts import lira as lira_mod
from engine.accounts import resp as resp_mod
from engine.accounts import rrif as rrif_mod
from engine.accounts import rrsp as rrsp_mod
from engine.accounts import taxable as taxable_mod
from engine.accounts import tfsa as tfsa_mod
from engine.benefits import cpp as cpp_mod
from engine.benefits import employment as employment_mod
from engine.benefits import gis as gis_mod
from engine.benefits import oas as oas_mod
from engine.benefits import pension as pension_mod
from engine.core import timeline
from engine.core.context import ByKind, MonthContext, MonthRecord, PersonInflows
from engine.core.indexation import RealParamYear
from engine.core.state import (
    BeneficiaryState,
    CashState,
    Elections,
    HouseholdState,
    PersonState,
    YearRecord,
    select_spending_level,
    updated,
)
from engine.mc.market import MarketInputs
from engine.policy.base import Decision, Policy, Transfer
from engine.scenario import CONTRIBUTION_KINDS, WITHDRAWAL_KINDS
from engine.tax import federal as federal_mod
from engine.tax import withholding as withholding_mod
from engine.tax.combined import household_assessment

#: Contribution kinds a per-person ByKind ledger tracks -- CONTRIBUTION_KINDS less "resp",
#: which is per-beneficiary and carried in its own tuple (MonthRecord.resp_contributions).
_ACCOUNT_CONTRIBUTION_KINDS = CONTRIBUTION_KINDS - {"resp"}


def advance_month(
    state: HouseholdState,
    month_returns: NDArray[np.float64],
    policy: Policy,
    market: MarketInputs,
    real_params: RealParamYear,
) -> HouseholdState:
    """Advance the household by one month, across every path. Order of operations:

    1. January: :func:`open_year`.
    2. Deaths: :func:`resolve_deaths`, a no-op until #36.
    3. Inflows to household cash: employment net of CPP and EI contributions, CPP, gross OAS, DB
       pensions with any bridge, and the RESP education draw for each enrolled beneficiary. The
       step applies no erosion factor: ``RealParamSet.amount`` and
       ``engine.benefits.pension.db_pension_monthly`` have already applied theirs. GIS is never
       paid (L2). Every amount is recorded into the ledger by kind, contributions and premiums
       included.
    4. Withholding: payroll withholding on employment and on each DB pension, remitted from cash
       into ``IncomeLedger.remitted`` (L16).
    5. Outflows from cash: spending at ``spending_monthly``, the education cost of each enrolled
       beneficiary, and in the filing month :func:`settle_tax_balance`.
    6. Forced withdrawals to cash: each RRIF's and LIF's ``minimum_still_required``, and where
       ``Elections.fill_pension_credit`` holds, the pension-credit fill as a forced RRIF
       withdrawal spread over the months left in the year, net of what is left of the LIF
       minimum; registered withholding on the excess above the minimum (L56). Recorded into
       the ledger.
    7. Policy: ``policy.decide(state, context, real_params)``, on the state as it stands after
       phase 6.
    8. Transfers: withdrawals to cash first, in the order given, each with its registered
       withholding; then contributions from cash, in the order given, capped at room and at the
       cash available. The RESP wind-up in the first month after the education window. The
       RRSP-to-RRIF and LIRA-to-LIF conversions are :func:`close_year` item 2's, not this phase's.
       Recorded into the ledger.
    9. Cash floor: where cash is negative, force-withdraw in ``policy.withdrawal_order()``, kind
       by kind and person by person, from non-RESP accounts, respecting LIF maxima, with no
       withholding (L58). Where still negative, reduce ``spending_achieved_ytd`` by the deficit,
       up to this month's spending, set cash to zero, and set ``depleted`` (L43). Recorded into
       the ledger.
    10. Growth: every account by this month's return under its allocation,
        ``market.weights(kind) @ month_returns``; the taxable account's distributions are
        computed on the balance growth applies to, recorded into the ledger and reinvested, and
        its price moves by ``price_growth``; the RESP grows into its income bucket; cash earns
        nothing (L38).
    11. December: :func:`close_year`.
    12. Advance the month, recomputing ``spending_monthly`` from the schedule at a year roll.

    Args:
        state: Opening state for ``state.year``/``state.month``.
        month_returns: This month's real return per path and asset class,
            ``(n_assets, n_paths)``, monthly and real.
        policy: Decision rules, given only information available at this point.
        market: The asset-class allocation and yield mix each account's
            balance grows under, so the step applies ``month_returns`` by
            kind rather than as one number.
        real_params: Parameters for the tax year ``state.year`` falls in,
            in the scenario's real-dollar view.

    Returns:
        Opening state for the following month.
    """
    return advance_month_traced(state, month_returns, policy, market, real_params)[0]


def advance_month_traced(
    state: HouseholdState,
    month_returns: NDArray[np.float64],
    policy: Policy,
    market: MarketInputs,
    real_params: RealParamYear,
) -> tuple[HouseholdState, MonthRecord]:
    """Do the work :func:`advance_month` describes, and also return the month's record.

    See :func:`advance_month` for the order of operations; this is the same computation,
    returning the resulting state together with a :class:`~engine.core.context.MonthRecord` of
    everything the month did — the context :func:`~engine.policy.base.Policy.decide` was shown,
    plus every transfer, forced withdrawal, and cash-floor draw the step applied on top of it.

    Args:
        state: Opening state for ``state.year``/``state.month``.
        month_returns: This month's real return per path and asset class,
            ``(n_assets, n_paths)``, monthly and real.
        policy: Decision rules, given only information available at this point.
        market: The asset-class allocation and yield mix each account's
            balance grows under.
        real_params: Parameters for the tax year ``state.year`` falls in,
            in the scenario's real-dollar view.

    Returns:
        ``(new_state, record)``.
    """
    n_paths = state.n_paths
    n_assets = len(market.asset_class_names)
    month_returns = np.asarray(month_returns, dtype=np.float64)
    assert month_returns.shape == (n_assets, n_paths), (
        f"month_returns shape {month_returns.shape} != ({n_assets}, {n_paths}) "
        f"(n_assets, n_paths)."
    )

    cash_opening = state.cash.balance
    january_month_index = state.month_index - (state.month - 1)
    months_remaining_in_year = 13 - state.month

    # Phase 1: January.
    if timeline.is_year_start(state.month):
        state = open_year(state, real_params)

    # Phase 2: deaths (no-op until #36).
    state = resolve_deaths(state, real_params)

    persons = list(state.persons)
    beneficiaries = list(state.beneficiaries)
    cash = state.cash

    # Phase 3: inflows to household cash.
    persons, cash, inflows, pension_amounts = _phase3_inflows(
        persons, cash, state.month_index, state.year, state.month, january_month_index, real_params
    )
    beneficiaries, cash, education_draws = _phase3_education_draws(
        beneficiaries, cash, state.month_index
    )

    # Phase 4: payroll withholding.
    persons, cash, payroll_withholding = _phase4_withholding(
        persons,
        inflows,
        pension_amounts,
        cash,
        state.year,
        state.province,
        real_params,
        january_month_index,
    )

    # Phase 5: outflows from cash.
    cash = cash_mod.pay(cash, np.full(n_paths, state.spending_monthly, dtype=np.float64))
    spending_achieved_ytd = state.spending_achieved_ytd + state.spending_monthly
    beneficiaries, cash, education_costs = _phase5_education_costs(
        beneficiaries, cash, state.month_index
    )
    if timeline.is_filing_month(state.month, real_params.federal):
        tax_settlement = tuple(p.balance_owing.copy() for p in persons)
        interim_state = updated(
            state, persons=tuple(persons), beneficiaries=tuple(beneficiaries), cash=cash
        )
        settled_state = settle_tax_balance(interim_state, real_params)
        persons = list(settled_state.persons)
        cash = settled_state.cash
    else:
        tax_settlement = tuple(np.zeros(n_paths, dtype=np.float64) for _ in persons)

    # Phase 6: forced RRIF/LIF withdrawals.
    persons, cash, forced_withdrawals, forced_withholding = _phase6_forced_withdrawals(
        persons,
        cash,
        state.elections,
        state.province,
        state.year,
        state.month_index,
        january_month_index,
        months_remaining_in_year,
        real_params,
    )

    cash_after_flows = cash.balance
    context = MonthContext(
        month_index=state.month_index,
        year=state.year,
        month=state.month,
        cash_opening=cash_opening,
        inflows=tuple(inflows),
        education_draws=tuple(education_draws),
        payroll_withholding=tuple(payroll_withholding),
        spending=np.full(n_paths, state.spending_monthly, dtype=np.float64),
        education_costs=tuple(education_costs),
        tax_settlement=tax_settlement,
        forced_withdrawals=tuple(forced_withdrawals),
        forced_withholding=tuple(forced_withholding),
        cash_after_flows=cash_after_flows,
    )

    # Phase 7: policy.
    state_after_phase6 = updated(
        state,
        persons=tuple(persons),
        beneficiaries=tuple(beneficiaries),
        cash=cash,
        spending_achieved_ytd=spending_achieved_ytd,
    )
    decision = policy.decide(state_after_phase6, context, real_params)

    # Phase 8: transfers, then RESP wind-up.
    (
        persons,
        beneficiaries,
        cash,
        withdrawals,
        withdrawal_withholding,
        contributions,
        resp_contributions,
        wind_up_to_cash,
    ) = _phase8_transfers(
        persons,
        beneficiaries,
        cash,
        decision,
        state.month_index,
        state.year,
        january_month_index,
        real_params,
    )

    # Phase 9: cash floor (no withholding on the forced sweep, L58). Where the sweep still
    # leaves cash short, the shortfall beyond this month's spending is forgiven (L43).
    persons, cash, floor_withdrawals = _phase9_cash_floor(persons, cash, policy.withdrawal_order())
    depletion_deficit = np.clip(-cash.balance, 0, None)
    spending_cut = np.minimum(depletion_deficit, state.spending_monthly)
    spending_achieved_ytd = spending_achieved_ytd - spending_cut
    depleted = state.depleted | (depletion_deficit > 0)
    cash = CashState(balance=np.clip(cash.balance, 0, None))

    record = MonthRecord(
        context=context,
        withdrawals=tuple(withdrawals),
        withdrawal_withholding=tuple(withdrawal_withholding),
        contributions=tuple(contributions),
        resp_contributions=tuple(resp_contributions),
        wind_up_to_cash=tuple(wind_up_to_cash),
        floor_withdrawals=tuple(floor_withdrawals),
        depletion_deficit=depletion_deficit,
        spending_cut=spending_cut,
        cash_close=cash.balance,
    )

    # Phase 10: growth.
    persons, beneficiaries = _phase10_growth(persons, beneficiaries, market, month_returns)

    new_state = updated(
        state,
        persons=tuple(persons),
        beneficiaries=tuple(beneficiaries),
        cash=cash,
        spending_achieved_ytd=spending_achieved_ytd,
        depleted=depleted,
    )

    # Phase 11: December.
    if timeline.is_year_end(state.month):
        new_state = close_year(new_state, real_params)

    # Phase 12: advance the month.
    next_year, next_month = timeline.next_month(state.year, state.month)
    next_month_index = state.month_index + 1
    if timeline.is_year_start(next_month):
        next_spending_monthly = select_spending_level(new_state.spending_schedule, next_year)
    else:
        next_spending_monthly = new_state.spending_monthly
    final_state = updated(
        new_state,
        year=next_year,
        month=next_month,
        month_index=next_month_index,
        spending_monthly=next_spending_monthly,
    )

    return final_state, record


def _phase3_inflows(
    persons: list[PersonState],
    cash: CashState,
    month_index: int,
    year: int,
    month: int,
    january_month_index: int,
    real_params: RealParamYear,
) -> tuple[list[PersonState], CashState, list[PersonInflows], list[list[NDArray[np.float64]]]]:
    """Employment (net of CPP/EI), CPP, gross OAS, and DB pensions, per person.

    The fourth return value is each person's pensions, in ``person.pensions`` order, at this
    month's amount -- phase 4 withholds on these directly rather than calling
    ``pension.db_pension_monthly`` a second time.
    """
    new_persons: list[PersonState] = []
    inflows: list[PersonInflows] = []
    pension_amounts: list[list[NDArray[np.float64]]] = []
    for person in persons:
        alive = person.alive
        employment = employment_mod.employment_income_monthly(
            person.employment, month_index, len(alive)
        )
        employment = np.where(alive, employment, 0.0)
        cpp_contributions = employment_mod.cpp_contributions_monthly(
            employment, person.income.employment, january_month_index, real_params.cpp
        )
        ei_premiums = employment_mod.ei_premium_monthly(
            employment, person.income.employment, january_month_index, real_params.federal
        )

        age_months = person.age_months(year, month)
        cpp_amount = cpp_mod.pension_monthly(person.cpp, age_months, month_index, real_params.cpp)
        cpp_amount = np.where(alive, cpp_amount, 0.0)
        oas_amount = oas_mod.gross_pension_monthly(
            person.oas, age_months, month_index, real_params.oas
        )
        oas_amount = np.where(alive, oas_amount, 0.0)

        db_pension_total = np.zeros_like(employment)
        this_person_pension_amounts: list[NDArray[np.float64]] = []
        for pension in person.pensions:
            pension_amount = pension_mod.db_pension_monthly(
                pension, month_index, alive, real_params.inflation_rate
            )
            db_pension_total = db_pension_total + pension_amount
            this_person_pension_amounts.append(pension_amount)
        pension_amounts.append(this_person_pension_amounts)

        new_income = updated(
            person.income,
            employment=person.income.employment + employment,
            cpp_base_contributions=person.income.cpp_base_contributions + cpp_contributions.base,
            cpp_enhanced_contributions=(
                person.income.cpp_enhanced_contributions + cpp_contributions.enhanced
            ),
            ei_premiums=person.income.ei_premiums + ei_premiums,
            cpp=person.income.cpp + cpp_amount,
            oas=person.income.oas + oas_amount,
            db_pension=person.income.db_pension + db_pension_total,
        )
        new_person = updated(
            person,
            income=new_income,
            cpp=updated(person.cpp, monthly_amount=cpp_amount),
            oas=updated(person.oas, monthly_amount=oas_amount),
        )
        new_persons.append(new_person)

        cpp_contributions_total = cpp_contributions.base + cpp_contributions.enhanced
        cash = cash_mod.deposit(cash, employment)
        cash = cash_mod.pay(cash, cpp_contributions_total)
        cash = cash_mod.pay(cash, ei_premiums)
        cash = cash_mod.deposit(cash, cpp_amount)
        cash = cash_mod.deposit(cash, oas_amount)
        cash = cash_mod.deposit(cash, db_pension_total)

        inflows.append(
            PersonInflows(
                employment=employment,
                cpp_contributions=cpp_contributions_total,
                ei_premiums=ei_premiums,
                cpp=cpp_amount,
                oas=oas_amount,
                db_pension=db_pension_total,
            )
        )
    return new_persons, cash, inflows, pension_amounts


def _in_education_window(beneficiary: BeneficiaryState, month_index: int) -> bool:
    resp = beneficiary.resp
    in_window = resp.education_start_month_index <= month_index < (
        resp.education_start_month_index + resp.education_months
    )
    return in_window and not np.all(resp.wound_up)


def _phase3_education_draws(
    beneficiaries: list[BeneficiaryState],
    cash: CashState,
    month_index: int,
) -> tuple[list[BeneficiaryState], CashState, list[NDArray[np.float64]]]:
    """The RESP education draw for each enrolled beneficiary. Nothing reaches any ledger (L30)."""
    new_beneficiaries: list[BeneficiaryState] = []
    draws: list[NDArray[np.float64]] = []
    for beneficiary in beneficiaries:
        if _in_education_window(beneficiary, month_index):
            new_resp, result = resp_mod.education_draw(beneficiary.resp)
            # education_draw itself now raises ValueError for a value more than float
            # dust below zero (engine.accounts.resp._clip_value_or_raise) and clips the
            # dust otherwise, so result.gross is never negative here; deposit it
            # directly rather than clipping it a second time.
            draw = result.gross
        else:
            new_resp = beneficiary.resp
            draw = np.zeros(cash.balance.shape, dtype=np.float64)
        cash = cash_mod.deposit(cash, draw)
        new_beneficiaries.append(updated(beneficiary, resp=new_resp))
        draws.append(draw)
    return new_beneficiaries, cash, draws


def _phase4_withholding(
    persons: list[PersonState],
    inflows: list[PersonInflows],
    pension_amounts: list[list[NDArray[np.float64]]],
    cash: CashState,
    year: int,
    province: str,
    real_params: RealParamYear,
    january_month_index: int,
) -> tuple[list[PersonState], CashState, list[NDArray[np.float64]]]:
    """Payroll withholding on employment and on each DB pension's own monthly amount (L16)."""
    new_persons: list[PersonState] = []
    withheld_list: list[NDArray[np.float64]] = []
    for person, person_inflows, this_person_pension_amounts in zip(
        persons, inflows, pension_amounts, strict=True
    ):
        age_end = timeline.age_at_end_of_year(person.birth_year, person.birth_month, year)
        withheld = withholding_mod.payroll_withholding_monthly(
            person_inflows.employment, age_end, True, province, real_params, january_month_index
        )
        for pension_amount in this_person_pension_amounts:
            withheld = withheld + withholding_mod.payroll_withholding_monthly(
                pension_amount, age_end, False, province, real_params, january_month_index
            )
        new_income = updated(person.income, remitted=person.income.remitted + withheld)
        new_persons.append(updated(person, income=new_income))
        withheld_list.append(withheld)
        cash = cash_mod.pay(cash, withheld)
    return new_persons, cash, withheld_list


def _phase5_education_costs(
    beneficiaries: list[BeneficiaryState],
    cash: CashState,
    month_index: int,
) -> tuple[list[BeneficiaryState], CashState, list[NDArray[np.float64]]]:
    """The education cost of each enrolled beneficiary, not counted toward spending achieved."""
    costs: list[NDArray[np.float64]] = []
    for beneficiary in beneficiaries:
        if _in_education_window(beneficiary, month_index):
            cost = np.full(
                cash.balance.shape, beneficiary.resp.education_monthly_cost, dtype=np.float64
            )
        else:
            cost = np.zeros(cash.balance.shape, dtype=np.float64)
        cash = cash_mod.pay(cash, cost)
        costs.append(cost)
    return beneficiaries, cash, costs


def _phase6_forced_withdrawals(
    persons: list[PersonState],
    cash: CashState,
    elections: Elections,
    province: str,
    year: int,
    month_index: int,
    january_month_index: int,
    months_remaining_in_year: int,
    real_params: RealParamYear,
) -> tuple[list[PersonState], CashState, list[ByKind], list[NDArray[np.float64]]]:
    """Each RRIF's and LIF's ``minimum_still_required``, plus the pension-credit fill (L56)."""
    fill_age = None
    target = None
    if elections.fill_pension_credit:
        fill_age = real_params.federal.number("eligible_pension_income.rrif_minimum_age_years")
        target = max(
            real_params.federal.annual_amount(
                "credits.pension_income_amount_annual", january_month_index
            ),
            real_params.province(province).annual_amount(
                "credits.pension_income_amount_annual", january_month_index
            ),
        )

    new_persons: list[PersonState] = []
    forced_withdrawals: list[ByKind] = []
    forced_withholding: list[NDArray[np.float64]] = []
    for person in persons:
        age_end = timeline.age_at_end_of_year(person.birth_year, person.birth_month, year)

        rrif_floor = rrif_mod.minimum_still_required(
            person.rrif.annual_minimum, person.rrif.withdrawn_ytd, months_remaining_in_year
        )
        if elections.fill_pension_credit and age_end >= fill_age:
            # Read before this month's own LIF withdrawal, below: the LIF minimum is fixed
            # in January, so this reads nothing from the future, only what is already
            # certain to become eligible pension income this year regardless of the fill.
            eligible_ytd = federal_mod.eligible_pension_income(
                person.income, age_end, real_params.federal
            )
            lif_min_remaining = np.clip(
                person.lif.annual_minimum - person.lif.withdrawn_ytd, 0, None
            )
            fill = (
                np.clip(target - eligible_ytd - lif_min_remaining, 0, None)
                / months_remaining_in_year
            )
            rrif_floor = np.maximum(rrif_floor, fill)

        new_rrif, rrif_result, rrif_above_min = rrif_mod.withdraw(
            person.rrif, np.zeros_like(rrif_floor), rrif_floor
        )

        lif_state = person.lif
        skip_lif = lif_state.jurisdiction == "" and bool(np.all(lif_state.balance == 0.0))
        if skip_lif:
            new_lif = lif_state
            lif_gross = np.zeros_like(rrif_floor)
            lif_above_min = np.zeros_like(rrif_floor)
        else:
            lif_floor = rrif_mod.minimum_still_required(
                lif_state.annual_minimum, lif_state.withdrawn_ytd, months_remaining_in_year
            )
            max_remaining = accounts_base.remaining_annual_allowance(
                lif_state.annual_maximum, lif_state.withdrawn_ytd
            )
            new_lif, lif_result, lif_above_min = lif_mod.withdraw(
                lif_state, np.zeros_like(lif_floor), lif_floor, max_remaining
            )
            lif_gross = lif_result.gross

        rrif_withholding = withholding_mod.registered_withholding(
            rrif_above_min, real_params.rrif, month_index
        )
        lif_withholding = withholding_mod.registered_withholding(
            lif_above_min, real_params.rrif, month_index
        )
        withheld = rrif_withholding + lif_withholding

        new_income = updated(
            person.income,
            rrif_lif_withdrawals=(
                person.income.rrif_lif_withdrawals + rrif_result.gross + lif_gross
            ),
            remitted=person.income.remitted + withheld,
        )
        new_persons.append(updated(person, rrif=new_rrif, lif=new_lif, income=new_income))

        cash = cash_mod.deposit(cash, rrif_result.gross + lif_gross)
        cash = cash_mod.pay(cash, withheld)

        forced_withdrawals.append(
            ByKind(
                rrsp=np.zeros_like(rrif_floor),
                rrif=rrif_result.gross,
                lif=lif_gross,
                tfsa=np.zeros_like(rrif_floor),
                taxable=np.zeros_like(rrif_floor),
            )
        )
        forced_withholding.append(withheld)
    return new_persons, cash, forced_withdrawals, forced_withholding


def _withdraw_from_kind(
    kind: str,
    person: PersonState,
    requested: NDArray[np.float64],
) -> tuple[PersonState, NDArray[np.float64], NDArray[np.float64]]:
    """Withdraw ``requested`` from account ``kind``, floor zero (LIF capped at its remaining
    maximum). Returns ``(new_person, gross, above_minimum)``; ``above_minimum`` is the
    registered-withholding basis for ``rrif``/``lif`` and zero for every other kind.
    """
    zeros = np.zeros_like(requested)
    if kind == "rrsp":
        new_rrsp, result = rrsp_mod.withdraw(person.rrsp, requested)
        new_income = updated(
            person.income, rrsp_withdrawals=person.income.rrsp_withdrawals + result.fully_taxable
        )
        return updated(person, rrsp=new_rrsp, income=new_income), result.gross, zeros
    if kind == "rrif":
        new_rrif, result, above_minimum = rrif_mod.withdraw(person.rrif, requested, zeros)
        new_income = updated(
            person.income,
            rrif_lif_withdrawals=person.income.rrif_lif_withdrawals + result.fully_taxable,
        )
        return updated(person, rrif=new_rrif, income=new_income), result.gross, above_minimum
    if kind == "lif":
        max_remaining = accounts_base.remaining_annual_allowance(
            person.lif.annual_maximum, person.lif.withdrawn_ytd
        )
        new_lif, result, above_minimum = lif_mod.withdraw(
            person.lif, requested, zeros, max_remaining
        )
        new_income = updated(
            person.income,
            rrif_lif_withdrawals=person.income.rrif_lif_withdrawals + result.fully_taxable,
        )
        return updated(person, lif=new_lif, income=new_income), result.gross, above_minimum
    if kind == "tfsa":
        new_tfsa, result = tfsa_mod.withdraw(person.tfsa, requested)
        return updated(person, tfsa=new_tfsa), result.gross, zeros
    if kind == "taxable":
        new_taxable, result = taxable_mod.withdraw(person.taxable, requested)
        new_income = updated(
            person.income, capital_gains=person.income.capital_gains + result.capital_gain
        )
        return updated(person, taxable=new_taxable, income=new_income), result.gross, zeros
    raise ValueError(f"{kind!r} is not a withdrawal account kind.")


def _validate_transfer(
    transfer: Transfer,
    n_paths: int,
    n_persons: int,
    beneficiaries: list[BeneficiaryState],
    month_index: int,
) -> None:
    """Raise ``ValueError``, naming ``transfer``, unless it is a legal instruction."""
    is_cash_from = transfer.from_kind == "cash"
    is_cash_to = transfer.to_kind == "cash"
    if is_cash_from == is_cash_to:
        raise ValueError(
            f"{transfer!r}: exactly one of from_kind/to_kind must be 'cash', got "
            f"from_kind={transfer.from_kind!r}, to_kind={transfer.to_kind!r}."
        )
    if is_cash_to:
        if transfer.from_kind not in WITHDRAWAL_KINDS:
            raise ValueError(
                f"{transfer!r}: from_kind {transfer.from_kind!r} is not in WITHDRAWAL_KINDS "
                f"({sorted(WITHDRAWAL_KINDS)})."
            )
        # "resp" is never reached here: it is not in WITHDRAWAL_KINDS, so the check above
        # already raised for it. A withdrawal's person_index is always into persons.
        limit = n_persons
    else:
        if transfer.to_kind not in CONTRIBUTION_KINDS:
            raise ValueError(
                f"{transfer!r}: to_kind {transfer.to_kind!r} is not in CONTRIBUTION_KINDS "
                f"({sorted(CONTRIBUTION_KINDS)})."
            )
        limit = len(beneficiaries) if transfer.to_kind == "resp" else n_persons

    if not 0 <= transfer.person_index < limit:
        raise ValueError(
            f"{transfer!r}: person_index {transfer.person_index} out of range "
            f"[0, {limit})."
        )

    if transfer.to_kind == "resp":
        resp_state = beneficiaries[transfer.person_index].resp
        threshold = resp_state.education_start_month_index + resp_state.education_months
        if month_index >= threshold:
            raise ValueError(
                f"{transfer!r}: the education window for beneficiary "
                f"{beneficiaries[transfer.person_index].beneficiary_id!r} ended at month "
                f"index {threshold!r}; no RESP contribution is accepted from month index "
                f"{threshold!r} on, wound up or not."
            )

    amount = np.asarray(transfer.amount, dtype=np.float64)
    if amount.shape != (n_paths,):
        raise ValueError(f"{transfer!r}: amount shape {amount.shape} != ({n_paths},).")
    if np.any(~np.isfinite(amount)):
        raise ValueError(f"{transfer!r}: amount is not finite on every path.")
    if np.any(amount < 0):
        raise ValueError(f"{transfer!r}: amount is negative on at least one path.")


def _phase8_transfers(
    persons: list[PersonState],
    beneficiaries: list[BeneficiaryState],
    cash: CashState,
    decision: Decision,
    month_index: int,
    year: int,
    january_month_index: int,
    real_params: RealParamYear,
) -> tuple[
    list[PersonState],
    list[BeneficiaryState],
    CashState,
    list[ByKind],
    list[NDArray[np.float64]],
    list[ByKind],
    list[NDArray[np.float64]],
    list[NDArray[np.float64]],
]:
    """Apply every withdrawal, in order, then every contribution, in order; then wind up any
    RESP whose education window has ended.
    """
    n_paths = cash.balance.shape[0]
    for transfer in decision.transfers:
        _validate_transfer(transfer, n_paths, len(persons), beneficiaries, month_index)

    withdrawal_amounts = [
        {kind: np.zeros(n_paths) for kind in WITHDRAWAL_KINDS} for _ in persons
    ]
    withdrawal_withholding_amounts = [np.zeros(n_paths) for _ in persons]

    for transfer in decision.transfers:
        if transfer.to_kind != "cash":
            continue
        i = transfer.person_index
        person, gross, above_minimum = _withdraw_from_kind(
            transfer.from_kind, persons[i], transfer.amount
        )
        if transfer.from_kind in ("rrif", "lif"):
            withholding_basis = above_minimum
        elif transfer.from_kind == "rrsp":
            withholding_basis = gross
        else:
            withholding_basis = None
        withholding = (
            withholding_mod.registered_withholding(withholding_basis, real_params.rrif, month_index)
            if withholding_basis is not None
            else np.zeros(n_paths)
        )
        new_income = updated(person.income, remitted=person.income.remitted + withholding)
        persons[i] = updated(person, income=new_income)
        cash = cash_mod.deposit(cash, gross)
        cash = cash_mod.pay(cash, withholding)
        withdrawal_amounts[i][transfer.from_kind] = (
            withdrawal_amounts[i][transfer.from_kind] + gross
        )
        withdrawal_withholding_amounts[i] = withdrawal_withholding_amounts[i] + withholding

    contribution_amounts = [
        {kind: np.zeros(n_paths) for kind in _ACCOUNT_CONTRIBUTION_KINDS} for _ in persons
    ]
    resp_contribution_amounts = [np.zeros(n_paths) for _ in beneficiaries]

    for transfer in decision.transfers:
        if transfer.from_kind != "cash":
            continue
        available = np.clip(cash.balance, 0, None)
        requested = np.minimum(transfer.amount, available)

        if transfer.to_kind == "resp":
            b = transfer.person_index
            beneficiary = beneficiaries[b]
            resp_mod.governing_income_year_offset(real_params.resp)
            family_income = sum(p.net_income_two_years_prior for p in persons)
            age_end = timeline.age_at_end_of_year(
                beneficiary.birth_year, beneficiary.birth_month, year
            )
            new_resp, contributed = resp_mod.contribute(
                beneficiary.resp,
                requested,
                family_income,
                age_end,
                january_month_index,
                real_params.resp,
            )
            beneficiaries[b] = updated(beneficiary, resp=new_resp)
            resp_contribution_amounts[b] = resp_contribution_amounts[b] + contributed
        else:
            i = transfer.person_index
            person = persons[i]
            if transfer.to_kind == "rrsp":
                new_rrsp, contributed = rrsp_mod.contribute(person.rrsp, requested)
                new_income = updated(
                    person.income, rrsp_deductions=person.income.rrsp_deductions + contributed
                )
                persons[i] = updated(person, rrsp=new_rrsp, income=new_income)
            elif transfer.to_kind == "tfsa":
                new_tfsa, contributed = tfsa_mod.contribute(person.tfsa, requested)
                persons[i] = updated(person, tfsa=new_tfsa)
            elif transfer.to_kind == "taxable":
                contributed = requested
                new_taxable = updated(
                    person.taxable,
                    balance=person.taxable.balance + contributed,
                    acb=person.taxable.acb + contributed,
                )
                persons[i] = updated(person, taxable=new_taxable)
            else:  # pragma: no cover - unreachable after _validate_transfer.
                raise ValueError(f"{transfer!r}: unreachable after validation.")
            contribution_amounts[i][transfer.to_kind] = (
                contribution_amounts[i][transfer.to_kind] + contributed
            )
        cash = cash_mod.pay(cash, contributed)

    wind_up_to_cash: list[NDArray[np.float64]] = []
    for b, beneficiary in enumerate(beneficiaries):
        resp_state = beneficiary.resp
        threshold = resp_state.education_start_month_index + resp_state.education_months
        if month_index >= threshold and not np.all(resp_state.wound_up):
            new_resp, to_cash_tax_free, accumulated, _grants_repaid = resp_mod.wind_up(resp_state)
            total_to_cash = to_cash_tax_free + accumulated
            cash = cash_mod.deposit(cash, total_to_cash)
            subscriber_index = resp_state.subscriber_index
            subscriber = persons[subscriber_index]
            new_income = updated(
                subscriber.income,
                resp_accumulated_income=subscriber.income.resp_accumulated_income + accumulated,
            )
            persons[subscriber_index] = updated(subscriber, income=new_income)
            beneficiaries[b] = updated(beneficiary, resp=new_resp)
            wind_up_to_cash.append(total_to_cash)
        else:
            wind_up_to_cash.append(np.zeros(n_paths))

    withdrawals = [
        ByKind(**{kind: amounts[kind] for kind in WITHDRAWAL_KINDS})
        for amounts in withdrawal_amounts
    ]
    contributions = [
        ByKind(
            rrsp=amounts["rrsp"],
            rrif=np.zeros(n_paths),
            lif=np.zeros(n_paths),
            tfsa=amounts["tfsa"],
            taxable=amounts["taxable"],
        )
        for amounts in contribution_amounts
    ]
    return (
        persons,
        beneficiaries,
        cash,
        withdrawals,
        withdrawal_withholding_amounts,
        contributions,
        resp_contribution_amounts,
        wind_up_to_cash,
    )


def _validate_withdrawal_order(order: tuple[str, ...]) -> None:
    """Raise ``ValueError`` unless ``order`` names each ``WITHDRAWAL_KINDS`` kind at most once."""
    seen: set[str] = set()
    for kind in order:
        if kind not in WITHDRAWAL_KINDS:
            raise ValueError(
                f"withdrawal_order(): {kind!r} is not in WITHDRAWAL_KINDS "
                f"({sorted(WITHDRAWAL_KINDS)})."
            )
        if kind in seen:
            raise ValueError(f"withdrawal_order(): {kind!r} is repeated.")
        seen.add(kind)


def _phase9_cash_floor(
    persons: list[PersonState],
    cash: CashState,
    policy_order: tuple[str, ...],
) -> tuple[list[PersonState], CashState, list[ByKind]]:
    """Force-withdraw, kind by kind and person by person, wherever cash is negative (L58)."""
    _validate_withdrawal_order(policy_order)
    remaining = tuple(sorted(kind for kind in WITHDRAWAL_KINDS if kind not in policy_order))
    full_order = tuple(policy_order) + remaining

    n_paths = cash.balance.shape[0]
    floor_amounts = [{kind: np.zeros(n_paths) for kind in WITHDRAWAL_KINDS} for _ in persons]

    for kind in full_order:
        for i, person in enumerate(persons):
            request = np.clip(-cash.balance, 0, None)
            new_person, gross, _above_minimum = _withdraw_from_kind(kind, person, request)
            persons[i] = new_person
            cash = cash_mod.deposit(cash, gross)
            floor_amounts[i][kind] = floor_amounts[i][kind] + gross

    floor_withdrawals = [
        ByKind(**{kind: amounts[kind] for kind in WITHDRAWAL_KINDS}) for amounts in floor_amounts
    ]
    return persons, cash, floor_withdrawals


def _phase10_growth(
    persons: list[PersonState],
    beneficiaries: list[BeneficiaryState],
    market: MarketInputs,
    month_returns: NDArray[np.float64],
) -> tuple[list[PersonState], list[BeneficiaryState]]:
    """Apply this month's return under each account's allocation."""
    r_rrsp = market.weights("rrsp") @ month_returns
    r_rrif = market.weights("rrif") @ month_returns
    r_lira = market.weights("lira") @ month_returns
    r_lif = market.weights("lif") @ month_returns
    r_tfsa = market.weights("tfsa") @ month_returns
    r_taxable = market.weights("taxable") @ month_returns
    r_resp = market.weights("resp") @ month_returns
    weighted_yields = market.weighted_yields("taxable")

    new_persons: list[PersonState] = []
    for person in persons:
        new_rrsp = updated(person.rrsp, balance=accounts_base.grow(person.rrsp.balance, r_rrsp))
        new_rrif = updated(person.rrif, balance=accounts_base.grow(person.rrif.balance, r_rrif))
        new_lira = updated(person.lira, balance=accounts_base.grow(person.lira.balance, r_lira))
        new_lif = updated(person.lif, balance=accounts_base.grow(person.lif.balance, r_lif))
        new_tfsa = updated(person.tfsa, balance=accounts_base.grow(person.tfsa.balance, r_tfsa))

        distributions = taxable_mod.distributions_monthly(person.taxable.balance, weighted_yields)
        interest, dividends, gains = distributions
        new_income = updated(
            person.income,
            interest=person.income.interest + interest,
            eligible_dividends=person.income.eligible_dividends + dividends,
            capital_gains=person.income.capital_gains + gains,
        )
        taxable_after_price = taxable_mod.price_growth(
            person.taxable, r_taxable, weighted_yields
        )
        new_taxable = taxable_mod.reinvest(taxable_after_price, distributions)

        new_persons.append(
            updated(
                person,
                rrsp=new_rrsp,
                rrif=new_rrif,
                lira=new_lira,
                lif=new_lif,
                tfsa=new_tfsa,
                taxable=new_taxable,
                income=new_income,
            )
        )

    new_beneficiaries = [
        updated(beneficiary, resp=resp_mod.grow(beneficiary.resp, r_resp))
        for beneficiary in beneficiaries
    ]
    return new_persons, new_beneficiaries


def open_year(state: HouseholdState, real_params: RealParamYear) -> HouseholdState:
    """January phase: grant room, fix the year's annual limits, reset the ledger.

    Called by :func:`advance_month`, never directly.

    1. Erode each nominal state balance by one year: ``erode_nominal`` of the RRSP, the TFSA,
       the taxable account, and each beneficiary's RESP (L57).
    2. TFSA: restore last year's withdrawals to room, then grant this year's room.
    3. RRSP room on last year's employment income, read from the ledger before item 6 resets it
       (L25).
    4. RESP grant room per beneficiary under the cessation age.
    5. RRIF and LIF minimum, and LIF maximum, from the opening 1 January balances by age at the
       start of the year. The LIF's maximum from
       ``real_params.jurisdiction(LifState.jurisdiction)``; where
       ``engine.accounts.lif.has_maximum`` is false, no maximum is stored
       (``annual_maximum`` is ``+inf``).
    6. Reset the year-to-date ledger, ``remitted`` included, the RRSP's ``contributed_ytd``,
       each RRIF's and LIF's ``withdrawn_ytd``, each RESP's ``grant_received_ytd`` and
       ``contributed_ytd``, and ``spending_achieved_ytd``. ``balance_owing`` is untouched: it is
       still owed until the filing month.

    At month index zero items 1-4 erode, restore and grant nothing (#33 item 5): the scenario's
    figures are already post-grant. Items 5 and 6 still run.

    Args:
        state: Opening state for January.
        real_params: Parameters for the new tax year, in the scenario's
            real-dollar view.

    Returns:
        State with the year's annual quantities established.
    """
    n_paths = state.n_paths
    zeros = np.zeros(n_paths, dtype=np.float64)
    first_year = state.month_index == 0
    year = state.year
    january_month_index = state.month_index
    inflation_rate = real_params.inflation_rate

    new_beneficiaries = []
    for beneficiary in state.beneficiaries:
        resp_state = beneficiary.resp
        if not first_year:
            resp_state = resp_mod.erode_nominal(resp_state, inflation_rate)
            age_end = timeline.age_at_end_of_year(
                beneficiary.birth_year, beneficiary.birth_month, year
            )
            grant_room_accrued = resp_mod.grant_room_accrued(
                age_end, real_params.resp, january_month_index
            )
            resp_state = updated(resp_state, grant_room=resp_state.grant_room + grant_room_accrued)
        resp_state = updated(resp_state, grant_received_ytd=zeros, contributed_ytd=zeros)
        new_beneficiaries.append(updated(beneficiary, resp=resp_state))

    new_persons = []
    for person in state.persons:
        rrsp_state = person.rrsp
        tfsa_state = person.tfsa
        taxable_state = person.taxable

        if not first_year:
            rrsp_state = rrsp_mod.erode_nominal(rrsp_state, inflation_rate)
            tfsa_state = tfsa_mod.erode_nominal(tfsa_state, inflation_rate)
            taxable_state = taxable_mod.erode_nominal(taxable_state, inflation_rate)

            tfsa_state = tfsa_mod.restore_room(tfsa_state, real_params.tfsa)
            age_end = timeline.age_at_end_of_year(person.birth_year, person.birth_month, year)
            tfsa_room_accrued = tfsa_mod.room_accrued(
                age_end, real_params.tfsa, january_month_index
            )
            tfsa_state = updated(tfsa_state, room=tfsa_state.room + tfsa_room_accrued)

            rrsp_room_accrued = rrsp_mod.room_accrued(
                person.income.employment, real_params.rrif, january_month_index
            )
            rrsp_state = updated(rrsp_state, room=rrsp_state.room + rrsp_room_accrued)

        age_start = timeline.age_at_start_of_year(person.birth_year, person.birth_month, year)

        rrif_state = person.rrif
        rrif_annual_minimum = rrif_mod.minimum_withdrawal(
            rrif_state.balance, age_start, rrif_state.opened_year, year, real_params.rrif
        )
        rrif_state = updated(
            rrif_state, annual_minimum=rrif_annual_minimum, withdrawn_ytd=zeros
        )

        lif_state = person.lif
        if lif_state.jurisdiction == "":
            lif_annual_minimum = zeros
            lif_annual_maximum = zeros
        else:
            lif_annual_minimum = rrif_mod.minimum_withdrawal(
                lif_state.balance, age_start, lif_state.opened_year, year, real_params.rrif
            )
            jurisdiction_params = real_params.jurisdiction(lif_state.jurisdiction)
            if lif_mod.has_maximum(jurisdiction_params):
                lif_annual_maximum = lif_mod.maximum_withdrawal(
                    lif_state.balance, age_start, jurisdiction_params
                )
            else:
                lif_annual_maximum = np.full(n_paths, np.inf, dtype=np.float64)
        lif_state = updated(
            lif_state,
            annual_minimum=lif_annual_minimum,
            annual_maximum=lif_annual_maximum,
            withdrawn_ytd=zeros,
        )

        rrsp_state = updated(rrsp_state, contributed_ytd=zeros)

        new_income = updated(
            person.income,
            employment=zeros,
            cpp=zeros,
            oas=zeros,
            db_pension=zeros,
            rrsp_withdrawals=zeros,
            rrif_lif_withdrawals=zeros,
            interest=zeros,
            eligible_dividends=zeros,
            capital_gains=zeros,
            resp_accumulated_income=zeros,
            rrsp_deductions=zeros,
            cpp_base_contributions=zeros,
            cpp_enhanced_contributions=zeros,
            ei_premiums=zeros,
            remitted=zeros,
        )

        new_persons.append(
            updated(
                person,
                rrsp=rrsp_state,
                tfsa=tfsa_state,
                taxable=taxable_state,
                rrif=rrif_state,
                lif=lif_state,
                income=new_income,
            )
        )

    return updated(
        state,
        persons=tuple(new_persons),
        beneficiaries=tuple(new_beneficiaries),
        spending_achieved_ytd=zeros,
    )


def close_year(state: HouseholdState, real_params: RealParamYear) -> HouseholdState:
    """December phase: convert, assess the year, record it, carry the balance forward.

    Called by :func:`advance_month`, never directly.

    1. Assertions. On every path: every RRIF's and LIF's minimum has been met --
       ``withdrawn_ytd >= annual_minimum`` (allowing float slack) or the account's balance is
       zero -- and every person's ``balance_owing`` is exactly zero, because the filing month
       has already settled it. Phase 6 of :func:`advance_month` already forces the RRIF/LIF
       shortfall out once the year is down to its last month, so by the time this runs both
       assertions should already hold; they exist to catch a violation of ordering rather than
       to force anything themselves.
    2. Conversions, per person, with ``age_end = timeline.age_at_end_of_year(birth_year,
       birth_month, state.year)``: ``statutory = rrsp_mod.must_convert(age_end,
       real_params.rrif)``; ``elected = age_end == state.elections.rrif_conversion_age_years``.
       If ``statutory``, convert fraction ``1.0``; otherwise, if ``elected``, convert
       ``state.elections.rrif_conversion_fraction``; otherwise no RRSP conversion. Likewise a
       LIRA whose ``jurisdiction`` is set converts to the LIF once ``lira_mod.must_convert``
       fires for that jurisdiction. **The conversion trigger does not vary by path**: age and
       the household-wide election are the only inputs, so after #36 a dead holder's balance is
       zero and converts to zero, needing no per-path mask here. A RRIF or LIF opened this year
       carries no minimum until next January (L26, ``receive_conversion``), so no December
       sweep of the newly converted balance is needed.
    3. Assess the year in one joint step, on this year's brackets
       (``engine.tax.combined.household_assessment``). This is one assessment per person, the
       dead included: that function already gates only the pension split on ``alive``, so
       nothing here filters persons out of the assessment.
    4. Assessment less ``remitted`` is **assigned** to ``balance_owing`` (not added to it,
       which is already zero per item 1): negative is a refund, settled in next year's filing
       month.
    5. The pair of net-income fields shifts, computed from the *old* values: for every person
       alive at any point in the calendar year (``person.death_month_index >
       january_month_index``, strictly greater -- never subtracted from, since the
       :data:`~engine.core.state.DEATH_NOT_DRAWN` sentinel overflows),
       ``net_income_two_years_prior`` takes the value ``prior_year_net_income`` held all year,
       and ``prior_year_net_income`` takes this year's net income after the social benefits
       repayment (line 23600,
       ``Assessment.net_income_after_repayment``). A person not alive at any point in the year
       keeps both figures exactly as they were: the pair freezes at the close of the last
       calendar year they were alive in.
    6. The GIS band indicator (L2's "living pensioner"), per person: a household-combined check
       (``engine.benefits.gis.in_band``) on ``combined_income`` and ``combined_oas`` -- each
       person's contribution zeroed for a path where they are not alive, so a spouse who died
       during the year drops out of both sums and the survivor is tested on the single
       threshold, as ``band_threshold_annual``'s docstring describes -- and ``has_spouse``
       (both persons alive, all-False for a household of one). The per-person indicator is
       true only where that person is alive, receives OAS, and the household is in the band.
    7. Append the year to ``history``: a :class:`~engine.core.state.YearRecord` with this
       year's ``net_worth`` -- cash plus, summed over every person, the RRSP, RRIF, LIRA, LIF,
       TFSA and taxable balances, less item 4's ``balance_owing`` (the RESP is excluded, L34) --
       ``after_tax_net_worth`` (a copy of ``net_worth`` until #36), this year's
       ``spending_achieved_ytd``, ``tax_assessed`` (the sum, over every person's assessment, of
       ``total`` -- the AIP penalty and the OAS repayment included, everyone included), item 3's
       ``net_income`` per person, item 6's ``gis_band`` per person, and ``depleted``.

    Args:
        state: State at the end of December, with twelve months accumulated.
        real_params: Parameters for the tax year being closed, in the
            scenario's real-dollar view.

    Returns:
        State with the year converted, assessed, and recorded.

    Raises:
        AssertionError: If, on any path, a person's RRIF or LIF has ``withdrawn_ytd`` short of
            ``annual_minimum`` and a non-zero balance, or a person's ``balance_owing`` is not
            exactly zero on entry.
    """
    for person in state.persons:
        for name, account in (("rrif", person.rrif), ("lif", person.lif)):
            slack = 1e-9 * np.maximum(1, account.annual_minimum)
            met = (account.withdrawn_ytd >= account.annual_minimum - slack) | (
                account.balance == 0.0
            )
            if not np.all(met):
                raise AssertionError(
                    f"person {person.person_id!r} {name}: withdrawn_ytd "
                    f"{account.withdrawn_ytd!r} short of annual_minimum "
                    f"{account.annual_minimum!r} with balance {account.balance!r} "
                    "still non-zero on at least one path."
                )
        if not np.all(person.balance_owing == 0.0):
            raise AssertionError(
                f"person {person.person_id!r}: balance_owing {person.balance_owing!r} is not "
                "exactly zero on entry to close_year; the filing month must already have "
                "settled the prior year's balance before this close runs."
            )

    year = state.year
    january_month_index = state.month_index - (state.month - 1)

    converted_persons: list[PersonState] = []
    for person in state.persons:
        age_end = timeline.age_at_end_of_year(person.birth_year, person.birth_month, year)

        rrsp_state = person.rrsp
        rrif_state = person.rrif
        statutory = rrsp_mod.must_convert(age_end, real_params.rrif)
        elected = age_end == state.elections.rrif_conversion_age_years
        if statutory:
            rrsp_state, moved = rrsp_mod.convert(rrsp_state, 1.0)
            rrif_state = rrif_mod.receive_conversion(rrif_state, moved, year)
        elif elected:
            rrsp_state, moved = rrsp_mod.convert(
                rrsp_state, state.elections.rrif_conversion_fraction
            )
            rrif_state = rrif_mod.receive_conversion(rrif_state, moved, year)

        lira_state = person.lira
        lif_state = person.lif
        if lira_state.jurisdiction != "" and lira_mod.must_convert(
            age_end, real_params.jurisdiction(lira_state.jurisdiction)
        ):
            lira_state, lif_state = lira_mod.convert_to_lif(lira_state, lif_state, year)

        converted_persons.append(
            updated(person, rrsp=rrsp_state, rrif=rrif_state, lira=lira_state, lif=lif_state)
        )

    state_after_conversions = updated(state, persons=tuple(converted_persons))
    assessments = household_assessment(state_after_conversions, real_params)

    final_persons: list[PersonState] = []
    for person, assessment in zip(converted_persons, assessments, strict=True):
        balance_owing = assessment.total - person.income.remitted

        mask = person.death_month_index > january_month_index
        new_two_years_prior = np.where(
            mask, person.prior_year_net_income, person.net_income_two_years_prior
        )
        new_prior_year = np.where(
            mask, assessment.net_income_after_repayment, person.prior_year_net_income
        )

        final_persons.append(
            updated(
                person,
                balance_owing=balance_owing,
                prior_year_net_income=new_prior_year,
                net_income_two_years_prior=new_two_years_prior,
            )
        )

    alive_flags = tuple(person.alive for person in final_persons)
    if len(final_persons) == 2:
        has_spouse = alive_flags[0] & alive_flags[1]
    else:
        has_spouse = np.zeros(state.n_paths, dtype=np.bool_)

    combined_income = np.zeros(state.n_paths, dtype=np.float64)
    combined_oas = np.zeros(state.n_paths, dtype=np.float64)
    for person, assessment, alive in zip(final_persons, assessments, alive_flags, strict=True):
        combined_income = combined_income + np.where(
            alive, assessment.net_income_after_repayment, 0.0
        )
        combined_oas = combined_oas + np.where(alive, person.income.oas, 0.0)

    household_in_band = gis_mod.in_band(
        combined_income, combined_oas, has_spouse, january_month_index, real_params.oas
    )
    gis_band = tuple(
        alive & (person.income.oas > 0) & household_in_band
        for person, alive in zip(final_persons, alive_flags, strict=True)
    )

    net_worth = state.cash.balance.copy()
    balance_owing_total = np.zeros(state.n_paths, dtype=np.float64)
    for person in final_persons:
        net_worth = (
            net_worth
            + person.rrsp.balance
            + person.rrif.balance
            + person.lira.balance
            + person.lif.balance
            + person.tfsa.balance
            + person.taxable.balance
        )
        balance_owing_total = balance_owing_total + person.balance_owing
    net_worth = net_worth - balance_owing_total

    tax_assessed = np.zeros(state.n_paths, dtype=np.float64)
    for assessment in assessments:
        tax_assessed = tax_assessed + assessment.total

    year_record = YearRecord(
        year=state.year,
        net_worth=net_worth,
        after_tax_net_worth=net_worth.copy(),
        spending=state.spending_achieved_ytd,
        tax_assessed=tax_assessed,
        net_income=tuple(assessment.net_income_after_repayment for assessment in assessments),
        gis_band=gis_band,
        depleted=state.depleted,
    )
    return updated(
        state_after_conversions,
        persons=tuple(final_persons),
        history=(*state.history, year_record),
    )


def settle_tax_balance(state: HouseholdState, real_params: RealParamYear) -> HouseholdState:
    """Filing-month phase: pay the prior year's balance owing in cash.

    Called by :func:`advance_month`, never directly. The month comes from
    ``real_params``; it is not written into the code.

    The settlement is a **debit** to household cash at phase 5 (a refund is a deposit); it
    realizes nothing itself. If paying it takes cash negative, phase 9's floor funds the
    shortfall in the policy's withdrawal order, and it is that forced withdrawal — not this
    phase — that can realize a capital gain.

    Per person: ``cash_mod.pay`` the amount ``max(balance_owing, 0)`` and ``cash_mod.deposit``
    the amount ``max(-balance_owing, 0)``. Both of those account functions refuse a negative
    argument, which is exactly why the amount owed is split into the two non-negative pieces
    rather than paid or deposited as one signed amount. ``balance_owing`` is then zeroed for
    every person, on every path, regardless of sign.

    Args:
        state: State in the filing month; ``balance_owing`` may be positive, negative, or
            zero on any path.
        real_params: Parameters for the current tax year, in the scenario's
            real-dollar view, supplying the filing month.

    Returns:
        State with every person's balance owing paid or refunded, and ``balance_owing`` zeroed.
    """
    cash = state.cash
    new_persons = []
    for person in state.persons:
        owing = person.balance_owing
        cash = cash_mod.pay(cash, np.clip(owing, 0, None))
        cash = cash_mod.deposit(cash, np.clip(-owing, 0, None))
        new_persons.append(updated(person, balance_owing=np.zeros_like(owing)))
    return updated(state, persons=tuple(new_persons), cash=cash)


def resolve_deaths(
    state: HouseholdState,
    real_params: RealParamYear,
) -> HouseholdState:
    """Apply mortality for this month and its immediate consequences.

    Monthly, not annual: a death in March stops three quarters of a year of OAS that an annual
    step would pay in full, and the final return covers only the part-year to death.

    Consequences in the same month: OAS ceases for the deceased from the following month (GIS is
    never paid, L2), a RRIF rolls to a surviving spouse tax-deferred, and with no survivor the
    registered balance comes fully into income in the year of death.

    Every ``death_month_index`` is **pre-resolved** by :func:`engine.core.build.draw_deaths`
    before the run starts, so this phase only ever *acts* on a month where
    ``death_month_index == month_index`` for some person; it does not draw anything itself.
    Returns ``state`` unchanged until #36.

    Args:
        state: Opening state for the month.
        real_params: Parameters for the current tax year, in the scenario's
            real-dollar view.

    Returns:
        State with ``alive`` and any rollover applied. Unchanged until #36.
    """
    return state

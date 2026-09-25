# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Household assessment: pension splitting, the OAS repayment, and the AIP special tax.

``HouseholdState`` fixtures here are small, private builders local to this
file (not imported from ``tests/core/test_state.py``), modelled on that
file's ``_zeros``/``_person``/``_income_ledger`` pattern. All tests run
against the real 2026 parameter files at zero inflation, so no relationship
asserted below restates a parameter value — every number it depends on is
read back out of the same ``RealParamSet``/``RealParamYear`` the
implementation reads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.core.indexation import RealParamYear, real_year
from engine.core.state import (
    DEATH_NOT_DRAWN,
    BenefitState,
    CashState,
    Elections,
    HouseholdState,
    IncomeLedger,
    LifState,
    LiraState,
    PersonState,
    RrifState,
    RrspState,
    SpendingLevel,
    TaxableState,
    TfsaState,
)
from engine.core.timeline import age_at_end_of_year
from engine.params.loader import load_year
from engine.tax import federal, provincial
from engine.tax.combined import (
    GRID_STEP,
    Assessment,
    household_assessment,
    oas_repayment,
    person_assessment,
)

START_YEAR = 2026

LEDGER_FIELDS = (
    "employment",
    "cpp",
    "oas",
    "db_pension",
    "rrsp_withdrawals",
    "rrif_lif_withdrawals",
    "interest",
    "eligible_dividends",
    "capital_gains",
    "resp_accumulated_income",
    "rrsp_deductions",
    "cpp_base_contributions",
    "cpp_enhanced_contributions",
    "ei_premiums",
    "remitted",
)


def _zeros(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float64)


def _bools(value: bool, n: int) -> np.ndarray:
    return np.full(n, value, dtype=np.bool_)


def _ledger(n: int, **overrides: object) -> IncomeLedger:
    fields = {name: _zeros(n) for name in LEDGER_FIELDS}
    for key, value in overrides.items():
        fields[key] = np.full(n, value, dtype=np.float64)
    return IncomeLedger(**fields)


def _person(
    person_id: str,
    n: int,
    *,
    birth_year: int = 1955,
    birth_month: int = 1,
    alive: np.ndarray | None = None,
    income: IncomeLedger | None = None,
) -> PersonState:
    alive_arr = alive if alive is not None else _bools(True, n)
    # Anyone not alive needs a real death_month_index: DEATH_NOT_DRAWN paired
    # with alive=False is a contradiction PersonState rejects.
    death_month_index = np.where(alive_arr, DEATH_NOT_DRAWN, 0).astype(np.int64)
    return PersonState(
        person_id=person_id,
        sex="f",
        birth_year=birth_year,
        birth_month=birth_month,
        alive=alive_arr,
        death_month_index=death_month_index,
        rrsp=RrspState(
            balance=_zeros(n),
            room=_zeros(n),
            contributed_ytd=_zeros(n),
            converted_fraction_applied=False,
        ),
        rrif=RrifState(
            balance=_zeros(n),
            annual_minimum=_zeros(n),
            withdrawn_ytd=_zeros(n),
            opened_year=None,
        ),
        lira=LiraState(balance=_zeros(n), jurisdiction=""),
        lif=LifState(
            balance=_zeros(n),
            jurisdiction="",
            annual_minimum=_zeros(n),
            annual_maximum=_zeros(n),
            withdrawn_ytd=_zeros(n),
            opened_year=None,
        ),
        tfsa=TfsaState(balance=_zeros(n), room=_zeros(n), withdrawn_this_year=_zeros(n)),
        taxable=TaxableState(balance=_zeros(n), acb=_zeros(n)),
        cpp=BenefitState(
            start_age_months=780,
            in_pay_monthly=None,
            contributory_history=0.5,
            monthly_amount=_zeros(n),
        ),
        oas=BenefitState(
            start_age_months=780,
            in_pay_monthly=None,
            contributory_history=None,
            monthly_amount=_zeros(n),
        ),
        employment=(),
        pensions=(),
        income=income if income is not None else _ledger(n),
        balance_owing=_zeros(n),
        prior_year_net_income=_zeros(n),
        net_income_two_years_prior=_zeros(n),
    )


def _household(
    persons: tuple[PersonState, ...],
    *,
    year: int,
    n: int,
    province: str = "ab",
    month: int = 12,
) -> HouseholdState:
    """A household at the close of ``month``. ``month_index`` counts from January ``START_YEAR``."""
    month_index = (year - START_YEAR) * 12 + (month - 1)
    return HouseholdState(
        year=year,
        month=month,
        month_index=month_index,
        n_paths=n,
        province=province,
        persons=persons,
        beneficiaries=(),
        cash=CashState(balance=_zeros(n)),
        elections=Elections(
            cpp_start_age_months=(780,) * len(persons),
            oas_start_age_months=(780,) * len(persons),
            rrif_conversion_age_years=71,
            rrif_conversion_fraction=1.0,
            fill_pension_credit=False,
        ),
        spending_schedule=(SpendingLevel(from_year=year, monthly_level=1000.0),),
        spending_monthly=1000.0,
        spending_survivor_share=0.6,
        spending_achieved_ytd=_zeros(n),
        depleted=_bools(False, n),
        estate_after_tax=np.full(n, np.nan),
        history=(),
    )


@pytest.fixture
def params():
    return real_year(load_year(2026), 0.0)


JANUARY = 0


# =============================================================================
# oas_repayment
# =============================================================================


def test_oas_repayment_zero_at_and_below_threshold(params) -> None:
    threshold = params.oas.annual_amount("recovery_tax.threshold_annual", JANUARY)
    assert oas_repayment(threshold, 5000.0, params.oas, JANUARY) == pytest.approx(0.0)
    assert oas_repayment(threshold - 1000.0, 5000.0, params.oas, JANUARY) == pytest.approx(0.0)


def test_oas_repayment_rate_times_excess_above_threshold(params) -> None:
    threshold = params.oas.annual_amount("recovery_tax.threshold_annual", JANUARY)
    rate = params.oas.number("recovery_tax.rate")
    result = oas_repayment(threshold + 10_000.0, 1_000_000.0, params.oas, JANUARY)
    assert result == pytest.approx(rate * 10_000.0)


def test_oas_repayment_capped_at_oas_received(params) -> None:
    threshold = params.oas.annual_amount("recovery_tax.threshold_annual", JANUARY)
    result = oas_repayment(threshold + 1_000_000.0, 500.0, params.oas, JANUARY)
    assert result == pytest.approx(500.0)


# =============================================================================
# person_assessment / household_assessment
# =============================================================================


def test_one_person_household_equals_person_assessment_at_zero_transfers(params) -> None:
    n = 2
    income = _ledger(n, employment=80_000.0, eligible_dividends=1000.0)
    person = _person("a", n, income=income)
    household = _household((person,), year=2026, n=n)

    (result,) = household_assessment(household, params)
    zero = np.zeros(n, dtype=np.float64)
    age = age_at_end_of_year(person.birth_year, person.birth_month, 2026)
    expected = person_assessment(income, age, zero, zero, "ab", params, JANUARY)

    np.testing.assert_allclose(result.total, expected.total)
    np.testing.assert_allclose(result.net_income, expected.net_income)


def test_assessment_total_equals_sum_of_parts(params) -> None:
    n = 3
    income = _ledger(n, employment=50_000.0, db_pension=10_000.0, oas=8_000.0)
    result = person_assessment(income, 66, np.zeros(n), np.zeros(n), "ab", params, JANUARY)
    np.testing.assert_allclose(
        result.total, result.federal + result.provincial + result.oas_repayment
    )


# =============================================================================
# Assessment.aip_penalty: the RESP accumulated-income special tax
# =============================================================================


def test_aip_penalty_is_the_rate_times_the_amount_per_path(params) -> None:
    rate = params.resp.number("aip.penalty_rate")
    n = 3
    fields = {name: _zeros(n) for name in LEDGER_FIELDS}
    fields["employment"] = np.full(n, 40_000.0, dtype=np.float64)
    fields["resp_accumulated_income"] = np.array([0.0, 10_000.0, 40_000.0], dtype=np.float64)
    income = IncomeLedger(**fields)

    result = person_assessment(income, 50, np.zeros(n), np.zeros(n), "ab", params, JANUARY)

    np.testing.assert_allclose(result.aip_penalty, income.resp_accumulated_income * rate)
    assert result.aip_penalty[0] == 0.0
    np.testing.assert_allclose(
        result.total,
        result.federal + result.provincial + result.oas_repayment + result.aip_penalty,
    )


def test_aip_penalty_is_isolated_from_every_other_line(params) -> None:
    # interest is plain line-12100 income with no credit of its own, so moving
    # the same amount from resp_accumulated_income to interest must leave
    # every other line unchanged — the penalty is the only difference.
    #
    # Employment 80,000 + OAS 5,000 + AIP 20,000 is chosen so the repayment
    # sits strictly between zero and OAS received (a repayment pinned at its
    # cap would not move if the penalty leaked into the income test) and so
    # net_income_after_repayment lands inside the federal age amount's
    # phase-out band (a penalty leaking into the credits' income argument
    # would otherwise go undetected).
    n = 1
    age = 66
    amount = 20_000.0
    income_a = _ledger(n, employment=80_000.0, oas=5_000.0, resp_accumulated_income=amount)
    income_b = _ledger(n, employment=80_000.0, oas=5_000.0, interest=amount)

    a = person_assessment(income_a, age, np.zeros(n), np.zeros(n), "ab", params, JANUARY)
    b = person_assessment(income_b, age, np.zeros(n), np.zeros(n), "ab", params, JANUARY)

    # Control: proves this test actually exercises the OAS repayment path —
    # a partial repayment, not one vacuously zero or pinned at its cap.
    assert np.all((b.oas_repayment > 0) & (b.oas_repayment < income_b.oas))

    # Control: proves this test actually exercises the age amount's
    # phase-out, not just its full or fully-clawed-back ends.
    fed = params.federal
    age_threshold = fed.annual_amount("credits.age_amount.reduction_threshold_annual", JANUARY)
    age_amount = fed.annual_amount("credits.age_amount.amount_annual", JANUARY)
    age_rate = fed.number("credits.age_amount.reduction_rate")
    assert np.all(
        (b.net_income_after_repayment > age_threshold)
        & (b.net_income_after_repayment < age_threshold + age_amount / age_rate)
    )

    np.testing.assert_allclose(a.federal, b.federal)
    np.testing.assert_allclose(a.provincial, b.provincial)
    np.testing.assert_allclose(a.oas_repayment, b.oas_repayment)
    np.testing.assert_allclose(a.net_income, b.net_income)
    np.testing.assert_allclose(a.net_income_after_repayment, b.net_income_after_repayment)
    np.testing.assert_allclose(a.total - b.total, a.aip_penalty)


def test_aip_penalty_is_isolated_from_the_alberta_age_amount_band(params) -> None:
    # A second isolation case alongside test_aip_penalty_is_isolated_from_every
    # _other_line: this one lands net_income_after_repayment inside Alberta's
    # own age amount phase-out band rather than the federal one, since
    # provincial.non_refundable_credits reads Alberta's own
    # credits.age_amount.* keys and its band need not coincide with the
    # federal band. No OAS is received, so the repayment is trivially zero
    # regardless of net income -- a control below, not an assumption.
    n = 1
    age = 66
    amount = 20_000.0
    ab = params.province("ab")
    threshold = ab.annual_amount("credits.age_amount.reduction_threshold_annual", JANUARY)
    age_amount = ab.annual_amount("credits.age_amount.amount_annual", JANUARY)
    reduction_rate = ab.number("credits.age_amount.reduction_rate")
    lower = threshold
    upper = threshold + age_amount / reduction_rate
    target_net_income = (lower + upper) / 2.0
    employment = target_net_income - amount

    income_a = _ledger(n, employment=employment, resp_accumulated_income=amount)
    income_b = _ledger(n, employment=employment, interest=amount)

    a = person_assessment(income_a, age, np.zeros(n), np.zeros(n), "ab", params, JANUARY)
    b = person_assessment(income_b, age, np.zeros(n), np.zeros(n), "ab", params, JANUARY)

    # Control: no OAS is received, so the repayment is trivially zero.
    assert np.all(a.oas_repayment == 0.0)
    assert np.all(b.oas_repayment == 0.0)

    # Control: proves this test actually exercises Alberta's age amount
    # phase-out, not just its full or fully-clawed-back ends.
    assert np.all((b.net_income_after_repayment > lower) & (b.net_income_after_repayment < upper))

    np.testing.assert_allclose(a.federal, b.federal)
    np.testing.assert_allclose(a.provincial, b.provincial)
    np.testing.assert_allclose(a.oas_repayment, b.oas_repayment)
    np.testing.assert_allclose(a.net_income, b.net_income)
    np.testing.assert_allclose(a.net_income_after_repayment, b.net_income_after_repayment)
    np.testing.assert_allclose(a.total - b.total, a.aip_penalty)


def test_aip_penalty_is_exactly_zero_with_no_aip(params) -> None:
    n = 1
    income = _ledger(n, employment=60_000.0, db_pension=5_000.0)
    result = person_assessment(income, 66, np.zeros(n), np.zeros(n), "ab", params, JANUARY)

    assert np.all(result.aip_penalty == 0.0)
    np.testing.assert_allclose(
        result.total, result.federal + result.provincial + result.oas_repayment
    )


def test_household_assessment_carries_the_aip_penalty_through_the_split_election(
    params,
) -> None:
    rate = params.resp.number("aip.penalty_rate")
    n = 1
    amount = 15_000.0
    db_pension = 60_000.0
    income0 = _ledger(n, db_pension=db_pension, resp_accumulated_income=amount)
    income1 = _ledger(n)
    person0 = _person("a", n, birth_year=1950, income=income0)
    person1 = _person("b", n, birth_year=1950, income=income1)
    household = _household((person0, person1), year=2026, n=n)
    a0, a1 = household_assessment(household, params)

    control_income0 = _ledger(n, db_pension=db_pension, interest=amount)
    control_person0 = _person("a", n, birth_year=1950, income=control_income0)
    control_household = _household((control_person0, person1), year=2026, n=n)
    control_a0, control_a1 = household_assessment(control_household, params)

    for field in ("federal", "provincial", "oas_repayment", "net_income"):
        np.testing.assert_allclose(getattr(a0, field), getattr(control_a0, field))
        np.testing.assert_allclose(getattr(a1, field), getattr(control_a1, field))

    np.testing.assert_allclose(a0.aip_penalty, amount * rate)
    assert np.all(a1.aip_penalty == 0.0)
    np.testing.assert_allclose(a0.total - control_a0.total, a0.aip_penalty)

    # Control case: the elected split must actually be exercised here, or the
    # equalities above would pass vacuously on a household that never splits.
    zero = np.zeros(n, dtype=np.float64)
    age = age_at_end_of_year(1950, 1, 2026)
    unsplit0 = person_assessment(income0, age, zero, zero, "ab", params, JANUARY)
    assert np.all(np.abs(a0.net_income - unsplit0.net_income) > 1e-6)


def _pension_split_grid(
    income0: IncomeLedger,
    income1: IncomeLedger,
    age: int,
    params: RealParamYear,
) -> tuple[list[Assessment], list[Assessment]]:
    """The candidate grid ``household_assessment`` searches, built from ``person_assessment``.

    Both transfer directions, in ``household_assessment``'s own grid order —
    the same construction ``test_household_assessment_elects_as_the_grid_does_with_no_aip``
    builds inline.

    Assumes both persons are alive on every path and share one age at the end
    of the year; household_assessment's both_alive mask and per-person ages are
    not reproduced.
    """
    epi0 = federal.eligible_pension_income(income0, age, params.federal)
    epi1 = federal.eligible_pension_income(income1, age, params.federal)

    maximum_share = params.federal.number("pension_splitting.maximum_transfer_share")
    magnitudes: list[float] = []
    k = 0
    while k * GRID_STEP < maximum_share:
        magnitudes.append(k * GRID_STEP)
        k += 1
    magnitudes.append(maximum_share)
    fractions: list[float] = [*magnitudes, *(-m for m in magnitudes[1:])]

    zero = np.zeros_like(epi0)
    candidates0: list[Assessment] = []
    candidates1: list[Assessment] = []
    for fraction in fractions:
        if fraction >= 0:
            transfer = fraction * epi0
            t_in0, t_out0 = zero, transfer
            t_in1, t_out1 = transfer, zero
        else:
            transfer = -fraction * epi1
            t_in0, t_out0 = transfer, zero
            t_in1, t_out1 = zero, transfer
        candidates0.append(person_assessment(income0, age, t_in0, t_out0, "ab", params, JANUARY))
        candidates1.append(person_assessment(income1, age, t_in1, t_out1, "ab", params, JANUARY))
    return candidates0, candidates1


def test_household_assessment_election_is_blind_to_the_aip_penalty(params) -> None:
    # A regression case: at these specific incomes, several split candidates
    # give mathematically equal household tax before the AIP penalty.
    # Because the penalty is the same for every candidate but
    # still perturbs float rounding of "total", electing on "total" instead
    # of the split-dependent lines picks a different tied candidate whenever
    # the penalty happens to be present. Moving the same amount from
    # resp_accumulated_income to interest removes the penalty without
    # changing anything the split election should react to, so a correct
    # election must land on the same split either way.
    n = 1
    db_pension0 = 137082.66972820694
    amount = 16135.661503842332
    db_pension1 = 132489.37310023705
    income0_aip = _ledger(n, db_pension=db_pension0, resp_accumulated_income=amount)
    income0_interest = _ledger(n, db_pension=db_pension0, interest=amount)
    income1 = _ledger(n, db_pension=db_pension1)
    person1 = _person("b", n, birth_year=1950, income=income1)

    household_aip = _household(
        (_person("a", n, birth_year=1950, income=income0_aip), person1), year=2026, n=n
    )
    household_interest = _household(
        (_person("a", n, birth_year=1950, income=income0_interest), person1), year=2026, n=n
    )
    a0_aip, a1_aip = household_assessment(household_aip, params)
    a0_interest, a1_interest = household_assessment(household_interest, params)

    for field in ("net_income", "federal", "provincial", "oas_repayment"):
        np.testing.assert_allclose(getattr(a0_aip, field), getattr(a0_interest, field))
        np.testing.assert_allclose(getattr(a1_aip, field), getattr(a1_interest, field))

    # Control: proves this couple still sits on a tie that electing on "total"
    # would resolve differently from electing on the split-dependent lines --
    # the bug this test guards against. If this fails after a parameter
    # change, the tie has dissolved and a new tied couple is needed.
    age = age_at_end_of_year(1950, 1, 2026)
    grid0, grid1 = _pension_split_grid(income0_aip, income1, age, params)
    total_argmin = int(
        np.argmin(
            np.stack([c0.total + c1.total for c0, c1 in zip(grid0, grid1, strict=True)], axis=0),
            axis=0,
        )[0]
    )
    elected_lines_argmin = int(
        np.argmin(
            np.stack(
                [
                    (c0.federal + c0.provincial + c0.oas_repayment)
                    + (c1.federal + c1.provincial + c1.oas_repayment)
                    for c0, c1 in zip(grid0, grid1, strict=True)
                ],
                axis=0,
            ),
            axis=0,
        )[0]
    )
    assert total_argmin != elected_lines_argmin

    # Ties the control above to the engine's actual election: proves the
    # engine elects the same candidate the helper's elected-lines argmin
    # does (so the helper's tie is the engine's tie), and that the two tied
    # candidates genuinely differ in net_income -- the quantity the main
    # assertion above compares.
    assert (a0_aip.net_income == grid0[elected_lines_argmin].net_income).all()
    assert (a1_aip.net_income == grid1[elected_lines_argmin].net_income).all()
    assert (grid0[total_argmin].net_income != grid0[elected_lines_argmin].net_income).all()


def test_household_assessment_elects_as_the_grid_does_with_no_aip(params) -> None:
    # A regression test: household_assessment must sum each person's
    # federal + provincial + oas_repayment left to right, in the same
    # grouping person_assessment uses for "total", or a mathematically
    # tied comparison elects a different split on rounding alone. With no
    # AIP the penalty is zero for every candidate, so the split that
    # minimises the elected lines is the same split that minimising "total"
    # gives -- the election here must be the one electing on "total" gives.
    n = 1
    age = age_at_end_of_year(1950, 1, 2026)
    db_pension0 = 119629.52337340865
    db_pension1 = 144581.80318130302
    income0 = _ledger(n, db_pension=db_pension0)
    income1 = _ledger(n, db_pension=db_pension1)
    person0 = _person("a", n, birth_year=1950, income=income0)
    person1 = _person("b", n, birth_year=1950, income=income1)
    household = _household((person0, person1), year=2026, n=n)

    a0, a1 = household_assessment(household, params)

    # The same candidate grid household_assessment searches, built entirely
    # from person_assessment: both directions of transfer, since both
    # persons here carry eligible pension income of their own.
    np.testing.assert_allclose(
        federal.eligible_pension_income(income0, age, params.federal), db_pension0
    )
    np.testing.assert_allclose(
        federal.eligible_pension_income(income1, age, params.federal), db_pension1
    )

    maximum_share = params.federal.number("pension_splitting.maximum_transfer_share")
    magnitudes: list[float] = []
    k = 0
    while k * GRID_STEP < maximum_share:
        magnitudes.append(k * GRID_STEP)
        k += 1
    magnitudes.append(maximum_share)
    fractions: list[float] = [*magnitudes, *(-m for m in magnitudes[1:])]

    zero = np.zeros(n, dtype=np.float64)
    candidates0: list[Assessment] = []
    candidates1: list[Assessment] = []
    for fraction in fractions:
        if fraction >= 0:
            transfer = np.full(n, fraction * db_pension0)
            t_in0, t_out0 = zero, transfer
            t_in1, t_out1 = transfer, zero
        else:
            transfer = np.full(n, -fraction * db_pension1)
            t_in0, t_out0 = transfer, zero
            t_in1, t_out1 = zero, transfer
        candidates0.append(person_assessment(income0, age, t_in0, t_out0, "ab", params, JANUARY))
        candidates1.append(person_assessment(income1, age, t_in1, t_out1, "ab", params, JANUARY))

    assert all(np.all(c.aip_penalty == 0.0) for c in (*candidates0, *candidates1))

    grid_totals = np.stack(
        [c0.total + c1.total for c0, c1 in zip(candidates0, candidates1, strict=True)], axis=0
    )
    best = int(np.argmin(grid_totals, axis=0)[0])

    assert (a0.net_income == candidates0[best].net_income).all()
    assert (a1.net_income == candidates1[best].net_income).all()

    # Control: proves this couple still sits on a tie that distinguishes the
    # grouping household_assessment sums in (each person's federal +
    # provincial + oas_repayment first, then across persons) from a wrong
    # grouping (each line across both persons first) that lands on the same
    # total mathematically but differs on float rounding. If this fails
    # after a parameter change, the tie has dissolved and a new tied couple
    # is needed.
    wrong = np.stack(
        [
            (c0.federal + c1.federal)
            + (c0.provincial + c1.provincial)
            + (c0.oas_repayment + c1.oas_repayment)
            for c0, c1 in zip(candidates0, candidates1, strict=True)
        ],
        axis=0,
    )
    assert int(np.argmin(wrong, axis=0)[0]) != int(np.argmin(grid_totals, axis=0)[0])


# =============================================================================
# Assessment.oas_repayment: a non-zero repayment inside a full assessment
# =============================================================================


def test_assessment_oas_repayment_uncapped_equals_rate_times_excess(params) -> None:
    threshold = params.oas.annual_amount("recovery_tax.threshold_annual", JANUARY)
    rate = params.oas.number("recovery_tax.rate")
    n = 1
    # employment alone puts net income well above the threshold; the OAS
    # received (50_000) is large enough that rate * excess does not reach it.
    income = _ledger(n, employment=threshold, oas=50_000.0)
    result = person_assessment(income, 66, np.zeros(n), np.zeros(n), "ab", params, JANUARY)

    excess = result.net_income - threshold
    assert np.all(excess > 0)
    np.testing.assert_allclose(result.oas_repayment, rate * excess)


def test_assessment_oas_repayment_capped_at_oas_received(params) -> None:
    threshold = params.oas.annual_amount("recovery_tax.threshold_annual", JANUARY)
    n = 1
    # Large income, small OAS: rate * excess vastly exceeds the OAS received,
    # so the repayment must be capped at it exactly.
    income = _ledger(n, employment=threshold + 1_000_000.0, oas=100.0)
    result = person_assessment(income, 66, np.zeros(n), np.zeros(n), "ab", params, JANUARY)

    np.testing.assert_allclose(result.oas_repayment, income.oas)
    assert np.all(result.net_income_after_repayment >= 0.0)


def test_assessment_net_income_after_repayment_is_zero_with_no_income(params) -> None:
    """Pins the net_income == 0 edge: repayment is zero, equal to net income, not below it."""
    n = 1
    income = _ledger(n)
    result = person_assessment(income, 66, np.zeros(n), np.zeros(n), "ab", params, JANUARY)

    assert np.all(result.net_income == 0.0)
    assert np.all(result.oas_repayment == 0.0)
    assert np.all(result.net_income_after_repayment == 0.0)


def test_assessment_net_income_after_repayment_equals_net_income_at_the_threshold(params) -> None:
    threshold = params.oas.annual_amount("recovery_tax.threshold_annual", JANUARY)
    n = 1
    oas_received = 5_000.0
    income = _ledger(n, employment=threshold - oas_received, oas=oas_received)
    result = person_assessment(income, 66, np.zeros(n), np.zeros(n), "ab", params, JANUARY)

    np.testing.assert_allclose(result.net_income, threshold)
    assert np.all(result.oas_repayment == 0.0)
    np.testing.assert_allclose(result.net_income_after_repayment, result.net_income)


def test_assessment_net_income_unchanged_by_a_binding_repayment(params) -> None:
    n = 1
    income = _ledger(n, employment=200_000.0, oas=100.0)
    result = person_assessment(income, 66, np.zeros(n), np.zeros(n), "ab", params, JANUARY)

    # The repayment binds (see the cap test above), but net income (line 23400)
    # is never adjusted for it: it must equal the figure federal.net_income
    # computes directly, with no split. Line 23600 is what does subtract it —
    # see net_income_after_repayment below.
    expected_net_income = federal.net_income(income, params.federal, 0.0, 0.0)
    np.testing.assert_allclose(result.net_income, expected_net_income)


def test_assessment_taxes_net_income_after_the_repayment(params) -> None:
    # The bug this issue fixes: gross_tax and the age amount were evaluated on
    # net income (line 23400) rather than net income less the OAS repayment
    # (line 23600). Built entirely from the engine's own functions, so it
    # would have caught the bug the worked figures in the issue describe.
    n = 1
    age = 66
    income = _ledger(n, employment=200_000.0, oas=100.0)
    result = person_assessment(income, age, np.zeros(n), np.zeros(n), "ab", params, JANUARY)

    assert np.all(result.oas_repayment > 0.0)
    base = result.net_income - result.oas_repayment
    epi = federal.eligible_pension_income(income, age, params.federal)

    expected_federal = federal.net_tax(
        federal.gross_tax(base, params.federal, JANUARY),
        federal.non_refundable_credits(
            base,
            age,
            epi,
            income.cpp_base_contributions,
            income.ei_premiums,
            income.eligible_dividends,
            params.federal,
            JANUARY,
        ),
    )
    expected_provincial = provincial.net_tax(
        provincial.gross_tax(base, params.province("ab"), JANUARY),
        provincial.non_refundable_credits(
            base,
            age,
            epi,
            income.cpp_base_contributions,
            income.ei_premiums,
            income.eligible_dividends,
            params.province("ab"),
            params.federal,
            JANUARY,
        ),
    )
    np.testing.assert_allclose(result.federal, expected_federal)
    np.testing.assert_allclose(result.provincial, expected_provincial)


def test_assessment_age_amount_taxed_after_the_repayment(params) -> None:
    # Every other case in this file sits at an income where the federal age
    # amount has already ground to zero on both net income and net income
    # less the repayment, so nothing pins the age-amount half of the fix.
    # This picks an income strictly between the OAS recovery threshold and
    # the point the federal age amount grinds out — the window where the
    # repayment binds but the age amount has not yet reached zero on either
    # basis — so the age amount computed on line 23600 differs from the age
    # amount that would have been computed on line 23400.
    fed = params.federal
    oas_recovery_threshold = params.oas.annual_amount("recovery_tax.threshold_annual", JANUARY)
    age_amount = fed.annual_amount("credits.age_amount.amount_annual", JANUARY)
    age_reduction_threshold = fed.annual_amount(
        "credits.age_amount.reduction_threshold_annual", JANUARY
    )
    age_reduction_rate = fed.number("credits.age_amount.reduction_rate")
    age_amount_grind_out = age_reduction_threshold + age_amount / age_reduction_rate
    assert oas_recovery_threshold < age_amount_grind_out

    target_net_income = (oas_recovery_threshold + age_amount_grind_out) / 2.0
    age = fed.number("credits.age_amount.eligibility_age_years")

    n = 1
    oas_received = 20_000.0
    income = _ledger(n, employment=target_net_income - oas_received, oas=oas_received)
    result = person_assessment(income, age, np.zeros(n), np.zeros(n), "ab", params, JANUARY)

    assert np.all(result.oas_repayment > 0.0)
    assert np.all(result.oas_repayment < income.oas)  # uncapped: isolates the age-amount effect
    base = result.net_income - result.oas_repayment
    epi = federal.eligible_pension_income(income, age, params.federal)

    expected_federal = federal.net_tax(
        federal.gross_tax(base, params.federal, JANUARY),
        federal.non_refundable_credits(
            base,
            age,
            epi,
            income.cpp_base_contributions,
            income.ei_premiums,
            income.eligible_dividends,
            params.federal,
            JANUARY,
        ),
    )
    np.testing.assert_allclose(result.federal, expected_federal)

    # The credit must actually move between the two income bases: at an
    # income where the age amount is already zero (or already unclawed) on
    # both, the assertion above would still pass without the fix this issue
    # makes, which is exactly the hole this test closes.
    credits_on_taxable_income = federal.non_refundable_credits(
        result.net_income_after_repayment,
        age,
        epi,
        income.cpp_base_contributions,
        income.ei_premiums,
        income.eligible_dividends,
        params.federal,
        JANUARY,
    )
    credits_on_net_income = federal.non_refundable_credits(
        result.net_income,
        age,
        epi,
        income.cpp_base_contributions,
        income.ei_premiums,
        income.eligible_dividends,
        params.federal,
        JANUARY,
    )
    assert np.all(credits_on_taxable_income > credits_on_net_income)

    # Alberta is not also asserted here: its age amount grinds out at its own
    # reduction_threshold_annual + amount_annual / reduction_rate, which sits
    # below the federal recovery threshold, so the provincial age amount is
    # structurally always zero whenever a repayment binds. A future change to
    # Alberta's parameters that moved its grind-out above the federal
    # recovery threshold would silently lose this case, so a provincial
    # assertion belongs here the day that stops holding.


def test_net_income_after_repayment_equals_net_income_at_zero_repayment(params) -> None:
    n = 1
    income = _ledger(n, employment=50_000.0, oas=8_000.0)
    result = person_assessment(income, 66, np.zeros(n), np.zeros(n), "ab", params, JANUARY)

    assert np.all(result.oas_repayment == 0.0)
    np.testing.assert_allclose(result.net_income_after_repayment, result.net_income)


def test_net_income_after_repayment_subtracts_the_repayment_when_owed(params) -> None:
    n = 1
    income = _ledger(n, employment=200_000.0, oas=100.0)
    result = person_assessment(income, 66, np.zeros(n), np.zeros(n), "ab", params, JANUARY)

    assert np.all(result.oas_repayment > 0.0)
    np.testing.assert_allclose(
        result.net_income_after_repayment, result.net_income - result.oas_repayment
    )
    assert np.all(result.net_income_after_repayment >= 0.0)


def test_assessment_unchanged_when_the_repayment_is_zero(params) -> None:
    # Regression: for a person who owes no repayment, net income and taxable
    # income coincide, so nothing about this assessment should have moved.
    n = 1
    age = 66
    income = _ledger(n, employment=50_000.0, db_pension=10_000.0)
    result = person_assessment(income, age, np.zeros(n), np.zeros(n), "ab", params, JANUARY)
    assert np.all(result.oas_repayment == 0.0)

    net = federal.net_income(income, params.federal, 0.0, 0.0)
    epi = federal.eligible_pension_income(income, age, params.federal)

    expected_federal = federal.net_tax(
        federal.gross_tax(net, params.federal, JANUARY),
        federal.non_refundable_credits(
            net,
            age,
            epi,
            income.cpp_base_contributions,
            income.ei_premiums,
            income.eligible_dividends,
            params.federal,
            JANUARY,
        ),
    )
    expected_provincial = provincial.net_tax(
        provincial.gross_tax(net, params.province("ab"), JANUARY),
        provincial.non_refundable_credits(
            net,
            age,
            epi,
            income.cpp_base_contributions,
            income.ei_premiums,
            income.eligible_dividends,
            params.province("ab"),
            params.federal,
            JANUARY,
        ),
    )
    np.testing.assert_allclose(result.federal, expected_federal)
    np.testing.assert_allclose(result.provincial, expected_provincial)


def test_couple_with_no_eligible_pension_income_is_unchanged_by_the_search(params) -> None:
    n = 2
    income0 = _ledger(n, employment=90_000.0)
    income1 = _ledger(n, employment=20_000.0)
    person0 = _person("a", n, birth_year=1955, income=income0)
    person1 = _person("b", n, birth_year=1957, income=income1)
    household = _household((person0, person1), year=2026, n=n)

    a0, a1 = household_assessment(household, params)
    zero = np.zeros(n, dtype=np.float64)
    age0 = age_at_end_of_year(person0.birth_year, person0.birth_month, 2026)
    age1 = age_at_end_of_year(person1.birth_year, person1.birth_month, 2026)
    expected0 = person_assessment(income0, age0, zero, zero, "ab", params, JANUARY)
    expected1 = person_assessment(income1, age1, zero, zero, "ab", params, JANUARY)

    np.testing.assert_allclose(a0.total, expected0.total)
    np.testing.assert_allclose(a1.total, expected1.total)


@pytest.mark.parametrize(
    ("db0", "db1", "employment0", "employment1"),
    [
        (60_000.0, 0.0, 0.0, 0.0),
        (40_000.0, 5_000.0, 0.0, 0.0),
        (30_000.0, 0.0, 20_000.0, 0.0),
    ],
    ids=["one-earner-db-pension", "unequal-db-pensions", "db-plus-employment"],
)
def test_splitting_never_increases_household_total(
    params, db0: float, db1: float, employment0: float, employment1: float
) -> None:
    n = 1
    income0 = _ledger(n, db_pension=db0, employment=employment0)
    income1 = _ledger(n, db_pension=db1, employment=employment1)
    person0 = _person("a", n, birth_year=1950, income=income0)
    person1 = _person("b", n, birth_year=1950, income=income1)
    household = _household((person0, person1), year=2026, n=n)

    a0, a1 = household_assessment(household, params)
    split_total = a0.total + a1.total

    zero = np.zeros(n, dtype=np.float64)
    age = age_at_end_of_year(1950, 1, 2026)
    unsplit0 = person_assessment(income0, age, zero, zero, "ab", params, JANUARY)
    unsplit1 = person_assessment(income1, age, zero, zero, "ab", params, JANUARY)
    unsplit_total = unsplit0.total + unsplit1.total

    assert np.all(split_total <= unsplit_total + 1e-6)


def test_splitting_strictly_reduces_total_for_a_one_sided_db_pension(params) -> None:
    n = 1
    income0 = _ledger(n, db_pension=60_000.0)
    income1 = _ledger(n)
    person0 = _person("a", n, birth_year=1950, income=income0)
    person1 = _person("b", n, birth_year=1950, income=income1)
    household = _household((person0, person1), year=2026, n=n)

    a0, a1 = household_assessment(household, params)
    split_total = a0.total + a1.total

    zero = np.zeros(n, dtype=np.float64)
    age = age_at_end_of_year(1950, 1, 2026)
    unsplit0 = person_assessment(income0, age, zero, zero, "ab", params, JANUARY)
    unsplit1 = person_assessment(income1, age, zero, zero, "ab", params, JANUARY)
    unsplit_total = unsplit0.total + unsplit1.total

    assert np.all(split_total < unsplit_total - 1e-6)


def test_mixed_alive_paths_only_split_where_both_are_alive(params) -> None:
    n = 3
    alive1 = np.array([True, False, True], dtype=np.bool_)
    income0 = _ledger(n, db_pension=60_000.0)
    income1 = _ledger(n)
    person0 = _person("a", n, birth_year=1950, income=income0)
    person1 = _person("b", n, birth_year=1950, income=income1, alive=alive1)
    household = _household((person0, person1), year=2026, n=n)

    a0, a1 = household_assessment(household, params)

    zero = np.zeros(n, dtype=np.float64)
    age = age_at_end_of_year(1950, 1, 2026)
    unsplit0 = person_assessment(income0, age, zero, zero, "ab", params, JANUARY)
    unsplit1 = person_assessment(income1, age, zero, zero, "ab", params, JANUARY)

    # Path 1 (index 1): person1 not alive, so the election must be forced to
    # zero and the assessment must match the unsplit one exactly.
    assert a0.total[1] == pytest.approx(unsplit0.total[1])
    assert a1.total[1] == pytest.approx(unsplit1.total[1])

    # Paths 0 and 2: both alive, large one-sided DB pension, splitting must
    # strictly help (same case as the strict test above).
    for path in (0, 2):
        assert a0.total[path] + a1.total[path] < unsplit0.total[path] + unsplit1.total[path] - 1e-6


def test_household_assessment_minimises_over_the_grid_with_a_binding_repayment(params) -> None:
    # Pins four things about a couple that carries a real OAS repayment: the
    # grid search still lands on the same minimum as an independently-built
    # grid of person_assessment calls; the negative-fraction half of that
    # search still collapses to the zero-transfer candidate already in the
    # positive half (see the comment below); household_assessment's _select
    # gathers every one of Assessment's six fields at the one winning
    # candidate index, never mixing fields from different candidates —
    # pinned by recomputing the winning grid candidate directly and
    # comparing every field of a0 against it; and person0's DB pension and
    # OAS are large enough that the repayment is still non-zero at the
    # elected split (the maximum share, not merely on the discarded unsplit
    # baseline). The objective itself — minimising fed(net - repay) +
    # prov(net - repay) + repay rather than fed(net) + prov(net) + repay — is
    # pinned elsewhere, by test_assessment_taxes_net_income_after_the_repayment
    # and test_assessment_age_amount_taxed_after_the_repayment.
    n = 1
    age = 70
    db_pension = 200_000.0
    oas_amount = 9_000.0
    income0 = _ledger(n, db_pension=db_pension, oas=oas_amount)
    income1 = _ledger(n)
    person0 = _person("a", n, birth_year=1956, income=income0)
    person1 = _person("b", n, birth_year=1956, income=income1)
    household = _household((person0, person1), year=2026, n=n)

    # The hand-built grid below transfers fraction * db_pension, matching
    # household_assessment's fraction * eligible_pension_income only because
    # eligible_pension_income returns the whole DB pension at this age.
    np.testing.assert_allclose(
        federal.eligible_pension_income(income0, age, params.federal), db_pension
    )

    zero = np.zeros(n, dtype=np.float64)
    unsplit0 = person_assessment(income0, age, zero, zero, "ab", params, JANUARY)
    unsplit1 = person_assessment(income1, age, zero, zero, "ab", params, JANUARY)
    assert np.all(unsplit0.oas_repayment > 0.0)

    a0, a1 = household_assessment(household, params)
    assert np.all(a0.oas_repayment > 0.0)
    np.testing.assert_allclose(a0.net_income_after_repayment, a0.net_income - a0.oas_repayment)
    np.testing.assert_allclose(a1.net_income_after_repayment, a1.net_income - a1.oas_repayment)
    elected_total = a0.total + a1.total

    # The same candidate grid household_assessment searches: person1 has no
    # eligible pension income, so the negative-fraction half of its search
    # (transferring from person1 to person0) collapses to the zero-transfer
    # candidate already in this list, and need not be built separately.
    maximum_share = params.federal.number("pension_splitting.maximum_transfer_share")
    magnitudes: list[float] = []
    k = 0
    while k * GRID_STEP < maximum_share:
        magnitudes.append(k * GRID_STEP)
        k += 1
    magnitudes.append(maximum_share)

    grid_totals = []
    for fraction in magnitudes:
        transfer = np.full(n, fraction * db_pension)
        candidate0 = person_assessment(income0, age, zero, transfer, "ab", params, JANUARY)
        candidate1 = person_assessment(income1, age, transfer, zero, "ab", params, JANUARY)
        grid_totals.append(candidate0.total + candidate1.total)
    grid_minimum = np.min(np.stack(grid_totals, axis=0), axis=0)

    np.testing.assert_allclose(elected_total, grid_minimum)
    assert np.all(elected_total < unsplit0.total + unsplit1.total - 1e-6)

    # Recover the winning grid index and recompute that candidate directly,
    # to pin every field of a0 -- not just total and the repayment identity
    # already asserted above -- at the one winning candidate.
    best = int(np.argmin(np.stack(grid_totals, axis=0), axis=0)[0])
    winning = person_assessment(
        income0, age, zero, np.full(n, magnitudes[best] * db_pension), "ab", params, JANUARY
    )
    for field in (
        "federal",
        "provincial",
        "oas_repayment",
        "total",
        "net_income",
        "net_income_after_repayment",
    ):
        np.testing.assert_allclose(getattr(a0, field), getattr(winning, field))


# =============================================================================
# household_assessment: the tie-break in the split search
# =============================================================================
#
# household_assessment relies on zero being the first candidate and
# np.argmin taking the first minimum, so a genuine tie goes to no split. This
# is only observable when *every* candidate fraction gives the same
# household total but different fractions give each person a different
# net_income — the case below, built from a synthetic federal/provincial
# pair with a single flat rate and no edges in either jurisdiction, no age
# amount in play, and both persons' pension credits already capped, so
# moving pension income between them changes neither person's tax nor
# either credit, only which of them it is booked to.

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
SYNTHETIC_FEDERAL_FLAT_RATE_TIE = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  brackets_and_credits:
    adjustment_months: [1]
    applies_to:
      - brackets.edges_annual
      - credits.basic_personal_amount_annual
      - credits.age_amount.amount_annual
      - credits.age_amount.reduction_threshold_annual
      - contribution_credit.cpp_maximum_annual
      - contribution_credit.ei_maximum_annual
  unindexed:
    adjustment_months: []
    applies_to:
      - credits.pension_income_amount_annual

brackets:
  edges_annual: []
  rates: [0.2]

credits:
  valuation_rate: 0.1
  basic_personal_amount_annual: 5000
  pension_income_amount_annual: 2000
  age_amount:
    eligibility_age_years: 999
    amount_annual: 1000
    reduction_threshold_annual: 100000
    reduction_rate: 0.1

contribution_credit:
  cpp_maximum_annual: 100
  ei_maximum_annual: 50

investment_income:
  capital_gains_inclusion_rate: 0.5
  eligible_dividend_gross_up_rate: 0.38
  eligible_dividend_credit_rate_of_gross_up: 0.3

pension_splitting:
  maximum_transfer_share: 0.5

eligible_pension_income:
  rrif_minimum_age_years: 65
"""

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
SYNTHETIC_AB_FLAT_RATE_TIE = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  brackets_and_credits:
    adjustment_months: [1]
    applies_to:
      - brackets.edges_annual
      - credits.basic_personal_amount_annual
      - credits.pension_income_amount_annual
      - credits.age_amount.amount_annual
      - credits.age_amount.reduction_threshold_annual

brackets:
  edges_annual: []
  rates: [0.1]

credits:
  valuation_rate: 0.08
  basic_personal_amount_annual: 4000
  pension_income_amount_annual: 1500
  age_amount:
    eligibility_age_years: 999
    amount_annual: 800
    reduction_threshold_annual: 100000
    reduction_rate: 0.1

investment_income:
  eligible_dividend_gross_up_rate: 0.38
  eligible_dividend_credit_rate_of_gross_up: 0.2
"""

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
SYNTHETIC_OAS_FLAT_RATE_TIE = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  recovery_tax_threshold:
    adjustment_months: [1]
    applies_to:
      - recovery_tax.threshold_annual

recovery_tax:
  threshold_annual: 1000000
  rate: 0.15
"""

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
SYNTHETIC_RESP_FLAT_RATE_TIE = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
aip:
  penalty_rate: 0.37
"""


@pytest.fixture
def synth_params_flat_rate_tie(tmp_path: Path):
    _write(tmp_path, "federal", SYNTHETIC_FEDERAL_FLAT_RATE_TIE)
    _write(tmp_path, "ab", SYNTHETIC_AB_FLAT_RATE_TIE)
    _write(tmp_path, "oas", SYNTHETIC_OAS_FLAT_RATE_TIE)
    _write(tmp_path, "resp", SYNTHETIC_RESP_FLAT_RATE_TIE)
    return real_year(load_year(2026, tmp_path), 0.0)


def test_household_assessment_ties_default_to_no_split(synth_params_flat_rate_tie) -> None:
    params = synth_params_flat_rate_tie
    n = 1
    # Both persons' own eligible pension income is far above the (already
    # capped) pension credit amount, so no candidate fraction in the grid
    # ever changes either credit; with a single flat rate and no edges in
    # either jurisdiction, moving pension income between them leaves the sum
    # of net incomes — and so the household total — exactly unchanged for
    # every candidate. Every candidate is a genuine tie.
    income0 = _ledger(n, db_pension=100_000.0)
    income1 = _ledger(n, db_pension=100_000.0)
    person0 = _person("a", n, birth_year=1950, income=income0)
    person1 = _person("b", n, birth_year=1950, income=income1)
    household = _household((person0, person1), year=2026, n=n)

    a0, a1 = household_assessment(household, params)

    zero = np.zeros(n, dtype=np.float64)
    age = age_at_end_of_year(1950, 1, 2026)
    unsplit0 = person_assessment(income0, age, zero, zero, "ab", params, JANUARY)
    unsplit1 = person_assessment(income1, age, zero, zero, "ab", params, JANUARY)

    # A total-only assertion would pass under any tie-break, since every
    # candidate ties on the total by construction. net_income is what
    # distinguishes the elected candidate from the others.
    np.testing.assert_allclose(a0.net_income, unsplit0.net_income)
    np.testing.assert_allclose(a1.net_income, unsplit1.net_income)

    # The flatness the test depends on, under observation rather than under
    # argument: a non-zero transfer in each direction must land on exactly
    # the same household total as the elected (unsplit) one. If a future
    # edit to the fixture broke this — a live age amount, a bracket edge, a
    # pension credit that stops being capped — this would fail, rather than
    # the test silently degrading into a duplicate of
    # test_couple_with_no_eligible_pension_income_is_unchanged_by_the_search.
    elected_total = a0.total + a1.total
    transfer = np.full(n, 0.25 * 100_000.0)
    forward0 = person_assessment(income0, age, zero, transfer, "ab", params, JANUARY)
    forward1 = person_assessment(income1, age, transfer, zero, "ab", params, JANUARY)
    backward0 = person_assessment(income0, age, transfer, zero, "ab", params, JANUARY)
    backward1 = person_assessment(income1, age, zero, transfer, "ab", params, JANUARY)

    # Exact, not approximate: at this scale the products involved land on
    # exactly representable binary values, and the two directions were
    # confirmed bit-for-bit equal to elected_total before writing this.
    assert (forward0.total + forward1.total == elected_total).all()
    assert (backward0.total + backward1.total == elected_total).all()


def test_aip_penalty_uses_the_rate_from_params_not_a_hardcoded_value(
    synth_params_flat_rate_tie, params
) -> None:
    synthetic_rate = synth_params_flat_rate_tie.resp.number("aip.penalty_rate")
    # Control: proves this test would notice person_assessment reading the
    # wrong file, rather than the synthetic and real rates coinciding.
    assert synthetic_rate != params.resp.number("aip.penalty_rate")

    n = 1
    amount = 10_000.0
    income = _ledger(n, employment=40_000.0, resp_accumulated_income=amount)
    result = person_assessment(
        income, 50, np.zeros(n), np.zeros(n), "ab", synth_params_flat_rate_tie, JANUARY
    )

    np.testing.assert_allclose(result.aip_penalty, amount * synthetic_rate)


# =============================================================================
# household_assessment: unsupported household sizes
# =============================================================================


def test_household_assessment_raises_for_zero_persons(params) -> None:
    n = 2
    household = _household((), year=2026, n=n)
    with pytest.raises(ValueError, match="one or two"):
        household_assessment(household, params)


def test_household_assessment_raises_for_three_persons(params) -> None:
    n = 2
    persons = tuple(_person(person_id, n) for person_id in ("a", "b", "c"))
    household = _household(persons, year=2026, n=n)
    with pytest.raises(ValueError, match="one or two"):
        household_assessment(household, params)


# =============================================================================
# household_assessment: the january_month_index derivation, at non-zero inflation
# =============================================================================
#
# At zero inflation, erosion_factor and unindexed_factor are both exactly 1
# for any legal january_month_index, so the tests above never distinguish a
# correct derivation from one that is off by a year. federal's pension income
# amount credit (credits.pension_income_amount_annual) is unindexed and
# db_pension is set above it, so its real value — and hence the assessment —
# genuinely differs between tax years below.


def test_household_assessment_uses_january_of_the_tax_year_at_nonzero_inflation() -> None:
    nonzero_params = real_year(load_year(2026), 0.03)
    n = 2
    income = _ledger(n, employment=40_000.0, db_pension=10_000.0)
    person = _person("a", n, birth_year=1950, income=income)
    year = START_YEAR + 3
    household = _household((person,), year=year, n=n)

    (result,) = household_assessment(household, nonzero_params)
    age = age_at_end_of_year(person.birth_year, person.birth_month, year)
    zero = np.zeros(n, dtype=np.float64)

    at_36 = person_assessment(income, age, zero, zero, "ab", nonzero_params, 36)
    at_24 = person_assessment(income, age, zero, zero, "ab", nonzero_params, 24)
    at_48 = person_assessment(income, age, zero, zero, "ab", nonzero_params, 48)

    np.testing.assert_allclose(result.total, at_36.total)
    # Negative assertions: an off-by-a-year derivation (e.g. reading
    # january_month_index as 36 +/- MONTHS_PER_YEAR) would pass an equality
    # check alone if it happened to match one of these by coincidence, so both
    # neighbouring years must differ from the correct one.
    assert np.all(np.abs(result.total - at_24.total) > 1e-6)
    assert np.all(np.abs(result.total - at_48.total) > 1e-6)


@pytest.mark.parametrize(
    ("year", "month"),
    [
        (2026, 1),
        (2026, 6),
        (2026, 12),
        (2027, 12),
        (2029, 3),
        (2032, 12),
    ],
)
def test_household_assessment_january_index_derivation_over_year_month_pairs(
    year: int, month: int
) -> None:
    """The derivation ``month_index - (month - 1)`` checked across (year, month) pairs.

    Independent of any particular tax outcome: for every pair the expected
    january_month_index is ``(year - START_YEAR) * 12``, by construction of
    ``_household``, regardless of which month within the year the household is
    assessed in.
    """
    nonzero_params = real_year(load_year(2026), 0.03)
    n = 1
    income = _ledger(n, employment=40_000.0, db_pension=10_000.0)
    person = _person("a", n, birth_year=1950, income=income)
    household = _household((person,), year=year, n=n, month=month)

    (result,) = household_assessment(household, nonzero_params)
    age = age_at_end_of_year(person.birth_year, person.birth_month, year)
    zero = np.zeros(n, dtype=np.float64)

    expected_january_month_index = (year - START_YEAR) * 12
    expected = person_assessment(
        income, age, zero, zero, "ab", nonzero_params, expected_january_month_index
    )

    np.testing.assert_allclose(result.total, expected.total)


# =============================================================================
# The pension income credit carried by the transfer
# =============================================================================
#
# eligible_pension_income - transfer_out + transfer_in in person_assessment
# carries the transferred amount's share of the pension income amount credit.
# The credit is capped at credits.pension_income_amount_annual — small next
# to the bracket effect household-level splitting tests above measure — so
# each test below isolates it by holding net income (and hence every other
# credit and both provinces' gross_tax) fixed between the two assessments
# compared, reaching the same total income through a non-pension source in
# the control.


def test_transferee_gains_the_pension_credit_the_transfer_carries(params) -> None:
    pension_amount = params.federal.annual_amount(
        "credits.pension_income_amount_annual", JANUARY
    )
    valuation_rate = params.federal.number("credits.valuation_rate")

    n = 1
    transfer_in = 5_000.0
    assert transfer_in > pension_amount  # the credit is capped, not proportional

    # Transferee has no pension income of their own; the only eligible
    # pension income they have is what the transfer carries.
    with_transfer = person_assessment(
        _ledger(n, employment=60_000.0),
        40,
        transfer_in,
        0.0,
        "ab",
        params,
        JANUARY,
    )
    # Control: identical net income (65,000), reached through employment
    # alone, so eligible pension income is zero and the credit is not earned.
    # The only difference between this and with_transfer is credit
    # eligibility.
    control = person_assessment(
        _ledger(n, employment=60_000.0 + transfer_in),
        40,
        0.0,
        0.0,
        "ab",
        params,
        JANUARY,
    )

    np.testing.assert_allclose(control.net_income, with_transfer.net_income)
    np.testing.assert_allclose(
        control.federal - with_transfer.federal, valuation_rate * pension_amount
    )
    assert np.all(with_transfer.total < control.total - 1e-6)


def test_transferor_loses_credit_in_step_with_remaining_eligible_pension_income(
    params,
) -> None:
    valuation_rate = params.federal.number("credits.valuation_rate")

    n = 1
    net_income_target = 70_000.0
    db_pension = 3_000.0
    # Both transfer amounts leave the transferor's remaining eligible pension
    # income (db_pension - transfer_out) below credits.pension_income_amount_annual,
    # so the credit tracks it dollar for dollar rather than sitting at the
    # cap. employment is adjusted so net income — and hence gross_tax and
    # every other credit — is identical in both cases; the only difference is
    # the remaining eligible pension income.
    transfer_out_more = 2_600.0  # remaining = 400
    transfer_out_less = 2_100.0  # remaining = 900
    employment_more = net_income_target - db_pension + transfer_out_more
    employment_less = net_income_target - db_pension + transfer_out_less

    more_transferred = person_assessment(
        _ledger(n, employment=employment_more, db_pension=db_pension),
        40,
        0.0,
        transfer_out_more,
        "ab",
        params,
        JANUARY,
    )
    less_transferred = person_assessment(
        _ledger(n, employment=employment_less, db_pension=db_pension),
        40,
        0.0,
        transfer_out_less,
        "ab",
        params,
        JANUARY,
    )

    np.testing.assert_allclose(more_transferred.net_income, [net_income_target])
    np.testing.assert_allclose(less_transferred.net_income, [net_income_target])

    remaining_more = db_pension - transfer_out_more
    remaining_less = db_pension - transfer_out_less
    # Less remaining eligible pension income earns less credit, so tax after
    # transferring more is higher, by exactly the credit lost.
    np.testing.assert_allclose(
        more_transferred.federal - less_transferred.federal,
        valuation_rate * (remaining_less - remaining_more),
    )


# =============================================================================
# The pension-split grid at a maximum share that is not a multiple of GRID_STEP
# =============================================================================

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
#: maximum_transfer_share (0.47) is deliberately not a multiple of GRID_STEP
#: (0.05), so the search's "append the maximum itself" branch is exercised
#: rather than landing on a fraction already in the grid.
SYNTHETIC_FEDERAL_NON_MULTIPLE_MAX_SHARE = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  brackets_and_credits:
    adjustment_months: [1]
    applies_to:
      - brackets.edges_annual
      - credits.basic_personal_amount_annual
      - credits.age_amount.amount_annual
      - credits.age_amount.reduction_threshold_annual
      - contribution_credit.cpp_maximum_annual
      - contribution_credit.ei_maximum_annual
  unindexed:
    adjustment_months: []
    applies_to:
      - credits.pension_income_amount_annual

brackets:
  edges_annual: [50000, 150000]
  rates: [0.1, 0.2, 0.4]

credits:
  valuation_rate: 0.2
  basic_personal_amount_annual: 10000
  pension_income_amount_annual: 2000
  age_amount:
    eligibility_age_years: 65
    amount_annual: 5000
    reduction_threshold_annual: 40000
    reduction_rate: 0.1

contribution_credit:
  cpp_maximum_annual: 100
  ei_maximum_annual: 50

investment_income:
  capital_gains_inclusion_rate: 0.5
  eligible_dividend_gross_up_rate: 0.38
  eligible_dividend_credit_rate_of_gross_up: 0.3

pension_splitting:
  maximum_transfer_share: 0.47

eligible_pension_income:
  rrif_minimum_age_years: 65
"""

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
SYNTHETIC_AB_FOR_MAX_SHARE = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  brackets_and_credits:
    adjustment_months: [1]
    applies_to:
      - brackets.edges_annual
      - credits.basic_personal_amount_annual
      - credits.pension_income_amount_annual
      - credits.age_amount.amount_annual
      - credits.age_amount.reduction_threshold_annual

brackets:
  edges_annual: [50000, 150000]
  rates: [0.05, 0.1, 0.2]

credits:
  valuation_rate: 0.1
  basic_personal_amount_annual: 8000
  pension_income_amount_annual: 1500
  age_amount:
    eligibility_age_years: 65
    amount_annual: 3000
    reduction_threshold_annual: 40000
    reduction_rate: 0.1

investment_income:
  eligible_dividend_gross_up_rate: 0.38
  eligible_dividend_credit_rate_of_gross_up: 0.2
"""

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
SYNTHETIC_OAS_FOR_MAX_SHARE = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  recovery_tax_threshold:
    adjustment_months: [1]
    applies_to:
      - recovery_tax.threshold_annual

recovery_tax:
  threshold_annual: 1000000
  rate: 0.15
"""

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
SYNTHETIC_RESP_FOR_MAX_SHARE = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
aip:
  penalty_rate: 0.42
"""


def _write(root: Path, name: str, text: str) -> Path:
    year_dir = root / "2026"
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def synth_params_non_multiple_max_share(tmp_path: Path):
    _write(tmp_path, "federal", SYNTHETIC_FEDERAL_NON_MULTIPLE_MAX_SHARE)
    _write(tmp_path, "ab", SYNTHETIC_AB_FOR_MAX_SHARE)
    _write(tmp_path, "oas", SYNTHETIC_OAS_FOR_MAX_SHARE)
    _write(tmp_path, "resp", SYNTHETIC_RESP_FOR_MAX_SHARE)
    return real_year(load_year(2026, tmp_path), 0.0)


def test_grid_reaches_a_maximum_share_that_is_not_a_multiple_of_grid_step(
    synth_params_non_multiple_max_share,
) -> None:
    params = synth_params_non_multiple_max_share
    maximum_share = params.federal.number("pension_splitting.maximum_transfer_share")
    assert maximum_share % GRID_STEP != 0  # the premise of this test

    n = 1
    # person0's marginal rate (0.4, top bracket) exceeds person1's (0.2 or
    # less) across the whole range transferred below, all the way to the
    # maximum share, so more transfer is always better and the search-bound
    # optimum sits exactly at the maximum share.
    income0 = _ledger(n, db_pension=300_000.0)
    income1 = _ledger(n)
    person0 = _person("a", n, birth_year=1950, income=income0)
    person1 = _person("b", n, birth_year=1950, income=income1)
    household = _household((person0, person1), year=2026, n=n)

    a0, a1 = household_assessment(household, params)

    zero = np.zeros(n, dtype=np.float64)
    age = age_at_end_of_year(1950, 1, 2026)

    # The last regular grid point strictly below the maximum share, mirroring
    # household_assessment's own grid construction.
    k = 0
    last_below_max = 0.0
    while k * GRID_STEP < maximum_share:
        last_below_max = k * GRID_STEP
        k += 1

    transfer_at_max = maximum_share * income0.db_pension
    transfer_at_last_grid_point = last_below_max * income0.db_pension

    at_max0 = person_assessment(income0, age, zero, transfer_at_max, "ab", params, JANUARY)
    at_max1 = person_assessment(income1, age, transfer_at_max, zero, "ab", params, JANUARY)
    at_last0 = person_assessment(
        income0, age, zero, transfer_at_last_grid_point, "ab", params, JANUARY
    )
    at_last1 = person_assessment(
        income1, age, transfer_at_last_grid_point, zero, "ab", params, JANUARY
    )

    np.testing.assert_allclose(a0.total, at_max0.total)
    np.testing.assert_allclose(a1.total, at_max1.total)
    assert np.all((a0.total + a1.total) < (at_last0.total + at_last1.total) - 1e-6)


# --- shape and dtype contracts ------------------------------------------------


def test_person_assessment_returns_float64_arrays(params) -> None:
    n = 4
    income = _ledger(n, employment=50_000.0)
    result = person_assessment(income, 50, np.zeros(n), np.zeros(n), "ab", params, JANUARY)
    assert isinstance(result, Assessment)
    for field in (
        "federal",
        "provincial",
        "oas_repayment",
        "aip_penalty",
        "total",
        "net_income",
        "net_income_after_repayment",
    ):
        value = getattr(result, field)
        assert value.dtype == np.float64
        assert value.shape == (n,)


def test_household_assessment_returns_float64_arrays_for_two_persons(params) -> None:
    n = 4
    income0 = _ledger(n, employment=50_000.0)
    income1 = _ledger(n, employment=30_000.0)
    person0 = _person("a", n, income=income0)
    person1 = _person("b", n, income=income1)
    household = _household((person0, person1), year=2026, n=n)

    a0, a1 = household_assessment(household, params)
    for assessment in (a0, a1):
        for field in (
            "federal",
            "provincial",
            "oas_repayment",
            "aip_penalty",
            "total",
            "net_income",
            "net_income_after_repayment",
        ):
            value = getattr(assessment, field)
            assert value.dtype == np.float64
            assert value.shape == (n,)


def test_oas_repayment_returns_float64_scalar_and_array(params) -> None:
    scalar = oas_repayment(100_000.0, 5000.0, params.oas, JANUARY)
    array = oas_repayment(
        np.array([100_000.0, 50_000.0]), np.array([5000.0, 5000.0]), params.oas, JANUARY
    )
    assert scalar.dtype == np.float64 and scalar.shape == ()
    assert array.dtype == np.float64 and array.shape == (2,)

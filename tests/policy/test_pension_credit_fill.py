# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The pension-credit fill (``engine.core.step``'s phase 6), end to end through a
:class:`~engine.policy.build.CompositePolicy`.

The fill itself is existing engine behaviour (``_phase6_forced_withdrawals``); this checks
that a policy built by :func:`~engine.policy.build.build_policy`, with ``taxable_ceiling_bracket``
off and an ``order`` that never names ``rrif``, still reaches the target every year from the
fill age -- the whole point being that the fill fires whether or not the policy itself ever
touches the RRIF.

A single person, born so that ``eligible_pension_income.rrif_minimum_age_years`` (age at *end*
of year) is crossed in the run's *second* calendar year, not its first -- "reaching the fill
age mid-run". Deterministic draws (``build_deterministic_draws``) survive to the life
table's latest age on their own path, which is what keeps this one person alive through every
year this test reads; no separate death-forcing is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.accounts import rrif as rrif_mod
from engine.core.build import (
    build_deterministic_draws,
    build_initial_state,
    build_market_inputs,
    draw_deaths,
)
from engine.core.indexation import real_year
from engine.core.timeline import age_at_end_of_year, age_at_start_of_year
from engine.mc.simulate import run
from engine.params.loader import load_year
from engine.policy.build import build_policy

#: SYNTHETIC balance: small enough that the ordinary annual minimum stays well below the
#: pension credit target every year this test reads, so the fill -- not the minimum -- is
#: what reaches it.
_RRIF_BALANCE = 10_000.0
#: Old enough that the person turns the fill age in the run's second year, not its first.
_BIRTH_YEAR = 1962
_BIRTH_MONTH = 6
_START_YEAR = 2026
_N_YEARS = 3  # 2026 (pre-fill), 2027-2028 (post-fill)


def _build_scenario(example_values, build_from_values, *, fill_pension_credit: bool):
    values = example_values()
    person = values["household"]["persons"][0]
    person["birth_year"] = _BIRTH_YEAR
    person["birth_month"] = _BIRTH_MONTH
    person["cpp"] = {"in_pay_monthly": 0}
    person["oas"] = {"in_pay_monthly": 0}
    person.pop("employment", None)
    person.pop("db_pensions", None)
    person["accounts"] = {
        "cash": {"balance": 20000},
        "rrsp": {"balance": 0, "room": 0},
        "rrif": {"balance": _RRIF_BALANCE},
        "tfsa": {"balance": 0, "room": 0},
        "taxable": {"balance": 500_000, "acb": 500_000},
    }
    values["household"]["beneficiaries"] = []
    values["spending"]["schedule"] = [{"from_year": _START_YEAR, "annual": 30000.0}]
    policy = values["policies"][0]
    policy["contribution"] = {
        "weights": {"rrsp": 0.0, "tfsa": 0.0, "taxable": 1.0, "resp": 0.0},
        "spill_order": ["taxable"],
    }
    # order never names "rrif": the only thing that can ever move it is phase 6's fill.
    policy["withdrawal"] = {
        "order": ["taxable"],
        "taxable_ceiling_bracket": None,
        "fill_pension_credit": fill_pension_credit,
    }
    policy["elections"]["cpp_start_age_years"] = {}
    policy["elections"]["oas_start_age_years"] = {}
    del values["grid"]

    scenario, _state = build_from_values(values, n_paths=1)
    return scenario


def _yearly_rrif_forced(trace, start_year: int, n_years: int) -> dict[int, float]:
    """Calendar year to that year's total forced RRIF withdrawal, from a month-by-month trace.

    ``order`` never names ``rrif`` and ``taxable_ceiling_bracket`` is off, so phase 6's forced
    withdrawal is the only thing that ever moves the RRIF: summing it per year is exactly that
    year's RRIF withdrawal.
    """
    totals: dict[int, float] = {year: 0.0 for year in range(start_year, start_year + n_years)}
    for record in trace:
        year = record.context.year
        if year in totals:
            totals[year] += float(record.context.forced_withdrawals[0].rrif[0])
    return totals


def _run_and_trace(scenario, n_years: int):
    market = build_market_inputs(scenario.assumptions)
    mortality = load_year(scenario.start_year)["mortality"]
    real_params = real_year(load_year(scenario.start_year), scenario.assumptions.inflation)
    draws = build_deterministic_draws(scenario, market, mortality)
    state = build_initial_state(scenario, n_paths=1)
    state = draw_deaths(state, draws, mortality)
    policy = build_policy(scenario.policies[0], scenario.household)

    result = run(state, policy, draws, market, real_params, trace_path=0)
    return result, _yearly_rrif_forced(result.trace, _START_YEAR, n_years)


def _minimum_for_year(balance: float, year: int, real_params) -> float:
    """The ordinary annual minimum for ``year``, from ``balance`` on 1 January of it.

    ``opened_year`` is fixed at ``_START_YEAR - 1``: the RRIF already held a balance at
    scenario build, so ``engine.core.build._build_person`` opens it the year before the run,
    and nothing here ever converts a new one in (no RRSP, no LIRA).
    """
    age_start = age_at_start_of_year(_BIRTH_YEAR, _BIRTH_MONTH, year)
    return float(
        rrif_mod.minimum_withdrawal(
            np.array([balance]), age_start, _START_YEAR - 1, year, real_params.rrif
        )[0]
    )


@pytest.fixture
def fill_age(real_params) -> int:
    return int(real_params.federal.number("eligible_pension_income.rrif_minimum_age_years"))


def _target_for_year(year: int, real_params) -> float:
    """``credits.pension_income_amount_annual``, the greater of federal and provincial, in
    real terms for ``year`` -- the figure is unindexed, so its real value decays year over
    year and must be read at that year's own ``january_month_index``, never a fixed one.
    """
    january_month_index = 12 * (year - _START_YEAR)
    return max(
        real_params.federal.annual_amount(
            "credits.pension_income_amount_annual", january_month_index
        ),
        real_params.province("ab").annual_amount(
            "credits.pension_income_amount_annual", january_month_index
        ),
    )


def test_the_fill_age_is_crossed_in_the_second_year_not_the_first(fill_age):
    assert age_at_end_of_year(_BIRTH_YEAR, _BIRTH_MONTH, _START_YEAR) == fill_age - 1
    assert age_at_end_of_year(_BIRTH_YEAR, _BIRTH_MONTH, _START_YEAR + 1) == fill_age


def test_the_ordinary_minimum_is_below_the_target_precondition(real_params):
    minimum = _minimum_for_year(_RRIF_BALANCE, _START_YEAR, real_params)
    assert minimum < _target_for_year(_START_YEAR, real_params)


def test_fill_on_before_and_from_the_fill_age(example_values, build_from_values, real_params):
    """Before the fill age, only the ordinary minimum is drawn; from it, the target is
    reached every year -- one run, both claims, since each reads a different year's total.
    """
    scenario = _build_scenario(example_values, build_from_values, fill_pension_credit=True)
    _result, yearly = _run_and_trace(scenario, _N_YEARS)

    expected_2026 = _minimum_for_year(_RRIF_BALANCE, _START_YEAR, real_params)
    assert yearly[_START_YEAR] == pytest.approx(expected_2026, abs=0.01)

    for year in range(_START_YEAR + 1, _START_YEAR + _N_YEARS):
        assert yearly[year] == pytest.approx(_target_for_year(year, real_params), abs=0.01)


def test_control_without_the_fill_draws_only_the_minimum_at_the_fill_age(
    example_values, build_from_values, real_params
):
    scenario = _build_scenario(example_values, build_from_values, fill_pension_credit=False)
    result, yearly = _run_and_trace(scenario, _N_YEARS)

    # January 2027's opening RRIF balance is exactly what December 2026 (month index 11)
    # closes with -- MonthRecord.balances_close's own contract.
    opening_2027 = float(result.trace[11].balances_close[0].rrif[0])
    expected_2027 = _minimum_for_year(opening_2027, _START_YEAR + 1, real_params)

    assert yearly[_START_YEAR + 1] == pytest.approx(expected_2027, abs=0.01)
    assert yearly[_START_YEAR + 1] < _target_for_year(_START_YEAR + 1, real_params)

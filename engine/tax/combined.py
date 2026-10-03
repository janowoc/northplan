# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Household tax assessment: pension splitting, the OAS repayment, and the AIP special tax.

The single entry point the December close calls, once per simulated year, on
income accumulated over that year's twelve monthly steps.
``household_assessment`` owns the household-level election that cannot be
evaluated one person at a time — pension income splitting — and assesses
every person at the elected split. The OAS repayment and the RESP
accumulated-income special tax both live here too, each as a line within
each person's :class:`Assessment`: there is no separate ``engine.tax.oas``
module.
"""

from __future__ import annotations

import dataclasses
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.accounts import resp
from engine.core.indexation import RealParamSet, RealParamYear
from engine.core.state import Assessment, HouseholdState, IncomeLedger
from engine.core.timeline import MONTHS_PER_YEAR
from engine.core.timeline import age_at_end_of_year as _age_at_end_of_year
from engine.tax import federal, provincial

#: Resolution of the pension-split search. Not a tax parameter: the statutory
#: rule is the maximum share in params/, and this is how finely the engine
#: looks between zero and it (docs/limitations.md L51).
GRID_STEP: Final[float] = 0.05


def oas_repayment(
    net_income_before_repayment: ArrayLike,
    oas_received: ArrayLike,
    params: RealParamSet,
    january_month_index: int,
) -> NDArray[np.float64]:
    """OAS recovery tax for the year.

    This is the current year's return line only, never a prior year's: it is
    computed once at the December close on this year's net income before the
    repayment (line 23400), which already includes this year's OAS (L23).

    Args:
        net_income_before_repayment: Net income for the year, real dollars.
        oas_received: Gross OAS received in the same calendar year.
        params: The ``oas`` parameter set for the tax year.
        january_month_index: Month index of January of the tax year.

    Returns:
        Repayment owed, non-negative, capped at ``oas_received``.
    """
    threshold = params.annual_amount("recovery_tax.threshold_annual", january_month_index)
    rate = params.number("recovery_tax.rate")
    net_income_arr = np.asarray(net_income_before_repayment, dtype=np.float64)
    oas_arr = np.asarray(oas_received, dtype=np.float64)
    return np.asarray(
        np.minimum(rate * np.clip(net_income_arr - threshold, 0, None), oas_arr),
        dtype=np.float64,
    )


def person_assessment(
    ledger: IncomeLedger,
    age_at_end_of_year: ArrayLike,
    transfer_in: ArrayLike,
    transfer_out: ArrayLike,
    province: str,
    params: RealParamYear,
    january_month_index: int,
    *,
    died_in_year: ArrayLike = False,
    transfer_in_credit_base: ArrayLike | None = None,
) -> Assessment:
    """One person's federal and provincial assessment, including the OAS repayment and AIP tax.

    Three figures, three different bases. The OAS repayment is tested against net income
    (line 23400). ``gross_tax`` -- federal and provincial brackets alike -- is applied to
    line 26000: line 23600 (net income less the OAS repayment) less the ITA 111(2)
    year-of-death capital loss deduction (``died_in_year``, L17). The non-refundable
    credits, age amount included, are tested against line 23600 itself, matching the CRA
    form's own basis for them. The deduction therefore changes neither line 23400 nor line
    23600 -- only the brackets see it. ``oas_received`` is not an argument: it is always
    ``ledger.oas``. The eligible pension income transferred carries its share of the
    pension income credit eligibility with it, which is what ``eligible_pension_income -
    transfer_out + transfer_in_credit_base`` does below; net income still counts the whole
    ``transfer_in``. The AIP special tax (line 41800) is computed from
    ``ledger.resp_accumulated_income`` and is additional tax only; the payment itself is
    already in net income.

    Args:
        ledger: This person's income components, accumulated over the year.
        age_at_end_of_year: Age in whole years on 31 December.
        transfer_in: Split-eligible pension income received from the other
            spouse, ``(n_paths,)`` or scalar. 0 for no split.
        transfer_out: Split-eligible pension income given to the other
            spouse, same shape. 0 for no split.
        province: Two-letter province code of residence.
        params: Every parameter file for the tax year, in real dollars.
        january_month_index: Month index of January of the tax year.
        died_in_year: Whether this is this person's year of death,
            ``(n_paths,)`` or scalar bool; default ``False`` reproduces
            every call site that omits it.
        transfer_in_credit_base: The part of ``transfer_in`` the pension
            income credit counts (ITA 118(7)), same shape; ``None`` means
            all of ``transfer_in``.

    Returns:
        This person's :class:`Assessment` at the given transfer.
    """
    fed = params.federal
    prov = params.province(province)
    net = federal.net_income(ledger, fed, transfer_in, transfer_out)
    credit_base = transfer_in if transfer_in_credit_base is None else transfer_in_credit_base
    epi = (
        federal.eligible_pension_income(ledger, age_at_end_of_year, fed)
        - np.asarray(transfer_out, dtype=np.float64)
        + np.asarray(credit_base, dtype=np.float64)
    )
    repay = oas_repayment(net, ledger.oas, params.oas, january_month_index)
    penalty = resp.aip_penalty(ledger.resp_accumulated_income, params.resp)
    line_23600 = federal.taxable_income(net, repay)
    deduction = federal.death_year_capital_loss_deduction(ledger, fed, died_in_year)
    line_26000 = np.maximum(line_23600 - deduction, 0.0)
    fed_tax = federal.net_tax(
        federal.gross_tax(line_26000, fed, january_month_index),
        federal.non_refundable_credits(
            line_23600,
            age_at_end_of_year,
            epi,
            ledger.cpp_base_contributions,
            ledger.ei_premiums,
            ledger.eligible_dividends,
            fed,
            january_month_index,
        ),
    )
    prov_tax = provincial.net_tax(
        provincial.gross_tax(line_26000, prov, january_month_index),
        provincial.non_refundable_credits(
            line_23600,
            age_at_end_of_year,
            epi,
            ledger.cpp_base_contributions,
            ledger.ei_premiums,
            ledger.eligible_dividends,
            prov,
            fed,
            january_month_index,
        ),
    )
    transfer_in_arr = np.broadcast_to(
        np.asarray(transfer_in, dtype=np.float64), fed_tax.shape
    ).copy()
    transfer_out_arr = np.broadcast_to(
        np.asarray(transfer_out, dtype=np.float64), fed_tax.shape
    ).copy()
    return Assessment(
        federal=fed_tax,
        provincial=prov_tax,
        oas_repayment=repay,
        aip_penalty=penalty,
        total=fed_tax + prov_tax + repay + penalty,
        net_income=net,
        net_income_after_repayment=line_23600,
        taxable_income=line_26000,
        transfer_in=transfer_in_arr,
        transfer_out=transfer_out_arr,
    )


#: Fields of :class:`Assessment`, in the order every stack below is built in.
#: Derived from the dataclass itself rather than hand-copied, so it cannot
#: drift from ``Assessment``'s actual fields.
_ASSESSMENT_FIELDS: Final[tuple[str, ...]] = tuple(f.name for f in dataclasses.fields(Assessment))


def _stack_all(assessments: list[Assessment]) -> dict[str, NDArray[np.float64]]:
    """Every field of a list of candidate ``Assessment``s, each stacked once.

    One ``(n_candidates, n_paths)`` array per field, built once regardless of
    how many times a caller needs to read from it — the ``argmin`` over the
    split-dependent lines and the later gather by ``best`` both read this
    same dict.
    """
    return {
        field: np.stack([getattr(assessment, field) for assessment in assessments], axis=0)
        for field in _ASSESSMENT_FIELDS
    }


def _select(
    stacked: dict[str, NDArray[np.float64]], best: NDArray[np.intp], idx: NDArray[np.intp]
) -> Assessment:
    return Assessment(**{field: stacked[field][best, idx] for field in _ASSESSMENT_FIELDS})


def household_assessment(state: HouseholdState, params: RealParamYear) -> tuple[Assessment, ...]:
    """Assess every person in the household, at the tax-minimising pension split.

    Elects pension income splitting to minimise, per path, the sum over both
    persons of ``federal + provincial + oas_repayment`` — the lines the split
    changes. The AIP penalty is excluded from that sum because it does not
    depend on the split; each returned :class:`Assessment` still carries it
    within ``total``. This is a within-year election on the completed year,
    which must not consider future years (L15). The search is a grid of
    resolution :data:`GRID_STEP` between zero and the statutory maximum share
    from params/, plus the maximum itself (L51); it does not search every real
    fraction. A household of one skips the search and elects zero.

    The split is available unless either person's death took effect before
    January of the year, so in the year of the first death too (L41). With ``m``
    a person's months of the year up to and including the month of death (a
    January death counts 1), the maximum transfer from ``x`` to ``y`` is the
    share times ``min(m_x, m_y) / m_x`` times ``x``'s eligible pension income
    (ITA 60.03(1); T1032 line 18). A transferee under the age
    ``eligible_pension_income.rrif_minimum_age_years`` at year end counts toward
    the pension credit no more than that maximum taken on ``x``'s qualified
    pension income alone (ITA 118(7); ``federal.qualified_pension_income``).

    Args:
        state: Household state at the December close.
        params: Every parameter file for the tax year, in real dollars.

    Returns:
        One :class:`Assessment` per person, in ``state.persons`` order, at
        the elected split.

    Raises:
        ValueError: If ``state.persons`` holds neither one nor two people —
            the only household sizes this model assesses.
    """
    january_month_index = state.month_index - (state.month - 1)
    ages = tuple(
        _age_at_end_of_year(person.birth_year, person.birth_month, state.year)
        for person in state.persons
    )
    # "This is the death year" -- compared, never subtracted, so
    # engine.core.state.DEATH_NOT_DRAWN's overflow never enters the arithmetic.
    died_in_year = tuple(
        (january_month_index <= person.death_month_index)
        & (person.death_month_index <= state.month_index)
        for person in state.persons
    )

    if len(state.persons) == 1:
        person = state.persons[0]
        zero = np.zeros(state.n_paths, dtype=np.float64)
        return (
            person_assessment(
                person.income,
                ages[0],
                zero,
                zero,
                state.province,
                params,
                january_month_index,
                died_in_year=died_in_year[0],
            ),
        )

    if len(state.persons) != 2:
        raise ValueError(
            f"household_assessment takes a household of one or two persons, "
            f"got {len(state.persons)}."
        )

    person0, person1 = state.persons
    age0, age1 = ages
    fed = params.federal
    epi0 = federal.eligible_pension_income(person0.income, age0, fed)
    epi1 = federal.eligible_pension_income(person1.income, age1, fed)
    qpi0 = federal.qualified_pension_income(person0.income)
    qpi1 = federal.qualified_pension_income(person1.income)

    maximum_share = fed.number("pension_splitting.maximum_transfer_share")
    magnitudes: list[float] = []
    k = 0
    while k * GRID_STEP < maximum_share:
        magnitudes.append(k * GRID_STEP)
        k += 1
    magnitudes.append(maximum_share)

    fractions: list[float] = [*magnitudes, *(-m for m in magnitudes[1:])]

    # Months of the tax year up to and including the month of death (a January death
    # counts 1): clip(death_month_index - january + 1, 0, 12), the minimum taken first
    # so engine.core.state.DEATH_NOT_DRAWN never enters the subtraction.
    december_month_index = january_month_index + MONTHS_PER_YEAR - 1
    months0, months1 = (
        np.clip(
            np.minimum(person.death_month_index, december_month_index) - january_month_index + 1,
            0,
            MONTHS_PER_YEAR,
        )
        for person in (person0, person1)
    )
    # min(m_x, m_y) / m_x, zero where either death took effect before January;
    # np.maximum only guards a zero m_x, whose numerator is zero too.
    shared = np.minimum(months0, months1).astype(np.float64)
    scale0 = shared / np.maximum(months0, 1)
    scale1 = shared / np.maximum(months1, 1)
    min_age = fed.number("eligible_pension_income.rrif_minimum_age_years")
    zeros = np.zeros(state.n_paths, dtype=np.float64)

    assessments0: list[Assessment] = []
    assessments1: list[Assessment] = []
    for fraction in fractions:
        if fraction >= 0:
            transfer = fraction * (scale0 * epi0)
            t_in0, t_out0 = zeros, transfer
            t_in1, t_out1 = transfer, zeros
            base0 = zeros
            base1 = (
                np.minimum(transfer, maximum_share * (scale0 * qpi0))
                if age1 < min_age
                else transfer
            )
        else:
            transfer = -fraction * (scale1 * epi1)
            t_in0, t_out0 = transfer, zeros
            t_in1, t_out1 = zeros, transfer
            base0 = (
                np.minimum(transfer, maximum_share * (scale1 * qpi1))
                if age0 < min_age
                else transfer
            )
            base1 = zeros
        assessments0.append(
            person_assessment(
                person0.income,
                age0,
                t_in0,
                t_out0,
                state.province,
                params,
                january_month_index,
                died_in_year=died_in_year[0],
                transfer_in_credit_base=base0,
            )
        )
        assessments1.append(
            person_assessment(
                person1.income,
                age1,
                t_in1,
                t_out1,
                state.province,
                params,
                january_month_index,
                died_in_year=died_in_year[1],
                transfer_in_credit_base=base1,
            )
        )

    stacked0 = _stack_all(assessments0)
    stacked1 = _stack_all(assessments1)
    totals = (stacked0["federal"] + stacked0["provincial"] + stacked0["oas_repayment"]) + (
        stacked1["federal"] + stacked1["provincial"] + stacked1["oas_repayment"]
    )
    best = np.argmin(totals, axis=0)
    idx = np.arange(state.n_paths)

    return (_select(stacked0, best, idx), _select(stacked1, best, idx))

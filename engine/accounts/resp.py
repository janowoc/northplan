# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registered Education Savings Plan — tracked per beneficiary.

**Never one pot.** Grant room, the lifetime contribution limit, and the
withdrawal window are all per-beneficiary and do not aggregate — an EAP paid
for one child cannot draw on another's grant. Every function here takes a
single beneficiary's state; the step iterates.

A withdrawal splits three ways and the split is not the caller's choice:
contributions come out tax-free, while grant and accumulated income come out
as an Educational Assistance Payment. In reality that EAP is taxable in the
*student's* hands; this engine does not model the student's own tax return
(``docs/limitations.md`` L30) and so :func:`education_draw` puts the whole
payment in ``WithdrawalResult.tax_free`` — none of it reaches the household
ledger.

Grant room accrues once a year, in January, **in grant dollars** — it is an
amount of grant entitlement, not of contribution. The annual grant maximum is
enforced against the year-to-date grant received, not a single month's.
Enrolment begins in a month, not on 1 January, and the EAP cap window is
measured in weeks from the start of enrolment — not modelled here
(``docs/limitations.md`` L31) — so the enrolment month is state the plan
carries but this module does not read.

The grant has two tiers: a basic match paid to everyone, bounded by an annual
maximum that carries unused room forward; and an additional match on the
first dollars of each year's contribution, at a rate that steps down as
family income rises, with no carry-forward and no separate annual cap — its
only ceiling is the lifetime maximum, shared with the basic tier.

Parameters come from ``params/{year}/resp.yaml``.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.accounts.base import WithdrawalResult
from engine.core.indexation import RealParamSet, nominal_carry_factor
from engine.core.state import RespState, updated

#: Net-income fields ``PersonState`` carries (``prior_year_net_income``,
#: ``net_income_two_years_prior``) — a structural fact about the state, not a
#: copy of the published ``grant.enhanced.income_year_offset``. The offset's
#: magnitude is this count; its sign is the file's own "years added to the
#: current year" convention and is policy, not structure, which is why
#: :func:`governing_income_year_offset` compares against
#: ``-_NET_INCOME_FIELDS`` rather than this constant directly.
_NET_INCOME_FIELDS: Final[int] = 2

#: Float-residue bound in dollars, not a tax parameter: erosion and growth arithmetic have
#: been observed to leave an exhausted plan's value about 1e-13 below zero, several orders
#: of magnitude inside this bound. Below it, a negative ``contributions + grants + income``
#: is not float dust -- it is an upstream error, named and raised rather than clipped away.
_VALUE_DUST_TOLERANCE_DOLLARS: Final[float] = -1e-6


def _clip_value_or_raise(value: NDArray[np.float64], caller: str) -> NDArray[np.float64]:
    """Clip ``value`` at zero, or raise if it is a real negative rather than float dust.

    Args:
        value: ``contributions + grants + income`` for one beneficiary, ``(n_paths,)``.
        caller: Name of the calling function, for the message.

    Returns:
        ``value``, clipped at zero.

    Raises:
        ValueError: If ``value`` is below :data:`_VALUE_DUST_TOLERANCE_DOLLARS` on any
            path, naming the value. Growth cannot take a plan's value below zero
            (:func:`grow` multiplies by a strictly positive factor, L35), so a value this
            far below zero did not come from here -- it is an upstream error.
    """
    if np.any(value < _VALUE_DUST_TOLERANCE_DOLLARS):
        raise ValueError(
            f"{caller}: contributions + grants + income is {value!r}, below "
            f"{_VALUE_DUST_TOLERANCE_DOLLARS!r} on at least one path. That is not "
            "float residue -- growth cannot take a plan's value below zero -- so it "
            "is an upstream error, named here rather than silently clipped away."
        )
    return np.clip(value, 0, None)


def grant_room_accrued(
    age_at_end_of_year: int,
    params: RealParamSet,
    january_month_index: int,
) -> float:
    """New grant room accrued in January, in grant dollars.

    Computes the year's quantity unconditionally — there is no start-year
    guard here; that lives once, in ``engine.core.step.open_year`` (#19).

    Args:
        age_at_end_of_year: Age in whole years on 31 December. Not per-path.
        params: The ``resp`` parameter set, supplying ``grant.room_annual``
            and ``grant.cessation_age_years``.
        january_month_index: Month index of January of the year the room is
            granted for.

    Returns:
        ``grant.room_annual`` while ``age_at_end_of_year <=
        grant.cessation_age_years``, else zero.
    """
    cessation_age = params.number("grant.cessation_age_years")
    if age_at_end_of_year > cessation_age:
        return 0.0
    return params.annual_amount("grant.room_annual", january_month_index)


def enhanced_grant_rate(
    family_income: ArrayLike,
    params: RealParamSet,
    january_month_index: int,
) -> NDArray[np.float64]:
    """Additional match rate for the first dollars of this year's contribution.

    A cliff table, not a phase-out: rates **descend** as income rises, and a
    dollar of income across a cut-off changes the rate on every eligible
    dollar, not just the dollars above the cut-off. Cut-offs and rates come
    from ``params`` under ``grant.enhanced``
    (``len(match_rates) == len(income_edges_annual) + 1``). A cut-off itself
    belongs to the **lower-income, higher-rate** band: income exactly at an
    edge is still "at or below" that edge, not above it. This resolves the
    **opposite** way from a tax bracket edge in ``engine.tax.brackets`` — that
    module's ``marginal_rate`` puts an exact edge in the *higher* bracket,
    because it is pricing the next dollar earned, which does fall there. A
    CESG cliff rates the whole eligible window at one rate; there is no next
    dollar to reason about, and family income landing exactly on the
    published cut-off has not yet crossed it.

    This returns a rate; the dollar cap on eligible contribution is applied
    by :func:`grant_on_contribution`.

    Args:
        family_income: Family income for the governing year, ``(n_paths,)`` —
            summed across every person in the household, including one who
            has died, whose last-written figure keeps counting
            (``docs/limitations.md`` L33) — not one person's. The governing
            year is set by ``grant.enhanced.income_year_offset`` in
            ``params`` and applied by the caller, not guessed here.
        params: The ``resp`` parameter set.
        january_month_index: Month index of January of the year the rate is
            being read for.

    Returns:
        Additional match rate as a bare fraction, ``(n_paths,)``, zero above
        the highest cut-off.
    """
    edges = params.annual_amounts("grant.enhanced.income_edges_annual", january_month_index)
    rates = params.numbers("grant.enhanced.match_rates")
    edges_arr = np.asarray(edges, dtype=np.float64)
    rates_arr = np.asarray(rates, dtype=np.float64)
    income_arr = np.asarray(family_income, dtype=np.float64)
    index = np.searchsorted(edges_arr, income_arr, side="left")
    return np.asarray(rates_arr[index], dtype=np.float64)


def governing_income_year_offset(params: RealParamSet) -> int:
    """Confirm ``resp.yaml``'s reach-back is still two years, and return it.

    Reads ``grant.enhanced.income_year_offset`` with :meth:`RealParamSet.get`,
    not :meth:`RealParamSet.number`: the value is a count of years, not a
    dollar or a rate, and the path is unrouted in the file's ``indexation``
    block, so ``number()`` would coerce it to ``float`` for no reason — and
    unlike ``number()``, ``get()`` returns the value's own type, so a caller
    can tell an ``int`` in the file from a ``float``.

    Does not touch :func:`enhanced_grant_rate`, whose docstring keeps the
    governing year "applied by the caller, not guessed here" — this function
    is for that caller (#19's ``open_year``) to consult when it decides which
    of ``PersonState``'s two net-income fields to sum.

    Args:
        params: The ``resp`` parameter set.

    Returns:
        The offset, always ``-2`` (``-_NET_INCOME_FIELDS``) if this returns
        at all.

    Raises:
        ValueError: If the value is not finite or not integral (``bool``,
            ``NaN``, and infinity all take this branch, never
            :class:`OverflowError`), or is integral but not
            ``-_NET_INCOME_FIELDS``. ``PersonState`` carries exactly that
            many net-income fields, so a two-year reach-back is the only one
            the state can express; if the published offset ever changes, the
            state gains or loses a field, and this is where that would first
            surface.
    """
    path = "grant.enhanced.income_year_offset"
    value = params.get(path)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not float(value).is_integer()
    ):
        raise ValueError(
            f"{params.raw.source}[{path}]: expected a whole number of years, found {value!r}."
        )
    offset = int(value)
    if offset != -_NET_INCOME_FIELDS:
        raise ValueError(
            f"{params.raw.source}[{path}]: is {offset}, not -{_NET_INCOME_FIELDS}. "
            "PersonState carries two net-income figures, so a two-year "
            "reach-back is the only one it can express. A genuinely changed "
            "offset means the state needs a field added or removed before "
            "this guard can be relaxed."
        )
    return offset


def basic_grant(
    contribution: ArrayLike,
    state: RespState,
    params: RealParamSet,
    january_month_index: int,
) -> NDArray[np.float64]:
    """Basic-tier grant matched to this month's contribution.

    Bounded four ways: the flat match rate, the unused annual room (which
    carries forward, ``state.grant_room``), the annual maximum against grant
    already received this calendar year (``state.grant_received_ytd``), and
    the lifetime maximum against grant received over all years
    (``state.grants_lifetime``) — ``grant.maximum_annual`` bounds this tier
    only (``params/2026/resp.yaml:67``).

    Args:
        contribution: This month's contribution, already capped at the
            lifetime contribution maximum, ``(n_paths,)``.
        state: That beneficiary's plan state.
        params: The ``resp`` parameter set.
        january_month_index: Month index of January of the current year.

    Returns:
        Basic-tier grant paid this month, ``(n_paths,)``, floored at zero.
    """
    match_rate = params.number("grant.match_rate")
    maximum_annual = params.annual_amount("grant.maximum_annual", january_month_index)
    maximum_lifetime = params.annual_amount("grant.maximum_lifetime", january_month_index)
    contribution_arr = np.asarray(contribution, dtype=np.float64)
    matched = match_rate * contribution_arr
    bound_room = state.grant_room
    bound_annual = maximum_annual - state.grant_received_ytd
    bound_lifetime = maximum_lifetime - state.grants_lifetime
    grant = np.minimum(np.minimum(matched, bound_room), np.minimum(bound_annual, bound_lifetime))
    return np.asarray(np.clip(grant, 0, None), dtype=np.float64)


def enhanced_grant(
    contribution: ArrayLike,
    family_income: ArrayLike,
    state: RespState,
    basic_grant_paid: ArrayLike,
    params: RealParamSet,
    january_month_index: int,
) -> NDArray[np.float64]:
    """Additional-tier grant matched to this month's contribution.

    Matches ``grant.enhanced`` rate against the first
    ``grant.enhanced.eligible_contribution_annual`` dollars contributed this
    calendar year (``state.contributed_ytd``, excluding this month's
    contribution). There is deliberately no annual cap or carry-forward for
    this tier — its only bound besides the eligible window is the lifetime
    maximum, **shared** with the basic tier: match basic first, then the
    enhanced tier takes what is left of it, which is why
    ``basic_grant_paid`` is an argument.

    Args:
        contribution: This month's contribution, already capped at the
            lifetime contribution maximum, ``(n_paths,)``.
        family_income: Family income for the year that governs the rate; see
            :func:`enhanced_grant_rate`.
        state: That beneficiary's plan state.
        basic_grant_paid: This month's basic-tier grant, from
            :func:`basic_grant`, so the shared lifetime cap accounts for it.
        params: The ``resp`` parameter set.
        january_month_index: Month index of January of the current year.

    Returns:
        Additional-tier grant paid this month, ``(n_paths,)``, floored at
        zero.
    """
    rate = enhanced_grant_rate(family_income, params, january_month_index)
    eligible_window = params.annual_amount(
        "grant.enhanced.eligible_contribution_annual", january_month_index
    )
    maximum_lifetime = params.annual_amount("grant.maximum_lifetime", january_month_index)
    contribution_arr = np.asarray(contribution, dtype=np.float64)
    remaining_window = np.clip(eligible_window - state.contributed_ytd, 0, None)
    eligible_contribution = np.minimum(contribution_arr, remaining_window)
    matched = rate * eligible_contribution
    remaining_lifetime = np.clip(
        maximum_lifetime - state.grants_lifetime - np.asarray(basic_grant_paid, dtype=np.float64),
        0,
        None,
    )
    grant = np.minimum(matched, remaining_lifetime)
    return np.asarray(np.clip(grant, 0, None), dtype=np.float64)


def grant_on_contribution(
    contribution: ArrayLike,
    family_income: ArrayLike,
    state: RespState,
    params: RealParamSet,
    january_month_index: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Both grant tiers matched to this month's contribution, for one beneficiary.

    Returns the pair rather than their sum: ``state.grant_received_ytd``
    tracks the basic half only (:func:`basic_grant`'s docstring).

    Args:
        contribution: This month's contribution, already capped at the
            lifetime contribution maximum, ``(n_paths,)``.
        family_income: Family income for the year that governs the enhanced
            rate; see :func:`enhanced_grant_rate`.
        state: That beneficiary's plan state.
        params: The ``resp`` parameter set.
        january_month_index: Month index of January of the current year.

    Returns:
        ``(basic, enhanced)``, each ``(n_paths,)``.
    """
    basic = basic_grant(contribution, state, params, january_month_index)
    enhanced = enhanced_grant(
        contribution, family_income, state, basic, params, january_month_index
    )
    return basic, enhanced


def contribute(
    state: RespState,
    requested: ArrayLike,
    family_income: ArrayLike,
    age_at_end_of_year: int,
    january_month_index: int,
    params: RealParamSet,
) -> tuple[RespState, NDArray[np.float64]]:
    """Contribute for one beneficiary this month and receive the matching grant.

    The contribution is capped at the lifetime contribution maximum
    **first**, and both grant tiers are computed on the capped amount. No
    grant at all once ``age_at_end_of_year > grant.cessation_age_years``
    (``docs/limitations.md`` L33) — the contribution itself is unaffected.

    Args:
        state: That beneficiary's plan state.
        requested: Desired contribution this month, ``(n_paths,)``. Negative
            on any path raises.
        family_income: Family income for the year that governs the enhanced
            grant rate; see :func:`enhanced_grant_rate`.
        age_at_end_of_year: Age in whole years on 31 December. Not per-path.
        january_month_index: Month index of January of the current year.
        params: The ``resp`` parameter set.

    Returns:
        ``(new_state, contributed)``. ``new_state`` has the contribution, both
        grant tiers, and the year-to-date and lifetime totals applied.
        ``state.income`` is untouched by a contribution.

    Raises:
        ValueError: If ``requested`` is negative on any path.
    """
    requested_arr = _non_negative(requested)
    lifetime_max = params.annual_amount("contributions.maximum_lifetime", january_month_index)
    remaining_lifetime = np.clip(lifetime_max - state.contributions_lifetime, 0, None)
    contributed = np.minimum(requested_arr, remaining_lifetime)

    cessation_age = params.number("grant.cessation_age_years")
    if age_at_end_of_year > cessation_age:
        zeros = np.zeros_like(contributed)
        basic, enhanced = zeros, zeros
    else:
        basic, enhanced = grant_on_contribution(
            contributed, family_income, state, params, january_month_index
        )

    new_state = updated(
        state,
        contributions=state.contributions + contributed,
        grants=state.grants + basic + enhanced,
        contributions_lifetime=state.contributions_lifetime + contributed,
        grants_lifetime=state.grants_lifetime + basic + enhanced,
        grant_room=state.grant_room - basic,
        grant_received_ytd=state.grant_received_ytd + basic,
        contributed_ytd=state.contributed_ytd + contributed,
    )
    return new_state, np.asarray(contributed, dtype=np.float64)


def grow(state: RespState, monthly_real_return: ArrayLike) -> RespState:
    """Apply one month of real return to the plan's whole value.

    The entire result accrues to ``state.income``: ``contributions`` and
    ``grants`` keep the nominal amounts the wind-up acts on, so
    ``state.income`` may go negative on a month of losses, and must be
    allowed to.

    Args:
        state: Opening RESP state.
        monthly_real_return: Real, monthly return as a bare fraction,
            ``(n_paths,)``.

    Returns:
        Updated state; only ``income`` changes.
    """
    monthly_real_return_arr = np.asarray(monthly_real_return, dtype=np.float64)
    value = state.contributions + state.grants + state.income
    growth = value * monthly_real_return_arr
    return updated(state, income=state.income + growth)


def education_draw(state: RespState) -> tuple[RespState, WithdrawalResult]:
    """Pay the scheduled monthly education cost, or the plan's whole value if less.

    Composition follows CESP Provider User Guide ch. 3-2: the payment draws
    grant and income **in proportion to their shares** of the ``grants +
    income`` pool while income is positive, and from grant alone when income
    is zero or negative; contributions are drawn only once grant and income
    are exhausted (``docs/limitations.md`` L31). No Canada Learning Bond is
    modelled (L33).

    Takes no ``months_remaining`` argument: dividing by the months left would
    drain the plan by construction, making :func:`wind_up`, the grant
    repayment, and :func:`aip_penalty` unreachable, and hiding the cost of
    over-funding.

    The whole payment goes in ``result.tax_free`` — see the module docstring's
    note on L30; ``result.fully_taxable`` is always zero.

    ``value = contributions + grants + income`` is clipped at zero before it is used:
    float residue can leave an exhausted plan's value a few ulps below zero after
    enough years of :func:`~engine.accounts.resp.erode_nominal` and :func:`grow` have
    shuffled amounts between the three buckets, and that dust is clipped rather than
    raised (see :func:`_clip_value_or_raise`). A value further below zero is not dust
    and raises instead (see Raises). Since a clipped ``value`` of zero forces
    ``payment`` to zero too, and both grant-and-income draws are floored at zero the
    same way (``from_pool`` while income is positive, the grant draw while it is not),
    the three buckets are left unchanged whenever there is nothing to pay.

    Args:
        state: Opening RESP state for one beneficiary.

    Returns:
        ``(new_state, result)``. ``result.shortfall`` is the scheduled cost
        not covered because the plan's value fell short of it.

    Raises:
        ValueError: If ``contributions + grants + income`` is below
            :data:`_VALUE_DUST_TOLERANCE_DOLLARS` on any path; see
            :func:`_clip_value_or_raise`.
    """
    value = _clip_value_or_raise(
        state.contributions + state.grants + state.income, "education_draw"
    )
    payment = np.minimum(state.education_monthly_cost, value)

    grants = state.grants
    income = state.income
    pool = grants + income
    income_positive = income > 0

    pool_safe = np.where(pool > 0, pool, 1.0)
    from_pool = np.clip(np.minimum(payment, pool), 0, None)
    from_grants_if_positive = from_pool * grants / pool_safe
    from_grants_if_nonpositive = np.clip(np.minimum(payment, grants), 0, None)

    from_grants = np.where(income_positive, from_grants_if_positive, from_grants_if_nonpositive)
    from_income = np.where(income_positive, from_pool - from_grants_if_positive, 0.0)
    from_contributions = payment - from_grants - from_income

    new_state = updated(
        state,
        contributions=state.contributions - from_contributions,
        grants=state.grants - from_grants,
        income=state.income - from_income,
    )
    shortfall = np.clip(state.education_monthly_cost - payment, 0, None)
    zeros = np.zeros_like(payment)
    result = WithdrawalResult(
        gross=payment,
        fully_taxable=zeros,
        capital_gain=zeros,
        tax_free=payment,
        shortfall=shortfall,
    )
    return new_state, result


def wind_up(
    state: RespState,
) -> tuple[RespState, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Wind up the plan: repay grant, pay contributions tax-free, tax the rest.

    Matches CESP Provider User Guide ch. 3-3: the grant repayment is the
    lesser of the grant account balance and the plan's fair market value at
    the time of the AIP. Every one of the three outputs is non-negative on
    every path, and they conserve dollars exactly: ``to_cash_free +
    accumulated + grants_repaid == max(value, 0)``, where ``value =
    contributions + grants + income``. ``value`` itself is never below zero by
    the time it is used here: a few ulps of float residue from years of erosion
    and growth are clipped (see :func:`_clip_value_or_raise`), and anything
    further below zero is an upstream error, raised there rather than reaching
    this point at all -- ``grow`` cannot take a plan's value below zero on its
    own. Called by the step in the month **after** the education window
    (``docs/limitations.md`` L32).

    Args:
        state: Opening RESP state for one beneficiary.

    Returns:
        ``(new_state, to_cash_tax_free, accumulated_income_to_subscriber,
        grants_repaid)``. ``new_state`` has all three buckets zeroed and
        ``wound_up`` set to ``True``.

    Raises:
        ValueError: If ``contributions + grants + income`` is below
            :data:`_VALUE_DUST_TOLERANCE_DOLLARS` on any path; see
            :func:`_clip_value_or_raise`.
    """
    value = _clip_value_or_raise(state.contributions + state.grants + state.income, "wind_up")
    grants_repaid = np.clip(np.minimum(state.grants, value), 0, None)
    remaining = value - grants_repaid
    to_cash_free = np.clip(np.minimum(state.contributions, remaining), 0, None)
    accumulated = remaining - to_cash_free

    new_state = updated(
        state,
        contributions=np.zeros_like(state.contributions),
        grants=np.zeros_like(state.grants),
        income=np.zeros_like(state.income),
        wound_up=np.ones_like(state.wound_up),
    )
    return (
        new_state,
        np.asarray(to_cash_free, dtype=np.float64),
        np.asarray(accumulated, dtype=np.float64),
        np.asarray(grants_repaid, dtype=np.float64),
    )


def aip_penalty(accumulated_income: ArrayLike, params: RealParamSet) -> NDArray[np.float64]:
    """Penalty on the accumulated-income portion of a wind-up.

    ``accumulated_income * aip.penalty_rate``. That is all this does —
    assessing the penalty (which line it lands on, whether it enters net
    income, wiring it through ``close_year``) is issue #50 and out of scope
    here.

    Args:
        accumulated_income: The AIP paid to the subscriber, from
            :func:`wind_up`, ``(n_paths,)``.
        params: The ``resp`` parameter set, supplying ``aip.penalty_rate``.

    Returns:
        Penalty amount, ``(n_paths,)``.
    """
    rate = params.number("aip.penalty_rate")
    return np.asarray(np.asarray(accumulated_income, dtype=np.float64) * rate, dtype=np.float64)


def erode_nominal(state: RespState, inflation_rate: float) -> RespState:
    """One January's decay of the plan's nominal buckets, conserving its value.

    Erodes ``grant_room``, ``contributions_lifetime``, ``grants_lifetime``,
    ``contributions``, and ``grants`` — the fields
    ``tests/core/test_state_nominal_or_real.py`` marks ``NOMINAL`` for this
    class. The amount ``contributions`` and ``grants`` lose moves into
    ``income``, so ``contributions + grants + income`` is unchanged:
    the buckets hold the nominal amounts the wind-up acts on, and the plan's
    value is the sum of the three. ``income`` may go negative and that is
    fine. The other two eroded fields, ``grant_room`` and the lifetime
    totals, are running counters against caps the parameter view decays, with
    no offsetting bucket. This applies one year's decay at a time — the
    erosion is annual, not monthly, which is ``docs/limitations.md`` L57.

    Args:
        state: Opening RESP state, before this January's erosion.
        inflation_rate: Assumed annual inflation as a bare fraction.

    Returns:
        Updated state.
    """
    factor = nominal_carry_factor(inflation_rate)
    new_contributions = state.contributions * factor
    new_grants = state.grants * factor
    lost = (state.contributions - new_contributions) + (state.grants - new_grants)
    return updated(
        state,
        contributions=new_contributions,
        grants=new_grants,
        income=state.income + lost,
        contributions_lifetime=state.contributions_lifetime * factor,
        grants_lifetime=state.grants_lifetime * factor,
        grant_room=state.grant_room * factor,
    )


def _non_negative(amount: ArrayLike) -> np.ndarray:
    """``amount`` as a float64 array, or raise if any path is negative."""
    amount_arr = np.asarray(amount, dtype=np.float64)
    if np.any(amount_arr < 0):
        raise ValueError(
            f"amount must be non-negative on every path, got {amount_arr!r}. "
            "A negative request is a caller bug, not a reverse transaction."
        )
    return amount_arr

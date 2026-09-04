"""Locked-In Retirement Account and the LIF it becomes.

Like an RRSP in tax treatment and unlike it in access: withdrawals are barred
until an unlocking age, and once converted to a LIF there is both a mandatory
minimum and a jurisdiction-specific *maximum* withdrawal. The maximum is the
part that distinguishes this from ``rrif.py``.

**The jurisdiction is not the province of residence.** A LIF is governed by the
pension legislation of the jurisdiction its originating pension was registered
under. Someone resident in Alberta may hold an Ontario-registered LIF: they
draw it under Ontario's table and file Alberta income tax. Every function here
that takes a parameter set takes the *registration* jurisdiction's, reached
through ``ParamYear.jurisdiction`` and named by
``engine.core.state.LockedInTerms.registration_jurisdiction`` — never through
``ParamYear.province(household.province)``. The two coincide for a household
that never moved, which is why the wrong one is easy to ship.

Both the minimum and the maximum are **annual** figures fixed in January from
the 1 January balance, and both are satisfied or consumed across the months of
the year. The maximum is the dangerous one: enforced per month it permits
twelve times what the jurisdiction allows, and nothing in the output looks
wrong. It is enforced against the year-to-date total.

Not every jurisdiction imposes a maximum. Some prescribe a RRIF-like account
with a minimum and no ceiling, and that is a rule rather than a gap in the
data. :func:`has_maximum` is what separates the two, so that a jurisdiction
with no ceiling does not look like one whose table nobody has entered yet.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.accounts.base import WithdrawalResult
from engine.params.loader import ParamSet


def withdrawals_permitted(
    age_at_start_of_year: ArrayLike,
    params: ParamSet,
) -> NDArray[np.bool_]:
    """Whether the account may be drawn on at all this year.

    The lock is what makes a LIRA a LIRA rather than an RRSP, and a policy that
    plans a withdrawal before the unlocking age is planning something that
    cannot happen. The unlocking age is jurisdictional and comes from
    ``params``.

    Args:
        age_at_start_of_year: Age in whole years on 1 January, ``(n_paths,)``.
            See ``engine.core.timeline.age_at_start_of_year``.
        params: The parameter set of the jurisdiction the account is
            **registered** in.

    Returns:
        Boolean mask, ``(n_paths,)``.
    """
    raise NotImplementedError


def has_maximum(params: ParamSet) -> bool:
    """Whether this jurisdiction caps annual LIF withdrawals.

    A scalar rule about a jurisdiction, not a per-path quantity, so this
    returns a plain ``bool`` rather than an array.

    Some jurisdictions prescribe a RRIF-like locked-in account with a minimum
    and no ceiling. Without an explicit flag in ``params``, such a jurisdiction
    is indistinguishable from one whose maximum table has not been entered
    yet — both produce a :class:`~engine.params.loader.MissingParameterError`
    on the table lookup — and the pressure at that moment is to invent a
    ceiling or to swallow the error. The flag makes "there is no maximum" a
    statement a human wrote down and verified.

    Args:
        params: The parameter set of the jurisdiction the account is
            **registered** in.

    Returns:
        True if :func:`maximum_withdrawal` should be consulted.

    Raises:
        MissingParameterError: If the flag itself is absent. An unverified
            jurisdiction is not assumed to be uncapped.
    """
    raise NotImplementedError


def maximum_withdrawal(
    opening_balance: ArrayLike,
    age_at_start_of_year: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Jurisdiction-specific maximum LIF withdrawal for the **whole year**.

    Called once, by the January phase of the step, and then drawn down. Call
    :func:`has_maximum` first: this raises for a jurisdiction that has no
    ceiling rather than returning infinity, so that an uncapped account is
    handled by the caller deciding not to cap it.

    Args:
        opening_balance: Balance on 1 January, before growth, ``(n_paths,)``.
        age_at_start_of_year: Age in whole years on 1 January.
        params: The parameter set of the jurisdiction the account is
            **registered** in — from ``ParamYear.jurisdiction`` keyed by
            ``LockedInTerms.registration_jurisdiction``. Not the household's
            province of residence, and not the set that
            ``engine.tax.provincial`` was handed for the same household.

    Returns:
        Maximum permitted withdrawal for the year, ``(n_paths,)``.

    Raises:
        MissingParameterError: If the jurisdiction has no maximum table. Guard
            with :func:`has_maximum`.
    """
    raise NotImplementedError


def withdraw(
    balance: ArrayLike,
    requested: ArrayLike,
    minimum_this_month: ArrayLike,
    maximum_remaining: ArrayLike,
) -> WithdrawalResult:
    """Withdraw this month within the annual bounds. Fully taxable.

    Args:
        balance: Balance at the start of this month, ``(n_paths,)``.
        requested: Policy-requested withdrawal for this month, clamped into
            ``[minimum_this_month, maximum_remaining]``.
        minimum_this_month: RRIF-equivalent floor for this month, from
            ``engine.accounts.rrif.minimum_still_required``.
        maximum_remaining: What is left of the year's maximum, from
            ``engine.accounts.base.remaining_annual_allowance`` applied to
            :func:`maximum_withdrawal`. Not the annual maximum itself. For a
            jurisdiction where :func:`has_maximum` is false, pass the balance:
            the account is still capped by what is in it.

    Returns:
        A :class:`~engine.accounts.base.WithdrawalResult` with everything in
        ``fully_taxable``.
    """
    raise NotImplementedError

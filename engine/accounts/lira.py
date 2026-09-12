# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Locked-In Retirement Account and the LIF it becomes.

Like an RRSP in tax treatment and unlike it in access: withdrawals are barred
until an unlocking age, and once converted to a LIF there is both a mandatory
minimum and a jurisdiction-specific *maximum* withdrawal — the part that
distinguishes this from ``rrif.py``.

**The jurisdiction is not the province of residence.** A LIF is governed by
the pension legislation of the jurisdiction its originating pension was
registered under: a household resident in Alberta may hold an
Ontario-registered LIF, drawn under Ontario's table while filing Alberta
income tax. Every function here that takes a parameter set takes the
*registration* jurisdiction's — ``ParamYear.jurisdiction`` keyed by
``LiraState.jurisdiction`` / ``LifState.jurisdiction`` — never
``ParamYear.province(household.province)``.

Both the minimum and the maximum are **annual** figures fixed in January from
the 1 January balance and enforced against the year-to-date total, not a
single month's amount.

Not every jurisdiction imposes a maximum; some prescribe a RRIF-like account
with a minimum and no ceiling. :func:`has_maximum` is what separates that
rule from a table nobody has entered yet.
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

    Without the flag, a jurisdiction with no ceiling is indistinguishable
    from one whose maximum table has not been entered yet — both would raise
    on lookup. The flag makes "there is no maximum" a fact a human verified,
    not an inference.

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
            **registered** in.

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

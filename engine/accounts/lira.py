# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Locked-In Retirement Account, and its conversion to a LIF.

Like an RRSP in tax treatment and unlike it in access: withdrawals are barred
until an unlocking age, and a LIRA itself pays out nothing — it only
accumulates until converted. The LIF the balance moves into, with its
mandatory minimum and jurisdiction-specific maximum, is
``engine.accounts.lif``.

**The jurisdiction is not the province of residence.** A LIRA is governed by
the pension legislation of the jurisdiction its originating pension was
registered under: a household resident in Alberta may hold an
Ontario-registered LIRA, converting under Ontario's rules while filing Alberta
income tax. Every function here that takes a parameter set takes the
*registration* jurisdiction's — ``ParamYear.jurisdiction`` keyed by
``LiraState.jurisdiction`` — never ``ParamYear.province(household.province)``.
``engine.accounts.lif`` carries the same warning for ``LifState.jurisdiction``.
"""

from __future__ import annotations

import numpy as np

from engine.core.indexation import RealParamSet
from engine.core.state import LifState, LiraState, updated


def withdrawals_permitted(age_at_start_of_year: int, params: RealParamSet) -> bool:
    """Whether the account may be drawn on at all this year.

    The lock is what makes a LIRA a LIRA rather than an RRSP, and a policy that
    plans a withdrawal before the unlocking age is planning something that
    cannot happen. The unlocking age is jurisdictional and comes from
    ``params``.

    Args:
        age_at_start_of_year: Age in whole years on 1 January. See
            ``engine.core.timeline.age_at_start_of_year``. Not per-path: a
            person's age does not vary by path.
        params: The parameter set of the jurisdiction the account is
            **registered** in.

    Returns:
        Whether the lock has lifted.
    """
    unlocking_age = params.number("lif.unlocking_age_years")
    return bool(age_at_start_of_year >= unlocking_age)


def convert_to_lif(lira: LiraState, lif: LifState, year: int) -> tuple[LiraState, LifState]:
    """Move the whole LIRA balance into the LIF, opening it if needed.

    Args:
        lira: The LIRA being converted; zeroed by this call.
        lif: The receiving LIF, possibly already holding a balance from an
            earlier partial conversion.
        year: The calendar year the conversion happens in.

    Returns:
        ``(new_lira, new_lif)``. ``new_lif.jurisdiction`` is the LIF's own if
        it was already set, the LIRA's otherwise. ``new_lif.opened_year`` is
        set to ``year`` when it was ``None``, otherwise left alone.

    Raises:
        ValueError: If both accounts name a jurisdiction and they differ — two
            jurisdictions do not merge (``docs/limitations.md`` L47).
    """
    if lira.jurisdiction and lif.jurisdiction and lira.jurisdiction != lif.jurisdiction:
        raise ValueError(
            f"LIRA jurisdiction {lira.jurisdiction!r} and LIF jurisdiction "
            f"{lif.jurisdiction!r} differ; two jurisdictions do not merge "
            f"(docs/limitations.md L47)."
        )
    jurisdiction = lif.jurisdiction or lira.jurisdiction
    opened_year = lif.opened_year if lif.opened_year is not None else year
    new_lif = updated(
        lif,
        balance=lif.balance + lira.balance,
        jurisdiction=jurisdiction,
        opened_year=opened_year,
    )
    new_lira = updated(lira, balance=np.zeros_like(lira.balance))
    return new_lira, new_lif


def must_convert(age_at_end_of_year: int, params: RealParamSet) -> bool:
    """Whether a LIRA must be converted to a LIF by the end of this year.

    A LIRA converts **at the deadline only** — the December close of the year
    the holder reaches ``lif.conversion_deadline_age_years``, alongside the
    RRSP conversion. In reality a LIRA may be converted at any time from the
    unlocking age; converting earlier is not modelled
    (``docs/limitations.md`` L55).

    Args:
        age_at_end_of_year: Age in whole years on 31 December. Not per-path.
        params: The parameter set of the jurisdiction the account is
            **registered** in, supplying ``lif.conversion_deadline_age_years``.

    Returns:
        Whether conversion is due this year.
    """
    deadline_age = params.number("lif.conversion_deadline_age_years")
    return bool(age_at_end_of_year >= deadline_age)

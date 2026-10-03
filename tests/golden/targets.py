# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Thin shims that reach an engine function a YAML case file cannot call directly.

A case's ``inputs`` mapping becomes keyword arguments to the target, and a
plain mapping cannot express an argument that is itself a constructed engine
type — here, ``engine.core.state.IncomeLedger``, a frozen dataclass of
sixteen ``(n_paths,)`` arrays. A target here exists only to close that gap,
never to make a call site more convenient: add one only when a YAML mapping
genuinely cannot express the target's real signature.

**A shim never computes.** No arithmetic operator and no conditional appears
below; each function here is pure construction — it builds the engine
argument the target actually wants and calls the target, unchanged. Every
per-path numeric argument arrives as a plain number and leaves as a
one-element ``(1,)`` array, via :func:`_one_path`, because
``engine.core.state.freeze`` asserts every carried array is one-dimensional.
There is no type check on that conversion, because a shim contains no
conditional to perform one. ``np.asarray([value], dtype=np.float64)``
converts as numpy does, and numpy's own conversion decides which mistake
raises and which one converts silently — this list is not exhaustive.
Raises: a non-numeric string, a mapping, a date. Converts silently: a
quoted numeric string such as ``"60000"`` becomes ``60000.0``; a bool
becomes ``1.0`` or ``0.0`` (``numpy`` treats a bool as a number); a YAML
null becomes ``NaN``.

A null usually makes the compared output ``NaN`` too and fails the case
loudly — but a null in a field that never reaches the compared output
passes silently: e.g. ``remitted`` on :func:`person_assessment`, which the
engine function never reads, or ``cpp_base_contributions`` or
``ei_premiums`` in a case whose ``expected`` compares only ``net_income``.
A null ``age_at_end_of_year`` is subtler still: every use of age in
``engine/tax/`` is a comparison, and ``NaN >= x`` is ``False``, so the
result is a finite number computed as if the person were under the age
threshold — not ``NaN`` — and it too can pass silently. The case author
writes plain unquoted numbers, the same way any other case author is
responsible for writing numbers.

Every parameter of a shim is keyword-only, ruling out a positional call;
since no function here accepts ``**kwargs``, a misspelled ``inputs`` key is
a ``TypeError`` naming it rather than a silently defaulted zero.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from engine.core.indexation import RealParamYear
from engine.core.state import IncomeLedger
from engine.tax import combined
from engine.tax.combined import Assessment


def _one_path(value: float) -> NDArray[np.float64]:
    """``value`` as a single-path ``(1,)`` array, the shape every carried engine array takes."""
    return np.asarray([value], dtype=np.float64)


def person_assessment(
    *,
    age_at_end_of_year: float,
    province: str,
    params: RealParamYear,
    january_month_index: int,
    transfer_in: float = 0.0,
    transfer_out: float = 0.0,
    employment: float = 0.0,
    cpp: float = 0.0,
    oas: float = 0.0,
    db_pension: float = 0.0,
    rrsp_withdrawals: float = 0.0,
    rrif_lif_withdrawals: float = 0.0,
    inherited_rrif_lif_withdrawals: float = 0.0,
    interest: float = 0.0,
    eligible_dividends: float = 0.0,
    capital_gains: float = 0.0,
    resp_accumulated_income: float = 0.0,
    rrsp_deductions: float = 0.0,
    cpp_base_contributions: float = 0.0,
    cpp_enhanced_contributions: float = 0.0,
    ei_premiums: float = 0.0,
    remitted: float = 0.0,
) -> Assessment:
    """Build a one-path ``IncomeLedger`` and call ``engine.tax.combined.person_assessment``.

    The signature exposes all sixteen ``IncomeLedger`` fields, in
    ``IncomeLedger`` field order, so it matches the dataclass field for
    field. ``remitted`` is accepted but has no effect on the returned
    ``Assessment``, because ``engine.tax.combined.person_assessment`` never
    reads it. See the module docstring for what a target here is, and is
    not, allowed to do.

    Args:
        age_at_end_of_year, province, params, january_month_index,
            transfer_in, transfer_out: Passed through to
            ``engine.tax.combined.person_assessment`` unchanged;
            ``age_at_end_of_year``, ``transfer_in``, and ``transfer_out`` go
            via :func:`_one_path`.
        employment .. remitted: The sixteen ``IncomeLedger`` fields, each via
            :func:`_one_path`.

    Returns:
        The ``Assessment`` ``engine.tax.combined.person_assessment`` returns,
        unchanged.
    """
    ledger = IncomeLedger(
        employment=_one_path(employment),
        cpp=_one_path(cpp),
        oas=_one_path(oas),
        db_pension=_one_path(db_pension),
        rrsp_withdrawals=_one_path(rrsp_withdrawals),
        rrif_lif_withdrawals=_one_path(rrif_lif_withdrawals),
        inherited_rrif_lif_withdrawals=_one_path(inherited_rrif_lif_withdrawals),
        interest=_one_path(interest),
        eligible_dividends=_one_path(eligible_dividends),
        capital_gains=_one_path(capital_gains),
        resp_accumulated_income=_one_path(resp_accumulated_income),
        rrsp_deductions=_one_path(rrsp_deductions),
        cpp_base_contributions=_one_path(cpp_base_contributions),
        cpp_enhanced_contributions=_one_path(cpp_enhanced_contributions),
        ei_premiums=_one_path(ei_premiums),
        remitted=_one_path(remitted),
    )
    return combined.person_assessment(
        ledger=ledger,
        age_at_end_of_year=_one_path(age_at_end_of_year),
        transfer_in=_one_path(transfer_in),
        transfer_out=_one_path(transfer_out),
        province=province,
        params=params,
        january_month_index=january_month_index,
    )

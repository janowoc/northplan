# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Structural check: a LIF's maximum factor never falls below the RRIF minimum factor.

A LIF's floor is the RRIF-equivalent minimum (``engine.accounts.rrif.minimum_factor``)
and its ceiling — where the jurisdiction imposes one — is
``engine.accounts.lif.maximum_withdrawal``. If the minimum ever exceeded the
maximum at the same age, ``engine.accounts.lif.withdraw`` would raise on every
path at that age, which is a parameter error rather than a household's
problem (see its docstring). This test asserts the ordering the real files
must hold for that never to happen — nothing about either factor's value.

Runs against every jurisdiction file under ``params/2026`` that declares
``lif.has_maximum: true``, not a hard-coded list, so a new jurisdiction file
is covered automatically.

Covers every age from the LIF table's own first row up to the RRIF's
terminal age, not just the ages the LIF table happens to list. Above the
LIF's own terminal age, :func:`~engine.accounts.lif.maximum_withdrawal`
applies its terminal row while the RRIF minimum keeps climbing under its own,
later terminal age — exactly the ages at which
``engine.accounts.lif.withdraw`` could raise if the ordering broke, so
stopping at the LIF table's last listed row would leave those ages unchecked.
"""

from __future__ import annotations

import numpy as np

from engine.accounts import lif, rrif
from engine.core.indexation import real_year
from engine.params.loader import load_year


def _jurisdictions_with_a_lif_maximum(year) -> list[str]:
    return [
        name
        for name in year.names()
        if year[name].has("lif.has_maximum") and year[name].get("lif.has_maximum")
    ]


def test_lif_maximum_factor_never_falls_below_the_rrif_minimum_factor() -> None:
    year = load_year(2026)
    real = real_year(year, 0.0)
    jurisdictions = _jurisdictions_with_a_lif_maximum(year)
    assert jurisdictions, "expected at least one jurisdiction file with a LIF maximum table"

    rrif_terminal_age = int(real.rrif.number("rrif.minimum_factors.terminal_age_years"))

    for name in jurisdictions:
        params = real.jurisdiction(name)
        table = params.get("lif.maximum_factors.by_age")
        first_lif_age = min(int(age_str) for age_str in table)
        for age in range(first_lif_age, rrif_terminal_age + 1):
            minimum_factor = rrif.minimum_factor(age, real.rrif)
            # balance of 1.0 turns maximum_withdrawal's dollar amount into the
            # bare factor, without asserting what either factor equals.
            maximum_factor = float(lif.maximum_withdrawal(np.array([1.0]), age, params)[0])
            assert minimum_factor <= maximum_factor, (
                f"jurisdiction {name!r}, age {age}: RRIF minimum factor "
                f"{minimum_factor} exceeds LIF maximum factor {maximum_factor}."
            )

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Total household tax: federal plus provincial, across all persons.

The single entry point the year-end close calls, once per simulated year, on
income accumulated over that year's twelve monthly steps. Household-level
elections that cannot be evaluated one person at a time — pension income
splitting above all — belong here, not in ``federal`` or ``provincial``.

Nothing in this module is called monthly. The election it makes is an annual
one made with the completed year in hand, which is the one point in the
simulation where the whole year *is* legitimately knowable.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from engine.core.state import IncomeLedger
from engine.params.loader import ParamYear


def person_tax(
    person_income: IncomeLedger,
    province: str,
    params: ParamYear,
) -> NDArray[np.float64]:
    """Total income tax for one person: federal plus provincial, after credits.

    Args:
        person_income: That person's income components accumulated over the
            full calendar year.
        province: Two-letter province code, e.g. ``"ab"``.
        params: Loaded parameters for the tax year.

    Returns:
        Combined tax payable, ``(n_paths,)``.
    """
    raise NotImplementedError


def household_tax(
    household_income: tuple[IncomeLedger, ...],
    province: str,
    params: ParamYear,
) -> NDArray[np.float64]:
    """Total income tax across every person in the household.

    Applies household-level elections before summing per-person tax. Pension
    income splitting is chosen to minimise combined tax, which is a joint
    optimisation over both persons and cannot be done per-person.

    Args:
        household_income: Income components for every person, accumulated over
            the full calendar year.
        province: Two-letter province code.
        params: Loaded parameters for the tax year.

    Returns:
        Combined household tax payable, ``(n_paths,)``.
    """
    raise NotImplementedError


def optimal_pension_split(
    household_income: tuple[IncomeLedger, ...],
    province: str,
    params: ParamYear,
) -> NDArray[np.float64]:
    """Fraction of eligible pension income to transfer to the lower earner.

    Bounded by the statutory maximum share. Chosen to minimise combined
    household tax for the year being closed only — this is a within-year
    election, not a multi-year optimisation, and must not consider future
    years.

    Elected once, at the year-end close, on the completed year. It is not a
    monthly decision and must not be recomputed as income accrues: a split
    chosen in March on a quarter of the year's income is not the split that
    minimises the year's tax.

    Args:
        household_income: Income components for every person, accumulated over
            the full calendar year.
        province: Two-letter province code.
        params: Loaded parameters for the tax year.

    Returns:
        Transfer fraction in ``[0, statutory maximum]``, ``(n_paths,)``.
    """
    raise NotImplementedError

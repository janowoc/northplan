"""Total household tax: federal plus provincial, across all persons.

The single entry point the annual loop calls. Household-level elections that
cannot be evaluated one person at a time — pension income splitting above all —
belong here, not in ``federal`` or ``provincial``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from engine.core.state import HouseholdIncome, PersonIncome
from engine.params.loader import ParamYear


def person_tax(
    person_income: PersonIncome,
    province: str,
    params: ParamYear,
) -> NDArray[np.float64]:
    """Total income tax for one person: federal plus provincial, after credits.

    Args:
        person_income: That person's income components for the year.
        province: Two-letter province code, e.g. ``"ab"``.
        params: Loaded parameters for the tax year.

    Returns:
        Combined tax payable, ``(n_paths,)``.
    """
    raise NotImplementedError


def household_tax(
    household_income: HouseholdIncome,
    province: str,
    params: ParamYear,
) -> NDArray[np.float64]:
    """Total income tax across every person in the household.

    Applies household-level elections before summing per-person tax. Pension
    income splitting is chosen to minimise combined tax, which is a joint
    optimisation over both persons and cannot be done per-person.

    Args:
        household_income: Income components for every person.
        province: Two-letter province code.
        params: Loaded parameters for the tax year.

    Returns:
        Combined household tax payable, ``(n_paths,)``.
    """
    raise NotImplementedError


def optimal_pension_split(
    household_income: HouseholdIncome,
    province: str,
    params: ParamYear,
) -> NDArray[np.float64]:
    """Fraction of eligible pension income to transfer to the lower earner.

    Bounded by the statutory maximum share. Chosen to minimise combined
    household tax for the current year only — this is a within-year election,
    not a multi-year optimisation, and must not consider future years.

    Args:
        household_income: Income components for every person.
        province: Two-letter province code.
        params: Loaded parameters for the tax year.

    Returns:
        Transfer fraction in ``[0, statutory maximum]``, ``(n_paths,)``.
    """
    raise NotImplementedError

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Employment income and the CPP and EI contributions it generates.

Employment income is a scenario input, carried as a tuple of
:class:`~engine.core.state.EmploymentBand`, resolved to a monthly figure by
:func:`employment_income_monthly` (L39). Contributions are computed on the
year-to-date total so that a tiered or capped rate applies to the year, never
to twelve times one month's amount: :func:`cpp_contributions_monthly` and
:func:`ei_premium_monthly` both take this month's income and the income
already earned this year, and return the *incremental* contribution the
month adds.

CPP contributions are taken on all employment income at every age, with no
account of whether the person's pension has started or the person's age
relative to 70 (L21): the amount a retirement-age worker's continued
contributions would earn toward a larger pension is not modelled either
(L20). :func:`cpp_contributions_monthly` splits the contribution into the
first tier's base-rate portion, which earns a federal and provincial credit,
and everything else, which is a deduction instead (L14).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.core.indexation import RealParamSet
from engine.core.state import EmploymentBand

__all__ = [
    "CppContributions",
    "cpp_contributions_monthly",
    "ei_premium_monthly",
    "employment_income_monthly",
]


class CppContributions(NamedTuple):
    """One month's incremental CPP contribution, split for the tax credit/deduction rules.

    Attributes:
        base: The first tier's base-rate portion; earns a non-refundable credit (L14).
        enhanced: Everything else — the rest of the first tier plus all of the
            second — a deduction rather than a credit (L14).
    """

    base: NDArray[np.float64]
    enhanced: NDArray[np.float64]


def employment_income_monthly(
    bands: tuple[EmploymentBand, ...],
    month_index: int,
    n_paths: int,
) -> NDArray[np.float64]:
    """This month's employment income, real dollars.

    Args:
        bands: The person's employment bands, in scenario order, possibly
            empty.
        month_index: Months since January of the scenario's start year.
        n_paths: Number of paths; needed because ``bands`` may be empty and
            the zero result must still carry the right shape.

    Returns:
        A copy of the amount of the band with ``from_month_index <=
        month_index <= to_month_index``, else zeros, ``(n_paths,)``.
    """
    for band in bands:
        if band.from_month_index <= month_index <= band.to_month_index:
            return np.array(band.monthly_amount, dtype=np.float64)
    return np.zeros(n_paths, dtype=np.float64)


def _tier1(
    earnings_annual: NDArray[np.float64], exemption: float, ceiling: float
) -> NDArray[np.float64]:
    return np.clip(earnings_annual - exemption, 0, ceiling - exemption)


def _tier2(
    earnings_annual: NDArray[np.float64], tier1_ceiling: float, tier2_ceiling: float
) -> NDArray[np.float64]:
    return np.clip(earnings_annual - tier1_ceiling, 0, tier2_ceiling - tier1_ceiling)


def cpp_contributions_monthly(
    monthly_income: ArrayLike,
    income_ytd: ArrayLike,
    january_month_index: int,
    params: RealParamSet,
) -> CppContributions:
    """This month's incremental CPP contributions, split into base and enhanced.

    Both tiers are computed on the year-to-date total including this month,
    less the year-to-date total excluding it, so a tier's ceiling is enforced
    against the year rather than twelve times a month's amount. The split
    between ``base`` and ``enhanced`` follows the credit/deduction rule the
    caller applies (L14).

    Args:
        monthly_income: This month's employment income, real dollars.
        income_ytd: Employment income earned earlier this year, excluding this
            month, real dollars.
        january_month_index: Month index of January of the tax year.
        params: The ``cpp`` parameter set.

    Returns:
        :class:`CppContributions`, both fields non-negative; still unpacks as
        a two-tuple.
    """
    exemption = params.annual_amount(
        "contributions.first_tier.basic_exemption_annual", january_month_index
    )
    tier1_ceiling = params.annual_amount(
        "contributions.first_tier.ceiling_annual", january_month_index
    )
    tier2_ceiling = params.annual_amount(
        "contributions.second_tier.ceiling_annual", january_month_index
    )
    base_rate = params.number("contributions.first_tier.base_rate")
    employee_rate = params.number("contributions.first_tier.employee_rate")
    tier2_rate = params.number("contributions.second_tier.employee_rate")

    ytd = np.asarray(income_ytd, dtype=np.float64)
    total = ytd + np.asarray(monthly_income, dtype=np.float64)

    tier1_month = employee_rate * (
        _tier1(total, exemption, tier1_ceiling) - _tier1(ytd, exemption, tier1_ceiling)
    )
    tier2_month = tier2_rate * (
        _tier2(total, tier1_ceiling, tier2_ceiling) - _tier2(ytd, tier1_ceiling, tier2_ceiling)
    )

    base = tier1_month * base_rate / employee_rate
    enhanced = tier1_month - base + tier2_month
    return CppContributions(
        base=np.asarray(base, dtype=np.float64), enhanced=np.asarray(enhanced, dtype=np.float64)
    )


def ei_premium_monthly(
    monthly_income: ArrayLike,
    income_ytd: ArrayLike,
    january_month_index: int,
    params: RealParamSet,
) -> NDArray[np.float64]:
    """This month's incremental EI premium.

    Args:
        monthly_income: This month's employment income, real dollars.
        income_ytd: Employment income earned earlier this year, excluding this
            month, real dollars.
        january_month_index: Month index of January of the tax year.
        params: The ``federal`` parameter set.

    Returns:
        This month's premium, real dollars, non-negative.
    """
    rate = params.number("ei.premium_rate")
    maximum = params.annual_amount("ei.maximum_insurable_earnings_annual", january_month_index)
    ytd = np.asarray(income_ytd, dtype=np.float64)
    total = ytd + np.asarray(monthly_income, dtype=np.float64)
    premium = rate * (np.minimum(total, maximum) - np.minimum(ytd, maximum))
    return np.asarray(premium, dtype=np.float64)

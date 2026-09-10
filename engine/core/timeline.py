# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Month arithmetic for the simulation timeline.

The monthly loop needs a small amount of calendar reasoning, and it needs
exactly one copy of it. Two modules that each work out "which tax year is this
month in" will eventually disagree by one month, and the disagreement will show
up as a benefit that starts a month early on some paths.

Conventions fixed here:

- ``month`` is ``1..12`` for January through December. There is no month zero.
- A *month index* is the number of whole months since the start of the
  simulation, counting from zero. It is what indexes the first axis of the
  random draws.
- A *tax year* is a calendar year. Canada does not offer individuals anything
  else.
- Age is carried in months. Whole years are derived from it, never stored
  alongside it, so the two cannot drift apart.

Statutory dates are not written into this file. The month a return is due, the
month a benefit year starts, and the months a benefit is indexed in are rules
that a human verified, so they arrive from ``params/`` like every other rule.
``MONTHS_PER_YEAR`` is the Gregorian calendar, not a tax parameter, and is the
only number here.
"""

from __future__ import annotations

from engine.params.loader import ParamSet

MONTHS_PER_YEAR = 12


def month_index(year: int, month: int, base_year: int) -> int:
    """Months elapsed from January of ``base_year`` to ``(year, month)``.

    Args:
        year: Calendar year.
        month: Month in ``1..12``.
        base_year: The simulation's first calendar year.

    Returns:
        Zero-based index into the first axis of the random draws.

    Raises:
        ValueError: If ``month`` is outside ``1..12``, or the result is
            negative.
    """
    if not 1 <= month <= MONTHS_PER_YEAR:
        raise ValueError(f"month must be in 1..{MONTHS_PER_YEAR}, got {month!r}.")
    index = (year - base_year) * MONTHS_PER_YEAR + (month - 1)
    if index < 0:
        raise ValueError(
            f"month_index would be negative for year={year!r}, month={month!r}, "
            f"base_year={base_year!r}: got {index!r}. month_index has no negative "
            f"rows; a date before the run's opening belongs to "
            f"engine.core.build._month_offset instead."
        )
    return index


def year_month(index: int, base_year: int) -> tuple[int, int]:
    """Calendar date ``index`` months after January of ``base_year``.

    Defined on every integer ``index``, not only the non-negative ones
    :func:`month_index` produces: it is the inverse of :func:`month_index`
    only on that non-negative half, where ``month_index`` has no negative
    rows to be the inverse of. A negative ``index`` is deliberately still
    answered rather than rejected, because it is the ordinary convention
    ``engine/core/state.py`` documents for a month index that predates the
    run — a pension already in payment, a bridge that already ended, an
    employment band begun years earlier — and those states need a way back
    to a calendar date as much as an ordinary one does.

    Args:
        index: Month index, zero in January of ``base_year``. May be
            negative.
        base_year: The simulation's first calendar year.

    Returns:
        ``(year, month)`` with ``month`` in ``1..12``.
    """
    year = base_year + index // MONTHS_PER_YEAR
    month = index % MONTHS_PER_YEAR + 1
    return year, month


def next_month(year: int, month: int) -> tuple[int, int]:
    """The month after ``(year, month)``, rolling the year over at December."""
    if month == MONTHS_PER_YEAR:
        return year + 1, 1
    return year, month + 1


def age_in_months(
    birth_year: int,
    birth_month: int,
    year: int,
    month: int,
) -> int:
    """Age in whole months at the start of ``(year, month)``.

    Whole months completed, so a person born in June is 0 months old for all of
    June and 1 month old in July. CPP's start adjustment is defined per month
    away from 65 and reads this directly; converting to years first and back
    loses the precision the adjustment is made of.

    Args:
        birth_year: Calendar year of birth.
        birth_month: Month of birth, ``1..12``.
        year: Current calendar year.
        month: Current month, ``1..12``.

    Returns:
        Age in whole months, non-negative.
    """
    return (year - birth_year) * MONTHS_PER_YEAR + (month - birth_month)


def age_in_years(
    birth_year: int,
    birth_month: int,
    year: int,
    month: int,
) -> int:
    """Age in whole years at the start of ``(year, month)``.

    Age on the birthday, which is what benefit eligibility keys off. This is
    *not* the "age at the end of the tax year" that some tax credits use, and
    it is not the "age at the start of the year" the RRIF factor uses. Those
    two have their own helpers below precisely so a call site has to say which
    it means.
    """
    return age_in_months(birth_year, birth_month, year, month) // MONTHS_PER_YEAR


def age_at_start_of_year(birth_year: int, birth_month: int, year: int) -> int:
    """Age in whole years on 1 January of ``year``.

    The age that selects the RRIF and LIF minimum factor. Fixed for the whole
    calendar year, which is why the factor can be resolved once in January and
    reused for the eleven months that follow.
    """
    return age_in_years(birth_year, birth_month, year, 1)


def age_at_end_of_year(birth_year: int, birth_month: int, year: int) -> int:
    """Age in whole years on 31 December of ``year``.

    The age several non-refundable credits are tested against, and the age that
    determines the year an RRSP must be converted to a RRIF.
    """
    return age_in_years(birth_year, birth_month, year, MONTHS_PER_YEAR)


def is_year_end(month: int) -> bool:
    """Whether this month closes the tax year, triggering the annual assessment."""
    return month == MONTHS_PER_YEAR


def is_year_start(month: int) -> bool:
    """Whether this month opens the tax year, granting room and fixing minimums."""
    return month == 1


def is_filing_month(month: int, params: ParamSet) -> bool:
    """Whether a balance owing for the prior tax year comes due this month.

    The month comes from ``params``, not from this file: a filing deadline is a
    statutory rule and gets a verification row like any other.

    Args:
        month: Current month, ``1..12``.
        params: The ``federal`` parameter set, supplying the filing month.

    Returns:
        True in the month the prior year's balance is settled.
    """
    return month == int(params.number("filing_month"))

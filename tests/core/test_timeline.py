# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Calendar arithmetic: ``engine.core.timeline``.

Every function here is pure integer arithmetic over ``(year, month)`` pairs
and month indices, so the tests are pinned to specific, hand-checkable dates
rather than round-trip-only properties — a round trip alone would not catch
an implementation that is internally consistent but off by a month against
the calendar everyone else uses.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from engine.core import timeline
from engine.params.loader import ParamSet

BASE_YEAR = 2026


def _federal(filing_month: int) -> ParamSet:
    """A synthetic ``federal`` parameter set naming only ``filing_month``.

    Built by hand rather than loaded from ``params/`` so the test can also
    exercise a *different* filing month than the real file's, which is the
    only way to prove ``is_filing_month`` reads the value from ``params``
    rather than having it inlined.
    """
    return ParamSet(
        name="federal",
        year=BASE_YEAR,
        source=Path("synthetic-federal.yaml"),
        values=MappingProxyType({"filing_month": filing_month}),
    )


class TestMonthIndex:
    def test_january_of_base_year_is_zero(self) -> None:
        """The origin of the index: if this drifts, every other index does too."""
        assert timeline.month_index(BASE_YEAR, 1, BASE_YEAR) == 0

    def test_counts_forward_across_a_year_boundary(self) -> None:
        """2031-04 is 63 months after 2026-01: (2031-2026)*12 + (4-1)."""
        assert timeline.month_index(2031, 4, BASE_YEAR) == 63

    def test_raises_on_month_zero(self) -> None:
        with pytest.raises(ValueError, match=r"1\.\.12"):
            timeline.month_index(BASE_YEAR, 0, BASE_YEAR)

    def test_raises_on_month_thirteen(self) -> None:
        with pytest.raises(ValueError, match=r"1\.\.12"):
            timeline.month_index(BASE_YEAR, 13, BASE_YEAR)

    def test_raises_on_a_year_before_base_year(self) -> None:
        """month_index has no negative rows; a date before the run is a hard error here."""
        with pytest.raises(ValueError):
            timeline.month_index(BASE_YEAR - 1, 12, BASE_YEAR)


class TestYearMonthRoundTrip:
    def test_round_trips_month_index(self) -> None:
        """Would catch either function computing a different index for the same date."""
        for index in range(0, 36):
            year, month = timeline.year_month(index, BASE_YEAR)
            assert timeline.month_index(year, month, BASE_YEAR) == index

    def test_round_trips_year_month_across_a_year_boundary(self) -> None:
        assert timeline.year_month(11, BASE_YEAR) == (BASE_YEAR, 12)
        assert timeline.year_month(12, BASE_YEAR) == (BASE_YEAR + 1, 1)

    def test_a_negative_index_is_a_deliberate_answer_not_a_round_trip(self) -> None:
        """Pins the asymmetry with month_index: this is defined on every
        integer, including the negative half month_index refuses, because
        engine/core/state.py's already-under-way states need a calendar date
        back from a negative index too."""
        assert timeline.year_month(-1, BASE_YEAR) == (BASE_YEAR - 1, 12)
        with pytest.raises(ValueError):
            timeline.month_index(BASE_YEAR - 1, 12, BASE_YEAR)


class TestNextMonth:
    def test_ordinary_month_increments(self) -> None:
        assert timeline.next_month(BASE_YEAR, 5) == (BASE_YEAR, 6)

    def test_december_rolls_to_january_of_the_next_year(self) -> None:
        assert timeline.next_month(BASE_YEAR, 12) == (BASE_YEAR + 1, 1)


class TestAgeInMonths:
    def test_born_in_june_is_zero_months_old_in_june(self) -> None:
        assert timeline.age_in_months(2000, 6, 2000, 6) == 0

    def test_born_in_june_is_one_month_old_in_july(self) -> None:
        """The whole-months convention CPP's start adjustment relies on directly."""
        assert timeline.age_in_months(2000, 6, 2000, 7) == 1

    def test_a_full_year_later_is_twelve_months(self) -> None:
        assert timeline.age_in_months(2000, 6, 2001, 6) == 12


class TestAgeInYears:
    def test_the_birthday_has_not_arrived_yet_this_calendar_year(self) -> None:
        # Born June 2000; by January 2026 they have had 25 birthdays (June
        # 2000..June 2025) and the 26th has not yet arrived.
        assert timeline.age_in_years(2000, 6, 2026, 1) == 25

    def test_on_the_birthday_month_the_age_has_incremented(self) -> None:
        assert timeline.age_in_years(2000, 6, 2026, 6) == 26


class TestAgeAtStartAndEndOfYear:
    def test_june_birthday_differs_across_the_year(self) -> None:
        """The ordinary case: the two ages differ by one for a mid-year birthday."""
        assert timeline.age_at_start_of_year(2000, 6, 2026) == 25
        assert timeline.age_at_end_of_year(2000, 6, 2026) == 26

    def test_january_birthday_agrees_across_the_year(self) -> None:
        """A January birthday has already happened by 1 January, so start and
        end of the same calendar year report the same age — unlike every
        other birth month."""
        assert timeline.age_at_start_of_year(2000, 1, 2026) == 26
        assert timeline.age_at_end_of_year(2000, 1, 2026) == 26

    def test_december_birthday_still_differs_across_the_year(self) -> None:
        """A December birthday has *not* happened by 1 January, so it behaves
        like the ordinary mid-year case, not like the January case above —
        the two are easy to conflate as both 'edge of the year'."""
        assert timeline.age_at_start_of_year(2000, 12, 2026) == 25
        assert timeline.age_at_end_of_year(2000, 12, 2026) == 26


class TestYearBoundaryFlags:
    def test_is_year_start_true_only_in_january(self) -> None:
        assert timeline.is_year_start(1) is True
        for month in range(2, 13):
            assert timeline.is_year_start(month) is False

    def test_is_year_end_true_only_in_december(self) -> None:
        assert timeline.is_year_end(12) is True
        for month in range(1, 12):
            assert timeline.is_year_end(month) is False


class TestIsFilingMonth:
    def test_true_only_in_the_month_params_names(self) -> None:
        params = _federal(filing_month=4)
        assert timeline.is_filing_month(4, params) is True
        for month in (1, 2, 3, 5, 12):
            assert timeline.is_filing_month(month, params) is False

    def test_a_different_params_value_gives_a_different_answer(self) -> None:
        """Proves the month is read from params, not inlined: the same call
        month is true against one synthetic filing month and false against
        another."""
        june_filer = _federal(filing_month=6)
        assert timeline.is_filing_month(4, june_filer) is False
        assert timeline.is_filing_month(6, june_filer) is True

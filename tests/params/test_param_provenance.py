"""Provenance of the real parameter files: does each one say where it came from?

Separate from ``test_param_file_structure.py`` because it asks a different kind
of question. Those tests assert invariants the engine depends on and fail hard,
because a rate table one entry short is broken now. This one asks whether a
human has recorded *when* they checked a value against its source, which is a
matter of audit completeness rather than correctness — the engine runs fine on
an undated value, right up until nobody can say whether it was read off the
2025 page or the 2026 one.

**These warn rather than fail, deliberately.** The source comments are the audit
record now that there is no separate ledger, and coverage of the date half is
still partial. A failing test on partially-complete bookkeeping gets suppressed
or deleted within a week; a warning that names the file every run stays visible
and shrinks as the gap closes. When coverage reaches every file, change
``warnings.warn`` to ``pytest.fail`` here and the ratchet holds.

The canonical form, matched below::

    # 2026-09-03 https://www.canada.ca/en/revenue-agency/...

Date first, then the URL, on its own comment line above the group of values it
sources. Date first because it sorts and greps: ``grep -h '^# 20' params/2026/*``
lists every check in the repository in date order, which is the query the
deleted ledger existed to answer.

``filterwarnings = ["error"]`` in ``pyproject.toml`` makes every warning a
failure, which is the right default for the rest of the suite. The
``filterwarnings("always")`` mark below re-enables warning behaviour for these
tests alone rather than weakening that setting globally.
"""

from __future__ import annotations

import datetime as dt
import re
import warnings
from pathlib import Path

import pytest

from engine.params.loader import DEFAULT_PARAMS_ROOT

#: ``# YYYY-MM-DD https://...`` — a source comment with the date it was checked.
DATED_SOURCE = re.compile(r"^\s*#\s*(\d{4}-\d{2}-\d{2})\s+(https?://\S+)")

#: Any comment carrying a URL, dated or not. Used to tell "no source recorded"
#: from "source recorded but not dated" — different problems, different fixes.
ANY_SOURCE = re.compile(r"^\s*#.*?(https?://\S+)")


class UndatedParameterSourceWarning(UserWarning):
    """A live parameter file records no date against any of its sources."""


class ImplausibleCheckDateWarning(UserWarning):
    """A source comment carries a date that is not a real past date."""


def _live_param_files() -> list[Path]:
    """Every YAML the loader can actually reach, i.e. inside a year directory.

    Drafts at the ``params/`` root are excluded for the same reason the
    placeholder guard excludes them: they are unfinished by definition, and
    warning about a draft's provenance every run would train the reader to
    ignore the warning that matters.
    """
    return sorted(
        path
        for directory in DEFAULT_PARAMS_ROOT.iterdir()
        if directory.is_dir() and directory.name.isdigit()
        for path in directory.rglob("*.yaml")
    )


LIVE_FILES = _live_param_files()
IDS = [f"{path.parent.name}/{path.name}" for path in LIVE_FILES]


@pytest.mark.filterwarnings("always")
@pytest.mark.parametrize("path", LIVE_FILES, ids=IDS)
def test_file_records_at_least_one_dated_source(path: Path) -> None:
    """Each live parameter file carries at least one ``# YYYY-MM-DD https://...``.

    One per file is the floor, not the goal — the convention in every file
    header asks for a source comment above each *group* of values. The floor is
    what a test can assert without deciding how the file ought to be divided
    into groups, which is a judgement about the source material rather than
    about the file.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    dated = [line for line in lines if DATED_SOURCE.match(line)]
    if dated:
        return

    undated = [line for line in lines if ANY_SOURCE.match(line)]
    where = f"{path.parent.name}/{path.name}"
    if undated:
        warnings.warn(
            f"{where}: {len(undated)} source URL(s), none with a check date. "
            f"A URL alone does not say whether it was read for this tax year. "
            f"Prefix each with the date it was checked: "
            f"'# YYYY-MM-DD https://...'.",
            UndatedParameterSourceWarning,
            stacklevel=1,
        )
    else:
        warnings.warn(
            f"{where}: no source comment at all. Every value in this file is "
            f"currently unattributable. Add '# YYYY-MM-DD https://...' above each "
            f"group of values.",
            UndatedParameterSourceWarning,
            stacklevel=1,
        )


@pytest.mark.filterwarnings("always")
@pytest.mark.parametrize("path", LIVE_FILES, ids=IDS)
def test_check_dates_are_real_and_not_in_the_future(path: Path) -> None:
    """Dates parse as calendar dates and do not postdate today.

    A future check date is a typo — a transposed year, or a date copied from
    the tax year rather than the day the page was opened. It is worth catching
    because it is the one error in this format that reads as more authoritative
    than the truth: an undated source looks unfinished, a 2062 date looks
    checked.
    """
    today = dt.date.today()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = DATED_SOURCE.match(line)
        if not match:
            continue
        stamp = match.group(1)
        try:
            checked = dt.date.fromisoformat(stamp)
        except ValueError:
            warnings.warn(
                f"{path.parent.name}/{path.name}:{number}: {stamp!r} is not a "
                f"calendar date.",
                ImplausibleCheckDateWarning,
                stacklevel=1,
            )
            continue
        if checked > today:
            warnings.warn(
                f"{path.parent.name}/{path.name}:{number}: check date {stamp} is in "
                f"the future. A source cannot have been checked on a day that has "
                f"not happened; this is usually the tax year copied in by mistake.",
                ImplausibleCheckDateWarning,
                stacklevel=1,
            )

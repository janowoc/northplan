# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared machinery for the golden test harness.

``test_cases.py`` (real cases under ``tests/golden/cases/``) and
``test_harness.py`` (synthetic cases in ``tmp_path``) both build on this
module rather than re-implementing it. The comparison tolerances in
:data:`ROUNDING_TOLERANCES` describe how a source *publishes* a figure, never
a tax parameter, so they stay here, not under ``params/``. A case file is a
human-authored YAML document of the form::

    target: some.module.doubles_its_input
    cases:
      - name: a human-readable description of what is being checked
        source: "https://example-calculator.invalid/... or a description"
        checked: 2026-09-05
        inputs:
          some_argument: 7.0
          params: {year: 2026, file: federal}
        expected:
          value: 14.0
        rounding: source_rounds_to_cent   # optional, this is the default

``source`` and ``checked`` are mandatory — an unrecorded or unchecked golden
value is exactly the kind of invented number this repository forbids — and
both are stored on the parsed case even though nothing downstream reads them
yet, so a field validated once cannot later rot unvalidated.

There is no numeric ``tolerance`` field; a case declares ``rounding`` instead,
one of the four members of :data:`ROUNDING_TOLERANCES`, defaulting to
``source_rounds_to_cent``. There is no numeric escape hatch — a source none of
the four describes gets a fifth member added here, in a diff checked against
the cited source — and an unrecognized field is rejected rather than dropped,
since a typo'd name is indistinguishable from a deliberate omission.

A ``params`` input resolves to a plain parameter set; ``real_params`` (stating
an inflation rate too) to one file's real-dollar view; and
``real_params_year`` (one file shallower, just ``year`` and ``inflation``) to
the *whole* tax year's real-dollar view. All three go under the keyword
``params``, so a case names exactly one; every malformed spec raises
:class:`GoldenCaseError` at discovery, naming the file and case.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import importlib
import math
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from engine.core.indexation import RealParamSet, RealParamYear, real_year
from engine.params.loader import DEFAULT_PARAMS_ROOT, ParamSet, ParamYear, load_year

__all__ = [
    "CASES_DIR",
    "ROUNDING_TOLERANCES",
    "GoldenCase",
    "GoldenCaseError",
    "discover_cases",
    "resolve_inputs",
    "resolve_params",
    "resolve_real_params",
    "resolve_real_params_year",
    "resolve_target",
    "run_case",
    "to_float",
]

#: Where the real, human-authored case files live.
CASES_DIR: Path = Path(__file__).parent / "cases"

#: The absolute tolerance implied by each declared ``rounding`` reason.
#:
#: These four numbers are properties of how a source *publishes* a figure —
#: to the cent, to the dollar, to ten dollars, or as a monthly amount
#: multiplied by twelve — never of the tax system itself: no bracket, rate,
#: threshold, or credit lives here. That is why they stay in this test
#: harness rather than moving under ``params/``, which ``CLAUDE.md`` reserves
#: for hand-populated tax parameters; putting a test-harness setting there
#: would be actively wrong, not merely misplaced.
#:
#: ``source_rounds_to_cent`` is the default (see ``_DEFAULT_ROUNDING`` below)
#: and its tolerance, 0.01, is the harness's original default tolerance from
#: issue #3, preserved unchanged. There is deliberately no fifth, numeric
#: escape hatch: a source whose imprecision none of these four describes
#: gets a new member added here, in a diff a reviewer can check against the
#: source cited in the case — see the module docstring.
#: Wrapped in :class:`types.MappingProxyType`: a case file can never reach
#: this table, but an imported target module could otherwise mutate it at
#: import time and silently widen every case discovered afterwards — exactly
#: the kind of widening route this module exists to close.
ROUNDING_TOLERANCES: Mapping[str, float] = types.MappingProxyType({
    # Half the last displayed digit is 0.005; 0.01 leaves room for float
    # representation.
    "source_rounds_to_cent": 0.01,
    # Half the last displayed digit.
    "source_rounds_to_dollar": 0.50,
    # Half the last displayed digit.
    "source_rounds_to_ten_dollars": 5.00,
    # A monthly figure rounded to the cent, multiplied by 12, compounds
    # ±0.005 twelve times.
    "monthly_cent_times_twelve": 0.06,
})

#: The ``rounding`` value assumed when a case omits the field entirely.
_DEFAULT_ROUNDING = "source_rounds_to_cent"

#: Short suffix rendered in a case's test id when its ``rounding`` is not the
#: default — see :attr:`GoldenCase.id`. The default member is intentionally
#: absent: a case at cent rounding keeps the plain ``<file stem>::<case
#: name>`` id, so ids do not churn for the common case.
#:
#: Every non-default key of :data:`ROUNDING_TOLERANCES` must have an entry
#: here — see the self-test asserting that in ``test_harness.py`` — or a
#: case using the missing member would run at a loosened tolerance while its
#: id stayed unsuffixed, which is exactly the invisible loosening the suffix
#: exists to prevent. Wrapped in :class:`types.MappingProxyType` for the same
#: reason as :data:`ROUNDING_TOLERANCES`.
_ROUNDING_ID_SUFFIXES: Mapping[str, str] = types.MappingProxyType({
    "source_rounds_to_dollar": "dollar",
    "source_rounds_to_ten_dollars": "ten_dollars",
    "monthly_cent_times_twelve": "monthly_cent_x12",
})

#: Fields a case entry may set. Anything else is a typo, not an extension —
#: see the module docstring on why an unknown field is rejected rather than
#: dropped. ``tolerance`` is deliberately absent: see ``discover_cases`` for
#: the dedicated message a case still setting it receives.
_ALLOWED_CASE_FIELDS = frozenset({"name", "source", "checked", "inputs", "expected", "rounding"})

#: Case-file extensions that are discovered, matched case-sensitively. A file
#: under ``cases/`` with any other suffix — including the same extension in
#: the wrong case, e.g. ``.YAML`` — is a stray file, not a silently-skipped
#: one; see ``discover_cases``.
_CASE_FILE_SUFFIXES = frozenset({".yaml", ".yml"})

#: The one non-case file `cases/` is allowed to hold.
_KEEPER_FILENAME = ".gitkeep"

#: Keys a ``params`` input spec must have — exactly these, no more, no
#: fewer, so a typo (``provice`` for ``province``) is caught here rather
#: than silently ignored.
_PARAMS_SPEC_KEYS = frozenset({"year", "file"})

#: Keys a ``real_params`` input spec must have — one more than ``params``:
#: the deflated real-terms view needs a stated inflation rate, and the
#: harness never assumes one. See ``_validate_real_params_spec``.
_REAL_PARAMS_SPEC_KEYS = frozenset({"year", "file", "inflation"})

#: Keys a ``real_params_year`` input spec must have — ``real_params`` without
#: ``file``: this resolves an entire ``RealParamYear``, every file for the
#: year, not one file picked out of it. See ``_validate_real_params_year_spec``.
_REAL_PARAMS_YEAR_SPEC_KEYS = frozenset({"year", "inflation"})


def _rounding_id_suffix(rounding: str) -> str:
    """The ``" (suffix)"`` text appended to an id for a non-default ``rounding``.

    Empty for the default. Shared by :attr:`GoldenCase.id` and by
    :func:`run_case`'s named-output failure label, so that a loosened case's
    suffix always lands at the very end of whatever is being reported —
    after a named output's ``[output_name]``, not wedged in front of it —
    rather than each call site rendering it in a different place.
    """
    suffix = _ROUNDING_ID_SUFFIXES.get(rounding)
    return "" if suffix is None else f" ({suffix})"


class GoldenCaseError(Exception):
    """A case file, or a case within it, is malformed.

    Raised during discovery — at module import time for the real cases, so
    that a broken file is a collection error rather than a test that quietly
    never ran.
    """


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """One case parsed out of a case file, ready to run.

    Attributes:
        file: The case file this came from, for error messages and for the
            ``<file stem>::<case name>`` test id.
        name: The case's human-readable name. Non-empty, unique within its
            file.
        target: Dotted path to the callable under test.
        source: Where the expected value was checked against. Recorded, not
            currently read by anything — see the module docstring.
        checked: The date it was checked. Recorded, not currently read by
            anything either.
        inputs: Raw keyword arguments as written in the YAML. ``params``,
            ``real_params``, and ``real_params_year`` entries are resolved
            lazily, at run time, by :func:`resolve_inputs` — not here, so
            that discovery does not itself need to read parameter files from
            disk. A ``real_params`` entry resolves to a
            :class:`~engine.core.indexation.RealParamSet` and a
            ``real_params_year`` entry to a
            :class:`~engine.core.indexation.RealParamYear`; both are passed
            to the target under the keyword ``params``, same as ``params``
            itself. A case names exactly one of the three; naming more than
            one fails at discovery instead, since they would claim the same
            argument.
        expected: Either ``{"value": <number>}`` or a mapping of output name
            to expected number. Never empty — see :func:`discover_cases`.
        rounding: The declared reason the source is imprecise — one of the
            keys of :data:`ROUNDING_TOLERANCES`. Recorded here, not just used
            to compute ``tolerance`` and discarded, so that it can also drive
            :attr:`id` and so a field that is validated once cannot later rot
            unread — see the module docstring on why ``source`` is recorded
            the same way.
        tolerance: Absolute tolerance for the comparison, derived from
            ``rounding`` — see :data:`ROUNDING_TOLERANCES`. A read-only
            property, not a constructor argument: storing it as an
            independent field would leave a way to construct a
            ``GoldenCase`` whose ``tolerance`` disagrees with its declared
            ``rounding``.
    """

    file: Path
    name: str
    target: str
    source: str
    checked: dt.date
    inputs: Mapping[str, Any] = field(default_factory=dict)
    expected: Mapping[str, Any] = field(default_factory=dict)
    rounding: str = _DEFAULT_ROUNDING

    @property
    def tolerance(self) -> float:
        """Absolute tolerance for the comparison, derived from ``rounding``.

        There is no constructor argument or setter for this: see the
        ``tolerance`` entry in the class docstring.
        """
        return ROUNDING_TOLERANCES[self.rounding]

    @property
    def id(self) -> str:
        """``<file stem>::<case name>``, plus a suffix if ``rounding`` is not the default.

        A case running at the default cent rounding keeps the plain
        ``<file stem>::<case name>`` id, so ids do not churn for the common
        case. A case running looser carries that in its id — see
        :data:`_ROUNDING_ID_SUFFIXES` — so that loosening a case changes its
        id in the diff, and every run's output shows which cases run loose.
        """
        return f"{self.file.stem}::{self.name}{_rounding_id_suffix(self.rounding)}"


def _validate_name(path: Path, index: int, name: Any) -> str:
    """A case's ``name`` must be a non-empty string; uniqueness is checked by the caller."""
    if not isinstance(name, str) or not name.strip():
        raise GoldenCaseError(
            f"{path}: case at index {index} has 'name' = {name!r}, which must be a "
            f"non-empty string."
        )
    return name


def _reject_reserved_suffix(path: Path, name: str) -> None:
    """Reject a case name ending in a rendered rounding suffix, e.g. ``" (dollar)"``.

    That parenthesized text is what the harness itself appends to an id to
    show a non-default ``rounding`` is in effect — see
    :data:`_ROUNDING_ID_SUFFIXES`. A case whose *name* independently ends the
    same way would either collide with another case's suffixed id (see the
    ``seen_ids`` check in :func:`discover_cases`) or, more insidiously,
    advertise loose rounding in every run's output while actually running at
    whatever ``rounding`` it declares — cent, most likely — which is exactly
    the kind of untrustworthy suffix this issue exists to prevent. This check
    runs regardless of the case's own declared ``rounding``, because the
    problem is the name looking like a suffix, not what it is paired with.
    """
    for suffix in _ROUNDING_ID_SUFFIXES.values():
        reserved = f" ({suffix})"
        if name.endswith(reserved):
            raise GoldenCaseError(
                f"{path}: case {name!r} ends with {reserved!r}, which is reserved "
                f"for the harness's own rounding-suffix rendering — see "
                f"_ROUNDING_ID_SUFFIXES. Rename the case; ending a name that way "
                f"would misleadingly advertise a non-default rounding regardless "
                f"of what 'rounding' the case actually declares."
            )


def _validate_source(path: Path, name: str, value: Any) -> str:
    """A case's ``source`` must be a non-empty string, not just present.

    A bare ``source:`` key parses to ``None`` in YAML, and an empty or
    whitespace-only string is no more a record of provenance than that is —
    both would otherwise satisfy a plain ``"source" not in entry`` check and
    leave the case with no real source at all.
    """
    if not isinstance(value, str) or not value.strip():
        raise GoldenCaseError(
            f"{path}: case {name!r} has 'source' = {value!r}, which must be a "
            f"non-empty string."
        )
    return value


def _parse_checked(path: Path, name: str, value: Any) -> dt.date:
    """Validate that a case's ``checked`` value is a bare ISO date, not in the future.

    PyYAML parses an unquoted ``2026-09-05`` into a :class:`datetime.date`
    already, so both a real ``date`` and a string in ISO form are accepted.
    An unquoted date with an out-of-range component (``2026-13-45``) never
    reaches here — PyYAML's own timestamp resolver raises constructing it,
    which :func:`discover_cases` catches at the file level, naming the file.

    A :class:`datetime.datetime` — what an unquoted ``2026-09-05 10:00:00``
    parses to — is rejected even though it subclasses ``date``: a case was
    "checked", not "checked at a particular time of day", and accepting a
    timestamp here silently would let one through with no way to tell it was
    not a plain date until something read the extra fields.

    A future date is rejected for the same reason
    ``tests/params/test_param_provenance.py`` rejects one on a parameter
    file: it reads as more authoritative than the truth, and it is the one
    error in this format that a careless human is likely to write by
    transposing a year.
    """
    if isinstance(value, dt.datetime):
        raise GoldenCaseError(
            f"{path}: case {name!r} has 'checked' = {value!r}, which includes a "
            f"time of day. Only a bare date (YYYY-MM-DD) is accepted."
        )
    checked: dt.date
    if isinstance(value, dt.date):
        checked = value
    elif isinstance(value, str):
        try:
            checked = dt.date.fromisoformat(value)
        except ValueError:
            raise GoldenCaseError(
                f"{path}: case {name!r} has 'checked' = {value!r}, which does not "
                f"parse as an ISO date (YYYY-MM-DD)."
            ) from None
    else:
        raise GoldenCaseError(
            f"{path}: case {name!r} has 'checked' = {value!r}, which does not parse "
            f"as an ISO date (YYYY-MM-DD)."
        )
    today = dt.date.today()
    if checked > today:
        raise GoldenCaseError(
            f"{path}: case {name!r} has 'checked' = {checked}, which is in the "
            f"future (today is {today}). A source cannot have been checked on a "
            f"day that has not happened yet."
        )
    return checked


def _validate_expected(path: Path, name: str, expected: Any) -> None:
    """Validate a case's ``expected``: a non-empty mapping of finite real numbers.

    An empty, absent, non-numeric, or non-finite ``expected`` is a case that
    either asserts nothing or can never truthfully pass, and both report
    green (or silently vacuous) if let through.
    """
    if not isinstance(expected, Mapping):
        raise GoldenCaseError(
            f"{path}: case {name!r} has 'expected' = {expected!r}, which must be a "
            f"mapping — either {{'value': <number>}} or {{<output name>: <number>, "
            f"...}}."
        )
    if not expected:
        raise GoldenCaseError(
            f"{path}: case {name!r} has an empty 'expected', which asserts nothing. "
            f"A golden case must name at least one expected number."
        )
    if "value" in expected and len(expected) > 1:
        others = sorted(key for key in expected if key != "value")
        raise GoldenCaseError(
            f"{path}: case {name!r} has 'expected' with both 'value' and named "
            f"output(s) {others}. 'value' means the whole result and cannot be "
            f"combined with named outputs — pick one form."
        )
    for output_name, value in expected.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GoldenCaseError(
                f"{path}: case {name!r} has expected[{output_name!r}] = {value!r}, "
                f"which is not a number."
            )
        if not math.isfinite(value):
            kind = "NaN" if math.isnan(value) else "infinite"
            raise GoldenCaseError(
                f"{path}: case {name!r} has expected[{output_name!r}] = {value!r}, "
                f"which is {kind}. A golden case must expect a finite number."
            )


def _parse_rounding(path: Path, name: str, raw_rounding: Any) -> str:
    """Validate a case's ``rounding``: a string naming one of :data:`ROUNDING_TOLERANCES`.

    Called only when the case sets ``rounding`` at all — see
    :func:`discover_cases`, which supplies :data:`_DEFAULT_ROUNDING` itself
    when the key is absent. That split is what lets a bare ``rounding:`` key
    (which parses to ``None``) be rejected here with the rest of the
    non-string values, rather than being treated as "omitted" and silently
    defaulted — the spec requires both to fail loudly, and only one of them
    to fail the same way as an absent key would.

    Booleans are numbers to Python (``isinstance(True, int)``) but are
    rejected here rather than accidentally matching a truthy check; they can
    only ever fail the "not a string" branch anyway, same as an int or float.
    """
    if not isinstance(raw_rounding, str):
        raise GoldenCaseError(
            f"{path}: case {name!r} has 'rounding' = {raw_rounding!r}, which must be "
            f"a string. Allowed values are {sorted(ROUNDING_TOLERANCES)}."
        )
    if raw_rounding not in ROUNDING_TOLERANCES:
        raise GoldenCaseError(
            f"{path}: case {name!r} has 'rounding' = {raw_rounding!r}, which is not "
            f"a recognized rounding. Allowed values are {sorted(ROUNDING_TOLERANCES)}."
        )
    return raw_rounding


def _validate_year(path: Path, name: str, key: str, spec: Mapping[str, Any]) -> None:
    """Validate a spec's ``year``: an int, not a bool. Shared by every spec kind that has one."""
    year = spec["year"]
    if isinstance(year, bool) or not isinstance(year, int):
        raise GoldenCaseError(
            f"{path}: case {name!r} has {key}.year = {year!r}, which must be an int."
        )


def _validate_inflation(path: Path, name: str, key: str, spec: Mapping[str, Any]) -> None:
    """Validate a spec's ``inflation``: an int or float, not a bool, and finite.

    Shared by ``real_params`` and ``real_params_year``. There is no default:
    a case states ``inflation`` explicitly, ``0.0`` for a nominal source.
    ``.nan``/``.inf``/``-.inf`` are rejected, and an int literal too large to
    convert to a float (``math.isfinite`` raising ``OverflowError`` rather
    than answering ``False``) is rejected too.
    """
    inflation = spec["inflation"]
    if isinstance(inflation, bool) or not isinstance(inflation, (int, float)):
        raise GoldenCaseError(
            f"{path}: case {name!r} has {key}.inflation = {inflation!r}, which must "
            f"be an int or a float. There is no default; state 0.0 explicitly for a "
            f"source that is a nominal calculator."
        )
    try:
        finite = math.isfinite(inflation)
    except OverflowError:
        # An int literal too large to convert to float (e.g. a 400-digit
        # number) makes math.isfinite raise rather than answer False.
        raise GoldenCaseError(
            f"{path}: case {name!r} has {key}.inflation = {inflation!r}, which is too "
            f"large to represent as a float. A golden case must state a finite "
            f"inflation rate."
        ) from None
    if not finite:
        kind = "NaN" if math.isnan(inflation) else "infinite"
        raise GoldenCaseError(
            f"{path}: case {name!r} has {key}.inflation = {inflation!r}, which is "
            f"{kind}. A golden case must state a finite inflation rate."
        )


def _validate_year_and_file(path: Path, name: str, key: str, spec: Mapping[str, Any]) -> None:
    """Validate the ``year``/``file`` pair common to both ``params`` and ``real_params``."""
    _validate_year(path, name, key, spec)
    file = spec["file"]
    if not isinstance(file, str):
        raise GoldenCaseError(
            f"{path}: case {name!r} has {key}.file = {file!r}, which must be a string."
        )


def _validate_params_spec(path: Path, name: str, key: str, spec: Any) -> None:
    """Validate a ``params`` input spec's shape, without resolving it.

    The actual :func:`load_year` call stays lazy, at run time, in
    :func:`resolve_params` — this only checks that the spec *could* resolve,
    so a malformed one (a string instead of a mapping, a missing ``file``, an
    extra key from a typo like ``provice``) is caught at discovery with the
    file and case named, instead of surfacing as a bare ``KeyError`` or
    ``TypeError`` with neither.
    """
    if not isinstance(spec, Mapping) or set(spec) != _PARAMS_SPEC_KEYS:
        raise GoldenCaseError(
            f"{path}: case {name!r} has {key!r} = {spec!r}, which must be a mapping "
            f"with exactly the keys {sorted(_PARAMS_SPEC_KEYS)}."
        )
    _validate_year_and_file(path, name, key, spec)


def _validate_real_params_spec(path: Path, name: str, key: str, spec: Any) -> None:
    """Validate a ``real_params`` input spec's shape, without resolving it.

    One key wider than :func:`_validate_params_spec`: the deflated real-terms
    view needs a stated inflation rate, and there is no default — a case
    reading a nominal calculator states ``inflation: 0.0`` explicitly rather
    than the harness assuming it. ``inflation`` must be an ``int`` or
    ``float``, not a ``bool`` (rejected the same way ``year`` is), and finite
    (``.nan``/``.inf``/``-.inf`` are rejected). The range check on the rate
    itself — at or below -1 — is left to ``engine.core.indexation``, which
    already raises at run time.
    """
    if not isinstance(spec, Mapping) or set(spec) != _REAL_PARAMS_SPEC_KEYS:
        raise GoldenCaseError(
            f"{path}: case {name!r} has {key!r} = {spec!r}, which must be a mapping "
            f"with exactly the keys {sorted(_REAL_PARAMS_SPEC_KEYS)}."
        )
    _validate_year_and_file(path, name, key, spec)
    _validate_inflation(path, name, key, spec)


def _validate_real_params_year_spec(path: Path, name: str, key: str, spec: Any) -> None:
    """Validate a ``real_params_year`` input spec's shape, without resolving it.

    ``real_params`` without ``file``: this resolves an entire
    :class:`~engine.core.indexation.RealParamYear`, every file for the year,
    not one file picked out of it. ``year`` and ``inflation`` are each
    validated exactly as they are on ``real_params`` — see
    :func:`_validate_year` and :func:`_validate_inflation`.
    """
    if not isinstance(spec, Mapping) or set(spec) != _REAL_PARAMS_YEAR_SPEC_KEYS:
        raise GoldenCaseError(
            f"{path}: case {name!r} has {key!r} = {spec!r}, which must be a mapping "
            f"with exactly the keys {sorted(_REAL_PARAMS_YEAR_SPEC_KEYS)}."
        )
    _validate_year(path, name, key, spec)
    _validate_inflation(path, name, key, spec)


def _validate_inputs(path: Path, name: str, raw_inputs: Any) -> Mapping[str, Any]:
    """Validate a case's ``inputs``: a mapping with string keys.

    A list under ``inputs`` (``[x: 7.0]`` instead of ``{x: 7.0}``, an easy
    slip) dies inside ``target(**kwargs)`` with a bare ``TypeError`` naming
    neither file nor case; this catches it here instead. Any ``params`` entry
    is additionally checked by :func:`_validate_params_spec`, any
    ``real_params`` entry by :func:`_validate_real_params_spec`, and any
    ``real_params_year`` entry by :func:`_validate_real_params_year_spec`.
    Naming two or three of them is rejected here too: all resolve to the same
    target keyword, ``params`` — see :func:`resolve_inputs` — so a case
    naming more than one would be claiming that argument twice.
    """
    if raw_inputs is None:
        raw_inputs = {}
    if not isinstance(raw_inputs, Mapping) or not all(isinstance(k, str) for k in raw_inputs):
        raise GoldenCaseError(
            f"{path}: case {name!r} has 'inputs' = {raw_inputs!r}, which must be a "
            f"mapping with string keys."
        )
    present = [key for key in ("params", "real_params", "real_params_year") if key in raw_inputs]
    if len(present) == 3:
        raise GoldenCaseError(
            f"{path}: case {name!r} has 'params', 'real_params' and 'real_params_year' "
            f"in 'inputs'. All three resolve to the target's 'params' keyword argument, "
            f"so a case names exactly one of them."
        )
    if len(present) == 2:
        a, b = present
        raise GoldenCaseError(
            f"{path}: case {name!r} has both {a!r} and {b!r} in 'inputs'. 'params', "
            f"'real_params' and 'real_params_year' all resolve to the target's "
            f"'params' keyword argument, so a case names exactly one of them, never "
            f"both."
        )
    if "params" in raw_inputs:
        _validate_params_spec(path, name, "params", raw_inputs["params"])
    if "real_params" in raw_inputs:
        _validate_real_params_spec(path, name, "real_params", raw_inputs["real_params"])
    if "real_params_year" in raw_inputs:
        _validate_real_params_year_spec(
            path, name, "real_params_year", raw_inputs["real_params_year"]
        )
    return raw_inputs


def _load_case_file(path: Path) -> Any:
    """Parse one case file's YAML, with the filename attached to any failure.

    Reading the whole file as text before handing it to ``yaml.safe_load``
    means a syntax error is otherwise reported against the string, not the
    file it came from. An out-of-range date written unquoted (``2026-13-45``)
    fails the same way: PyYAML's implicit timestamp resolver constructs a
    ``datetime.date`` while parsing and raises a bare ``ValueError`` with no
    filename attached, so that is caught here too.
    """
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, ValueError) as exc:
        raise GoldenCaseError(f"{path}: not valid YAML: {exc}") from exc


def _find_case_files(cases_dir: Path) -> list[Path]:
    """Every case file under ``cases_dir``, recursively, or a loud error.

    Recursive because a subdirectory is the first thing a human reaches for
    once there are more than a handful of cases, and a case tucked inside one
    must not silently stop being discovered. Anything found under
    ``cases_dir`` that is not a recognized case file and not ``.gitkeep`` — a
    stray editor backup, a case file under the wrong extension or the wrong
    case, a document dropped in the wrong directory — is rejected rather than
    quietly ignored, because a case file the harness cannot see is exactly as
    dangerous as a case within one that asserts nothing.
    """
    paths: list[Path] = []
    for candidate in sorted(cases_dir.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.name == _KEEPER_FILENAME:
            continue
        if candidate.suffix in _CASE_FILE_SUFFIXES:
            paths.append(candidate)
            continue
        raise GoldenCaseError(
            f"{candidate}: found under {cases_dir}, but is neither a case file "
            f"({sorted(_CASE_FILE_SUFFIXES)}) nor {_KEEPER_FILENAME!r}. Remove it, "
            f"rename it, or move it out of the cases directory."
        )
    return paths


def discover_cases(cases_dir: Path) -> list[GoldenCase]:
    """Parse every case file in ``cases_dir`` into a flat list of cases.

    A directory holding no case files at all discovers an empty list, with no
    error and no warning: that is the legitimate state of a freshly bootstrapped
    ``tests/golden/cases/`` before the human has written a case, or of a
    ``tmp_path`` a harness self-test has not populated yet.

    A case file that *is* present but yields no cases — an empty or missing
    ``cases:`` list, most likely from a copy-paste that dropped the list — is
    different: it is evidence of a mistake, not an empty state, so it raises
    :class:`GoldenCaseError` naming the file rather than silently
    contributing zero tests. See :func:`_find_case_files` for the other way a
    case can go missing: a subdirectory, a wrong extension, or a stray file.

    Raises:
        GoldenCaseError: If ``cases_dir`` does not exist, if it holds a file
            that is neither a case file nor ``.gitkeep``, if any case file is
            not valid YAML or not a mapping with a ``target``, if a file
            defines no cases, or if any case is missing a required field,
            sets an unrecognized field (including the removed ``tolerance``),
            or fails validation of ``name``, ``source``, ``checked``,
            ``expected``, ``rounding``, or ``inputs``.
    """
    if not cases_dir.is_dir():
        raise GoldenCaseError(f"{cases_dir}: cases directory does not exist.")

    cases: list[GoldenCase] = []
    for path in _find_case_files(cases_dir):
        raw = _load_case_file(path)
        if not isinstance(raw, Mapping) or "target" not in raw:
            raise GoldenCaseError(
                f"{path}: expected a mapping with a 'target' key at the top level."
            )
        target = raw["target"]
        raw_cases = raw.get("cases")
        if not raw_cases:
            raise GoldenCaseError(
                f"{path}: defines no cases. A case file with an empty or missing "
                f"'cases:' list is a broken file, not a legitimately empty one — "
                f"fill it in by hand or delete the file."
            )
        seen_names: dict[str, int] = {}
        seen_ids: dict[str, int] = {}
        for index, entry in enumerate(raw_cases):
            if not isinstance(entry, Mapping) or "name" not in entry:
                raise GoldenCaseError(f"{path}: case at index {index} has no 'name'.")
            name = _validate_name(path, index, entry["name"])
            _reject_reserved_suffix(path, name)
            if name in seen_names:
                raise GoldenCaseError(
                    f"{path}: name {name!r} is used by more than one case (indices "
                    f"{seen_names[name]} and {index}). Names must be unique within "
                    f"a file — they are how a failure is grepped back to its case."
                )
            seen_names[name] = index

            unknown = set(entry) - _ALLOWED_CASE_FIELDS
            if unknown:
                if "tolerance" in unknown:
                    raise GoldenCaseError(
                        f"{path}: case {name!r} sets 'tolerance', which was removed. "
                        f"Declare 'rounding' instead — one of "
                        f"{sorted(ROUNDING_TOLERANCES)} — and the harness derives the "
                        f"tolerance from it."
                    )
                raise GoldenCaseError(
                    f"{path}: case {name!r} has unrecognized field(s) {sorted(unknown)}. "
                    f"Allowed fields are {sorted(_ALLOWED_CASE_FIELDS)}."
                )
            if "source" not in entry:
                raise GoldenCaseError(
                    f"{path}: case {name!r} is missing required field 'source'."
                )
            source = _validate_source(path, name, entry["source"])
            if "checked" not in entry:
                raise GoldenCaseError(
                    f"{path}: case {name!r} is missing required field 'checked'."
                )
            checked = _parse_checked(path, name, entry["checked"])
            if "expected" not in entry:
                raise GoldenCaseError(
                    f"{path}: case {name!r} is missing required field 'expected'."
                )
            _validate_expected(path, name, entry["expected"])
            if "rounding" in entry:
                rounding = _parse_rounding(path, name, entry["rounding"])
            else:
                rounding = _DEFAULT_ROUNDING
            inputs = _validate_inputs(path, name, entry.get("inputs"))

            case = GoldenCase(
                file=path,
                name=name,
                target=target,
                source=source,
                checked=checked,
                inputs=inputs,
                expected=entry["expected"],
                rounding=rounding,
            )
            # Unreachable while `_reject_reserved_suffix` stands: the only way
            # two differing names can render one id is for one of them to end
            # in a rendered suffix, and that name is rejected before its id is
            # ever computed. Kept as the check that survives a change to the
            # suffix scheme, since an id collision silently makes two cases'
            # failure messages identical. No test can reach it today, and none
            # pretends to.
            if case.id in seen_ids:
                raise GoldenCaseError(
                    f"{path}: id {case.id!r} is used by more than one case (indices "
                    f"{seen_ids[case.id]} and {index}), even though their names "
                    f"differ. Two cases must not render the same test id — it is "
                    f"what every comparison and lookup failure message carries."
                )
            seen_ids[case.id] = index
            cases.append(case)
    return cases


def resolve_target(dotted: str) -> Callable[..., Any]:
    """Import a dotted path to a callable, e.g. ``engine.tax.brackets.tax_on_income``.

    Raises:
        GoldenCaseError: If ``dotted`` has no module component, if the module
            cannot be imported, if the module has no attribute of that name,
            or if the attribute exists but is not callable.
    """
    module_name, sep, attr = dotted.rpartition(".")
    if not sep:
        raise GoldenCaseError(f"{dotted!r} is not a dotted path to a callable.")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise GoldenCaseError(f"{dotted!r}: cannot import module {module_name!r}: {exc}") from exc
    try:
        target = getattr(module, attr)
    except AttributeError as exc:
        raise GoldenCaseError(
            f"{dotted!r}: module {module_name!r} has no attribute {attr!r}."
        ) from exc
    if not callable(target):
        raise GoldenCaseError(
            f"{dotted!r} resolves to a {type(target).__name__}, which is not callable."
        )
    return target


@cache
def _load_year_cached(year: int) -> ParamYear:
    """``load_year``, cached: a scenario's parameter files are read once.

    Safe because :func:`load_year`'s return is deeply immutable — see
    ``engine/params/loader.py`` — so nothing downstream can observe the
    difference between a fresh read and a cached one. Always reads from
    :data:`DEFAULT_PARAMS_ROOT`, the only root a real or synthetic case ever
    names.
    """
    return load_year(year, DEFAULT_PARAMS_ROOT)


def resolve_params(spec: Mapping[str, Any]) -> ParamSet:
    """Resolve a case's ``params: {year, file}`` input to a real :class:`ParamSet`."""
    return _load_year_cached(spec["year"])[spec["file"]]


def resolve_real_params(spec: Mapping[str, Any]) -> RealParamSet:
    """Resolve a case's ``real_params: {year, file, inflation}`` input to a real
    :class:`~engine.core.indexation.RealParamSet`.

    ``inflation`` is the case's own stated rate — a bare fraction, e.g. ``0.0``
    for a nominal source — and reaches
    :func:`~engine.core.indexation.real_year` unchanged; there is no default
    and this never assumes one. Reuses :func:`_load_year_cached`, the same
    cache :func:`resolve_params` uses.

    Raises:
        ParamYearMissingError: If ``spec["year"]`` has no parameter files.
        ParamFileMissingError: If ``spec["file"]`` does not exist for that year.
        ValueError: If ``spec["inflation"]`` is at or below -1 — see
            :func:`~engine.core.indexation.real_year`.
    """
    return real_year(_load_year_cached(spec["year"]), float(spec["inflation"]))[spec["file"]]


def resolve_real_params_year(spec: Mapping[str, Any]) -> RealParamYear:
    """Resolve a case's ``real_params_year: {year, inflation}`` input to a
    :class:`~engine.core.indexation.RealParamYear`.

    One file shallower than :func:`resolve_real_params`: the whole tax year,
    in real dollars, rather than one file picked out of it — for a target
    that reaches more than one parameter set itself, e.g.
    ``engine.tax.combined.person_assessment``. Reuses the same
    :func:`_load_year_cached` cache.

    Raises:
        ParamYearMissingError: If ``spec["year"]`` has no parameter files.
        ValueError: If ``spec["inflation"]`` is at or below -1 — see
            :func:`~engine.core.indexation.real_year`.
    """
    return real_year(_load_year_cached(spec["year"]), float(spec["inflation"]))


def resolve_inputs(raw_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Turn a case's raw ``inputs`` mapping into real keyword arguments.

    A ``real_params`` entry resolves through :func:`resolve_real_params` and
    a ``real_params_year`` entry through :func:`resolve_real_params_year`;
    both are stored under the key ``"params"``, not their own name — see the
    module docstring on why a case names exactly one of the three.
    """
    resolved: dict[str, Any] = {}
    for key, value in raw_inputs.items():
        if key == "params":
            resolved["params"] = resolve_params(value)
        elif key == "real_params":
            resolved["params"] = resolve_real_params(value)
        elif key == "real_params_year":
            resolved["params"] = resolve_real_params_year(value)
        else:
            resolved[key] = value
    return resolved


def to_float(value: Any) -> float:
    """Normalize a target's result to a Python float for comparison.

    The engine vectorizes across paths, so a scalar-input call can come back
    as a numpy scalar or a 0-d or single-element array. Anything with more
    than one element cannot stand for a single expected number and fails
    loudly rather than silently comparing against element 0. A boolean is
    rejected in every form it can arrive in — a Python ``bool``, a numpy
    ``bool_`` scalar (which is a :class:`numpy.generic` but not a Python
    ``bool``), or a boolean-dtype array — because ``True``/``False`` silently
    becoming ``1.0``/``0.0`` would let an eligibility mask compare as though
    it were a dollar figure.

    Raises:
        TypeError: If ``value`` is boolean in any of those forms, or is not a
            recognized numeric type.
        ValueError: If ``value`` is an array with more than one element.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"cannot compare a bool result ({value!r}) to a golden number.")
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            raise TypeError(
                f"cannot compare a boolean array result ({value!r}) to a golden number."
            )
        if value.size != 1:
            raise ValueError(
                f"expected a single-element result, got an array of shape {value.shape}."
            )
        return float(value.reshape(-1)[0])
    if isinstance(value, (int, float, np.generic)):
        return float(value)
    raise TypeError(f"cannot interpret {value!r} ({type(value).__name__}) as a number.")


def _declared_field_names(result: Any) -> frozenset[str] | None:
    """The field names a named-output lookup may address on ``result``, or ``None``.

    Restricted to a dataclass instance's declared fields or a NamedTuple's
    ``_fields`` — never a bare ``getattr``, which would resolve ``imag``,
    ``real``, ``ndim``, or ``size`` on an ordinary float or numpy scalar and
    let a named-output case pass against a bare number forever, vacuously.
    """
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return frozenset(f.name for f in dataclasses.fields(result))
    if isinstance(result, tuple) and hasattr(result, "_fields"):
        return frozenset(result._fields)  # type: ignore[attr-defined]
    return None


def _lookup_output(case: GoldenCase, result: Any, output_name: str) -> Any:
    if isinstance(result, Mapping):
        try:
            return result[output_name]
        except KeyError:
            raise GoldenCaseError(
                f"{case.id}: result has no key {output_name!r}."
            ) from None

    field_names = _declared_field_names(result)
    if field_names is None:
        raise GoldenCaseError(
            f"{case.id}: a named output requires a mapping, dataclass, or "
            f"NamedTuple result; got {type(result).__name__}. A bare scalar result "
            f"must be addressed with 'value:' instead of a named output."
        )
    if output_name not in field_names:
        raise GoldenCaseError(f"{case.id}: result has no field {output_name!r}.")
    return getattr(result, output_name)


def _compare_one(case: GoldenCase, label: str, expected_value: Any, actual_raw: Any) -> None:
    actual = to_float(actual_raw)
    expected_float = float(expected_value)
    assert actual == pytest.approx(expected_float, abs=case.tolerance), (
        f"{label}: expected {expected_float}, got {actual} (tolerance ±{case.tolerance})."
    )


def run_case(case: GoldenCase) -> None:
    """Call a case's target and assert its result matches ``expected``.

    ``expected`` is checked for emptiness here too, not just in
    :func:`discover_cases`: that function is what every real and synthetic
    case file goes through, but a :class:`GoldenCase` built directly (as a
    self-test might) bypasses it, and the whole point of rejecting an empty
    ``expected`` is that there is no way to construct one that asserts
    nothing.

    Raises:
        GoldenCaseError: If ``expected`` is empty, if ``case.target`` cannot
            be resolved (see :func:`resolve_target`), or if an output named
            in ``expected`` cannot be found on the result (see
            :func:`_lookup_output`).
        AssertionError: If a value does not match, with a message naming both
            the expected and the actual number.
        TypeError: If a result cannot be interpreted as a number (see
            :func:`to_float`).
        ValueError: Two distinct causes: a result is a multi-element array
            (see :func:`to_float`), or a ``real_params``/``real_params_year``
            input names an inflation rate at or below -1 (raised by
            ``engine.core.indexation``, via :func:`resolve_real_params` or
            :func:`resolve_real_params_year`).
        ParamError: From resolving a ``params``, ``real_params``, or
            ``real_params_year`` input (see :func:`resolve_params`,
            :func:`resolve_real_params`, :func:`resolve_real_params_year`):
            the year or file it names does not exist, any file in that
            year's directory is malformed, or — for ``real_params``/
            ``real_params_year`` — a file's ``indexation`` block is
            malformed. The loader's own errors all subclass it.
        Exception: Whatever ``case.target`` itself raises, uncaught — a golden
            case is not responsible for turning the target's own failures
            into anything friendlier.
    """
    if not case.expected:
        raise GoldenCaseError(f"{case.id}: 'expected' is empty; nothing to assert.")

    target = resolve_target(case.target)
    kwargs = resolve_inputs(case.inputs)
    result = target(**kwargs)

    expected = case.expected
    if list(expected.keys()) == ["value"]:
        _compare_one(case, case.id, expected["value"], result)
        return

    for output_name, expected_value in expected.items():
        actual_raw = _lookup_output(case, result, output_name)
        # Deliberately not `f"{case.id}[{output_name}]"`: case.id already
        # carries the rounding suffix, which would then land *before*
        # `[output_name]` instead of at the end, reading confusingly
        # (`name (dollar)[output]`). Rebuilding the label from the base name
        # keeps `[output_name]` adjacent to the name it qualifies, with the
        # suffix trailing both.
        label = (
            f"{case.file.stem}::{case.name}[{output_name}]"
            f"{_rounding_id_suffix(case.rounding)}"
        )
        _compare_one(case, label, expected_value, actual_raw)

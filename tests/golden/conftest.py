"""Shared machinery for the golden test harness.

Everything a golden case file needs turned into a runnable assertion lives
here, and lives here once, because both ``test_cases.py`` (which discovers the
real cases under ``tests/golden/cases/``) and ``test_harness.py`` (which
exercises the discovery and comparison logic against synthetic cases in
``tmp_path``) need exactly the same behaviour. Neither module re-implements
any of it.

Nothing in this file ever supplies a tax or benefit number. The one numeric
constant here, :data:`DEFAULT_TOLERANCE`, is a comparison tolerance for the
test harness itself, fixed by the specification this module implements
(issue #3) — not a parameter of the tax system.

A case file is a human-authored YAML document of the form::

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
        tolerance: 0.01   # optional, defaults to DEFAULT_TOLERANCE

(``7.0``/``14.0`` here are placeholders standing in for whatever the target
actually computes; a real case's numbers come from the source it cites, never
from this docstring.)

``source`` and ``checked`` are mandatory on every case: a golden value with no
record of where it came from or when it was checked is exactly the kind of
plausible-looking number this repository forbids inventing, and a harness
that let it through silently would be as bad as inventing one. Both are also
stored on the parsed :class:`GoldenCase`, even though nothing downstream reads
them yet — a field that is validated once and then discarded is exactly how a
later refactor forgets to validate it, which is what happened to ``source``
the first time round.

Discovery is strict well beyond ``source`` and ``checked``, and deliberately
so: every hole here is a way for a case to look green without actually
asserting anything, which is the one thing a golden test exists to do.
``name`` must be a non-empty string, unique within its file, because a blank
or duplicated name breaks the ``<file>::<name>`` id a failure is grepped back
by. ``source`` must be a non-empty string after stripping whitespace — a bare
or blank key is the same as no source at all. ``checked`` must be a bare date,
not a timestamp, and it must not be in the future, for the same reason
``tests/params/test_param_provenance.py`` rejects a future date on a
parameter file: a check that has not happened yet is worse than no claim of
one. ``expected`` must be a non-empty mapping of finite real numbers (not a
bare scalar, not ``null``, not a boolean, not a string masquerading as one,
not ``inf`` or ``nan``); ``value`` cannot appear next to a named output,
because its meaning silently changes when a sibling key is added.
``tolerance`` must be a finite number greater than zero, because an unbounded
or non-positive tolerance passes anything or nothing. ``inputs`` must be a
mapping with string keys, and a ``params``/``real_params`` value within it
must be a mapping with exactly the keys ``year`` (an int) and ``file`` (a
str) — no more, no fewer. An unrecognized field on a case is rejected rather
than silently ignored, because a typo'd field name (``tolerance`` misspelled
as ``tolerence``) is indistinguishable from a deliberately omitted one
otherwise. And every file actually found under ``cases/`` — recursively, so a
subdirectory is not a place to hide from discovery — must be a case file
(``.yaml``/``.yml``, matched case-sensitively) or ``.gitkeep``; anything else
is a stray file rather than a silently-ignored one. Every one of these is
raised as :class:`GoldenCaseError` naming the file and the case, at discovery
time, so a malformed case file cannot even reach collection — never mind a
run.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import importlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest
import yaml

from engine.params.loader import DEFAULT_PARAMS_ROOT, ParamSet, ParamYear, load_year

__all__ = [
    "CASES_DIR",
    "DEFAULT_TOLERANCE",
    "GoldenCase",
    "GoldenCaseError",
    "discover_cases",
    "resolve_inputs",
    "resolve_params",
    "resolve_real_params",
    "resolve_target",
    "run_case",
    "to_float",
]

#: Where the real, human-authored case files live.
CASES_DIR: Path = Path(__file__).parent / "cases"

#: Comparison tolerance used when a case does not set its own ``tolerance``.
#: A harness setting, not a tax parameter — see the module docstring.
DEFAULT_TOLERANCE: float = 0.01

#: Fields a case entry may set. Anything else is a typo, not an extension —
#: see the module docstring on why an unknown field is rejected rather than
#: dropped.
_ALLOWED_CASE_FIELDS = frozenset({"name", "source", "checked", "inputs", "expected", "tolerance"})

#: Case-file extensions that are discovered, matched case-sensitively. A file
#: under ``cases/`` with any other suffix — including the same extension in
#: the wrong case, e.g. ``.YAML`` — is a stray file, not a silently-skipped
#: one; see ``discover_cases``.
_CASE_FILE_SUFFIXES = frozenset({".yaml", ".yml"})

#: The one non-case file `cases/` is allowed to hold.
_KEEPER_FILENAME = ".gitkeep"

#: Keys a ``params``/``real_params`` input spec must have — exactly these,
#: no more, no fewer, so a typo (``provice`` for ``province``) is caught here
#: rather than silently ignored.
_PARAMS_SPEC_KEYS = frozenset({"year", "file"})


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
        inputs: Raw keyword arguments as written in the YAML. ``params`` and
            ``real_params`` entries are resolved lazily, at run time, by
            :func:`resolve_inputs` — not here, so that a case naming a
            ``real_params`` input still discovers cleanly and only fails when
            actually run.
        expected: Either ``{"value": <number>}`` or a mapping of output name
            to expected number. Never empty — see :func:`discover_cases`.
        tolerance: Absolute tolerance for the comparison. Finite and > 0.
    """

    file: Path
    name: str
    target: str
    source: str
    checked: dt.date
    inputs: Mapping[str, Any] = field(default_factory=dict)
    expected: Mapping[str, Any] = field(default_factory=dict)
    tolerance: float = DEFAULT_TOLERANCE

    @property
    def id(self) -> str:
        """``<file stem>::<case name>``, the parametrized test id."""
        return f"{self.file.stem}::{self.name}"


def _validate_name(path: Path, index: int, name: Any) -> str:
    """A case's ``name`` must be a non-empty string; uniqueness is checked by the caller."""
    if not isinstance(name, str) or not name.strip():
        raise GoldenCaseError(
            f"{path}: case at index {index} has 'name' = {name!r}, which must be a "
            f"non-empty string."
        )
    return name


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

    See the module docstring for why each of these is checked: an empty,
    absent, non-numeric, or non-finite ``expected`` is a case that either
    asserts nothing or can never truthfully pass, and both report green (or
    silently vacuous) if let through.
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


def _parse_tolerance(path: Path, name: str, raw_tolerance: Any) -> float:
    """Validate and convert a case's ``tolerance``: finite and strictly positive.

    An unbounded tolerance (``1e12``) passes any comparison; a non-positive
    one either passes nothing or is nonsensical. Both are as much a way to
    fake a passing case as emptying ``expected`` is, so both are rejected here
    rather than left to fail confusingly inside ``pytest.approx`` later.
    """
    if isinstance(raw_tolerance, bool):
        raise GoldenCaseError(
            f"{path}: case {name!r} has 'tolerance' = {raw_tolerance!r}, which is "
            f"not a number."
        )
    try:
        tolerance = float(raw_tolerance)
    except (TypeError, ValueError) as exc:
        raise GoldenCaseError(
            f"{path}: case {name!r} has 'tolerance' = {raw_tolerance!r}, which is "
            f"not a number."
        ) from exc
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise GoldenCaseError(
            f"{path}: case {name!r} has 'tolerance' = {raw_tolerance!r}, which must "
            f"be a finite number greater than zero."
        )
    return tolerance


def _validate_params_spec(path: Path, name: str, key: str, spec: Any) -> None:
    """Validate a ``params``/``real_params`` input spec's shape, without resolving it.

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
    year = spec["year"]
    if isinstance(year, bool) or not isinstance(year, int):
        raise GoldenCaseError(
            f"{path}: case {name!r} has {key}.year = {year!r}, which must be an int."
        )
    file = spec["file"]
    if not isinstance(file, str):
        raise GoldenCaseError(
            f"{path}: case {name!r} has {key}.file = {file!r}, which must be a string."
        )


def _validate_inputs(path: Path, name: str, raw_inputs: Any) -> Mapping[str, Any]:
    """Validate a case's ``inputs``: a mapping with string keys.

    A list under ``inputs`` (``[x: 7.0]`` instead of ``{x: 7.0}``, an easy
    slip) dies inside ``target(**kwargs)`` with a bare ``TypeError`` naming
    neither file nor case; this catches it here instead. Any ``params`` or
    ``real_params`` entry is additionally checked by
    :func:`_validate_params_spec`.
    """
    if raw_inputs is None:
        raw_inputs = {}
    if not isinstance(raw_inputs, Mapping) or not all(isinstance(k, str) for k in raw_inputs):
        raise GoldenCaseError(
            f"{path}: case {name!r} has 'inputs' = {raw_inputs!r}, which must be a "
            f"mapping with string keys."
        )
    for key in ("params", "real_params"):
        if key in raw_inputs:
            _validate_params_spec(path, name, key, raw_inputs[key])
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
            sets an unrecognized field, or fails validation of ``name``,
            ``source``, ``checked``, ``expected``, ``tolerance``, or
            ``inputs``.
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
        for index, entry in enumerate(raw_cases):
            if not isinstance(entry, Mapping) or "name" not in entry:
                raise GoldenCaseError(f"{path}: case at index {index} has no 'name'.")
            name = _validate_name(path, index, entry["name"])
            if name in seen_names:
                raise GoldenCaseError(
                    f"{path}: name {name!r} is used by more than one case (indices "
                    f"{seen_names[name]} and {index}). Names must be unique within "
                    f"a file — they are how a failure is grepped back to its case."
                )
            seen_names[name] = index

            unknown = set(entry) - _ALLOWED_CASE_FIELDS
            if unknown:
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
            tolerance = _parse_tolerance(path, name, entry.get("tolerance", DEFAULT_TOLERANCE))
            inputs = _validate_inputs(path, name, entry.get("inputs"))

            cases.append(
                GoldenCase(
                    file=path,
                    name=name,
                    target=target,
                    source=source,
                    checked=checked,
                    inputs=inputs,
                    expected=entry["expected"],
                    tolerance=tolerance,
                )
            )
    return cases


def resolve_target(dotted: str) -> Callable[..., Any]:
    """Import a dotted path to a callable, e.g. ``engine.tax.federal.tax_on_income``.

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


def resolve_real_params(spec: Mapping[str, Any]) -> NoReturn:
    """Hook for the deflated real-terms parameter view.

    That view does not exist yet — it is roadmap issue 8's job, implemented
    against ``engine.core.indexation``. A case that needs ``real_params`` must
    wait for that issue; this raises rather than inventing one.
    """
    raise NotImplementedError(
        "real_params is not resolvable yet: the deflated real-terms parameter "
        "view is roadmap issue 8, to be implemented in engine.core.indexation. "
        f"Requested spec: {spec!r}."
    )


def resolve_inputs(raw_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Turn a case's raw ``inputs`` mapping into real keyword arguments."""
    resolved: dict[str, Any] = {}
    for key, value in raw_inputs.items():
        if key == "params":
            resolved[key] = resolve_params(value)
        elif key == "real_params":
            resolved[key] = resolve_real_params(value)
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
        ValueError: If a result is a multi-element array (see :func:`to_float`).
        NotImplementedError: If an input named ``real_params`` is present (see
            :func:`resolve_real_params`).
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
        _compare_one(case, f"{case.id}[{output_name}]", expected_value, actual_raw)

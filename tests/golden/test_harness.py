"""Self-tests of the golden harness itself.

These are not golden cases and carry no ``@pytest.mark.golden`` — they check
that ``conftest.discover_cases`` and ``conftest.run_case`` behave correctly,
using synthetic case files written into ``tmp_path`` and a trivial function
defined right here, resolved by its real dotted path. Every number below is
obviously synthetic arithmetic (doubling), never a value shaped like a tax
parameter.

The dotted path a synthetic case needs is this module's own runtime name —
``golden.test_harness`` (see ``tests/golden/__init__.py`` for why the package
exists at all) — captured once, at import time, as :data:`_THIS_MODULE`,
rather than hard-coded, so these tests do not silently start resolving the
wrong module if pytest's import naming ever changes.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from engine.params.loader import ParamFileMissingError, ParamSet, ParamYearMissingError

from .conftest import (
    GoldenCase,
    GoldenCaseError,
    _load_year_cached,
    _lookup_output,
    discover_cases,
    resolve_inputs,
    resolve_params,
    resolve_real_params,
    resolve_target,
    run_case,
    to_float,
)

_THIS_MODULE = __name__

#: A fixed "today" reference for the future-``checked`` test, safely in the
#: future no matter when this suite runs.
_FAR_FUTURE_DATE = "9999-01-01"

#: A module attribute that is not callable, for resolve_target's non-callable check.
NOT_CALLABLE = 42.0


def double(x: float) -> float:
    """Obviously synthetic: doubles its argument. Not a tax function."""
    return x * 2.0


def double_named(x: float) -> dict[str, float]:
    """Same idea, returned under a named output rather than bare ``value``."""
    return {"doubled": x * 2.0}


def _write(tmp_path: Path, filename: str, text: str) -> Path:
    path = tmp_path / filename
    path.write_text(text, encoding="utf-8")
    return path


def _well_formed_case(name: str = "seven doubled", *, expected: float = 14.0) -> str:
    return f"""
target: {_THIS_MODULE}.double
cases:
  - name: {name}
    source: "a synthetic unit test, not a real calculator"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: {expected}
"""


def test_well_formed_case_discovers_one_case_and_passes(tmp_path: Path) -> None:
    _write(tmp_path, "synthetic.yaml", _well_formed_case())

    cases = discover_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0].id == "synthetic::seven doubled"
    run_case(cases[0])  # must not raise


def test_named_output_case_passes(tmp_path: Path) -> None:
    text = f"""
target: {_THIS_MODULE}.double_named
cases:
  - name: seven doubled, named
    source: "a synthetic unit test, not a real calculator"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      doubled: 14.0
"""
    _write(tmp_path, "synthetic.yaml", text)

    cases = discover_cases(tmp_path)

    assert len(cases) == 1
    run_case(cases[0])  # must not raise


def test_wrong_expected_value_fails_with_both_numbers_in_message(tmp_path: Path) -> None:
    _write(tmp_path, "synthetic.yaml", _well_formed_case(expected=999.0))

    cases = discover_cases(tmp_path)

    with pytest.raises(AssertionError) as excinfo:
        run_case(cases[0])

    message = str(excinfo.value)
    assert "999" in message, message
    assert "14" in message, message


def test_case_missing_source_fails_discovery_naming_file_and_case(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: an unattributed number
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "an unattributed number" in message
    # The case name deliberately does not contain the word "source", so this
    # assertion can only pass by the message actually naming the missing
    # field, not by an accidental substring match on the case name.
    assert "source" in message


def test_case_missing_checked_fails_discovery_naming_file_and_case(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: an undated number
    source: "a synthetic unit test, not a real calculator"
    inputs:
      x: 7.0
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "an undated number" in message
    # Same reasoning as above: "checked" is absent from the case name, so this
    # can only pass by the message naming the missing field.
    assert "checked" in message


def test_case_with_non_date_checked_fails_discovery(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: checked is not a date
    source: "a synthetic unit test, not a real calculator"
    checked: "not a date"
    inputs:
      x: 7.0
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "checked is not a date" in message


def test_future_checked_date_fails_discovery(tmp_path: Path) -> None:
    """A future check date reads as more authoritative than the truth."""
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: checked in the future
    source: "synthetic"
    checked: {_FAR_FUTURE_DATE}
    inputs:
      x: 7.0
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert str(path) in str(excinfo.value)


def test_checked_with_a_time_of_day_is_rejected(tmp_path: Path) -> None:
    """A timestamp, not a bare date: 'checked', not 'checked at a given hour'."""
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: checked with a timestamp
    source: "synthetic"
    checked: 2026-01-01 10:00:00
    inputs:
      x: 7.0
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "checked with a timestamp" in str(excinfo.value)


# --- BLOCKER 1: source must be a real, non-empty string, not just present --


def test_null_source_fails_discovery(tmp_path: Path) -> None:
    """A bare ``source:`` key parses to ``None``, satisfying a presence check."""
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: source is null
    source:
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "source is null" in str(excinfo.value)


def test_empty_string_source_fails_discovery(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: source is an empty string
    source: ""
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "source is an empty string" in str(excinfo.value)


def test_whitespace_only_source_fails_discovery(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: source is whitespace only
    source: "   "
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "source is whitespace only" in str(excinfo.value)


# --- name must be a real, non-empty, unique-within-file string -------------


def test_null_name_fails_discovery(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name:
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert str(path) in str(excinfo.value)


def test_empty_string_name_fails_discovery(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: ""
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError):
        discover_cases(tmp_path)


def test_duplicate_name_within_a_file_fails_discovery(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: dup
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: 14.0
  - name: dup
    source: "synthetic, second occurrence"
    checked: 2026-01-01
    inputs:
      x: 8.0
    expected:
      value: 16.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "dup" in message


def test_empty_cases_directory_discovers_nothing(tmp_path: Path) -> None:
    assert discover_cases(tmp_path) == []


def test_yaml_file_present_but_yielding_no_cases_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "empty.yaml",
        f"""
target: {_THIS_MODULE}.double
cases: []
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert str(path) in str(excinfo.value)


def test_yml_extension_is_also_discovered(tmp_path: Path) -> None:
    """A case file saved as ``.yml`` must not silently vanish from discovery."""
    _write(tmp_path, "synthetic.yml", _well_formed_case())

    cases = discover_cases(tmp_path)

    assert len(cases) == 1
    run_case(cases[0])  # must not raise


def test_missing_cases_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path / "does_not_exist")

    assert "does_not_exist" in str(excinfo.value)


# --- Blocker 2: discovery must see every file under cases/, recursively, or
# --- reject it loudly rather than silently skip it. ------------------------


def test_case_in_a_subdirectory_is_discovered(tmp_path: Path) -> None:
    """A subdirectory is the first thing a human reaches for past a handful of cases."""
    (tmp_path / "federal").mkdir()
    _write(tmp_path / "federal", "brackets.yaml", _well_formed_case())

    cases = discover_cases(tmp_path)

    assert len(cases) == 1
    run_case(cases[0])  # must not raise


def test_uppercase_extension_is_rejected_not_silently_skipped(tmp_path: Path) -> None:
    path = _write(tmp_path, "c.YAML", _well_formed_case())

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert str(path) in str(excinfo.value)


def test_stray_backup_file_is_rejected_not_silently_skipped(tmp_path: Path) -> None:
    path = _write(tmp_path, "c.yaml.bak", _well_formed_case())

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert str(path) in str(excinfo.value)


def test_gitkeep_itself_is_not_treated_as_a_stray_file(tmp_path: Path) -> None:
    _write(tmp_path, ".gitkeep", "")

    assert discover_cases(tmp_path) == []


# --- The blocker: an empty `expected` must not silently assert nothing -----


def test_empty_expected_fails_discovery_naming_file_and_case(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: asserts absolutely nothing
    source: "synthetic"
    checked: 2026-01-01
    inputs: {{x: 7.0}}
    expected: {{}}
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "asserts absolutely nothing" in message


def test_null_expected_fails_discovery(tmp_path: Path) -> None:
    """A bare ``expected:`` key parses as ``None`` in YAML, not ``{}``."""
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: bare expected key
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "bare expected key" in str(excinfo.value)


def test_scalar_expected_fails_discovery(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: expected is a bare number
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "expected is a bare number" in str(excinfo.value)


def test_null_expected_value_fails_discovery(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: expected value is null
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value:
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "expected value is null" in str(excinfo.value)


def test_boolean_expected_value_fails_discovery(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: expected value is a boolean
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: true
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "expected value is a boolean" in str(excinfo.value)


def test_string_expected_value_fails_discovery(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: expected value is a string
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: "14.0"
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "expected value is a string" in str(excinfo.value)


def test_value_mixed_with_named_output_fails_discovery(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: value mixed with a named output
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: 14.0
      other: 2.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "value mixed with a named output" in message
    assert "value" in message


def test_infinite_expected_value_fails_discovery(tmp_path: Path) -> None:
    """An overflowing target must not report green via inf == approx(inf)."""
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: expected value is infinite
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: .inf
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "expected value is infinite" in str(excinfo.value)


def test_nan_expected_value_fails_discovery(tmp_path: Path) -> None:
    """NaN can never truthfully compare equal to anything at run time."""
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: expected value is nan
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: .nan
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "expected value is nan" in str(excinfo.value)


# --- inputs and params/real_params specs are validated at discovery too ----


def test_list_inputs_fails_discovery(tmp_path: Path) -> None:
    """``[x: 7.0]`` instead of ``{x: 7.0}`` must not die inside ``target(**kwargs)``."""
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: inputs is a list
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      - x: 7.0
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert str(path) in str(excinfo.value)


def test_params_spec_missing_a_key_fails_discovery(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: params spec is missing file
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      params: {{year: 2026}}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "params spec is missing file" in str(excinfo.value)


def test_params_spec_not_a_mapping_fails_discovery(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: params spec is a string
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      params: federal
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "params spec is a string" in str(excinfo.value)


def test_params_spec_with_an_extra_key_fails_discovery(tmp_path: Path) -> None:
    """A typo one level into ``inputs`` must be as loud as one at the case level."""
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: params spec has a typo'd key
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      params: {{year: 2026, file: federal, provice: ab}}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "params spec has a typo'd key" in str(excinfo.value)


# --- tolerance validation ----------------------------------------------------


def test_non_numeric_tolerance_fails_discovery(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: tolerance is not a number
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: 14.0
    tolerance: "abc"
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "tolerance is not a number" in str(excinfo.value)


def test_negative_tolerance_fails_discovery(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: tolerance is negative
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: 14.0
    tolerance: -1
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "tolerance is negative" in str(excinfo.value)


def test_zero_tolerance_fails_discovery(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: tolerance is zero
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: 14.0
    tolerance: 0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "tolerance is zero" in str(excinfo.value)


def test_infinite_tolerance_fails_discovery(tmp_path: Path) -> None:
    """The green-washing route: an unbounded tolerance passes anything."""
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: tolerance is infinite
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: 14.0
    tolerance: .inf
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert "tolerance is infinite" in str(excinfo.value)


def test_generous_tolerance_override_makes_a_would_be_failure_pass(tmp_path: Path) -> None:
    """A legitimate, finite, per-case override still works as documented."""
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: generous tolerance
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: 14.5
    tolerance: 1.0
""",
    )

    cases = discover_cases(tmp_path)

    run_case(cases[0])  # 14.0 is within 1.0 of 14.5; must not raise


# --- unrecognized fields ------------------------------------------------------


def test_unknown_field_fails_discovery(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: a typo'd field
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: 14.0
    tolerence: 5.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "a typo'd field" in message
    assert "tolerence" in message


# --- YAML-layer failures escape without a filename, unless caught here ------


def test_malformed_yaml_fails_discovery_naming_file(tmp_path: Path) -> None:
    path = _write(tmp_path, "broken.yaml", "target: [unterminated\ncases:\n")

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert str(path) in str(excinfo.value)


def test_out_of_range_unquoted_date_fails_discovery_naming_file(tmp_path: Path) -> None:
    """PyYAML's own timestamp resolver raises a bare ValueError on this input."""
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: an impossible date
    source: "synthetic"
    checked: 2026-13-45
    inputs:
      x: 7.0
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    assert str(path) in str(excinfo.value)


# --- resolve_params / resolve_real_params / resolve_target -------------------


def test_resolve_params_returns_a_real_paramset() -> None:
    """Only the *type* is asserted here — never a number out of the file.

    Asserting a tax figure in this test would be inventing a golden case
    outside the harness the rest of this suite exists to gate; the harness's
    job is only to prove it can hand a real, hand-populated ParamSet to a
    target, not to check what is in it.
    """
    result = resolve_params({"year": 2026, "file": "federal"})

    assert isinstance(result, ParamSet)


def test_resolve_params_routes_year_and_file_to_the_matching_paramset() -> None:
    """Identity fields only — ``.name``/``.year``/the source filename.

    Asserting these is not inventing a tax number: a harness that ignored
    ``year`` or ``file`` and always returned the same set would pass a test
    that only checked ``isinstance``, and every case naming ``params: {year,
    file: ab}`` would then silently run against ``federal.yaml`` instead.
    Checking that two different requested files actually come back as two
    different, correctly labelled ParamSets is the minimum that rules that
    out.
    """
    federal = resolve_params({"year": 2026, "file": "federal"})
    ab = resolve_params({"year": 2026, "file": "ab"})

    assert federal.name == "federal"
    assert federal.year == 2026
    assert federal.source.name == "federal.yaml"

    assert ab.name == "ab"
    assert ab.year == 2026
    assert ab.source.name == "ab.yaml"

    assert federal.source != ab.source


def test_resolve_params_rejects_an_unknown_year() -> None:
    with pytest.raises(ParamYearMissingError):
        resolve_params({"year": 1900, "file": "federal"})


def test_resolve_params_rejects_an_unknown_file() -> None:
    with pytest.raises(ParamFileMissingError):
        resolve_params({"year": 2026, "file": "not_a_real_jurisdiction"})


def test_load_year_cached_returns_the_same_paramyear_object() -> None:
    """The caching added for issue #3's finding 12: no correctness change.

    ``ParamYear`` is deeply immutable, so returning the identical object on a
    second call for the same year is safe and is exactly what the cache is
    for.
    """
    assert _load_year_cached(2026) is _load_year_cached(2026)


def test_resolve_real_params_raises_not_implemented_naming_issue_8() -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        resolve_real_params({"year": 2026, "file": "federal"})

    message = str(excinfo.value)
    assert "issue 8" in message
    assert "engine.core.indexation" in message


def test_resolve_inputs_dispatches_params_and_leaves_others_alone() -> None:
    resolved = resolve_inputs({"x": 7.0, "params": {"year": 2026, "file": "federal"}})

    assert resolved["x"] == 7.0
    assert isinstance(resolved["params"], ParamSet)


def test_resolve_target_rejects_a_non_dotted_path() -> None:
    with pytest.raises(GoldenCaseError):
        resolve_target("double")


def test_resolve_target_rejects_a_missing_attribute() -> None:
    with pytest.raises(GoldenCaseError):
        resolve_target(f"{_THIS_MODULE}.this_function_does_not_exist")


def test_resolve_target_rejects_an_unimportable_module() -> None:
    """A typo in the module half must not surface as a bare ModuleNotFoundError."""
    with pytest.raises(GoldenCaseError):
        resolve_target("no_such_module_at_all_xyz.double")


def test_resolve_target_rejects_a_non_callable_attribute() -> None:
    with pytest.raises(GoldenCaseError):
        resolve_target(f"{_THIS_MODULE}.NOT_CALLABLE")


# --- to_float -----------------------------------------------------------------


def test_to_float_accepts_a_0d_array() -> None:
    assert to_float(np.array(5.0)) == 5.0


def test_to_float_accepts_a_single_element_array() -> None:
    assert to_float(np.array([5.0])) == 5.0


def test_to_float_rejects_none() -> None:
    with pytest.raises(TypeError):
        to_float(None)


def test_to_float_rejects_a_string() -> None:
    with pytest.raises(TypeError):
        to_float("5.0")


def test_to_float_rejects_a_list_as_the_wrong_type_not_the_wrong_size() -> None:
    """A plain list is rejected by the "not a recognized type" branch.

    Both ``[1.0, 2.0]`` and ``[1.0]`` fail identically here — a Python list is
    never a valid result regardless of its length, only a numpy array is
    size-checked — so this pins the branch with ``match=`` rather than just
    asserting *some* ``TypeError``, which a rename of the size-guard's message
    could otherwise satisfy by accident.
    """
    with pytest.raises(TypeError, match="cannot interpret"):
        to_float([1.0, 2.0])


def test_to_float_rejects_a_multi_element_array() -> None:
    with pytest.raises(ValueError, match=r"\(3,\)"):
        to_float(np.array([1.0, 2.0, 3.0]))


def test_to_float_rejects_a_python_bool() -> None:
    with pytest.raises(TypeError):
        to_float(True)


def test_to_float_rejects_a_numpy_bool_scalar() -> None:
    with pytest.raises(TypeError):
        to_float(np.bool_(True))


def test_to_float_rejects_a_boolean_array() -> None:
    with pytest.raises(TypeError):
        to_float(np.array([True]))


# --- _lookup_output ------------------------------------------------------------


@dataclass(frozen=True)
class _Result:
    """A trivial dataclass result, standing in for a real engine return type."""

    doubled: float


def _dummy_case() -> GoldenCase:
    return GoldenCase(
        file=Path("synthetic.yaml"),
        name="dummy",
        target=f"{_THIS_MODULE}.double",
        source="synthetic",
        checked=dt.date(2026, 1, 1),
        expected={"doubled": 14.0},
    )


def test_lookup_output_reads_a_mapping_key() -> None:
    assert _lookup_output(_dummy_case(), {"doubled": 14.0}, "doubled") == 14.0


def test_lookup_output_reads_a_dataclass_field() -> None:
    assert _lookup_output(_dummy_case(), _Result(doubled=14.0), "doubled") == 14.0


def test_lookup_output_missing_mapping_key_raises() -> None:
    with pytest.raises(GoldenCaseError):
        _lookup_output(_dummy_case(), {"other": 1.0}, "doubled")


def test_lookup_output_missing_dataclass_field_raises() -> None:
    with pytest.raises(GoldenCaseError):
        _lookup_output(_dummy_case(), _Result(doubled=14.0), "not_a_field")


def test_lookup_output_rejects_a_result_with_no_declared_fields() -> None:
    """Neither a mapping nor a dataclass/NamedTuple: nothing to address by name.

    This is the finding-7 regression: without this restriction, ``getattr``
    would resolve ``imag``/``real``/``ndim``/``size`` on an ordinary float or
    numpy scalar and let a named-output case pass against a bare number
    forever, vacuously.
    """
    with pytest.raises(GoldenCaseError, match="mapping, dataclass, or NamedTuple"):
        _lookup_output(_dummy_case(), 3.0, "imag")


def test_named_output_against_a_bare_float_result_is_rejected(tmp_path: Path) -> None:
    """End-to-end version of the finding-7 regression, through run_case."""
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: named output against a bare scalar
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      imag: 0.0
""",
    )
    cases = discover_cases(tmp_path)

    with pytest.raises(GoldenCaseError, match="mapping, dataclass, or NamedTuple"):
        run_case(cases[0])


# --- run_case defends the empty-expected invariant even off the discovery path


def test_run_case_rejects_a_hand_built_case_with_empty_expected() -> None:
    case = GoldenCase(
        file=Path("synthetic.yaml"),
        name="built by hand",
        target=f"{_THIS_MODULE}.double",
        source="synthetic",
        checked=dt.date(2026, 1, 1),
        inputs={"x": 7.0},
        expected={},
    )

    with pytest.raises(GoldenCaseError):
        run_case(case)

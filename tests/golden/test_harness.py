# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

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
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from engine.core.indexation import RealParamSet, RealParamYear, erosion_factor
from engine.params.loader import ParamFileMissingError, ParamSet, ParamYearMissingError

from .conftest import (
    _DEFAULT_ROUNDING,
    _ROUNDING_ID_SUFFIXES,
    ROUNDING_TOLERANCES,
    GoldenCase,
    GoldenCaseError,
    _load_year_cached,
    _lookup_output,
    discover_cases,
    resolve_inputs,
    resolve_params,
    resolve_real_params,
    resolve_real_params_year,
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


def read_basic_amount_annual(params: RealParamSet) -> float:
    """Reads the synthetic ``credits.basic_amount_annual`` via ``annual_amount``.

    Obviously synthetic, same as ``double`` above: exercises that a
    ``real_params`` input reaches a target as the keyword ``params``, not
    that the number itself means anything.
    """
    return params.annual_amount("credits.basic_amount_annual", 0)


def read_basic_amount_annual_from_year(params: RealParamYear) -> float:
    """Same idea as :func:`read_basic_amount_annual`, but for a whole tax year.

    Exercises that a ``real_params_year`` input reaches a target as the
    keyword ``params``, resolved to a :class:`RealParamYear` rather than one
    file's :class:`RealParamSet`.
    """
    return params.federal.annual_amount("credits.basic_amount_annual", 0)


#: The synthetic tax year the ``real_params`` self-tests load, kept out of the
#: way of any year a real case under ``params/`` names.
_SYNTHETIC_REAL_PARAMS_YEAR = 2030

_SYNTHETIC_FEDERAL_YAML = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were, see
# tests/core/test_indexation.py.
indexation:
  yearly:
    adjustment_months: [1]
    applies_to:
      - credits.basic_amount_annual

credits:
  basic_amount_annual: 1000
"""


def _write(tmp_path: Path, filename: str, text: str) -> Path:
    path = tmp_path / filename
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def synthetic_params_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ``tmp_path`` params root holding one synthetic file at a synthetic year.

    Kept as a subdirectory separate from any cases directory a test writes
    into (``tmp_path / "params"``, never ``tmp_path`` itself), because
    ``discover_cases`` walks its own directory recursively and would trip
    over a params YAML sitting inside it. Patches ``DEFAULT_PARAMS_ROOT`` on
    this package's own ``conftest`` module, imported relatively — the name
    ``_load_year_cached`` actually reads at call time — rather than adding a
    root parameter to any harness function. The cache is cleared both before
    and after, so neither a stale entry from an earlier test nor this
    fixture's own entry leaks into a test that did not ask for it.
    """
    from . import conftest as conftest_module

    root = tmp_path / "params"
    year_dir = root / str(_SYNTHETIC_REAL_PARAMS_YEAR)
    year_dir.mkdir(parents=True)
    (year_dir / "federal.yaml").write_text(_SYNTHETIC_FEDERAL_YAML, encoding="utf-8")

    conftest_module._load_year_cached.cache_clear()
    monkeypatch.setattr(conftest_module, "DEFAULT_PARAMS_ROOT", root)
    yield root
    conftest_module._load_year_cached.cache_clear()


def _well_formed_case(
    name: str = "seven doubled",
    *,
    expected: float = 14.0,
    rounding: str | None = None,
    tolerance: float | None = None,
) -> str:
    """A synthetic ``double`` case, optionally declaring ``rounding`` or the removed ``tolerance``.

    ``rounding`` and ``tolerance`` are inserted verbatim as written, not
    validated here, so a caller can also use them to construct a
    deliberately malformed value (a number, a list, an empty string standing
    in for a bare ``rounding:`` key) for a discovery-rejection test, without
    a second near-duplicate helper to keep in sync with this one.
    """
    extra = ""
    if rounding is not None:
        extra += f"\n    rounding: {rounding}"
    if tolerance is not None:
        extra += f"\n    tolerance: {tolerance}"
    return f"""
target: {_THIS_MODULE}.double
cases:
  - name: {name}
    source: "a synthetic unit test, not a real calculator"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: {expected}{extra}
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


# --- real_params: {year, file, inflation}, and inflation has no default ----


def test_real_params_spec_missing_inflation_fails_discovery(tmp_path: Path) -> None:
    """The one key ``real_params`` has beyond ``params``, and it is required."""
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: real_params spec is missing inflation
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      real_params: {{year: 2026, file: federal}}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "real_params spec is missing inflation" in message
    assert str(path) in message
    assert "exactly the keys" in message


def test_real_params_spec_with_a_string_inflation_fails_discovery(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: real_params inflation is a string
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      real_params: {{year: 2026, file: federal, inflation: "2%"}}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "real_params inflation is a string" in message
    assert str(path) in message
    assert "must be an int or a float" in message


def test_real_params_spec_with_a_boolean_inflation_fails_discovery(tmp_path: Path) -> None:
    """A bool is an int to Python, and rejected the same way ``year`` rejects one."""
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: real_params inflation is a boolean
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      real_params: {{year: 2026, file: federal, inflation: true}}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "real_params inflation is a boolean" in message
    assert str(path) in message
    assert "must be an int or a float" in message


def test_real_params_spec_with_a_nan_inflation_fails_discovery(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: real_params inflation is nan
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      real_params: {{year: 2026, file: federal, inflation: .nan}}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "real_params inflation is nan" in message
    assert str(path) in message
    assert "NaN" in message


def test_real_params_spec_with_an_infinite_inflation_fails_discovery(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: real_params inflation is infinite
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      real_params: {{year: 2026, file: federal, inflation: .inf}}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "real_params inflation is infinite" in message
    assert str(path) in message
    # Bare "infinite" also appears in the case name above, so the assertion
    # below uses the longer, branch-specific phrase instead.
    assert "which is infinite" in message


def test_real_params_spec_with_an_extra_key_fails_discovery(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: real_params spec has a typo'd key
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      real_params: {{year: 2026, file: federal, inflation: 0.0, provice: ab}}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "real_params spec has a typo'd key" in message
    assert str(path) in message
    assert "exactly the keys" in message


def test_inflation_on_a_plain_params_spec_fails_discovery_as_an_extra_key(
    tmp_path: Path,
) -> None:
    """``inflation`` belongs to ``real_params``; on ``params`` it is just an extra key."""
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: params spec has inflation, which is not one of its keys
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      params: {{year: 2026, file: federal, inflation: 0.0}}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "params spec has inflation, which is not one of its keys" in message
    assert str(path) in message
    assert "exactly the keys" in message


def test_real_params_spec_with_an_inflation_too_large_for_a_float_fails_discovery(
    tmp_path: Path,
) -> None:
    """A 400-digit YAML integer makes ``math.isfinite`` raise ``OverflowError``, not answer False."""
    huge = "9" * 400
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: real_params inflation overflows a float
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      real_params: {{year: 2026, file: federal, inflation: {huge}}}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "real_params inflation overflows a float" in message
    assert str(path) in message
    assert "too large to represent as a float" in message


def test_naming_both_params_and_real_params_fails_discovery(tmp_path: Path) -> None:
    """Both would claim the target's ``params`` keyword — a case names one or the other."""
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: both params and real_params
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      params: {{year: 2026, file: federal}}
      real_params: {{year: 2026, file: federal, inflation: 0.0}}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "both params and real_params" in message
    assert str(path) in message
    assert "never both" in message


def test_real_params_case_reaches_annual_amount_at_zero_inflation(
    tmp_path: Path, synthetic_params_root: Path
) -> None:
    """The success criterion: a synthetic case reaches the real-terms view and passes.

    ``cases/`` and ``params/`` are kept as separate subdirectories of
    ``tmp_path`` — ``discover_cases`` walks its own directory recursively and
    would reject a params YAML sitting inside it as a stray file.
    """
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    _write(
        cases_dir,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.read_basic_amount_annual
cases:
  - name: real_params reaches annual_amount at zero inflation
    source: "synthetic unit test, not a real calculator"
    checked: 2026-01-01
    inputs:
      real_params: {{year: {_SYNTHETIC_REAL_PARAMS_YEAR}, file: federal, inflation: 0.0}}
    expected:
      value: 1000.0
""",
    )

    cases = discover_cases(cases_dir)

    assert len(cases) == 1
    run_case(cases[0])  # must not raise


def test_real_params_case_reaches_annual_amount_at_a_non_zero_inflation(
    tmp_path: Path, synthetic_params_root: Path
) -> None:
    """The stated rate actually reaches the target, end to end through ``run_case``.

    The expected value is computed with :func:`erosion_factor` here, in the
    test, and written into the case YAML — the case expects what the
    arithmetic gives, not a transcribed float that could drift from it.
    """
    rate = 4095
    expected = 1000 * erosion_factor(rate, 1)
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    _write(
        cases_dir,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.read_basic_amount_annual
cases:
  - name: real_params reaches annual_amount at a non-zero inflation
    source: "synthetic unit test, not a real calculator"
    checked: 2026-01-01
    inputs:
      real_params: {{year: {_SYNTHETIC_REAL_PARAMS_YEAR}, file: federal, inflation: {rate}}}
    expected:
      value: {expected!r}
""",
    )

    cases = discover_cases(cases_dir)

    assert len(cases) == 1
    run_case(cases[0])  # must not raise


def test_real_params_case_missing_inflation_fails_discovery(
    tmp_path: Path, synthetic_params_root: Path
) -> None:
    """The twin of the case above, with ``inflation`` dropped: fails at discovery."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    path = _write(
        cases_dir,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.read_basic_amount_annual
cases:
  - name: real_params missing inflation reaches nothing
    source: "synthetic unit test, not a real calculator"
    checked: 2026-01-01
    inputs:
      real_params: {{year: {_SYNTHETIC_REAL_PARAMS_YEAR}, file: federal}}
    expected:
      value: 1000.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(cases_dir)

    message = str(excinfo.value)
    assert "real_params missing inflation reaches nothing" in message
    assert str(path) in message
    assert "exactly the keys" in message


# --- real_params_year: {year, inflation}, one file shallower than real_params -


def test_real_params_year_spec_with_an_extra_file_key_fails_discovery(tmp_path: Path) -> None:
    """``real_params_year`` has no ``file``: it resolves a whole year, not one file."""
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: real_params_year spec has a file key
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      real_params_year: {{year: 2026, file: federal, inflation: 0.0}}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "real_params_year spec has a file key" in message
    assert str(path) in message
    assert "exactly the keys" in message


def test_real_params_year_spec_missing_inflation_fails_discovery(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: real_params_year spec is missing inflation
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      real_params_year: {{year: 2026}}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "real_params_year spec is missing inflation" in message
    assert str(path) in message
    assert "exactly the keys" in message


def test_real_params_year_spec_with_a_boolean_year_fails_discovery(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: real_params_year year is a boolean
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      real_params_year: {{year: true, inflation: 0.0}}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "real_params_year year is a boolean" in message
    assert str(path) in message
    assert "must be an int" in message


@pytest.mark.parametrize(
    ("label", "raw_inflation", "expected_phrase"),
    [
        ("a string", '"2%"', "must be an int or a float"),
        ("a boolean", "true", "must be an int or a float"),
        ("nan", ".nan", "NaN"),
        ("infinite", ".inf", "which is infinite"),
    ],
)
def test_real_params_year_spec_with_an_invalid_inflation_fails_discovery(
    tmp_path: Path, label: str, raw_inflation: str, expected_phrase: str
) -> None:
    name = f"real_params_year inflation is {label}"
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: {name}
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      real_params_year: {{year: 2026, inflation: {raw_inflation}}}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert name in message
    assert str(path) in message
    assert expected_phrase in message


def test_real_params_year_spec_not_a_mapping_fails_discovery(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: real_params_year spec is a plain number
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      real_params_year: 2026
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "real_params_year spec is a plain number" in message
    assert str(path) in message
    assert "which must be a mapping with exactly the keys ['inflation', 'year']" in message


def test_real_params_year_spec_with_an_inflation_too_large_for_a_float_fails_discovery(
    tmp_path: Path,
) -> None:
    """A 400-digit YAML integer makes ``math.isfinite`` raise ``OverflowError``, not answer False."""
    huge = "9" * 400
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: real_params_year inflation overflows a float
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      real_params_year: {{year: 2026, inflation: {huge}}}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "real_params_year inflation overflows a float" in message
    assert str(path) in message
    assert "too large to represent as a float" in message


# --- naming more than one of params / real_params / real_params_year ----------

_SPEC_SNIPPETS = {
    "params": "{year: 2026, file: federal}",
    "real_params": "{year: 2026, file: federal, inflation: 0.0}",
    "real_params_year": "{year: 2026, inflation: 0.0}",
}


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("params", "real_params"),
        ("params", "real_params_year"),
        ("real_params", "real_params_year"),
    ],
)
def test_naming_two_parameter_inputs_fails_discovery_naming_both_keys(
    tmp_path: Path, first: str, second: str
) -> None:
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: names two parameter inputs
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      {first}: {_SPEC_SNIPPETS[first]}
      {second}: {_SPEC_SNIPPETS[second]}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert f"has both {first!r} and {second!r} in 'inputs'" in message
    assert repr("names two parameter inputs") in message


def test_naming_all_three_parameter_inputs_fails_discovery_naming_all_three(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: names all three parameter inputs
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
      params: {_SPEC_SNIPPETS["params"]}
      real_params: {_SPEC_SNIPPETS["real_params"]}
      real_params_year: {_SPEC_SNIPPETS["real_params_year"]}
    expected:
      value: 14.0
""",
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "has 'params', 'real_params' and 'real_params_year' in 'inputs'" in message
    assert "has both" not in message
    assert "never both" not in message
    assert repr("names all three parameter inputs") in message


def test_real_params_year_case_reaches_annual_amount_at_zero_inflation(
    tmp_path: Path, synthetic_params_root: Path
) -> None:
    """The success criterion: a synthetic case reaches the whole-year real-terms view and passes."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    _write(
        cases_dir,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.read_basic_amount_annual_from_year
cases:
  - name: real_params_year reaches annual_amount at zero inflation
    source: "synthetic unit test, not a real calculator"
    checked: 2026-01-01
    inputs:
      real_params_year: {{year: {_SYNTHETIC_REAL_PARAMS_YEAR}, inflation: 0.0}}
    expected:
      value: 1000.0
""",
    )

    cases = discover_cases(cases_dir)

    assert len(cases) == 1
    run_case(cases[0])  # must not raise


def test_resolve_real_params_year_returns_the_stated_rate_and_the_cached_raw_year(
    synthetic_params_root: Path,
) -> None:
    """``.inflation_rate`` is the case's own stated rate, and ``.raw`` is the cached ``ParamYear``.

    ``.raw`` identity, not just equality, proves this reuses
    ``_load_year_cached`` rather than reading the year a second time.
    """
    rate = 0.25
    result = resolve_real_params_year({"year": _SYNTHETIC_REAL_PARAMS_YEAR, "inflation": rate})

    assert isinstance(result, RealParamYear)
    assert result.inflation_rate == rate
    assert result.raw is _load_year_cached(_SYNTHETIC_REAL_PARAMS_YEAR)


# --- rounding validation and derivation ---------------------------------------
#
# Issue #25: `tolerance` was replaced by a declared `rounding`, from which the
# harness derives the absolute tolerance. There is no numeric field any more —
# see `conftest.ROUNDING_TOLERANCES` for the derivation table and its rationale.


def test_omitting_rounding_gives_the_cent_tolerance(tmp_path: Path) -> None:
    _write(tmp_path, "synthetic.yaml", _well_formed_case("plain case", expected=14.0))

    cases = discover_cases(tmp_path)

    assert cases[0].rounding == "source_rounds_to_cent"
    assert cases[0].tolerance == pytest.approx(0.01)
    assert cases[0].id == "synthetic::plain case"  # no suffix at the default


@pytest.mark.parametrize(
    ("rounding", "expected_tolerance"),
    [
        ("source_rounds_to_cent", 0.01),
        ("source_rounds_to_dollar", 0.50),
        ("source_rounds_to_ten_dollars", 5.00),
        ("monthly_cent_times_twelve", 0.06),
    ],
)
def test_each_rounding_member_derives_its_documented_tolerance(
    tmp_path: Path, rounding: str, expected_tolerance: float
) -> None:
    _write(
        tmp_path,
        "synthetic.yaml",
        _well_formed_case("declared rounding", expected=14.0, rounding=rounding),
    )

    cases = discover_cases(tmp_path)

    assert cases[0].rounding == rounding
    assert cases[0].tolerance == pytest.approx(expected_tolerance)


def test_rounding_tolerances_and_id_suffixes_stay_in_step(tmp_path: Path) -> None:
    """Guards the one extension path the issue sanctions: adding a fifth member.

    ``GoldenCase.id`` falls back to the unsuffixed id on a lookup miss in
    ``_ROUNDING_ID_SUFFIXES``, so a new key added to ``ROUNDING_TOLERANCES``
    without a matching entry here would run at a loosened tolerance while
    still rendering a plain id — the exact invisible loosening the suffix
    exists to prevent. This does not touch a case file at all; it pins the
    two tables against each other directly, which is the only place this
    invariant is checkable.
    """
    assert set(ROUNDING_TOLERANCES) - {_DEFAULT_ROUNDING} == set(_ROUNDING_ID_SUFFIXES)
    assert _DEFAULT_ROUNDING in ROUNDING_TOLERANCES


def test_every_rounding_tolerance_is_finite_and_positive() -> None:
    """The invariant the deleted numeric-``tolerance`` validation used to guard.

    A case file can no longer reach an unbounded or non-positive tolerance —
    ``rounding`` only ever selects one of these four fixed numbers — but the
    table itself is still where that number ultimately comes from, so the
    invariant is pinned here instead of left unreachable and unchecked.
    """
    for rounding, tolerance in ROUNDING_TOLERANCES.items():
        assert math.isfinite(tolerance), rounding
        assert tolerance > 0, rounding


@pytest.mark.parametrize(
    ("rounding", "suffix"),
    [
        ("source_rounds_to_dollar", "dollar"),
        ("source_rounds_to_ten_dollars", "ten_dollars"),
        ("monthly_cent_times_twelve", "monthly_cent_x12"),
    ],
)
def test_non_default_rounding_carries_its_suffix_in_the_id(
    tmp_path: Path, rounding: str, suffix: str
) -> None:
    _write(
        tmp_path,
        "synthetic.yaml",
        _well_formed_case("suffixed case", expected=14.0, rounding=rounding),
    )

    cases = discover_cases(tmp_path)

    assert cases[0].id == f"synthetic::suffixed case ({suffix})"


def test_fails_at_cent_rounding_but_passes_at_dollar_rounding(tmp_path: Path) -> None:
    """Proves the derivation is actually applied, not decoratively ignored.

    ``double(7.0)`` is ``14.0``; the case expects ``14.2`` — 0.2 off, which is
    more than the cent tolerance (0.01) and less than the dollar tolerance
    (0.50). Without this test, every ``rounding`` member could silently
    resolve to the same default tolerance and the rest of this suite would
    still be green.
    """
    at_cent = _well_formed_case("off by twenty cents", expected=14.2)
    at_dollar = _well_formed_case(
        "off by twenty cents", expected=14.2, rounding="source_rounds_to_dollar"
    )

    _write(tmp_path, "synthetic.yaml", at_cent)
    cent_case = discover_cases(tmp_path)[0]
    assert cent_case.tolerance == pytest.approx(0.01)
    with pytest.raises(AssertionError):
        run_case(cent_case)

    _write(tmp_path, "synthetic.yaml", at_dollar)
    dollar_case = discover_cases(tmp_path)[0]
    assert dollar_case.tolerance == pytest.approx(0.50)
    run_case(dollar_case)  # must not raise


def test_unrecognized_rounding_fails_discovery_naming_file_and_case(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "synthetic.yaml",
        _well_formed_case(
            "rounding is not a real member", expected=14.0, rounding="source_rounds_to_nickel"
        ),
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "rounding is not a real member" in message
    assert "source_rounds_to_nickel" in message
    for member in (
        "source_rounds_to_cent",
        "source_rounds_to_dollar",
        "source_rounds_to_ten_dollars",
        "monthly_cent_times_twelve",
    ):
        assert member in message


def test_numeric_rounding_fails_discovery(tmp_path: Path) -> None:
    """``rounding: 0.5`` (unquoted) parses as a YAML float, not a string."""
    _write(
        tmp_path,
        "synthetic.yaml",
        _well_formed_case("rounding is a number", expected=14.0, rounding="0.5"),
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "rounding is a number" in message
    assert "must be a string" in message


def test_list_rounding_fails_discovery(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "synthetic.yaml",
        _well_formed_case("rounding is a list", expected=14.0, rounding="[source_rounds_to_cent]"),
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "rounding is a list" in message
    assert "must be a string" in message


def test_bare_rounding_key_fails_discovery(tmp_path: Path) -> None:
    """A bare ``rounding:`` key parses to ``None``, not "omitted"."""
    _write(
        tmp_path,
        "synthetic.yaml",
        _well_formed_case("rounding key is bare", expected=14.0, rounding=""),
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "rounding key is bare" in message
    assert "must be a string" in message


def test_case_still_setting_tolerance_is_rejected_by_the_dedicated_message(
    tmp_path: Path,
) -> None:
    """Must fail via the dedicated ``tolerance``-was-removed branch, not the generic one.

    ``rounding`` is a member of ``_ALLOWED_CASE_FIELDS`` and ``tolerance`` is
    the one unrecognized field being reported, so the *generic* "unrecognized
    field(s)" message also contains both the literal substrings "rounding"
    and "tolerance" — asserting only on those, as a first pass at this test
    did, passes even if the dedicated ``if "tolerance" in unknown`` branch in
    ``discover_cases`` is deleted outright. This pins language unique to that
    branch, all four members by name, and the explicit *absence* of the
    generic message, so deleting the branch fails this test again.
    """
    _write(
        tmp_path,
        "synthetic.yaml",
        _well_formed_case("still sets tolerance", expected=14.0, tolerance=1.0),
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert "still sets tolerance" in message
    assert "was removed" in message
    assert "unrecognized field" not in message
    for member in (
        "source_rounds_to_cent",
        "source_rounds_to_dollar",
        "source_rounds_to_ten_dollars",
        "monthly_cent_times_twelve",
    ):
        assert member in message


# --- id collisions and the reserved rounding-suffix pattern -------------------


def test_a_name_colliding_with_a_suffixed_sibling_is_rejected(
    tmp_path: Path,
) -> None:
    """Two cases that would render one id are rejected — by the name guard.

    A duplicate ``name`` check alone does not catch this, because the names
    differ: ``basic rate`` at dollar rounding and ``basic rate (dollar)`` at
    the default both render ``synthetic::basic rate (dollar)``. ``id`` is what
    every comparison and lookup failure message carries, so two cases sharing
    one would make either's failure text indistinguishable from the other's.

    The rejection comes from ``_reject_reserved_suffix``, which fires on the
    second case's *name* before any id is computed — not from the ``seen_ids``
    uniqueness check in ``discover_cases``, which no input can reach while the
    name guard stands. This test asserts which mechanism fired, so that
    removing the name guard fails here rather than silently falling through to
    a check that would then be doing the work unannounced.
    """
    path = _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double
cases:
  - name: basic rate
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      value: 14.0
    rounding: source_rounds_to_dollar
  - name: basic rate (dollar)
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
    assert "basic rate (dollar)" in message
    assert "reserved" in message, message
    assert "is used by more than one case" not in message, message


def test_case_name_ending_in_a_reserved_suffix_fails_discovery(tmp_path: Path) -> None:
    """A name that merely *looks* suffixed is rejected regardless of its own rounding.

    Left unchecked, a case named this way would advertise loose rounding in
    every run's output while actually running at whatever ``rounding`` it
    declares (cent, here, since none is set), making the suffix untrustworthy
    as evidence of anything.
    """
    path = _write(
        tmp_path,
        "synthetic.yaml",
        _well_formed_case("looks loosened (dollar)", expected=14.0),
    )

    with pytest.raises(GoldenCaseError) as excinfo:
        discover_cases(tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "looks loosened (dollar)" in message
    assert "reserved" in message


@pytest.mark.parametrize("suffix", ["dollar", "ten_dollars", "monthly_cent_x12"])
def test_case_name_ending_in_any_reserved_suffix_fails_discovery(
    tmp_path: Path, suffix: str
) -> None:
    _write(
        tmp_path,
        "synthetic.yaml",
        _well_formed_case(f"a case named like it is loose ({suffix})", expected=14.0),
    )

    with pytest.raises(GoldenCaseError):
        discover_cases(tmp_path)


# --- tolerance is a derived property, not an independently settable field ----


def test_golden_case_cannot_be_constructed_with_an_explicit_tolerance() -> None:
    """The inconsistent state — a tolerance that disagrees with ``rounding`` — is unrepresentable.

    Before this was a property, ``GoldenCase(..., tolerance=1e9)`` would
    construct without complaint and ``run_case`` would honour the bogus
    tolerance; that escape hatch existed one level above any case file, in
    code that hand-builds a ``GoldenCase`` directly.
    """
    with pytest.raises(TypeError):
        GoldenCase(
            file=Path("synthetic.yaml"),
            name="mismatched tolerance",
            target=f"{_THIS_MODULE}.double",
            source="synthetic",
            checked=dt.date(2026, 1, 1),
            expected={"value": 14.0},
            tolerance=1e9,  # type: ignore[call-arg]
        )


def test_rounding_tolerances_table_is_immutable() -> None:
    with pytest.raises(TypeError):
        ROUNDING_TOLERANCES["source_rounds_to_cent"] = 1e9  # type: ignore[index]


def test_rounding_id_suffixes_table_is_immutable() -> None:
    with pytest.raises(TypeError):
        _ROUNDING_ID_SUFFIXES["source_rounds_to_dollar"] = "x"  # type: ignore[index]


# --- named-output failure labels read naturally, suffix trailing the output --


def double_named_off_by_twenty_cents(x: float) -> dict[str, float]:
    """Same shape as ``double_named``, but deliberately wrong by 0.2 for the message test."""
    return {"doubled": x * 2.0 + 0.2}


def test_named_output_failure_message_keeps_output_name_adjacent_to_case_name(
    tmp_path: Path,
) -> None:
    """The rounding suffix must trail ``[output_name]``, not sit in front of it.

    ``case.id`` alone (``name (monthly_cent_x12)``) is not what a
    named-output failure should render, because gluing ``[doubled]`` onto the
    end of that reads as ``name (monthly_cent_x12)[doubled]``, with the
    suffix wedged between the name and the output it has nothing to do with.
    The 0.2 mismatch is deliberately larger than ``monthly_cent_times_twelve``'s
    0.06 tolerance, so the case fails and there is a message to inspect.
    """
    _write(
        tmp_path,
        "synthetic.yaml",
        f"""
target: {_THIS_MODULE}.double_named_off_by_twenty_cents
cases:
  - name: named output with loose rounding
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      x: 7.0
    expected:
      doubled: 14.0
    rounding: monthly_cent_times_twelve
""",
    )
    cases = discover_cases(tmp_path)

    with pytest.raises(AssertionError) as excinfo:
        run_case(cases[0])

    message = str(excinfo.value)
    assert "[doubled] (monthly_cent_x12)" in message
    assert "(monthly_cent_x12)[doubled]" not in message


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


@pytest.mark.parametrize("inflation", [0.0, 0.25])
def test_resolve_real_params_returns_a_real_paramset_at_the_stated_rate(
    inflation: float,
) -> None:
    """Identity fields only — never a tax number out of the real ``federal.yaml``.

    Loading ``params/2026/federal.yaml`` here is fine because nothing from it
    is asserted; only ``.name``, ``.year``, and ``.inflation_rate`` are
    checked. Run once at 0.0 and once at a non-zero rate: a harness that
    ignored ``inflation`` and always built the view at, say, zero would pass
    every 0.0 case and still pass an ``isinstance`` check, so the non-zero run
    is what proves the case's own stated rate actually reaches the view.
    """
    result = resolve_real_params({"year": 2026, "file": "federal", "inflation": inflation})

    assert isinstance(result, RealParamSet)
    assert result.name == "federal"
    assert result.year == 2026
    assert result.inflation_rate == inflation


def test_resolve_inputs_dispatches_params_and_leaves_others_alone() -> None:
    resolved = resolve_inputs({"x": 7.0, "params": {"year": 2026, "file": "federal"}})

    assert resolved["x"] == 7.0
    assert isinstance(resolved["params"], ParamSet)


def test_resolve_inputs_dispatches_real_params_under_the_params_key(
    synthetic_params_root: Path,
) -> None:
    """A ``real_params`` input resolves under the keyword ``"params"``, not its own name.

    Every engine function names its parameter-set argument ``params``, so a
    case must be able to hand a ``RealParamSet`` to that name directly — see
    the module docstring. The rate is non-zero, and checked: a harness that
    silently dropped the case's stated ``inflation`` and always built the
    view at zero would still pass an ``isinstance`` check.
    """
    rate = 0.25
    resolved = resolve_inputs(
        {
            "x": 7.0,
            "real_params": {
                "year": _SYNTHETIC_REAL_PARAMS_YEAR,
                "file": "federal",
                "inflation": rate,
            },
        }
    )

    assert resolved["x"] == 7.0
    assert isinstance(resolved["params"], RealParamSet)
    assert resolved["params"].inflation_rate == rate
    assert "real_params" not in resolved


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

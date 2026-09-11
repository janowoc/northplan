# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the parameter loader — the guardrail behind the parameter rule.

The rule "never invent a tax parameter" is only as strong as the loader's
refusal to produce a value nobody wrote down. These tests assert that refusal
directly, including the absence of any defaulting API, because an added
``default=`` argument would quietly turn every missing parameter into a
plausible wrong number.
"""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path
from typing import Final

import pytest

from engine.params.loader import (
    DEFAULT_PARAMS_ROOT,
    MalformedParamFileError,
    MissingParameterError,
    ParamError,
    ParamFileMissingError,
    ParamSet,
    ParamYearMissingError,
    load_year,
)

#: Scaffolding marker left on every unverified line of a draft parameter file.
#: Spelled by concatenation so that this file does not itself trip the scan.
MARKER = "PLACE" + "HOLDER"

#: The parameter sets that must exist for 2026. Recording the names is
#: deliberate friction: dropping a file into a year directory without adding
#: its name here fails, which is the point. ``"mortality"`` arrived with issue
#: 5; the template at the ``params/`` root stays out of ``load_year``'s reach
#: and is not a member.
EXPECTED_2026_SETS: Final[tuple[str, ...]] = (
    "ab",
    "cpp",
    "federal",
    "mortality",
    "oas",
    "resp",
    "rrif",
    "tfsa",
)


def _files_with_markers(root: Path) -> list[str]:
    """Every ``path:line: text`` under ``root``'s year directories still marked.

    Scoped to the year directories because those are the only files
    ``load_year`` reads. Takes the root as an argument so the scan itself can
    be exercised against a tree that deliberately contains an offender.
    """
    year_directories = sorted(
        path for path in root.iterdir() if path.is_dir() and path.name.isdigit()
    )
    assert year_directories, f"No parameter year directories under {root}."
    return [
        f"{path.relative_to(root)}:{number}: {line.strip()}"
        for directory in year_directories
        for path in sorted(directory.rglob("*.yaml"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if MARKER in line
    ]


@pytest.fixture
def params_root(tmp_path: Path) -> Path:
    """A params tree with one populated year and one stub year."""
    year = tmp_path / "2030"
    year.mkdir()
    (year / "federal.yaml").write_text(
        "# header comment\n"
        "basic_personal_amount: 1234.5  # <source-url> checked 2030-01-01\n"
        "brackets:\n"
        "  edges: [0, 100, 200]\n"
        "  rates: [0.1, 0.2, 0.3]\n"
        "indexed: true\n"
        "name: federal\n",
        encoding="utf-8",
    )
    (year / "ab.yaml").write_text("# stub, no values yet\n", encoding="utf-8")
    return tmp_path


# --- Loading -----------------------------------------------------------------


def test_loads_every_yaml_file_in_the_year_directory(params_root: Path) -> None:
    year = load_year(2030, params_root)
    assert year.names() == ("ab", "federal")
    assert year.year == 2030


def test_missing_year_raises_named_error(params_root: Path) -> None:
    with pytest.raises(ParamYearMissingError) as exc:
        load_year(1999, params_root)
    assert "1999" in str(exc.value)
    assert "2030" in str(exc.value), "The error should list the years that do exist."


def test_missing_file_raises_named_error(params_root: Path) -> None:
    year = load_year(2030, params_root)
    with pytest.raises(ParamFileMissingError) as exc:
        year["bc"]
    assert "bc" in str(exc.value)
    assert "ab" in str(exc.value), "The error should list the files that do exist."


def test_malformed_file_raises(tmp_path: Path) -> None:
    year = tmp_path / "2030"
    year.mkdir()
    (year / "federal.yaml").write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(MalformedParamFileError):
        load_year(2030, tmp_path)


def test_non_utf8_file_raises_malformed_naming_that_file(tmp_path: Path) -> None:
    """A hand-edited file saved in the wrong encoding names itself, not `UnicodeDecodeError`.

    The bad file (``federal.yaml``) sorts between two valid ones (``ab.yaml``
    and ``on.yaml``) in ``load_year``'s ``sorted(glob(...))`` order, so the
    test shows the loader naming the file that actually failed rather than
    the first or last one it happened to read.
    """
    year = tmp_path / "2030"
    year.mkdir()
    (year / "ab.yaml").write_text("x: 1\n", encoding="utf-8")
    bad_file = year / "federal.yaml"
    bad_file.write_bytes("# Québec\nx: 1\n".encode("cp1252"))
    (year / "on.yaml").write_text("x: 1\n", encoding="utf-8")

    with pytest.raises(MalformedParamFileError) as excinfo:
        load_year(2030, tmp_path)

    message = str(excinfo.value)
    assert str(bad_file) in message
    assert "UTF-8" in message
    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)
    assert "ab.yaml" not in message
    assert "on.yaml" not in message


def test_every_loader_error_is_a_param_error() -> None:
    """One exception root, so a caller can catch the whole category."""
    for error in (
        MissingParameterError,
        ParamFileMissingError,
        ParamYearMissingError,
        MalformedParamFileError,
    ):
        assert issubclass(error, ParamError)


# --- The refusal to invent a value -------------------------------------------


def test_missing_parameter_raises_rather_than_returning_none(params_root: Path) -> None:
    federal = load_year(2030, params_root)["federal"]
    with pytest.raises(MissingParameterError):
        federal.get("age_amount")


def test_missing_parameter_error_names_the_file_and_the_key(params_root: Path) -> None:
    """The error has to tell a human which file to edit, or it will be worked around."""
    federal = load_year(2030, params_root)["federal"]
    with pytest.raises(MissingParameterError) as exc:
        federal.get("age_amount")
    message = str(exc.value)
    assert "federal.yaml" in message
    assert "age_amount" in message
    assert "source" in message, "The error should say where a real value comes from."
    assert "Never substitute" in message, (
        "The error should close off the estimated-value escape hatch explicitly."
    )


def test_stub_file_lookup_raises_and_says_it_is_a_stub(params_root: Path) -> None:
    """An empty file is a valid stub, not a load failure — but reading it still raises."""
    alberta = load_year(2030, params_root)["ab"]
    assert alberta.is_stub()
    with pytest.raises(MissingParameterError) as exc:
        alberta.get("brackets.rates")
    assert "stub" in str(exc.value).lower()


def test_the_repository_params_load_and_are_populated() -> None:
    """The real 2026 files parse and hold values.

    This replaced an assertion that every file was still an empty stub, which
    guarded the period before any value had been entered by hand. That period
    ended when the 2026 files were populated. What is worth guarding now is
    that no file silently reverts to empty — a stub reads as "parameter
    missing" at every lookup, which is loud, but a file emptied by a bad merge
    should fail here rather than at the first simulation — and that the set of
    jurisdictions is deliberate rather than whatever happens to be on disk.

    Adding a province or a program is therefore a two-part change: the file,
    and this tuple. That is the intended friction. The tuple itself lives in
    ``EXPECTED_2026_SETS`` so that a change to it is visible at the top of this
    module rather than buried in an assertion.
    """
    year = load_year(2026, DEFAULT_PARAMS_ROOT)
    assert year.names() == EXPECTED_2026_SETS
    assert not any(year[name].is_stub() for name in year.names())


def test_no_repository_param_file_contains_a_placeholder_marker() -> None:
    """No value under ``params/`` is still carrying its scaffolding marker.

    Draft parameter files are written with implausible repdigit values and a
    trailing ``# PLACEHOLDER`` on every unverified line, so that
    ``grep -c PLACEHOLDER`` counts the work remaining. A file promoted into
    ``params/`` with markers still on it is the failure mode that workflow
    exists to prevent: the numbers around them look finished, and a plausible
    fake reads exactly like a verified value at the call site.

    This asserts the last step of that workflow rather than trusting it to be
    remembered. It scans the text, not the loaded values, because the marker
    lives in a YAML comment and the loader discards comments by design.

    Scoped to the year directories, which are the only files ``load_year``
    reads. A draft parked at the ``params/`` root — ``gis_not_implemented.yaml``
    is one — is unreachable by the engine and is *expected* to be full of
    markers; that is what makes it a draft. The line this test draws is the one
    that matters: a file is allowed to be unfinished right up until it is moved
    into a year directory, and from that moment it must be clean.
    """
    offenders = _files_with_markers(DEFAULT_PARAMS_ROOT)
    assert not offenders, (
        "Placeholder markers remain in files under params/. Verify each value "
        "against its source, then replace the number and the marker together, "
        "leaving a comment with the source URL and check date:\n" + "\n".join(offenders)
    )


def test_the_marker_scan_catches_a_template_promoted_into_a_year_directory(
    tmp_path: Path,
) -> None:
    """The scan above only means something if it can fail, so make it fail.

    Copying ``params/mortality-template.yaml`` into a year directory with its
    markers still on it is the exact mistake issue 5 could make: the file loads,
    every structural test passes, and every number in it is fake. This proves
    the guard fires on that.
    """
    year = tmp_path / "2030"
    year.mkdir()
    shutil.copyfile(DEFAULT_PARAMS_ROOT / "mortality-template.yaml", year / "mortality.yaml")

    offenders = _files_with_markers(tmp_path)
    assert offenders, "A template full of markers was copied in and the scan found nothing."
    assert any("mortality.yaml" in offender for offender in offenders), (
        f"The scan must name the offending file; it reported {offenders[:3]}."
    )


def test_get_has_no_default_argument() -> None:
    """No defaulting API, at all.

    A ``default=`` on any lookup would make "the parameter is missing" a silent
    condition. This asserts the absence structurally so it cannot be added back
    without a test failing.
    """
    for method in (ParamSet.get, ParamSet.number, ParamSet.sequence, ParamSet.numbers):
        parameters = set(inspect.signature(method).parameters)
        assert parameters == {"self", "path"}, (
            f"{method.__qualname__} must take only (self, path). A default value "
            f"argument would let a caller proceed without a hand-verified parameter."
        )


def test_loader_module_exposes_no_defaulting_helper() -> None:
    import engine.params.loader as loader

    for name in ("get_or", "get_default", "getdefault", "fallback"):
        assert not hasattr(loader, name), f"{name!r} would undermine the parameter rule."


# --- Lookup behaviour --------------------------------------------------------


def test_nested_lookup_by_dotted_path(params_root: Path) -> None:
    federal = load_year(2030, params_root)["federal"]
    assert federal.get("brackets.rates") == (0.1, 0.2, 0.3)


def test_traversing_through_a_scalar_raises(params_root: Path) -> None:
    federal = load_year(2030, params_root)["federal"]
    with pytest.raises(MissingParameterError) as exc:
        federal.get("basic_personal_amount.nested")
    assert "not a mapping" in str(exc.value)


def test_number_returns_a_float(params_root: Path) -> None:
    federal = load_year(2030, params_root)["federal"]
    assert federal.number("basic_personal_amount") == pytest.approx(1234.5)


def test_number_rejects_a_boolean(params_root: Path) -> None:
    """YAML ``true`` must never silently become ``1.0`` in a tax calculation."""
    federal = load_year(2030, params_root)["federal"]
    with pytest.raises(MalformedParamFileError):
        federal.number("indexed")


def test_number_rejects_a_string(params_root: Path) -> None:
    federal = load_year(2030, params_root)["federal"]
    with pytest.raises(MalformedParamFileError):
        federal.number("name")


def test_number_on_a_missing_key_raises_missing_not_malformed(params_root: Path) -> None:
    federal = load_year(2030, params_root)["federal"]
    with pytest.raises(MissingParameterError):
        federal.number("age_amount")


def test_numbers_returns_a_tuple_of_floats(params_root: Path) -> None:
    federal = load_year(2030, params_root)["federal"]
    edges = federal.numbers("brackets.edges")
    assert edges == (0.0, 100.0, 200.0)
    assert all(isinstance(edge, float) for edge in edges)


def test_numbers_rejects_a_scalar(params_root: Path) -> None:
    """A one-entry table is a data error worth surfacing, not a value to wrap."""
    federal = load_year(2030, params_root)["federal"]
    with pytest.raises(MalformedParamFileError):
        federal.numbers("basic_personal_amount")


def test_numbers_reports_the_offending_element_index(tmp_path: Path) -> None:
    year = tmp_path / "2030"
    year.mkdir()
    (year / "federal.yaml").write_text("rates: [0.1, oops, 0.3]\n", encoding="utf-8")
    federal = load_year(2030, tmp_path)["federal"]
    with pytest.raises(MalformedParamFileError) as exc:
        federal.numbers("rates")
    assert "element 1" in str(exc.value)


def test_has_reports_presence_without_raising(params_root: Path) -> None:
    federal = load_year(2030, params_root)["federal"]
    assert federal.has("basic_personal_amount")
    assert not federal.has("age_amount")


# --- Immutability ------------------------------------------------------------


def test_param_set_is_frozen(params_root: Path) -> None:
    federal = load_year(2030, params_root)["federal"]
    with pytest.raises((AttributeError, TypeError)):
        federal.name = "tampered"  # type: ignore[misc]


def test_values_cannot_be_mutated(params_root: Path) -> None:
    """Parameters are shared across every path and every policy evaluation."""
    federal = load_year(2030, params_root)["federal"]
    with pytest.raises(TypeError):
        federal.values["basic_personal_amount"] = 0  # type: ignore[index]


def test_nested_mappings_cannot_be_mutated(params_root: Path) -> None:
    federal = load_year(2030, params_root)["federal"]
    with pytest.raises(TypeError):
        federal.get("brackets")["rates"] = ()  # type: ignore[index]


def test_sequences_load_as_tuples(params_root: Path) -> None:
    federal = load_year(2030, params_root)["federal"]
    assert isinstance(federal.get("brackets.edges"), tuple)


# --- Convenience accessors ---------------------------------------------------


def test_named_accessors_reach_the_right_files(params_root: Path) -> None:
    year = load_year(2030, params_root)
    assert year.federal.name == "federal"
    assert year.province("AB").name == "ab", "Province codes are case-insensitive."


def test_named_accessor_for_a_missing_file_raises(params_root: Path) -> None:
    year = load_year(2030, params_root)
    with pytest.raises(ParamFileMissingError):
        _ = year.cpp

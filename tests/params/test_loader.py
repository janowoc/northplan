"""Tests for the parameter loader — the guardrail behind the parameter rule.

The rule "never invent a tax parameter" is only as strong as the loader's
refusal to produce a value nobody wrote down. These tests assert that refusal
directly, including the absence of any defaulting API, because an added
``default=`` argument would quietly turn every missing parameter into a
plausible wrong number.
"""

from __future__ import annotations

import inspect
from pathlib import Path

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
    assert "VERIFICATION" in message, "The error should point at the verification ledger."


def test_stub_file_lookup_raises_and_says_it_is_a_stub(params_root: Path) -> None:
    """An empty file is a valid stub, not a load failure — but reading it still raises."""
    alberta = load_year(2030, params_root)["ab"]
    assert alberta.is_stub()
    with pytest.raises(MissingParameterError) as exc:
        alberta.get("brackets.rates")
    assert "stub" in str(exc.value).lower()


def test_the_repository_params_load_and_are_all_stubs() -> None:
    """The real 2026 files parse, and none has been populated by anything but a human.

    This will start failing the moment a value is added, at which point the
    assertion below is the thing to update — deliberately, by a human, alongside
    a row in docs/VERIFICATION.md.
    """
    year = load_year(2026, DEFAULT_PARAMS_ROOT)
    assert year.names() == ("ab", "cpp", "federal", "oas", "rrif")
    assert all(year[name].is_stub() for name in year.names())


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

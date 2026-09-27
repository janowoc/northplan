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

import datetime
import inspect
import shutil
import sys
from pathlib import Path
from typing import Final

import pytest
import yaml

from engine.params.loader import (
    DEFAULT_PARAMS_ROOT,
    DuplicateYamlKeyError,
    MalformedParamFileError,
    MissingParameterError,
    ParamError,
    ParamFileMissingError,
    ParamSet,
    ParamYearMissingError,
    YamlConstructionError,
    load_year,
    parse_yaml,
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


# --- Repeated keys ------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "line", "key"),
    [
        pytest.param("71: a\n71: b\n", 2, 71, id="int"),
        pytest.param("1.5: a\n1.5: b\n", 2, 1.5, id="float"),
        pytest.param("yes: a\ntrue: b\n", 2, True, id="bool"),
        pytest.param("~: a\nnull: b\n", 2, None, id="null"),
        pytest.param("2026-01-01: a\n2026-01-01: b\n", 2, datetime.date(2026, 1, 1), id="date"),
        pytest.param("!!binary aGk=: a\n!!binary aGk=: b\n", 2, b"hi", id="binary"),
        pytest.param("a: &k foo\n*k : 1\nfoo: 2\n", 3, "foo", id="anchor-alias"),
        pytest.param("71: a\n71.0: b\n", 2, 71.0, id="int-then-float"),
        pytest.param("1: a\ntrue: b\n", 2, True, id="int-then-bool"),
    ],
)
def test_parse_yaml_refuses_a_repeat_under_each_scalar_tag(
    text: str, line: int, key: object
) -> None:
    with pytest.raises(DuplicateYamlKeyError) as excinfo:
        parse_yaml(text)

    message = str(excinfo.value)
    assert f"line {line}" in message
    assert f"key {key!r} appears" in message


def test_repeated_top_level_key_raises_malformed_naming_line_and_key(tmp_path: Path) -> None:
    year = tmp_path / "2030"
    year.mkdir()
    path = year / "federal.yaml"
    path.write_text(
        "zzz_synthetic_key: 1\nzzz_synthetic_key: 2\n",
        encoding="utf-8",
    )

    with pytest.raises(MalformedParamFileError) as excinfo:
        load_year(2030, tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "line 2" in message
    assert repr("zzz_synthetic_key") in message
    assert "not valid YAML" not in message
    assert isinstance(excinfo.value.__cause__, DuplicateYamlKeyError)


def test_repeated_key_in_a_nested_mapping_raises_malformed_naming_line_and_key(
    tmp_path: Path,
) -> None:
    year = tmp_path / "2030"
    year.mkdir()
    path = year / "federal.yaml"
    path.write_text(
        "outer:\n  zzz_synthetic_key: 1\n  zzz_synthetic_key: 2\n",
        encoding="utf-8",
    )

    with pytest.raises(MalformedParamFileError) as excinfo:
        load_year(2030, tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "line 3" in message
    assert repr("zzz_synthetic_key") in message
    assert "not valid YAML" not in message
    assert isinstance(excinfo.value.__cause__, DuplicateYamlKeyError)


def test_repeated_age_row_in_a_factor_table_raises_malformed(tmp_path: Path) -> None:
    """A repeated integer age row is refused through ``load_year``, naming the file, line and key.

    The committed tables keyed by integer age are ``rrif.yaml``'s
    ``rrif.minimum_factors.by_age``, ``ab.yaml``'s ``lif.maximum_factors.by_age``
    and ``mortality.yaml``'s ``q_x.f`` and ``q_x.m``.
    """
    year = tmp_path / "2030"
    year.mkdir()
    path = year / "rrif.yaml"
    path.write_text("factors:\n  71: 0.5\n  72: 0.6\n  71: 0.7\n", encoding="utf-8")

    with pytest.raises(MalformedParamFileError) as excinfo:
        load_year(2030, tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "line 4" in message
    assert f"key {71!r} appears" in message


def test_same_key_in_two_sibling_mappings_loads_both_values(tmp_path: Path) -> None:
    year = tmp_path / "2030"
    year.mkdir()
    (year / "federal.yaml").write_text(
        "a: {zzz_synthetic_key: 1}\nb: {zzz_synthetic_key: 2}\n",
        encoding="utf-8",
    )

    federal = load_year(2030, tmp_path)["federal"]

    assert federal.get("a.zzz_synthetic_key") == 1
    assert federal.get("b.zzz_synthetic_key") == 2


def test_duplicate_yaml_key_error_is_a_value_error_not_a_yaml_error() -> None:
    assert issubclass(DuplicateYamlKeyError, ValueError)
    assert not issubclass(DuplicateYamlKeyError, yaml.YAMLError)


def test_parse_yaml_syntax_error_raises_yaml_error_not_duplicate_key_error() -> None:
    with pytest.raises(yaml.YAMLError) as excinfo:
        parse_yaml("a: [unterminated\n")
    assert not isinstance(excinfo.value, DuplicateYamlKeyError)
    assert not isinstance(excinfo.value, YamlConstructionError)


def test_parse_yaml_reports_a_construction_error_ahead_of_a_later_syntax_error() -> None:
    """Pins the docstring's claim: a construction error is possibly reported
    ahead of a syntax error later in the same text.

    ``yaml.safe_load`` of the same text raises a ``yaml.YAMLError`` at the
    unterminated flow node in ``b`` — not a :class:`YamlConstructionError` —
    while ``parse_yaml`` never gets that far, because building the malformed
    ``!!int`` key fails first.
    """
    text = "a:\n  !!int abc: 1\nb: [\n"

    with pytest.raises(YamlConstructionError) as excinfo:
        parse_yaml(text)
    assert isinstance(excinfo.value.__cause__, ValueError)

    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(text)


@pytest.mark.parametrize(
    ("text", "cause_type"),
    [
        pytest.param("x: !!int abc\n", ValueError, id="int"),
        pytest.param("x: !!bool maybe\n", KeyError, id="bool"),
        pytest.param("x: !!timestamp abc\n", AttributeError, id="timestamp"),
        pytest.param("rate: !!float\n", IndexError, id="float-no-value"),
        pytest.param('x: !!int ""\n', IndexError, id="int-empty-string"),
        pytest.param("x: !!float 1" + ":1" * 400 + "\n", OverflowError, id="base-60-overflow"),
        pytest.param("[" * (sys.getrecursionlimit() + 1), RecursionError, id="deep-nesting"),
    ],
)
def test_parse_yaml_wraps_a_bare_construction_error_once(
    text: str, cause_type: type[Exception]
) -> None:
    with pytest.raises(YamlConstructionError) as excinfo:
        parse_yaml(text)

    assert isinstance(excinfo.value, yaml.YAMLError)
    assert isinstance(excinfo.value.__cause__, cause_type)


def test_parse_yaml_does_not_wrap_a_base_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """``parse_yaml``'s ``except Exception`` must not catch a ``BaseException``.

    ``KeyboardInterrupt`` is not an ``Exception``, so it passes through the
    ``except Exception`` wrap unchanged. A guard widened to ``BaseException``
    would turn it into a ``YamlConstructionError``.
    """
    import engine.params.loader as loader

    def _raise_keyboard_interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(loader._UniqueKeyLoader, "compose_mapping_node", _raise_keyboard_interrupt)

    with pytest.raises(KeyboardInterrupt):
        parse_yaml("a: 1\n")


def test_parse_yaml_reports_a_nested_repeat_ahead_of_a_later_syntax_error() -> None:
    """A repeated key in a nested mapping is refused before a later syntax error.

    ``yaml.safe_load`` of the same text raises a ``yaml.YAMLError`` at the
    unterminated flow node in ``c``; ``parse_yaml`` never gets that far,
    because the repeated key ``b`` is refused first.
    """
    text = "a:\n  b: 1\n  b: 2\nc: [\n"

    with pytest.raises(DuplicateYamlKeyError):
        parse_yaml(text)

    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(text)


def test_parse_yaml_reports_an_alias_key_repeat_at_the_anchor() -> None:
    """Pins documented behaviour: a repeat made through an alias key is
    reported at the anchor's location, not the alias's.
    """
    text = "&k foo: 1\nx: 0\n*k : 2\n"

    with pytest.raises(DuplicateYamlKeyError) as excinfo:
        parse_yaml(text)

    message = str(excinfo.value)
    assert "line 1" in message
    assert repr("foo") in message


def test_parse_yaml_unhashable_key_raises_yaml_error_not_type_error() -> None:
    with pytest.raises(yaml.YAMLError) as excinfo:
        parse_yaml("? [a, b]\n: 1\n")
    assert not isinstance(excinfo.value, YamlConstructionError)


def test_load_year_on_a_file_with_an_unhashable_key_raises_malformed_naming_that_file(
    tmp_path: Path,
) -> None:
    year = tmp_path / "2030"
    year.mkdir()
    path = year / "federal.yaml"
    path.write_text("? [a, b]\n: 1\n", encoding="utf-8")

    with pytest.raises(MalformedParamFileError) as excinfo:
        load_year(2030, tmp_path)

    assert str(path) in str(excinfo.value)


@pytest.mark.parametrize(
    ("text", "cause_type"),
    [
        pytest.param("x: !!int abc\n", ValueError, id="int"),
        pytest.param("x: !!bool maybe\n", KeyError, id="bool"),
        pytest.param("x: !!timestamp abc\n", AttributeError, id="timestamp"),
        pytest.param("by_age:\n  2026-13-45: 1\n", ValueError, id="out-of-range-date-key"),
        pytest.param("rate: !!float\n", IndexError, id="float-no-value"),
    ],
)
def test_load_year_on_a_construction_error_raises_malformed_naming_that_file(
    tmp_path: Path, text: str, cause_type: type[Exception]
) -> None:
    """PyYAML's scalar constructors raise a bare ``ValueError``, ``KeyError``,
    ``AttributeError`` or ``IndexError`` on a malformed tagged or date scalar,
    with no filename attached; ``load_year`` must still name the file rather
    than let one escape. ``parse_yaml`` converts the bare exception to
    :class:`YamlConstructionError` once, so it is that class's ``__cause__``
    that carries the original.
    """
    year = tmp_path / "2030"
    year.mkdir()
    path = year / "federal.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(MalformedParamFileError) as excinfo:
        load_year(2030, tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "not valid YAML" in message
    assert isinstance(excinfo.value.__cause__, YamlConstructionError)
    assert isinstance(excinfo.value.__cause__.__cause__, cause_type)


def test_parse_yaml_allows_a_merge_key_matching_safe_load() -> None:
    text = (
        "base: &base\n"
        "  zzz_synthetic_key: 1\n"
        "  other_synthetic_key: 2\n"
        "merged:\n"
        "  <<: *base\n"
        "  other_synthetic_key: 3\n"
    )
    assert parse_yaml(text) == yaml.safe_load(text)


def test_parse_yaml_still_refuses_an_explicit_repeat_alongside_a_merge_key() -> None:
    text = (
        "base: &base\n"
        "  zzz_synthetic_key: 1\n"
        "  other_synthetic_key: 2\n"
        "merged:\n"
        "  <<: *base\n"
        "  other_synthetic_key: 3\n"
        "  other_synthetic_key: 4\n"
    )
    with pytest.raises(DuplicateYamlKeyError) as excinfo:
        parse_yaml(text)

    message = str(excinfo.value)
    assert "line 7" in message
    assert repr("other_synthetic_key") in message


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "a: &a {k: 1}\nx:\n  b: &b\n    <<: *a\n    k: 2\nc:\n  <<: *b\n  k: 3\n",
            id="nested merge source",
        ),
        pytest.param(
            "a: &a {k: 1}\nx:\n  - &b\n    <<: *a\n    k: 2\nc:\n  <<: [*b]\n  k: 3\n",
            id="merge source in a sequence",
        ),
    ],
)
def test_parse_yaml_matches_safe_load_when_a_merge_source_is_itself_a_merge(
    text: str,
) -> None:
    """A merge source built by its own merge must not look like a repeat.

    ``flatten_mapping`` rewrites a mapping node in place, and it builds nested
    mappings lazily. A duplicate check that runs after that rewrite can see a
    merge source's own flattened key alongside its override and refuse a
    document ``safe_load`` accepts.
    """
    assert parse_yaml(text) == yaml.safe_load(text)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("m:\n  <<: {x: 1, x: 2}\n  y: 0\n", id="inline merge source"),
        pytest.param("m:\n  <<: [{x: 1, x: 2}]\n", id="merge source in a sequence"),
    ],
)
def test_parse_yaml_refuses_a_repeat_inside_a_merge_source(text: str) -> None:
    """A repeat inside the mapping a merge key points at is still a repeat."""
    with pytest.raises(DuplicateYamlKeyError) as excinfo:
        parse_yaml(text)

    message = str(excinfo.value)
    assert "line 2" in message
    assert repr("x") in message


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("=: 1\n", id="= at the top level"),
        pytest.param("a:\n  =: 1\n", id="= in a nested mapping"),
    ],
)
def test_parse_yaml_matches_safe_load_for_a_bare_equals_key(text: str) -> None:
    """A bare ``=`` key loads as it does under ``safe_load``.

    PyYAML tags a bare ``=`` key ``tag:yaml.org,2002:value``, which SafeLoader
    has no constructor for until ``flatten_mapping`` rewrites it. The check
    must not construct it directly.
    """
    assert parse_yaml(text) == yaml.safe_load(text)


def test_parse_yaml_matches_safe_load_with_two_merge_keys_in_one_mapping() -> None:
    """Two ``<<`` keys in one mapping are allowed; neither is a repeat."""
    text = "a: &a {x: 1}\nb: &b {y: 2}\nm:\n  <<: *a\n  <<: *b\n"

    expected = {"a": {"x": 1}, "b": {"y": 2}, "m": {"x": 1, "y": 2}}
    assert yaml.safe_load(text) == expected
    assert parse_yaml(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("!!map a: 1\n", id="map-tagged key"),
        pytest.param("!!set a: 1\n", id="set-tagged key"),
        pytest.param("x: {!!seq a: 1}\n", id="seq-tagged key, nested"),
    ],
)
def test_parse_yaml_on_a_collection_tagged_key_raises_yaml_error_not_type_error(
    text: str,
) -> None:
    """A scalar key with a collection tag builds an unhashable object.

    ``safe_load`` refuses it with ``ConstructorError``. The check must not let
    a bare ``TypeError`` from ``key in seen`` escape instead.
    """
    with pytest.raises(yaml.YAMLError):
        parse_yaml(text)


def test_load_year_on_a_collection_tagged_key_raises_malformed_naming_that_file(
    tmp_path: Path,
) -> None:
    year = tmp_path / "2030"
    year.mkdir()
    path = year / "federal.yaml"
    path.write_text("!!map a: 1\n", encoding="utf-8")

    with pytest.raises(MalformedParamFileError) as excinfo:
        load_year(2030, tmp_path)

    assert str(path) in str(excinfo.value)


def test_parse_yaml_matches_safe_load_with_two_value_tagged_keys() -> None:
    text = "!!value a: 1\n!!value b: 2\n"
    assert parse_yaml(text) == yaml.safe_load(text)


def test_parse_yaml_refuses_a_value_tagged_key_matching_a_plain_one_by_text() -> None:
    """A ``!!value`` key is compared by its text, not the literal ``"="``."""
    text = "!!value a: 1\na: 2\n"
    with pytest.raises(DuplicateYamlKeyError) as excinfo:
        parse_yaml(text)

    message = str(excinfo.value)
    assert "line 2" in message
    assert repr("a") in message


def test_top_level_key_collision_after_str_cast_raises_malformed(tmp_path: Path) -> None:
    """``"1"`` and ``1`` are different YAML keys but the same key once stored as text."""
    year = tmp_path / "2030"
    year.mkdir()
    path = year / "federal.yaml"
    text = '"1": a\n1: b\n'
    path.write_text(text, encoding="utf-8")

    with pytest.raises(MalformedParamFileError) as excinfo:
        load_year(2030, tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert repr("1") in message
    assert repr(1) in message
    assert "top level" in message
    assert f"{'1'!r} and {1!r}" in message
    assert isinstance(excinfo.value.__cause__, MalformedParamFileError)

    # parse_yaml alone, with no str-cast, sees two distinct keys.
    assert parse_yaml(text) == {"1": "a", 1: "b"}


def test_nested_key_collision_after_str_cast_names_the_path(tmp_path: Path) -> None:
    year = tmp_path / "2030"
    year.mkdir()
    path = year / "federal.yaml"
    path.write_text('outer:\n  - {"65": a, 65: b}\n', encoding="utf-8")

    with pytest.raises(MalformedParamFileError) as excinfo:
        load_year(2030, tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "outer.0" in message


def test_nested_key_collision_after_str_cast_names_the_depth_two_path(tmp_path: Path) -> None:
    year = tmp_path / "2030"
    year.mkdir()
    path = year / "federal.yaml"
    path.write_text('outer:\n  inner:\n    - {"65": a, 65: b}\n', encoding="utf-8")

    with pytest.raises(MalformedParamFileError) as excinfo:
        load_year(2030, tmp_path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "outer.inner.0" in message


def test_same_key_text_in_sibling_mappings_loads(tmp_path: Path) -> None:
    """Control: the same text in two different mappings is not a collision."""
    year = tmp_path / "2030"
    year.mkdir()
    (year / "federal.yaml").write_text(
        'a: {"65": 1}\nb: {65: 2}\n',
        encoding="utf-8",
    )

    federal = load_year(2030, tmp_path)["federal"]

    assert federal.get("a.65") == 1
    assert federal.get("b.65") == 2


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

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
import errno
import inspect
import os
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest
import yaml

from engine.params.loader import (
    DEFAULT_PARAMS_ROOT,
    DuplicateYamlKeyError,
    MalformedParamFileError,
    MissingParameterError,
    ParamDirectoryUnreadableError,
    ParamError,
    ParamFileMissingError,
    ParamSet,
    ParamYearMissingError,
    YamlConstructionError,
    check_params_root,
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


def test_a_root_that_is_a_file_is_reported_as_not_a_directory(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(ParamYearMissingError) as exc:
        load_year(2026, root)
    assert "the params root is not a directory" in str(exc.value)
    assert "itself does not exist" not in str(exc.value)


def test_a_root_that_does_not_exist_is_reported_as_missing(tmp_path: Path) -> None:
    with pytest.raises(ParamYearMissingError) as exc:
        load_year(2026, tmp_path / "absent")
    assert "the params root itself does not exist" in str(exc.value)


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

    def _raise_keyboard_interrupt(*_args: object, **_kwargs: object) -> None:
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
        ParamDirectoryUnreadableError,
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


# =============================================================================
# Merge keys, ordered maps
# =============================================================================


def _merge_bomb(levels: int) -> str:
    """Anchors ``b0`` to ``b<levels>``: each merges ten copies of the one before."""
    lines = ["b0: &b0 {" + ", ".join(f"k{j}: {j}" for j in range(10)) + "}"]
    for i in range(1, levels + 1):
        lines.append(f"b{i}: &b{i} {{<<: [{', '.join([f'*b{i - 1}'] * 10)}]}}")
    return "\n".join(lines) + "\n"


def test_parse_yaml_merge_bomb_is_linear() -> None:
    text = _merge_bomb(6)
    assert len(text) == 469

    start = time.perf_counter()
    loaded = parse_yaml(text)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0
    assert loaded["b6"] == loaded["b0"]


_MERGE_CORPUS = [
    pytest.param(
        "base: &base\n  zzz_synthetic_key: 1\n  other_synthetic_key: 2\n"
        "merged:\n  <<: *base\n  other_synthetic_key: 3\n",
        id="override after a merge",
    ),
    pytest.param(
        "a: &a {k: 1}\nx:\n  b: &b\n    <<: *a\n    k: 2\nc:\n  <<: *b\n  k: 3\n",
        id="nested merge source",
    ),
    pytest.param(
        "a: &a {k: 1}\nx:\n  - &b\n    <<: *a\n    k: 2\nc:\n  <<: [*b]\n  k: 3\n",
        id="merge source in a sequence",
    ),
    pytest.param("a: &a {x: 1}\nb: &b {y: 2}\nm:\n  <<: *a\n  <<: *b\n", id="two merge keys"),
    pytest.param(
        "a: &a {x: 1, y: 1, z: 1}\nb: &b {y: 2, w: 2}\nm:\n  <<: [*a, *b]\n",
        id="precedence across a merge list",
    ),
    pytest.param(
        "a: &a {x: 1, y: 1, z: 1}\nb: &b {y: 2, w: 2}\nm:\n  <<: [*b, *a]\n",
        id="precedence across a reversed merge list",
    ),
    pytest.param(
        "a: &a {x: 1, y: 1}\nm:\n  y: 9\n  <<: *a\n  q: 0\n",
        id="explicit key before the merge",
    ),
    pytest.param(
        "a: &a {x: 1, y: 1}\nm:\n  q: 0\n  <<: *a\n  y: 9\n",
        id="explicit key after the merge",
    ),
    pytest.param(
        "a: &a {x: 1}\nb: &b {<<: *a, y: 2}\nc: &c {<<: *b, z: 3}\nd: {<<: [*c, *a], x: 7}\n",
        id="a merge whose source is a merge",
    ),
    pytest.param(
        "A: &A {1: a}\nB: &B {1.0: b}\nx: {<<: [*B, *A], 1: c}\n",
        id="1 and 1.0 across a merge list",
    ),
    pytest.param("A: &A {~: a}\nB: &B {null: b}\nx: {<<: [*B, *A], ~: c}\n", id="~ and null"),
    pytest.param(
        "A: &A {true: a}\nB: &B {yes: b}\nx: {<<: [*B, *A], true: c}\n", id="true and yes"
    ),
    pytest.param(
        "A: &A {1: a}\nB: &B {+1: b}\nC: &C {0x1: c}\nx: {<<: [*C, *B, *A], 1: d}\n",
        id="1, +1 and 0x1",
    ),
    pytest.param(
        "m0: &m0 {1.0: v00}\nm1: &m1 {+1: v10, <<: [*m0]}\nm2: {<<: [*m0, *m1], a: z}\n",
        id="three-level spellings",
    ),
    pytest.param(_merge_bomb(1), id="bomb level 1"),
    pytest.param(_merge_bomb(2), id="bomb level 2"),
    pytest.param(_merge_bomb(3), id="bomb level 3"),
]


@pytest.mark.parametrize("text", _MERGE_CORPUS)
def test_parse_yaml_matches_safe_load_on_merges(text: str) -> None:
    """The differential oracle: same value, and same key order at every level."""
    loaded = parse_yaml(text)
    assert loaded == yaml.safe_load(text)
    assert repr(loaded) == repr(yaml.safe_load(text))


_RANDOM_KEYS = ["1", "1.0", "+1", "0x1", "true", "yes", "~", "null", "a", "'1'"]


def _random_merge_document(rng: random.Random) -> str:
    lines = []
    for m in range(4):
        keys = rng.sample(_RANDOM_KEYS, rng.randint(1, 3))
        pairs = [f"{key}: v{m}{j}" for j, key in enumerate(keys)]
        if m > 0 and rng.random() < 0.8:
            earlier = [f"*m{i}" for i in range(m)]
            chosen = rng.sample(earlier, rng.randint(1, len(earlier)))
            pairs.insert(rng.randint(0, len(pairs)), f"<<: [{', '.join(chosen)}]")
        lines.append(f"m{m}: &m{m} {{{', '.join(pairs)}}}")
    return "\n".join(lines) + "\n"


def test_parse_yaml_matches_safe_load_on_random_merges() -> None:
    """Draw documents until 2,000 are accepted; a draw with a repeat is legitimately refused.

    Most keys in the list read the same as another once built (``1``, ``1.0``, ``+1``,
    ``0x1``, ``true`` and ``yes`` are one key), so many draws are refused.
    """
    rng = random.Random(7)
    compared = 0
    for _ in range(20_000):
        text = _random_merge_document(rng)
        try:
            loaded = parse_yaml(text)
        except DuplicateYamlKeyError:
            continue
        assert repr(loaded) == repr(yaml.safe_load(text)), text
        compared += 1
        if compared == 2_000:
            break
    assert compared >= 1_000


def _tracked_yaml_files() -> list[str]:
    root = Path(__file__).resolve().parents[2]
    listing = subprocess.run(
        ["git", "ls-files", "*.yaml", "*.yml"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return [str(root / name) for name in listing]


@pytest.mark.parametrize("path", _tracked_yaml_files())
def test_parse_yaml_matches_safe_load_on_every_tracked_yaml(path: str) -> None:
    text = Path(path).read_text(encoding="utf-8")
    assert repr(parse_yaml(text)) == repr(yaml.safe_load(text))


def test_tracked_yaml_listing_is_not_empty() -> None:
    assert len(_tracked_yaml_files()) > 10


def test_parse_yaml_refuses_a_repeated_key_in_an_omap() -> None:
    with pytest.raises(DuplicateYamlKeyError) as excinfo:
        parse_yaml("o: !!omap [{a: 1}, {a: 2}]\n")

    message = str(excinfo.value)
    assert "line 1" in message
    assert repr("a") in message
    assert "ordered map" in message


def test_parse_yaml_refuses_an_omap_repeat_that_reads_the_same() -> None:
    with pytest.raises(DuplicateYamlKeyError) as excinfo:
        parse_yaml("o: !!omap [{1: x}, {1.0: y}]\n")

    assert "ordered map" in str(excinfo.value)


def test_parse_yaml_keeps_an_omap_without_repeats() -> None:
    text = "o: !!omap [{a: 1}, {b: 2}, {c: 3}]\n"
    assert parse_yaml(text) == yaml.safe_load(text)
    assert parse_yaml(text) == {"o": [("a", 1), ("b", 2), ("c", 3)]}


def test_parse_yaml_keeps_repeated_keys_in_pairs() -> None:
    text = "o: !!pairs [{a: 1}, {a: 2}]\n"
    assert parse_yaml(text) == yaml.safe_load(text)
    assert parse_yaml(text) == {"o": [("a", 1), ("a", 2)]}


def test_omap_override_leaves_safe_load_alone() -> None:
    assert yaml.safe_load("o: !!omap [{a: 1}, {a: 2}]\n") == {"o": [("a", 1), ("a", 2)]}


# --- Directories the operating system will not read ---------------------------

DENIED = os.strerror(errno.EACCES)

Lock = Callable[[Path, int], None]


def _assert_refused(action: Callable[[], object]) -> None:
    """The operating system, not the loader, refuses ``action`` with a ``PermissionError``."""
    with pytest.raises(PermissionError):
        action()


def _lookup_message(year: int | str, root: Path) -> str:
    return f"Cannot look up tax year {year} in the parameters directory {root}: {DENIED}."


def _listing_message(directory: Path) -> str:
    return f"Cannot list the parameters directory {directory}: {DENIED}."


def _root_with_a_year(tmp_path: Path, params_root: Path, name: str = "root") -> Path:
    """A fresh root inside ``tmp_path``, which is also the fixture's root.

    It holds a copy of the fixture's 2030 directory.
    """
    root = tmp_path / name
    shutil.copytree(params_root / "2030", root / "2030")
    return root


def test_a_root_that_cannot_be_entered_or_read_is_unreadable(params_root: Path, lock: Lock) -> None:
    lock(params_root, 0o000)
    with pytest.raises(ParamDirectoryUnreadableError) as exc:
        load_year(2030, params_root)
    assert str(exc.value) == _lookup_message(2030, params_root)


def test_a_root_that_can_be_read_but_not_entered_is_unreadable(
    params_root: Path, lock: Lock
) -> None:
    lock(params_root, 0o444)
    with pytest.raises(ParamDirectoryUnreadableError) as exc:
        load_year(2030, params_root)
    assert str(exc.value) == _lookup_message(2030, params_root)


@pytest.mark.parametrize("present", [True, False], ids=["present", "absent"])
def test_a_root_whose_parent_cannot_be_entered_is_unreadable(
    tmp_path: Path, params_root: Path, lock: Lock, present: bool
) -> None:
    parent = tmp_path / "outer"
    root = parent / "params"
    if present:
        shutil.copytree(params_root / "2030", root / "2030")
    else:
        parent.mkdir()
        assert not root.exists()
    lock(parent, 0o000)
    _assert_refused(root.stat)
    with pytest.raises(ParamDirectoryUnreadableError) as exc:
        load_year(2030, root)
    assert str(exc.value) == _lookup_message(2030, root)


def test_a_root_that_can_be_entered_but_not_read_still_loads(params_root: Path, lock: Lock) -> None:
    before = load_year(2030, params_root).names()
    assert before == ("ab", "federal")
    lock(params_root, 0o111)
    _assert_refused(lambda: list(params_root.iterdir()))
    assert load_year(2030, params_root).names() == before


@pytest.mark.parametrize("mode", [0o000, 0o111], ids=["000", "111"])
def test_a_year_directory_that_cannot_be_listed_is_unreadable(
    params_root: Path, lock: Lock, mode: int
) -> None:
    year_dir = params_root / "2030"
    lock(year_dir, mode)
    _assert_refused(lambda: list(year_dir.iterdir()))
    with pytest.raises(ParamDirectoryUnreadableError) as exc:
        load_year(2030, params_root)
    assert str(exc.value) == _listing_message(year_dir)


def test_a_year_directory_that_can_be_read_but_not_entered_names_the_file(
    params_root: Path, lock: Lock
) -> None:
    year_dir = params_root / "2030"
    first = sorted(year_dir.glob("*.yaml"))[0]
    lock(year_dir, 0o444)
    assert first in list(year_dir.iterdir())  # the listing works
    with pytest.raises(ParamFileMissingError) as exc:
        load_year(2030, params_root)
    assert str(exc.value).startswith(f"Cannot read parameter file {first}: ")


def test_an_unreadable_parameter_file_is_a_file_error(params_root: Path, lock: Lock) -> None:
    path = params_root / "2030" / "ab.yaml"
    lock(path, 0o000)
    with pytest.raises(PermissionError) as refused:
        path.read_text(encoding="utf-8")
    with pytest.raises(ParamFileMissingError) as exc:
        load_year(2030, params_root)
    assert str(exc.value) == f"Cannot read parameter file {path}: {refused.value}"


def test_a_missing_year_under_a_root_that_cannot_be_listed_says_unknown(
    params_root: Path, lock: Lock
) -> None:
    lock(params_root, 0o111)
    _assert_refused(lambda: list(params_root.iterdir()))
    with pytest.raises(ParamYearMissingError) as exc:
        load_year(1999, params_root)
    assert str(exc.value).endswith(
        f"Available years: unknown — {params_root} cannot be listed: {DENIED}."
    )


def test_a_missing_year_under_a_root_that_cannot_be_entered_is_unreadable(
    params_root: Path, lock: Lock
) -> None:
    lock(params_root, 0o444)
    with pytest.raises(ParamDirectoryUnreadableError) as exc:
        load_year(1999, params_root)
    assert str(exc.value) == _lookup_message(1999, params_root)


def _root_with_an_unlookable_year(tmp_path: Path, params_root: Path, lock: Lock) -> Path:
    """A root whose ``2031`` is a symlink into a directory the process cannot enter."""
    root = _root_with_a_year(tmp_path, params_root)
    other = tmp_path / "other"
    (other / "inner").mkdir(parents=True)
    (root / "2031").symlink_to(other / "inner", target_is_directory=True)
    lock(other, 0o000)
    _assert_refused(lambda: (root / "2031").stat())
    return root


def test_a_year_entry_that_cannot_be_looked_up_is_left_out_of_the_available_years(
    tmp_path: Path, params_root: Path, lock: Lock
) -> None:
    root = _root_with_an_unlookable_year(tmp_path, params_root, lock)
    with pytest.raises(ParamYearMissingError) as with_link:
        load_year(1999, root)
    (root / "2031").unlink()
    with pytest.raises(ParamYearMissingError) as without_link:
        load_year(1999, root)
    assert "2030" in str(with_link.value)
    assert "2031" not in str(with_link.value)
    assert str(with_link.value) == str(without_link.value)


def test_a_root_that_is_a_symlink_loop_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "loop"
    root.symlink_to("loop")
    with pytest.raises(OSError) as raised:
        root.stat()
    assert raised.value.errno == errno.ELOOP
    with pytest.raises(ParamYearMissingError) as exc:
        load_year(2030, root)
    assert str(exc.value) == (
        f"No parameters for tax year 2030: {root / '2030'} does not exist. "
        "Available years: none — the params root itself does not exist."
    )


def test_a_year_that_is_a_symlink_loop_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "1999").symlink_to(root / "1999")
    with pytest.raises(ParamYearMissingError) as exc:
        load_year(1999, root)
    assert "does not exist" in str(exc.value)


def test_a_root_with_a_nul_byte_is_missing(tmp_path: Path) -> None:
    with pytest.raises(ParamYearMissingError) as exc:
        load_year(2030, tmp_path / "a\0b")
    assert "the params root itself does not exist" in str(exc.value)


def test_the_year_listing_matches_glob(tmp_path: Path) -> None:
    year_dir = tmp_path / "2030"
    year_dir.mkdir()
    created = [
        "a.yaml",
        ".h.yaml",
        ".yaml",
        "b.YAML",
        "c.yaml.bak",
        "d.yml",
        "n\n.yaml",
        "a[1].yaml",
        "*.yaml",
        "x.yaml ",
        "é.yaml",
    ]
    for name in created:
        (year_dir / name).write_text("", encoding="utf-8")
    (year_dir / "link.yaml").symlink_to(year_dir / "a.yaml")
    (year_dir / "sub").mkdir()
    (year_dir / "sub" / "nested.yaml").write_text("", encoding="utf-8")
    matched = sorted(path.stem for path in year_dir.glob("*.yaml"))
    assert len(matched) >= 8
    assert sum(1 for name in [*created, "sub"] if Path(name).stem not in matched) >= 4
    assert load_year(2030, tmp_path).names() == tuple(matched)


@pytest.mark.parametrize("kind", ["directory", "broken link"])
def test_a_directory_or_broken_link_named_yaml_is_a_file_error(tmp_path: Path, kind: str) -> None:
    year_dir = tmp_path / "2030"
    year_dir.mkdir()
    path = year_dir / ("dir.yaml" if kind == "directory" else "broken.yaml")
    if kind == "directory":
        path.mkdir()
    else:
        path.symlink_to(year_dir / "absent")
    with pytest.raises(ParamFileMissingError) as exc:
        load_year(2030, tmp_path)
    assert str(exc.value).startswith(f"Cannot read parameter file {path}: ")


# --- check_params_root --------------------------------------------------------


def test_check_passes_a_readable_root(tmp_path: Path, params_root: Path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(DEFAULT_PARAMS_ROOT, copy)
    assert check_params_root(params_root) is None
    assert check_params_root(copy) is None


def test_check_passes_an_empty_readable_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    assert check_params_root(root) is None


@pytest.mark.parametrize("kind", ["absent", "file"])
def test_check_passes_a_root_that_does_not_exist_or_is_a_file(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "root"
    if kind == "file":
        root.write_text("not a directory\n", encoding="utf-8")
    assert check_params_root(root) is None


@pytest.mark.parametrize("present", [True, False], ids=["present", "absent"])
def test_check_refuses_a_root_whose_parent_cannot_be_entered(
    tmp_path: Path, params_root: Path, lock: Lock, present: bool
) -> None:
    parent = tmp_path / "outer"
    root = parent / "params"
    if present:
        shutil.copytree(params_root / "2030", root / "2030")
    else:
        parent.mkdir()
        assert not root.exists()
    lock(parent, 0o000)
    _assert_refused(root.stat)
    with pytest.raises(ParamDirectoryUnreadableError) as exc:
        check_params_root(root)
    assert str(exc.value) == f"Cannot look up the parameters directory {root}: {DENIED}."


def test_check_refuses_a_root_that_cannot_be_entered_or_read(params_root: Path, lock: Lock) -> None:
    lock(params_root, 0o000)
    with pytest.raises(ParamDirectoryUnreadableError) as exc:
        check_params_root(params_root)
    assert str(exc.value) == _listing_message(params_root)


def test_check_passes_a_root_that_can_be_entered_but_not_read(
    params_root: Path, lock: Lock
) -> None:
    lock(params_root, 0o111)
    _assert_refused(lambda: list(params_root.iterdir()))
    assert check_params_root(params_root) is None


def test_check_passes_a_root_that_can_be_entered_but_not_read_holding_an_unreadable_year(
    params_root: Path, lock: Lock
) -> None:
    lock(params_root / "2030", 0o000)
    lock(params_root, 0o111)
    _assert_refused(lambda: list(params_root.iterdir()))
    _assert_refused(lambda: list((params_root / "2030").iterdir()))
    assert check_params_root(params_root) is None


def test_check_passes_a_root_that_can_be_entered_but_not_read_holding_an_unlookable_year(
    tmp_path: Path, params_root: Path, lock: Lock
) -> None:
    root = _root_with_an_unlookable_year(tmp_path, params_root, lock)
    lock(root, 0o111)
    _assert_refused(lambda: list(root.iterdir()))
    _assert_refused(lambda: (root / "2031").stat())
    assert check_params_root(root) is None


def test_check_refuses_a_root_that_can_be_read_but_not_entered(
    tmp_path: Path, params_root: Path, lock: Lock
) -> None:
    root = _root_with_a_year(tmp_path, params_root)
    lock(root, 0o444)
    assert [entry.name for entry in root.iterdir()] == ["2030"]
    with pytest.raises(ParamDirectoryUnreadableError) as exc:
        check_params_root(root)
    assert str(exc.value) == _lookup_message(2030, root)


def test_check_refuses_a_root_that_can_be_read_but_not_entered_holding_a_year_file(
    tmp_path: Path, lock: Lock
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "2031").write_text("", encoding="utf-8")
    lock(root, 0o444)
    with pytest.raises(ParamDirectoryUnreadableError) as exc:
        check_params_root(root)
    assert str(exc.value) == _lookup_message(2031, root)


@pytest.mark.parametrize("mode", [0o000, 0o111], ids=["000", "111"])
def test_check_refuses_a_year_directory_that_cannot_be_listed(
    params_root: Path, lock: Lock, mode: int
) -> None:
    year_dir = params_root / "2030"
    lock(year_dir, mode)
    _assert_refused(lambda: list(year_dir.iterdir()))
    with pytest.raises(ParamDirectoryUnreadableError) as exc:
        check_params_root(params_root)
    assert str(exc.value) == _listing_message(year_dir)


def test_check_skips_entries_not_named_as_years(tmp_path: Path, lock: Lock) -> None:
    root = tmp_path / "root"
    names = ["lost+found", "templates", "²"]
    for name in names:
        (root / name).mkdir(parents=True)
    (root / "2031").write_text("not a directory\n", encoding="utf-8")
    for name in names:
        lock(root / name, 0o000)
    _assert_refused(lambda: list((root / "lost+found").iterdir()))
    assert check_params_root(root) is None


@pytest.mark.parametrize("kind", ["year directory", "file"])
def test_check_does_not_read_files(params_root: Path, lock: Lock, kind: str) -> None:
    year_dir = params_root / "2030"
    lock(
        year_dir if kind == "year directory" else year_dir / "ab.yaml",
        0o444 if kind == "year directory" else 0o000,
    )
    assert check_params_root(params_root) is None
    with pytest.raises(ParamFileMissingError):
        load_year(2030, params_root)


def test_check_refuses_a_year_entry_that_cannot_be_looked_up(
    tmp_path: Path, params_root: Path, lock: Lock
) -> None:
    root = _root_with_an_unlookable_year(tmp_path, params_root, lock)
    with pytest.raises(ParamDirectoryUnreadableError) as exc:
        check_params_root(root)
    assert str(exc.value) == _lookup_message(2031, root)


def _too_long(tmp_path: Path) -> Path:
    name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
    assert name_max > 0, "the filesystem reports no limit on a name's length"
    return tmp_path / ("x" * (name_max + 1))


def test_a_root_whose_name_is_too_long_is_unreadable(tmp_path: Path) -> None:
    root = _too_long(tmp_path)
    with pytest.raises(ParamDirectoryUnreadableError) as exc:
        load_year(2030, root)
    reason = os.strerror(errno.ENAMETOOLONG)
    assert str(exc.value) == (
        f"Cannot look up tax year 2030 in the parameters directory {root}: {reason}."
    )


def test_check_refuses_a_root_whose_name_is_too_long(tmp_path: Path) -> None:
    root = _too_long(tmp_path)
    with pytest.raises(ParamDirectoryUnreadableError) as exc:
        check_params_root(root)
    reason = os.strerror(errno.ENAMETOOLONG)
    assert str(exc.value) == f"Cannot look up the parameters directory {root}: {reason}."


def test_check_passes_a_root_that_can_be_read_but_not_entered_with_no_year(
    tmp_path: Path, lock: Lock
) -> None:
    root = tmp_path / "root"
    (root / "templates").mkdir(parents=True)
    lock(root, 0o444)
    _assert_refused((root / "templates").stat)
    assert check_params_root(root) is None
    with pytest.raises(ParamDirectoryUnreadableError) as exc:
        load_year(2030, root)
    assert str(exc.value) == _lookup_message(2030, root)


@pytest.mark.parametrize("kind", ["loop", "nul"])
def test_check_passes_a_root_that_is_a_symlink_loop_or_holds_a_nul_byte(
    tmp_path: Path, kind: str
) -> None:
    if kind == "loop":
        root = tmp_path / "loop"
        root.symlink_to(root.name)
    else:
        root = tmp_path / "a\0b"
    assert check_params_root(root) is None


def test_check_passes_a_year_entry_that_is_a_symlink_loop(
    tmp_path: Path, params_root: Path
) -> None:
    root = _root_with_a_year(tmp_path, params_root)
    (root / "2031").symlink_to("2031")
    with pytest.raises(OSError) as raised:
        (root / "2031").stat()
    assert raised.value.errno == errno.ELOOP
    assert check_params_root(root) is None

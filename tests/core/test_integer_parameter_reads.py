# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""No engine module reads an integer-like key with ``number()``.

An age in years, a count of months, the filing month and the TFSA recontribution
lag are whole numbers. They are read with ``integer()``, which refuses NaN, an
infinity and a fraction with an error naming the file and the key. ``number()``
followed by ``int()`` would raise a bare ``ValueError`` on NaN or silently
truncate a fraction. This scan reads every ``engine`` module's syntax tree and
finds two spellings: a ``number()`` call whose first positional argument is a
string or f-string naming an integer-like key (each placeholder read as a word),
and ``int()`` applied directly to a ``number()`` call. Among the reads not
checked: a key passed by keyword or through a variable, a key assembled other
than as one string or f-string, a placeholder that supplies the integer-like
suffix itself, a ``numbers()`` or ``sequence()`` call, and a ``get()`` read.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Final

ENGINE: Final[Path] = Path(__file__).resolve().parents[2] / "engine"

#: Keys that name a whole number: an age, a count of months, the filing month,
#: the recontribution lag.
INTEGER_LIKE: Final[re.Pattern[str]] = re.compile(
    r"(^|\.)\w*(_age_years|_months)$|(^|\.)filing_month$|restored_after_n_year_ends$"
)

#: Measured: the number of modules under ``engine/`` on this tree. A floor, so
#: that a scan that finds no files cannot pass.
MIN_FILES_SCANNED: Final[int] = 53


def _is_number_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "number"
    )


def _key_text(node: ast.expr) -> str | None:
    """A string constant's text, an f-string's text with each placeholder as ``x``, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value if isinstance(part, ast.Constant) else "x" for part in node.values
        )
    return None


def number_reads_of_integer_keys(tree: ast.AST) -> list[int]:
    """Line numbers of ``X.number(<string or f-string naming an integer-like key>)`` calls."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if _is_number_call(node) and isinstance(node, ast.Call) and node.args:
            key = _key_text(node.args[0])
            if key is not None and INTEGER_LIKE.search(key):
                lines.append(node.lineno)
    return lines


def int_of_number_calls(tree: ast.AST) -> list[int]:
    """Line numbers of ``int(X.number(...))`` calls."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "int"
        and node.args
        and _is_number_call(node.args[0])
    ]


def _scan() -> tuple[int, list[str], list[str]]:
    files = sorted(ENGINE.rglob("*.py"))
    reads: list[str] = []
    casts: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        where = path.relative_to(ENGINE.parent)
        reads += [f"{where}:{line}" for line in number_reads_of_integer_keys(tree)]
        casts += [f"{where}:{line}" for line in int_of_number_calls(tree)]
    return len(files), reads, casts


def test_no_engine_module_reads_an_integer_like_key_with_number() -> None:
    scanned, reads, _ = _scan()
    assert scanned >= MIN_FILES_SCANNED
    assert reads == []


def test_no_engine_module_casts_a_number_read_to_int() -> None:
    scanned, _, casts = _scan()
    assert scanned >= MIN_FILES_SCANNED
    assert casts == []


# --- Synthetic controls: each pattern is flagged, and the neighbours are not ----


def test_the_scan_flags_a_number_read_of_an_age_key() -> None:
    tree = ast.parse('x = params.rrif.number("conversion_age_years")')
    assert number_reads_of_integer_keys(tree) == [1]


def test_the_scan_flags_each_integer_like_spelling() -> None:
    for key in (
        "start_age.earliest_months",
        "filing_month",
        "recontribution.restored_after_n_year_ends",
        "a.b_age_years",
    ):
        assert number_reads_of_integer_keys(ast.parse(f'params.number("{key}")')) == [1], key


def test_the_scan_flags_int_of_a_number_read() -> None:
    tree = ast.parse('x = int(params.number("anything"))')
    assert int_of_number_calls(tree) == [1]


def test_the_scan_does_not_flag_a_rate_per_month_or_an_integer_read() -> None:
    tree = ast.parse(
        'a = params.number("deferral.increment_rate_per_month")\n'
        'b = params.integer("conversion_age_years")\n'
        "c = int(1.5)\n"
    )
    assert number_reads_of_integer_keys(tree) == []
    assert int_of_number_calls(tree) == []


def test_the_scan_flags_an_f_string_naming_an_integer_like_key() -> None:
    for source in (
        'params.number(f"{prefix}.terminal_age_years")',
        'params.number(f"{prefix}.start_age.earliest_months")',
    ):
        assert number_reads_of_integer_keys(ast.parse(source)) == [1], source


def test_the_scan_does_not_flag_an_f_string_row_key() -> None:
    for source in (
        'params.number(f"rrif.minimum_factors.by_age.{age}")',
        'table.number(f"q_x.{sex}.{age}")',
    ):
        assert number_reads_of_integer_keys(ast.parse(source)) == [], source

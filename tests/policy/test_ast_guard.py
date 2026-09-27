# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guard 1: nothing under ``engine/policy/`` names a future fact.

A policy reads only the opening state, the month context, and the parameters (see
``engine/policy/__init__.py``'s module docstring). It must never read
``death_month_index`` (a future fact), and it has no business naming the Monte Carlo
machinery at all -- ``n_months``, ``draws``, ``real_returns``, ``mortality``,
``draw_deaths`` -- nor importing ``engine.mc.returns`` or ``engine.core.mortality``.

This walks the AST rather than importing, exactly as ``tests/test_layering.py`` does for the
API/CLI boundary, and for the same reason: a name buried in a function body or an f-string
argument is caught the same way a top-level one is. A ``Name``, an ``Attribute``, a keyword
argument, and a non-docstring string constant are the four ways a forbidden word can appear
in source without being an import; each has its own control case below, proving the walker
actually flags it, not merely that no committed file happens to trip it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_ROOT = REPO_ROOT / "engine" / "policy"

#: Names a policy must never read or reference, even as a string.
FORBIDDEN_NAMES = frozenset(
    {"death_month_index", "n_months", "draws", "real_returns", "mortality", "draw_deaths"}
)

#: Modules a policy must never import, in any form.
FORBIDDEN_IMPORTS = frozenset({"engine.mc.returns", "engine.core.mortality"})


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    """``id()`` of every string constant that is a genuine docstring -- the first statement
    of a module, class, or function body -- exempt from the string-constant check below.
    """
    exempt: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                exempt.add(id(body[0].value))
    return exempt


def find_violations(tree: ast.AST) -> list[str]:
    """Every forbidden ``Name``, ``Attribute``, keyword argument, non-docstring string
    constant, or forbidden import in ``tree``, each described with its line number.
    """
    exempt_docstrings = _docstring_constant_ids(tree)
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            violations.append(f"line {node.lineno}: Name {node.id!r}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            violations.append(f"line {node.lineno}: Attribute {node.attr!r}")
        elif isinstance(node, ast.keyword) and node.arg in FORBIDDEN_NAMES:
            lineno = getattr(node, "lineno", "?")
            violations.append(f"line {lineno}: keyword {node.arg!r}")
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in FORBIDDEN_NAMES
            and id(node) not in exempt_docstrings
        ):
            violations.append(f"line {node.lineno}: string constant {node.value!r}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORTS:
                    violations.append(f"line {node.lineno}: import {alias.name!r}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module in FORBIDDEN_IMPORTS:
                violations.append(f"line {node.lineno}: from-import {node.module!r}")
            for alias in node.names:
                combined = f"{node.module}.{alias.name}"
                if combined in FORBIDDEN_IMPORTS:
                    violations.append(f"line {node.lineno}: from-import {combined!r}")

    return violations


def _policy_modules() -> list[Path]:
    return sorted(POLICY_ROOT.glob("*.py"))


def test_the_walk_visits_at_least_one_file() -> None:
    modules = _policy_modules()
    assert modules, f"no modules found under {POLICY_ROOT}; the guard is vacuous."


@pytest.mark.parametrize("module", _policy_modules(), ids=lambda p: p.name)
def test_engine_policy_names_no_future_fact(module: Path) -> None:
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    violations = find_violations(tree)
    assert not violations, f"{module.relative_to(REPO_ROOT)}:\n  " + "\n  ".join(violations)


# --- Control cases: the walker must actually flag each node kind. -----------------------


def test_control_name_is_flagged() -> None:
    tree = ast.parse("x = death_month_index\n")
    assert find_violations(tree)


def test_control_attribute_is_flagged() -> None:
    tree = ast.parse("x = state.death_month_index\n")
    assert find_violations(tree)


def test_control_keyword_is_flagged() -> None:
    tree = ast.parse("f(mortality=1)\n")
    assert find_violations(tree)


def test_control_string_constant_is_flagged() -> None:
    tree = ast.parse('x = "mortality"\n')
    assert find_violations(tree)


@pytest.mark.parametrize("forbidden_import", sorted(FORBIDDEN_IMPORTS))
def test_control_plain_import_is_flagged(forbidden_import: str) -> None:
    tree = ast.parse(f"import {forbidden_import}\n")
    assert find_violations(tree)


def test_control_from_import_of_engine_mc_returns_is_flagged() -> None:
    tree = ast.parse("from engine.mc import returns\n")
    assert find_violations(tree)


def test_control_from_import_of_engine_core_mortality_is_flagged() -> None:
    tree = ast.parse("from engine.core import mortality\n")
    assert find_violations(tree)


def test_control_a_docstring_mention_is_not_flagged() -> None:
    """A module docstring that happens to say ``"mortality"`` is exempt."""
    tree = ast.parse('"""mortality"""\n')
    assert not find_violations(tree)


def test_control_a_function_docstring_mention_is_not_flagged() -> None:
    tree = ast.parse('def f():\n    """mortality"""\n    return 1\n')
    assert not find_violations(tree)

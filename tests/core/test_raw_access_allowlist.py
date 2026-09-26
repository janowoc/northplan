# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Every ``.raw`` access under ``engine/`` is on an explicit allowlist, with a reason.

``RealParamSet.raw`` and ``RealParamYear.raw`` (``engine/core/indexation.py``) are public by
design: tests and diagnostics legitimately need the published figure. But engine code that
reaches through ``.raw`` to a *dollar amount* is doing by hand the thing
:class:`~engine.core.indexation.RealParamSet` exists to prevent — a future
``params.raw.number("credits.basic_amount_annual")`` would silently return an undeflated figure
and raise nothing.

This module walks the AST (not text — a ``.raw`` access mentioned inside a docstring is not code
and must not be seen) of every file under ``engine/`` and asserts every ``.raw`` access it finds
is in :data:`ALLOWLIST`, each entry carrying a one-line reason the thing reached is not a dollar.
``engine/core/indexation.py`` is excluded wholesale: it is the module that *defines*
``RealParamSet``/``RealParamYear``, and every one of its ``self.raw`` accesses is the
implementation reading its own wrapped object.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "engine"

#: engine/core/indexation.py defines RealParamSet and RealParamYear: every one of its 22
#: `self.raw` accesses is the implementation reading its own wrapped object. Listing them would
#: be 22 allowlist entries all saying "this is the implementation." Excluded by path, and nothing
#: else is.
_EXCLUDED_MODULE: Final[str] = "engine/core/indexation.py"

#: One entry per allowlisted access, keyed by (module path relative to the repo root, the
#: rendered access expression). See the module docstring for how an access is found and keyed.
ALLOWLIST: Final[dict[tuple[str, str], str]] = {
    ("engine/accounts/resp.py", "params.raw.source"): (
        "Reaches the parameter file's Path for an error message, not a dollar amount, and has "
        "no real-terms equivalent."
    ),
    ("engine/benefits/oas.py", "params.raw.source"): (
        "Reaches the parameter file's Path for an error message, not a dollar amount, and has "
        "no real-terms equivalent."
    ),
}

#: A floor, not an exact count: high enough that a collector that silently stopped finding
#: accesses could not pass the first test vacuously. The real tree has exactly 2 today.
MINIMUM_RAW_ACCESSES: Final[int] = 2


def find_raw_accesses(source: str, module_path: str) -> set[tuple[str, str]]:
    """Every ``.raw`` access expression in ``source``, keyed by ``(module_path, expression)``.

    An AST walk, not a grep: a ``.raw`` access mentioned inside a string (a docstring, for
    example) is not an :class:`ast.Attribute` node and is never found here.

    For each ``ast.Attribute`` node with ``attr == "raw"``, climbs outward while the parent is an
    ``ast.Attribute`` or ``ast.Subscript`` whose ``.value`` is the current node, then, if the
    parent of the result is an ``ast.Call`` whose ``func`` is that node, takes the ``Call`` too —
    so ``params.raw.number("mortality.terminal_age_years")`` keys on the parameter path it
    actually reads, while ``params.raw.source`` (not a call) is unaffected.

    Args:
        source: Python source text.
        module_path: Identifies the source for the returned keys and for
            :func:`ast.parse`'s error messages; not read from disk here.

    Returns:
        One entry per distinct access expression; a repeated identical access in the same module
        collapses to one entry.

    Raises:
        SyntaxError: If ``source`` is not valid Python.
    """
    tree = ast.parse(source, filename=module_path)

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    found: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr == "raw"):
            continue
        current: ast.AST = node
        while True:
            parent = parents.get(current)
            climbs = isinstance(parent, (ast.Attribute, ast.Subscript)) and parent.value is current
            if not climbs:
                break
            current = parent
        parent = parents.get(current)
        if isinstance(parent, ast.Call) and parent.func is current:
            current = parent
        found.add((module_path, ast.unparse(current)))
    return found


def _real_accesses() -> set[tuple[str, str]]:
    """Every ``.raw`` access under ``engine/``, unioned across every file but the excluded one."""
    found: set[tuple[str, str]] = set()
    for path in sorted(ENGINE_ROOT.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative == _EXCLUDED_MODULE:
            continue
        found |= find_raw_accesses(path.read_text(encoding="utf-8"), relative)
    return found


class TestEveryRawAccessIsAllowlisted:
    def test_every_access_found_is_allowlisted(self) -> None:
        unlisted = _real_accesses() - set(ALLOWLIST)
        assert not unlisted, (
            f"the following .raw accesses under engine/ are not in ALLOWLIST: "
            f"{sorted(unlisted)}. For each: either it reaches a dollar amount, and belongs "
            f"through RealParamSet.amount()/annual_amount() instead, or it genuinely is not a "
            f"dollar, and needs one line in ALLOWLIST saying why."
        )

    def test_no_stale_entry_names_an_access_that_no_longer_exists(self) -> None:
        stale = set(ALLOWLIST) - _real_accesses()
        assert not stale, (
            f"ALLOWLIST names the following accesses, but they are no longer found under "
            f"engine/ (removed, or the expression changed): {sorted(stale)}. Update ALLOWLIST "
            f"to match."
        )

    def test_the_walk_is_not_vacuous(self) -> None:
        """A floor on the collector's own output, so a collector that silently stopped finding
        accesses could not pass test_every_access_found_is_allowlisted vacuously."""
        assert len(_real_accesses()) >= MINIMUM_RAW_ACCESSES
        assert len(ALLOWLIST) >= MINIMUM_RAW_ACCESSES

    def test_the_floor_can_actually_bite(self) -> None:
        """Shows the assertion above can fail, not just that it currently passes.

        A collector restricted to a module with no `.raw` access at all finds far fewer than the
        floor, the way ``tests/core/test_state_nominal_or_real.py``'s equivalent test does.
        """
        state_source = (ENGINE_ROOT / "core" / "state.py").read_text(encoding="utf-8")
        restricted = find_raw_accesses(state_source, "engine/core/state.py")
        assert len(restricted) < MINIMUM_RAW_ACCESSES


class TestTheGuardBites:
    def test_an_unlisted_access_on_a_synthetic_module_is_reported(self) -> None:
        """The success criterion: a new .raw access is not allowlisted until it is added.

        A synthetic source string, run through the same collector the real test uses, keyed
        against a module path that cannot appear in the real tree — the real ``ALLOWLIST`` does
        not, and cannot, know about it.
        """
        source = 'def f(params):\n    return params.raw.number("credits.basic_amount_annual")\n'
        found = find_raw_accesses(source, "synthetic/not_yet_allowlisted.py")
        assert found == {
            (
                "synthetic/not_yet_allowlisted.py",
                "params.raw.number('credits.basic_amount_annual')",
            )
        }
        assert not (found & set(ALLOWLIST))

    def test_a_raw_access_inside_a_docstring_is_not_seen(self) -> None:
        """AST, not grep: a `.raw` access that only ever appears inside a string is invisible.

        Demonstrated against a synthetic module rather than against
        ``engine/optimize/search.py`` by name, even though that file's docstring contains
        ``real_params.raw["mortality"]`` today — asserting against the real file by name would
        make this test depend on a sentence staying exactly as worded.
        """
        source = (
            "def f() -> None:\n"
            '    """Uses real_params.raw["mortality"] as an example of the convention."""\n'
        )
        assert find_raw_accesses(source, "synthetic/docstring_only.py") == set()

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The layering rule: ``engine/`` may not depend on the layers above it.

``engine/`` must be importable and fully testable with FastAPI absent. That is
what keeps the tax and simulation logic verifiable on its own terms, and what
stops HTTP concerns from leaking into a bracket calculation.

This walks the AST rather than importing, so a forbidden import is caught even
inside a function body, a ``TYPE_CHECKING`` block, or a module that would fail
to import for an unrelated reason.

``report/`` sits between the engine and the two front ends: it may import the
engine, and neither front end nor FastAPI, so the command line and the API
share it without depending on each other.

It also pins the one permitted dependency of ``engine/scenario`` on
``engine/mc`` — the leaf ``engine.mc.moments``, which imports nothing from
``engine.scenario``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = REPO_ROOT / "engine"

#: Top-level modules ``engine/`` must never import, directly or as a submodule.
FORBIDDEN_ROOTS = frozenset({"api", "fastapi", "cli", "report", "starlette", "uvicorn"})

REPORT_ROOT = REPO_ROOT / "report"

#: Top-level modules ``report/`` must never import.
REPORT_FORBIDDEN_ROOTS = frozenset({"api", "cli", "fastapi", "starlette", "uvicorn"})


def engine_modules() -> list[Path]:
    """Every Python module under ``engine/``."""
    return sorted(ENGINE_ROOT.rglob("*.py"))


def imported_roots(tree: ast.AST) -> set[tuple[str, int]]:
    """Top-level package name of every import in ``tree``, with its line number.

    Relative imports are ignored: they cannot reach outside the package being walked.
    """
    found: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add((alias.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, stays inside the package
                continue
            if node.module:
                found.add((node.module.split(".")[0], node.lineno))
    return found


def test_engine_directory_exists() -> None:
    """Guard against the walk silently passing because it found no files."""
    modules = engine_modules()
    assert modules, f"No modules found under {ENGINE_ROOT}; the layering test is vacuous."


@pytest.mark.parametrize("module", engine_modules(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_engine_does_not_import_upper_layers(module: Path) -> None:
    """No module under ``engine/`` imports the API or CLI layer."""
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))

    violations = [
        f"{module.relative_to(REPO_ROOT)}:{lineno} imports {root!r}"
        for root, lineno in sorted(imported_roots(tree), key=lambda pair: pair[1])
        if root in FORBIDDEN_ROOTS
    ]

    assert not violations, (
        "engine/ must be importable and testable with FastAPI absent, and must not "
        "depend on the layers above it:\n  " + "\n  ".join(violations)
    )


def report_modules() -> list[Path]:
    """Every Python module under ``report/``."""
    return sorted(REPORT_ROOT.rglob("*.py"))


def test_report_directory_exists() -> None:
    """Guard against the walk silently passing because it found no files."""
    modules = report_modules()
    assert modules, f"No modules found under {REPORT_ROOT}; the layering test is vacuous."


@pytest.mark.parametrize("module", report_modules(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_report_does_not_import_the_front_ends(module: Path) -> None:
    """No module under ``report/`` imports the API or CLI layer, or FastAPI."""
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))

    violations = [
        f"{module.relative_to(REPO_ROOT)}:{lineno} imports {root!r}"
        for root, lineno in sorted(imported_roots(tree), key=lambda pair: pair[1])
        if root in REPORT_FORBIDDEN_ROOTS
    ]

    assert not violations, (
        "report/ is shared by cli/ and api/ and must depend on neither:\n  "
        + "\n  ".join(violations)
    )


API_ROOT = REPO_ROOT / "api"
CLI_ROOT = REPO_ROOT / "cli"


def test_front_end_directories_exist() -> None:
    """Guard against the front-end walks silently passing because they found no files."""
    for root in (API_ROOT, CLI_ROOT):
        assert sorted(root.rglob("*.py")), f"No modules found under {root}; the test is vacuous."


def front_end_violations(module: Path, forbidden: str) -> list[str]:
    """Each import in ``module`` whose root is ``forbidden``, as ``path:line imports 'root'``."""
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    return [
        f"{module.relative_to(REPO_ROOT)}:{lineno} imports {root!r}"
        for root, lineno in sorted(imported_roots(tree), key=lambda pair: pair[1])
        if root == forbidden
    ]


@pytest.mark.parametrize(
    "module", sorted(API_ROOT.rglob("*.py")), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_api_does_not_import_the_command_line(module: Path) -> None:
    """No module under ``api/`` imports ``cli``."""
    assert not front_end_violations(module, "cli")


@pytest.mark.parametrize(
    "module", sorted(CLI_ROOT.rglob("*.py")), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_command_line_does_not_import_the_api(module: Path) -> None:
    """No module under ``cli/`` imports ``api``."""
    assert not front_end_violations(module, "api")


SCENARIO_ROOT = ENGINE_ROOT / "scenario"


def scenario_modules() -> list[Path]:
    """Every Python module under ``engine/scenario/``."""
    return sorted(SCENARIO_ROOT.rglob("*.py"))


def module_package(path: Path) -> str:
    """The dotted package ``path`` lives in, e.g. ``engine.scenario``.

    Both ``engine/scenario/schema.py`` and ``engine/scenario/__init__.py``
    give ``"engine.scenario"``: the module's own filename is always the
    last path component, and this drops it.
    """
    parts = path.relative_to(REPO_ROOT).with_suffix("").parts
    return ".".join(parts[:-1])


def imported_modules(tree: ast.AST, package: str) -> set[tuple[str, int]]:
    """Full dotted ``module.name`` of every import in ``tree``, with its line number.

    Unlike :func:`imported_roots`, this keeps the whole dotted path rather
    than only the top-level package, so ``engine.mc.moments`` and
    ``engine.mc.returns`` are told apart. A from-import is recorded as
    ``module.name`` for each imported name -- never the bare module on its
    own -- with ``asname`` ignored and ``*`` recorded literally as
    ``module.*``. A relative from-import is resolved against ``package``,
    the dotted package the source file lives in (see :func:`module_package`),
    rather than skipped, so it is caught exactly like an absolute one.
    """
    found: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module
            else:
                parts = package.split(".")
                drop = node.level - 1
                base_parts = parts[: len(parts) - drop] if drop else parts
                base = ".".join(base_parts)
                if node.module:
                    base = f"{base}.{node.module}"
            for alias in node.names:
                found.add((f"{base}.{alias.name}", node.lineno))
    return found


def reaches_mc_beyond_moments(name: str) -> bool:
    """True when ``name`` names ``engine.mc`` or any submodule other than
    ``engine.mc.moments`` (or one of its own submodules)."""
    in_mc = name == "engine.mc" or name.startswith("engine.mc.")
    in_moments = name == "engine.mc.moments" or name.startswith("engine.mc.moments.")
    return in_mc and not in_moments


def reaches_scenario(name: str) -> bool:
    """True when ``name`` names ``engine.scenario`` or any of its submodules."""
    return name == "engine.scenario" or name.startswith("engine.scenario.")


def test_moments_imports_nothing_from_the_scenario_package() -> None:
    """``engine.mc.moments`` is the leaf: the scenario package may depend on
    it, but it must depend on nothing from ``engine.scenario``."""
    module = ENGINE_ROOT / "mc" / "moments.py"
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))

    violations = [
        f"{module.relative_to(REPO_ROOT)}:{lineno} imports {name!r}"
        for name, lineno in sorted(
            imported_modules(tree, module_package(module)), key=lambda pair: pair[1]
        )
        if reaches_scenario(name)
    ]

    assert not violations, (
        "engine.mc.moments must import nothing from engine.scenario, or the "
        "dependency between the two packages would run in both directions:\n  "
        + "\n  ".join(violations)
    )


@pytest.mark.parametrize("module", scenario_modules(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_the_scenario_package_imports_only_moments_from_mc(module: Path) -> None:
    """The scenario package's one permitted dependency on ``engine/mc`` is
    ``engine.mc.moments``, in any import form, ``from engine.mc import
    moments`` included. ``engine.mc`` itself and every other submodule are
    refused."""
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    package = module_package(module)

    violations = [
        f"{module.relative_to(REPO_ROOT)}:{lineno} imports {name!r}"
        for name, lineno in sorted(imported_modules(tree, package), key=lambda pair: pair[1])
        if reaches_mc_beyond_moments(name)
    ]

    assert not violations, (
        "engine/scenario may depend on engine.mc.moments only, never on "
        "engine.mc itself or any other engine.mc submodule:\n  " + "\n  ".join(violations)
    )


def test_module_detector_is_not_vacuous() -> None:
    """Runs the real predicates, ``reaches_mc_beyond_moments`` and
    ``reaches_scenario``, over every import form the two tests above must
    tell apart -- plain, from-import, aliased, relative, and a leaf-only
    from-import -- so those tests are not vacuous passes over an empty
    violation set. Also pins ``module_package``."""
    package = module_package(SCENARIO_ROOT / "schema.py")
    assert package == module_package(SCENARIO_ROOT / "__init__.py")
    assert package == "engine.scenario"
    assert module_package(ENGINE_ROOT / "mc" / "moments.py") == "engine.mc"

    source = (
        "from engine.mc.returns import generate\n"
        "import engine.mc\n"
        "from engine import mc\n"
        "from engine.mc import returns\n"
        "from ..mc import returns as r\n"
        "from engine.mc import moments\n"
        "from engine.mc.moments import monthly_log_moments\n"
        "import engine.mc.moments\n"
        "from engine import scenario\n"
        "from engine.scenario.schema import Scenario\n"
    )
    tree = ast.parse(source)
    found = imported_modules(tree, package)

    assert {lineno for name, lineno in found if reaches_mc_beyond_moments(name)} == {1, 2, 3, 4, 5}
    assert {lineno for name, lineno in found if reaches_scenario(name)} == {9, 10}


def test_forbidden_roots_are_actually_detected() -> None:
    """The detector catches a forbidden import; the test above is not vacuous.

    Without this, a bug in ``imported_roots`` would make every module pass.
    """
    source = (
        "import fastapi\n"
        "from api.schemas import ScenarioRequest\n"
        "from . import sibling\n"
        "def f():\n"
        "    import cli.main\n"
    )
    roots = {root for root, _ in imported_roots(ast.parse(source))}

    assert "fastapi" in roots, "A plain `import fastapi` must be detected."
    assert "api" in roots, "A `from api...` import must be detected."
    assert "cli" in roots, "An import inside a function body must be detected."
    assert "sibling" not in roots, "Relative imports must be ignored."

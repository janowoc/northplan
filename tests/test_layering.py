"""The layering rule: ``engine/`` may not depend on the layers above it.

``engine/`` must be importable and fully testable with FastAPI absent. That is
what keeps the tax and simulation logic verifiable on its own terms, and what
stops HTTP concerns from leaking into a bracket calculation.

This walks the AST rather than importing, so a forbidden import is caught even
inside a function body, a ``TYPE_CHECKING`` block, or a module that would fail
to import for an unrelated reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = REPO_ROOT / "engine"

#: Top-level modules ``engine/`` must never import, directly or as a submodule.
FORBIDDEN_ROOTS = frozenset({"api", "fastapi", "cli", "starlette", "uvicorn"})


def engine_modules() -> list[Path]:
    """Every Python module under ``engine/``."""
    return sorted(ENGINE_ROOT.rglob("*.py"))


def imported_roots(tree: ast.AST) -> set[tuple[str, int]]:
    """Top-level package name of every import in ``tree``, with its line number.

    Relative imports are ignored: they cannot reach outside ``engine/``.
    """
    found: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add((alias.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, stays inside engine/
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

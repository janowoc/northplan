# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The gate every test that needs Node and the pyodide npm package goes through.

Without Node, or without ``tests/browser/node_modules/pyodide``, the test is skipped with
the way to install them; with ``NORTHPLAN_REQUIRE_NODE=1`` (CI) it fails instead, so that a
runner that lost Node does not report green over a differential test that never ran.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

BROWSER_TESTS = Path(__file__).resolve().parent
PYODIDE_PACKAGE = BROWSER_TESTS / "node_modules" / "pyodide"
REQUIRE_VARIABLE = "NORTHPLAN_REQUIRE_NODE"


def require_node(
    *,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    package: Path = PYODIDE_PACKAGE,
) -> str:
    """The path of the ``node`` executable, when it and the pyodide package are present.

    Raises:
        pytest.skip.Exception: Node or the package is missing and ``NORTHPLAN_REQUIRE_NODE``
            is not ``1``.
        pytest.fail.Exception: Node or the package is missing and it is ``1``.
    """
    environment = os.environ if environ is None else environ
    node = which("node")
    missing = []
    if node is None:
        missing.append("node is not on PATH")
    if not package.is_dir():
        missing.append(f"{package} is absent")
    if not missing:
        assert node is not None
        return node
    reason = (
        f"{' and '.join(missing)}; install Node and run `npm ci --ignore-scripts` in "
        f"{BROWSER_TESTS} to run the browser differential tests."
    )
    if environment.get(REQUIRE_VARIABLE) == "1":
        pytest.fail(f"{REQUIRE_VARIABLE}=1 and {reason}")
    pytest.skip(reason)

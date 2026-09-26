# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Every Python file is exactly as ``ruff format`` would leave it.

With the whole tree formatted, formatting a whole file changes only the lines
its author touched, so a diff shows the change and nothing else. The ruff
version is pinned in ``pyproject.toml`` because the formatter's style changes
between releases; the installed ruff must be the pinned one, or this check
reports a style the repository never adopted.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ruff(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_installed_ruff_is_the_pinned_version() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    pins = [
        requirement
        for requirement in pyproject["project"]["optional-dependencies"]["dev"]
        if re.match(r"ruff\b", requirement)
    ]
    assert len(pins) == 1 and pins[0].startswith("ruff=="), pins
    pinned = pins[0].removeprefix("ruff==")

    installed = _ruff("--version").stdout.split()[-1]
    assert installed == pinned, f"ruff {installed} installed; pyproject.toml pins {pinned}."


def test_every_python_file_is_formatted() -> None:
    completed = _ruff("format", "--check", ".")
    assert completed.returncode == 0, (
        "Run `ruff format .` and commit the result.\n" + completed.stdout + completed.stderr
    )

    # Tracked or untracked but not ignored: the set ruff walks. An exclude
    # that dropped a file would otherwise pass as "already formatted".
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert listed
    checked = re.search(r"(\d+) files? already formatted", completed.stdout)
    assert checked is not None, completed.stdout
    assert int(checked.group(1)) == len(listed), (completed.stdout, len(listed))

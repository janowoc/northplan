# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build the deployable browser site: ``python browser/build.py OUT_DIR``.

Standard library only, so that it runs on a CI runner's python3 as well as on the
project's Python. OUT_DIR is created, and must not exist or must be empty. It receives
exactly ``index.html`` and ``worker.js`` (copied from this directory), ``editor.js`` (copied
from ``api/web/``), ``northplan.zip``, ``example.yaml`` and ``minimal.yaml`` (byte copies of
those files under ``scenarios/``) and ``build-info.json``.

The archive holds the files ``git ls-files`` lists under ``engine``, ``report`` and
``params`` (tracked or intent-to-add, content read from the working tree) at their
repository-relative paths, plus ``browser/northplan_browser.py`` as
``northplan_browser.py``. Names are sorted and timestamps fixed, so two builds of one tree
are byte-identical; nothing untracked is shipped.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIRECTORIES = ("engine", "report", "params")
ENTRY_POINT = "northplan_browser.py"
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)


class BuildError(Exception):
    """The site cannot be built; the message says why."""


def _git(root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise BuildError(f"git {' '.join(arguments)} failed in {root}: {error}") from error
    return completed.stdout


def archive_members(root: Path) -> dict[str, Path]:
    """The archive's names mapped to the files that supply them, in sorted order.

    Raises:
        BuildError: A listed file is missing on disk.
    """
    listed = _git(root, "ls-files", "-z", *ARCHIVE_DIRECTORIES).decode("utf-8").split("\0")
    members = {name: root / name for name in listed if name}
    members[ENTRY_POINT] = root / "browser" / ENTRY_POINT
    missing = sorted(name for name, path in members.items() if not path.is_file())
    if missing:
        raise BuildError("files listed for the archive are missing on disk: " + ", ".join(missing))
    return dict(sorted(members.items()))


def write_archive(members: dict[str, Path], target: Path) -> None:
    """Write ``members`` to the zip at ``target``, deflated, with fixed timestamps."""
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in members.items():
            info = zipfile.ZipInfo(name, _FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def build(out_dir: Path, root: Path = ROOT) -> None:
    """Build the site under ``root`` into ``out_dir``.

    Raises:
        BuildError: ``out_dir`` exists and is not empty, a file listed for the archive is
            missing on disk, or git fails.
    """
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise BuildError(f"{out_dir} exists and is not an empty directory; give a new one.")
    members = archive_members(root)
    commit = _git(root, "rev-parse", "HEAD").decode("utf-8").strip()
    years = sorted(
        {
            parts[1]
            for parts in (name.split("/") for name in members)
            if len(parts) > 2 and parts[0] == "params" and len(parts[1]) == 4 and parts[1].isdigit()
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "worker.js"):
        shutil.copyfile(root / "browser" / name, out_dir / name)
    shutil.copyfile(root / "api" / "web" / "editor.js", out_dir / "editor.js")
    write_archive(members, out_dir / "northplan.zip")
    for name in ("example.yaml", "minimal.yaml"):
        shutil.copyfile(root / "scenarios" / name, out_dir / name)
    info = {"parameter_years": years, "commit": commit}
    (out_dir / "build-info.json").write_text(json.dumps(info) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python browser/build.py OUT_DIR", file=sys.stderr)
        return 2
    try:
        build(Path(argv[1]))
    except BuildError as error:
        print(f"build.py: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

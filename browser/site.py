# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Assemble the deployable site from its two halves: ``python browser/site.py OUT_DIR MAIN_REF``.

Standard library only, so that it runs on a CI runner's python3. The site is the release at
``OUT_DIR/`` and the head of main at ``OUT_DIR/main/``, each built by its own tree's
``browser/build.py`` called as ``build.py DIR``.

The root is built from the newest annotated tag named ``vN.N.N``: the tag whose creation date
is greatest. Lightweight tags and every other name are ignored. MAIN_REF is any name git
resolves to a commit, such as ``origin/main`` or a SHA. Both trees are checked out as
temporary detached worktrees outside the repository, and are removed on every exit.

The script refuses, and does not fall back to another tag or to main's ``build.py``, when:

* ``OUT_DIR`` exists and is not an empty directory;
* MAIN_REF does not resolve to a commit;
* no tag qualifies, a qualifying tag's creation date is not a number, or two or more share
  the newest creation date;
* the chosen tag does not peel to a commit;
* the tag's ``pyproject.toml`` is missing, unreadable, has no string ``[project] version``,
  or has a version that, after a ``v``, is not the tag's name;
* the tag's tree has no ``browser/build.py``;
* either build exits non-zero. The message names the half and carries the build's stderr, and
  whatever the builds had written is left in ``OUT_DIR``.

Every refusal before the first build leaves ``OUT_DIR`` as it was. The final ``git worktree
prune`` runs in ``root``, so it also drops any other unlocked worktree of that repository whose
directory is gone.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_TAG = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
BUILD_SCRIPT = "browser/build.py"


class SiteError(Exception):
    """The site cannot be assembled; the message says why."""


def _run(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(["git", *arguments], cwd=root, capture_output=True)
    except OSError as error:
        raise SiteError(f"git {' '.join(arguments)} failed in {root}: {error}") from error


def _git(root: Path, *arguments: str) -> bytes:
    completed = _run(root, *arguments)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise SiteError(f"git {' '.join(arguments)} failed in {root}: {detail}")
    return completed.stdout


def _commit_of(root: Path, name: str) -> str | None:
    """The commit that ``name`` peels to, or None when it does not resolve to one."""
    completed = _run(
        root, "rev-parse", "--verify", "--quiet", "--end-of-options", f"{name}^{{commit}}"
    )
    if completed.returncode == 0:
        return completed.stdout.decode("utf-8").strip()
    if completed.returncode == 1:
        return None
    raise SiteError(f"git rev-parse {name} failed in {root}: exit status {completed.returncode}")


def _newest_release_tag(root: Path) -> str:
    """The annotated ``vN.N.N`` tag with the newest creation date.

    Raises:
        SiteError: No tag qualifies, a qualifying tag's creation date is not a number, or
            several share the newest creation date.
    """
    listing = _git(
        root,
        "for-each-ref",
        "--format=%(refname:strip=2) %(objecttype) %(creatordate:unix)",
        "refs/tags",
    ).decode("utf-8", errors="surrogateescape")
    dated: list[tuple[int, str]] = []
    for line in filter(None, listing.split("\n")):
        name, kind, created = line.split(" ", 2)
        if kind != "tag" or not RELEASE_TAG.fullmatch(name):
            continue
        if not (created.isascii() and created.isdigit()):
            raise SiteError(f"the creation date of the tag {name} is not a number: {created!r}")
        dated.append((int(created), name))
    if not dated:
        raise SiteError("no annotated tag named vN.N.N exists, so there is no release to serve")
    newest = max(created for created, _ in dated)
    tied = sorted(name for created, name in dated if created == newest)
    if len(tied) > 1:
        raise SiteError("the tags " + ", ".join(tied) + " share the newest creation date")
    return tied[0]


def _check_release(root: Path, tag: str) -> str:
    """The commit of ``tag``, after checking that its tree can build a release.

    Raises:
        SiteError: The tag does not peel to a commit, its ``pyproject.toml`` is missing or
            unreadable or does not give a string version equal to the tag without its ``v``,
            or its tree has no ``browser/build.py``.
    """
    commit = _commit_of(root, f"refs/tags/{tag}")
    if commit is None:
        raise SiteError(f"the tag {tag} does not point at a commit")
    completed = _run(root, "show", f"{commit}:pyproject.toml")
    if completed.returncode != 0:
        raise SiteError(f"the tree of {tag} has no pyproject.toml")
    try:
        document = tomllib.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise SiteError(f"cannot read the version from pyproject.toml at {tag}: {error}") from error
    project = document.get("project")
    version = project.get("version") if isinstance(project, dict) else None
    if not isinstance(version, str):
        raise SiteError(f"pyproject.toml at {tag} has no [project] version that is a string")
    if f"v{version}" != tag:
        raise SiteError(
            f"the tag {tag} does not match {version!r}, the version in its pyproject.toml"
        )
    if _run(root, "cat-file", "-e", f"{commit}:{BUILD_SCRIPT}").returncode != 0:
        raise SiteError(f"the tree of {tag} has no {BUILD_SCRIPT}")
    return commit


def _build(worktree: Path, target: Path, half: str) -> None:
    script = worktree / BUILD_SCRIPT
    try:
        completed = subprocess.run(
            [sys.executable, str(script), str(target)],
            cwd=worktree,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise SiteError(f"cannot run {script} for {half}: {error}") from error
    if completed.returncode != 0:
        raise SiteError(f"the build of {half} failed:\n{completed.stderr.strip()}")


def assemble(out_dir: Path, main_ref: str, root: Path = ROOT) -> None:
    """Build the release into ``out_dir`` and the head of ``main_ref`` into ``out_dir/main``.

    ``root`` is a git repository holding every tag. Tags and files are read with git, not from
    the working tree.

    Raises:
        SiteError: Any condition in the module docstring.
    """
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise SiteError(f"{out_dir} exists and is not an empty directory; give a new one.")
    main_commit = _commit_of(root, main_ref)
    if main_commit is None:
        raise SiteError(f"{main_ref} does not resolve to a commit")
    tag = _newest_release_tag(root)
    tag_commit = _check_release(root, tag)
    target = out_dir.absolute()
    added: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(prefix="northplan-site-") as scratch:
            try:
                halves = (
                    (f"the root at {tag}", tag_commit, Path(scratch) / "release", target),
                    (
                        f"main at {main_commit}",
                        main_commit,
                        Path(scratch) / "main",
                        target / "main",
                    ),
                )
                for half, commit, worktree, destination in halves:
                    added.append(worktree)
                    _git(root, "worktree", "add", "--detach", str(worktree), commit)
                    _build(worktree, destination, half)
            finally:
                for worktree in added:
                    _run(root, "worktree", "remove", "--force", "--force", str(worktree))
    finally:
        _run(root, "worktree", "prune")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python browser/site.py OUT_DIR MAIN_REF", file=sys.stderr)
        return 2
    try:
        assemble(Path(argv[1]), argv[2])
    except SiteError as error:
        print(f"site.py: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

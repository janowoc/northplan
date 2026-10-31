# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""browser/site.py: which tag is the root, what it refuses, and that it cleans up."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from test_browser_build import REPO_ROOT, commit_all, git, make_repository

OLD = "2030-01-01T00:00:00Z"
MIDDLE = "2031-01-01T00:00:00Z"
NEWEST = "2032-01-01T00:00:00Z"
NEWER = "2033-01-01T00:00:00Z"
IDENTITY = ("-c", "user.name=t", "-c", "user.email=t@example.invalid")
MARKER_LINE = "    out_dir.mkdir(parents=True, exist_ok=True)\n"


def load_site() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "northplan_site_under_test", REPO_ROOT / "browser" / "site.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def site() -> ModuleType:
    return load_site()


def isolated_git(repository: Path, *arguments: str, date: str | None = None) -> None:
    """Run git with the user's and the system's configuration out of the way."""
    environment = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    if date is not None:
        environment |= {"GIT_COMMITTER_DATE": date, "GIT_AUTHOR_DATE": date}
    subprocess.run(
        ["git", *IDENTITY, *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        env=environment,
    )


def head(repository: Path) -> str:
    return git(repository, "rev-parse", "HEAD").strip()


def tag(
    repository: Path, name: str, date: str, target: str = "HEAD", annotated: bool = True
) -> None:
    arguments = ["tag", "-a", name, "-m", name, target] if annotated else ["tag", name, target]
    isolated_git(repository, *arguments, date=date)


def creation_date(repository: Path, name: str) -> int:
    out = git(repository, "for-each-ref", "--format=%(creatordate:unix)", f"refs/tags/{name}")
    return int(out)


def scratch_repository(root: Path) -> Path:
    """A scratch repository whose tree holds the real browser/build.py, version 7.7.7."""
    make_repository(root)
    (root / "browser" / "build.py").write_bytes((REPO_ROOT / "browser" / "build.py").read_bytes())
    git(root, "add", ".")
    commit_all(root)
    return root


def set_version(repository: Path, version: str, date: str | None = None) -> None:
    (repository / "pyproject.toml").write_text(
        f'[project]\nname = "scratch"\nversion = "{version}"\n', encoding="utf-8"
    )
    git(repository, "add", ".")
    isolated_git(repository, "commit", "-q", "--allow-empty", "-m", "x", date=date)


def write_build_script(repository: Path, text: str) -> None:
    (repository / "browser" / "build.py").write_text(text, encoding="utf-8")
    git(repository, "add", ".")
    commit_all(repository)


def released_repository(tmp_path: Path) -> Path:
    """Scratch repository with v7.7.7 on its only commit."""
    repository = scratch_repository(tmp_path / "repository")
    tag(repository, "v7.7.7", MIDDLE)
    return repository


def info(directory: Path) -> dict[str, object]:
    return json.loads((directory / "build-info.json").read_text(encoding="utf-8"))


def worktree_count(repository: Path) -> int:
    listing = git(repository, "worktree", "list", "--porcelain")
    return sum(1 for line in listing.splitlines() if line.startswith("worktree "))


def record_commands(site: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Every command ``site`` runs from now on is appended to the returned list."""
    calls: list[list[str]] = []
    real_run = subprocess.run

    def recording(command: list[str], *args: object, **kwargs: object) -> object:
        calls.append(command)
        return real_run(command, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(site.subprocess, "run", recording)
    return calls


def worktree_verbs(calls: list[list[str]]) -> list[str]:
    return [call[2] for call in calls if call[1] == "worktree"]


def assert_refuses(
    site: ModuleType, repository: Path, out: Path, main_ref: str, message: str
) -> None:
    with pytest.raises(site.SiteError) as caught:
        site.assemble(out, main_ref, repository)
    assert message in str(caught.value)
    assert not out.exists()
    assert worktree_count(repository) == 1


@pytest.mark.parametrize(
    ("name", "version", "annotated", "commit_date"),
    [
        ("v7.7.8", "7.7.8", False, NEWER),
        ("v7.8.0-rc1", "7.8.0-rc1", True, None),
        ("vtest", "test", True, None),
        ("v07.7.8", "07.7.8", True, None),
    ],
)
def test_the_root_is_the_highest_annotated_release_tag_and_nothing_else(
    tmp_path: Path,
    site: ModuleType,
    name: str,
    version: str,
    annotated: bool,
    commit_date: str | None,
) -> None:
    repository = released_repository(tmp_path)
    released = head(repository)
    set_version(repository, version, commit_date)
    tag(repository, name, NEWEST, annotated=annotated)
    assert creation_date(repository, name) > creation_date(repository, "v7.7.7")
    out = tmp_path / "out"
    site.assemble(out, "HEAD", repository)
    assert info(out)["commit"] == released
    assert info(out)["version"] == "7.7.7"
    assert info(out / "main")["version"] == version


def test_a_higher_annotated_release_tag_is_the_root(tmp_path: Path, site: ModuleType) -> None:
    repository = released_repository(tmp_path)
    set_version(repository, "7.7.8")
    tag(repository, "v7.7.8", NEWEST)
    out = tmp_path / "out"
    site.assemble(out, "HEAD", repository)
    assert info(out)["version"] == "7.7.8"
    assert info(out)["commit"] == head(repository)


def test_the_higher_version_wins_over_the_newest_creation_date(
    tmp_path: Path, site: ModuleType
) -> None:
    repository = scratch_repository(tmp_path / "repository")
    tag(repository, "v7.7.7", MIDDLE)
    higher_version = head(repository)
    set_version(repository, "7.7.2")
    tag(repository, "v7.7.2", NEWEST)
    out = tmp_path / "out"
    site.assemble(out, "HEAD", repository)
    assert info(out)["commit"] == higher_version
    assert info(out)["version"] == "7.7.7"


def test_no_qualifying_tag_is_refused(tmp_path: Path, site: ModuleType) -> None:
    repository = scratch_repository(tmp_path / "repository")
    tag(repository, "v7.7.7", MIDDLE, annotated=False)
    tag(repository, "v7.8.0-rc1", MIDDLE)
    tag(repository, "vtest", MIDDLE)
    assert_refuses(site, repository, tmp_path / "out", "HEAD", "no annotated tag named vN.N.N")


def test_a_tag_whose_pyproject_version_is_not_its_name_is_refused(
    tmp_path: Path, site: ModuleType
) -> None:
    repository = scratch_repository(tmp_path / "repository")
    tag(repository, "v7.7.8", NEWEST)
    assert_refuses(
        site, repository, tmp_path / "out", "HEAD", "the tag v7.7.8 does not match '7.7.7'"
    )


def test_a_tag_whose_tree_has_no_pyproject_is_refused(tmp_path: Path, site: ModuleType) -> None:
    repository = scratch_repository(tmp_path / "repository")
    has_pyproject = head(repository)
    git(repository, "rm", "-q", "pyproject.toml")
    commit_all(repository)
    tag(repository, "v7.7.7", NEWEST)
    git(repository, "checkout", "-q", has_pyproject, "--", "pyproject.toml")
    commit_all(repository)
    assert (repository / "pyproject.toml").is_file()
    assert_refuses(
        site, repository, tmp_path / "out", "HEAD", "the tree of v7.7.7 has no pyproject.toml"
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[project\nversion = 1\n", "cannot read the version from pyproject.toml at v7.7.7"),
        ("[tool]\nx = 1\n", "has no [project] version that is a string"),
        ("[project]\nname = 'x'\n", "has no [project] version that is a string"),
        ("[project]\nversion = 7\n", "has no [project] version that is a string"),
    ],
)
def test_a_tag_whose_pyproject_gives_no_string_version_is_refused(
    tmp_path: Path, site: ModuleType, text: str, message: str
) -> None:
    repository = scratch_repository(tmp_path / "repository")
    (repository / "pyproject.toml").write_text(text, encoding="utf-8")
    git(repository, "add", ".")
    commit_all(repository)
    tag(repository, "v7.7.7", NEWEST)
    assert_refuses(site, repository, tmp_path / "out", "HEAD", message)


def test_a_tag_whose_tree_has_no_build_script_is_refused_without_falling_back(
    tmp_path: Path, site: ModuleType
) -> None:
    repository = scratch_repository(tmp_path / "repository")
    has_build_script = head(repository)
    git(repository, "rm", "-q", "browser/build.py")
    commit_all(repository)
    tag(repository, "v7.7.7", NEWEST)
    git(repository, "checkout", "-q", has_build_script, "--", "browser/build.py")
    commit_all(repository)
    assert (repository / "browser" / "build.py").is_file()
    assert_refuses(
        site,
        repository,
        tmp_path / "out",
        head(repository),
        "the tree of v7.7.7 has no browser/build.py",
    )


def test_a_tag_that_points_at_a_tree_is_refused(tmp_path: Path, site: ModuleType) -> None:
    repository = released_repository(tmp_path)
    tag(repository, "v7.7.8", NEWEST, target="HEAD^{tree}")
    assert_refuses(
        site, repository, tmp_path / "out", "HEAD", "the tag v7.7.8 does not point at a commit"
    )


def test_a_main_ref_that_does_not_resolve_is_refused(tmp_path: Path, site: ModuleType) -> None:
    repository = released_repository(tmp_path)
    assert_refuses(
        site, repository, tmp_path / "out", "origin/nowhere", "origin/nowhere does not resolve"
    )


def test_a_main_ref_that_is_a_tree_is_refused(tmp_path: Path, site: ModuleType) -> None:
    repository = released_repository(tmp_path)
    assert_refuses(
        site, repository, tmp_path / "out", "HEAD^{tree}", "HEAD^{tree} does not resolve"
    )


def test_a_non_empty_out_dir_is_refused_before_git_is_touched(
    tmp_path: Path, site: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = released_repository(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "stray.txt").write_text("keep", encoding="utf-8")
    calls = record_commands(site, monkeypatch)
    with pytest.raises(site.SiteError, match="is not an empty directory"):
        site.assemble(out, "HEAD", repository)
    assert [path.name for path in out.iterdir()] == ["stray.txt"]
    assert calls == []
    assert worktree_count(repository) == 1


def test_an_out_dir_that_is_a_file_is_refused_before_git_is_touched(
    tmp_path: Path, site: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = released_repository(tmp_path)
    out = tmp_path / "out"
    out.write_text("file", encoding="utf-8")
    calls = record_commands(site, monkeypatch)
    with pytest.raises(site.SiteError, match="is not an empty directory"):
        site.assemble(out, "HEAD", repository)
    assert out.read_text(encoding="utf-8") == "file"
    assert calls == []


def test_an_empty_existing_out_dir_is_accepted(tmp_path: Path, site: ModuleType) -> None:
    repository = released_repository(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    site.assemble(out, "HEAD", repository)
    assert info(out)["version"] == "7.7.7"
    assert info(out / "main")["version"] == "7.7.7"


def test_a_relative_out_dir_is_built_where_it_names(
    tmp_path: Path, site: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = released_repository(tmp_path)
    monkeypatch.chdir(tmp_path)
    site.assemble(Path("relative-out"), "HEAD", repository)
    assert info(tmp_path / "relative-out")["version"] == "7.7.7"
    assert info(tmp_path / "relative-out" / "main")["version"] == "7.7.7"


def test_each_half_is_built_by_its_own_trees_build_script(tmp_path: Path, site: ModuleType) -> None:
    repository = released_repository(tmp_path)
    tagged = head(repository)
    original = (repository / "browser" / "build.py").read_text(encoding="utf-8")
    changed = original.replace(
        MARKER_LINE, MARKER_LINE + '    (out_dir / "marker.txt").write_text("m")\n'
    )
    assert changed != original
    write_build_script(repository, changed)
    out = tmp_path / "out"
    site.assemble(out, head(repository), repository)
    assert (out / "main" / "marker.txt").is_file()
    assert not (out / "marker.txt").exists()
    assert info(out) | {"parameter_years": None} == {
        "parameter_years": None,
        "commit": tagged,
        "version": "7.7.7",
        "released": True,
    }
    assert info(out / "main")["commit"] == head(repository)
    assert info(out / "main")["released"] is False


def test_main_at_the_tagged_commit_is_released(tmp_path: Path, site: ModuleType) -> None:
    repository = released_repository(tmp_path)
    out = tmp_path / "out"
    site.assemble(out, head(repository), repository)
    assert info(out / "main")["released"] is True
    assert info(out / "main")["commit"] == info(out)["commit"]


def test_a_branch_name_as_main_ref_is_resolved_to_its_commit(
    tmp_path: Path, site: ModuleType
) -> None:
    repository = released_repository(tmp_path)
    set_version(repository, "7.8.0")
    git(repository, "update-ref", "refs/remotes/origin/main", head(repository))
    git(repository, "reset", "-q", "--hard", "HEAD^")
    out = tmp_path / "out"
    site.assemble(out, "origin/main", repository)
    assert info(out / "main")["version"] == "7.8.0"
    assert info(out)["version"] == "7.7.7"


def test_a_failing_main_build_is_reported_with_its_stderr_and_cleaned_up(
    tmp_path: Path, site: ModuleType
) -> None:
    repository = released_repository(tmp_path)
    write_build_script(
        repository, 'import sys\nprint("boom from the main build", file=sys.stderr)\nsys.exit(1)\n'
    )
    failing = head(repository)
    with pytest.raises(site.SiteError) as caught:
        site.assemble(tmp_path / "out", failing, repository)
    assert f"the build of main at {failing} failed" in str(caught.value)
    assert "boom from the main build" in str(caught.value)
    assert worktree_count(repository) == 1


def test_a_failing_root_build_is_reported_and_main_is_not_built(
    tmp_path: Path, site: ModuleType
) -> None:
    repository = scratch_repository(tmp_path / "repository")
    write_build_script(
        repository, 'import sys\nprint("boom from the root build", file=sys.stderr)\nsys.exit(1)\n'
    )
    tag(repository, "v7.7.7", NEWEST)
    out = tmp_path / "out"
    with pytest.raises(site.SiteError) as caught:
        site.assemble(out, "HEAD", repository)
    assert "the build of the root at v7.7.7 failed" in str(caught.value)
    assert "boom from the root build" in str(caught.value)
    assert not (out / "main").exists()
    assert worktree_count(repository) == 1


def test_the_worktrees_are_removed_after_a_success(tmp_path: Path, site: ModuleType) -> None:
    repository = released_repository(tmp_path)
    site.assemble(tmp_path / "out", "HEAD", repository)
    assert worktree_count(repository) == 1
    assert not (repository / ".git" / "worktrees").exists()


def test_a_worktree_whose_directory_a_build_deleted_is_dropped(
    tmp_path: Path, site: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The removal drops the registration of a worktree whose directory is already gone."""
    repository = released_repository(tmp_path)
    real_build = site._build

    def deleting(worktree: Path, target: Path, half: str) -> None:
        real_build(worktree, target, half)
        subprocess.run(["rm", "-rf", str(worktree)], check=True)

    real_run = site._run

    def run_without_prune(root: Path, *arguments: str) -> object:
        if arguments[:2] == ("worktree", "prune"):
            return subprocess.CompletedProcess(arguments, 0, b"", b"")
        return real_run(root, *arguments)

    monkeypatch.setattr(site, "_build", deleting)
    monkeypatch.setattr(site, "_run", run_without_prune)
    site.assemble(tmp_path / "out", "HEAD", repository)
    assert worktree_count(repository) == 1
    assert not list((repository / ".git" / "worktrees").glob("*"))


def test_the_prune_drops_registrations_that_the_removal_did_not(
    tmp_path: Path, site: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With every removal failing, only the prune after the scratch directory is gone helps."""
    repository = released_repository(tmp_path)
    real_run = site._run

    def run_without_removal(root: Path, *arguments: str) -> object:
        if arguments[:2] == ("worktree", "remove"):
            return subprocess.CompletedProcess(arguments, 1, b"", b"")
        return real_run(root, *arguments)

    monkeypatch.setattr(site, "_run", run_without_removal)
    site.assemble(tmp_path / "out", "HEAD", repository)
    assert worktree_count(repository) == 1


def test_an_interrupt_in_a_build_propagates_and_the_worktrees_are_removed(
    tmp_path: Path, site: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = released_repository(tmp_path)

    def interrupted(_worktree: Path, _target: Path, _half: str) -> None:
        raise KeyboardInterrupt

    calls = record_commands(site, monkeypatch)
    monkeypatch.setattr(site, "_build", interrupted)
    with pytest.raises(KeyboardInterrupt):
        site.assemble(tmp_path / "out", "HEAD", repository)
    assert worktree_verbs(calls).count("add") == 1
    assert worktree_verbs(calls).count("remove") == 1
    assert worktree_count(repository) == 1


def test_an_unexpected_error_in_a_build_propagates_and_the_worktrees_are_removed(
    tmp_path: Path, site: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = released_repository(tmp_path)

    def broken(_worktree: Path, _target: Path, _half: str) -> None:
        raise RuntimeError("unexpected")

    calls = record_commands(site, monkeypatch)
    monkeypatch.setattr(site, "_build", broken)
    with pytest.raises(RuntimeError, match="unexpected"):
        site.assemble(tmp_path / "out", "HEAD", repository)
    assert worktree_verbs(calls).count("add") == 1
    assert worktree_verbs(calls).count("remove") == 1
    assert worktree_count(repository) == 1


def test_a_worktree_add_that_fails_after_registering_a_locked_entry_is_cleaned_up(
    tmp_path: Path, site: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second add that registers and locks its worktree, then fails, as a killed add can."""
    repository = released_repository(tmp_path)
    real_git = site._git
    adds: list[str] = []

    def failing_second_add(root: Path, *arguments: str) -> bytes:
        output = real_git(root, *arguments)
        if arguments[:2] == ("worktree", "add"):
            adds.append(arguments[-2])
            if len(adds) == 2:
                real_git(root, "worktree", "lock", arguments[-2])
                raise site.SiteError("worktree add failed")
        return output

    monkeypatch.setattr(site, "_git", failing_second_add)
    with pytest.raises(site.SiteError, match="worktree add failed"):
        site.assemble(tmp_path / "out", "HEAD", repository)
    assert len(adds) == 2
    assert worktree_count(repository) == 1


def test_a_tag_name_that_is_not_utf8_is_ignored(
    site: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listing = b"v7.7.\xff tag\nv7.7.8 commit\nv7.7.7 tag\n"
    monkeypatch.setattr(site, "_git", lambda _root, *_arguments: listing)
    assert site._highest_release_tag(tmp_path) == "v7.7.7"


def test_a_tag_name_with_a_line_separator_character_is_one_line(
    site: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listing = "x\u0085y tag\nx\u2028y tag\nv7.7.7 tag\n".encode()
    monkeypatch.setattr(site, "_git", lambda _root, *_arguments: listing)
    assert site._highest_release_tag(tmp_path) == "v7.7.7"


def test_the_workflow_and_the_script_agree_on_what_a_release_tag_is(site: ModuleType) -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "docker-image.yml").read_text(
        encoding="utf-8"
    )
    found = re.findall(r're\.fullmatch\(r"([^"]+)", tag\)', workflow)
    assert found == [site.RELEASE_TAG.pattern]


def test_the_worktrees_are_pruned_after_they_are_removed(
    tmp_path: Path, site: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = released_repository(tmp_path)
    calls = record_commands(site, monkeypatch)
    site.assemble(tmp_path / "out", "HEAD", repository)
    worktree_calls = worktree_verbs(calls)
    assert worktree_calls.count("add") == 2
    assert worktree_calls.count("remove") == 2
    assert worktree_calls[-1] == "prune"


def test_wrong_argument_count_prints_usage_and_exits_2(
    site: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    usage = (
        "usage: python browser/site.py OUT_DIR MAIN_REF\n"
        "       python browser/site.py --image-tags TAG\n"
    )
    for argv in (
        ["site.py"],
        ["site.py", "out"],
        ["site.py", "out", "main", "extra"],
        ["site.py", "--image-tags"],
        ["site.py", "--image-tags", "v7.7.7", "extra"],
    ):
        assert site.main(argv) == 2
        assert capsys.readouterr().err == usage


def test_a_refusal_exits_1_with_the_message_on_stderr(
    tmp_path: Path, site: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "stray.txt").write_text("keep", encoding="utf-8")
    assert site.main(["site.py", str(out), "HEAD"]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("site.py: ")
    assert "is not an empty directory" in captured.err
    assert captured.out == ""
    assert [path.name for path in out.iterdir()] == ["stray.txt"]


WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker-image.yml"


def release_tags(site: ModuleType, repository: Path) -> list[str]:
    """The names of the annotated release tags of ``repository``."""
    return sorted(name for name, version in site._tags(repository).items() if version is not None)


def backport_repository(tmp_path: Path) -> Path:
    """v7.7.7 oldest, v7.8.0 in the middle, and the backport v7.7.8 tagged last."""
    repository = scratch_repository(tmp_path / "repository")
    set_version(repository, "7.7.7")
    tag(repository, "v7.7.7", OLD)
    set_version(repository, "7.8.0")
    tag(repository, "v7.8.0", MIDDLE)
    set_version(repository, "7.7.8")
    tag(repository, "v7.7.8", NEWEST)
    return repository


def lightweight_newer_repository(tmp_path: Path) -> Path:
    """An annotated v7.8.0 whose pyproject matches, and a lightweight v7.9.0."""
    repository = scratch_repository(tmp_path / "repository")
    set_version(repository, "7.8.0")
    tag(repository, "v7.8.0", MIDDLE)
    tag(repository, "v7.9.0", NEWEST, annotated=False)
    return repository


def two_digit_repository(tmp_path: Path, lower: str, higher: str) -> Path:
    """The higher version tagged long ago, and the lower one tagged last."""
    repository = scratch_repository(tmp_path / "repository")
    set_version(repository, higher)
    tag(repository, f"v{higher}", OLD)
    set_version(repository, lower)
    tag(repository, f"v{lower}", NEWEST)
    return repository


VERSION_PAIRS = [
    ("7.7.9", "7.7.10", (False, False)),
    ("7.9.9", "7.10.0", (False, True)),
    ("9.9.9", "10.0.0", (False, True)),
]


@pytest.mark.parametrize(("lower", "higher", "lower_gets"), VERSION_PAIRS)
def test_versions_are_compared_as_integers(
    tmp_path: Path, site: ModuleType, lower: str, higher: str, lower_gets: tuple[bool, bool]
) -> None:
    repository = two_digit_repository(tmp_path, lower, higher)
    assert f"v{lower}" > f"v{higher}"
    assert creation_date(repository, f"v{lower}") > creation_date(repository, f"v{higher}")
    out = tmp_path / "out"
    site.assemble(out, "HEAD", repository)
    assert info(out)["version"] == higher
    assert site.image_tags(f"v{higher}", repository) == (True, True)
    assert site.image_tags(f"v{lower}", repository) == lower_gets


def test_a_backport_tagged_after_a_newer_release_is_neither_the_root_nor_latest(
    tmp_path: Path, site: ModuleType
) -> None:
    repository = backport_repository(tmp_path)
    out = tmp_path / "out"
    site.assemble(out, "HEAD", repository)
    assert info(out)["version"] == "7.8.0"
    assert site.image_tags("v7.8.0", repository) == (True, True)
    assert site.image_tags("v7.7.8", repository) == (False, True)
    assert site.image_tags("v7.7.7", repository) == (False, False)


def test_x_y_is_the_series_of_both_numbers(tmp_path: Path, site: ModuleType) -> None:
    repository = scratch_repository(tmp_path / "repository")
    for name in ("v7.7.7", "v7.7.8", "v7.8.0", "v8.7.9"):
        tag(repository, name, MIDDLE)
    assert site.image_tags("v7.7.8", repository) == (False, True)
    assert site.image_tags("v7.8.0", repository) == (False, True)
    assert site.image_tags("v8.7.9", repository) == (True, True)
    assert site.image_tags("v7.7.7", repository) == (False, False)


def test_a_lightweight_tag_gets_neither_and_does_not_count(
    tmp_path: Path, site: ModuleType
) -> None:
    repository = scratch_repository(tmp_path / "repository")
    tag(repository, "v7.8.0", MIDDLE)
    tag(repository, "v7.9.0", NEWEST, annotated=False)
    tag(repository, "v7.8.1", NEWEST, annotated=False)
    assert site.image_tags("v7.9.0", repository) == (False, False)
    assert site.image_tags("v7.8.1", repository) == (False, False)
    assert site.image_tags("v7.8.0", repository) == (True, True)


@pytest.mark.parametrize("name", ["v7.8.0-rc1", "vtest", "v07.7.8", "", "7.7.7"])
def test_image_tags_refuses_a_name_that_is_not_a_release_name(
    tmp_path: Path, site: ModuleType, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    repository = released_repository(tmp_path)
    if name:
        tag(repository, name, NEWEST)
        assert name in git(repository, "tag", "--list", name).split()
    calls = record_commands(site, monkeypatch)
    with pytest.raises(site.SiteError, match=r"is not named vN\.N\.N"):
        site.image_tags(name, repository)
    assert calls == []


def test_image_tags_refuses_a_tag_that_does_not_exist(tmp_path: Path, site: ModuleType) -> None:
    repository = released_repository(tmp_path)
    git(repository, "branch", "v7.7.9")
    with pytest.raises(site.SiteError, match=r"no tag named v7\.7\.9 exists"):
        site.image_tags("v7.7.9", repository)


def test_the_only_release_gets_both(tmp_path: Path, site: ModuleType) -> None:
    repository = released_repository(tmp_path)
    assert site.image_tags("v7.7.7", repository) == (True, True)


def test_image_latest_agrees_with_the_root(tmp_path: Path, site: ModuleType) -> None:
    corpus = {
        "backport": backport_repository(tmp_path / "a"),
        "lightweight": lightweight_newer_repository(tmp_path / "b"),
        "integers": two_digit_repository(tmp_path / "c", "7.7.9", "7.7.10"),
    }
    for label, repository in corpus.items():
        out = tmp_path / f"out-{label}"
        site.assemble(out, "HEAD", repository)
        names = release_tags(site, repository)
        latest = [name for name in names if site.image_tags(name, repository)[0]]
        assert len(latest) == 1, label
        assert latest[0] == f"v{info(out)['version']}", label


@pytest.mark.parametrize(
    ("pair", "printed"),
    [
        ((True, False), "latest=true\nminor=false\n"),
        ((False, True), "latest=false\nminor=true\n"),
    ],
)
def test_image_tags_prints_two_lines_and_exits_0(
    site: ModuleType,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    pair: tuple[bool, bool],
    printed: str,
) -> None:
    seen: list[str] = []

    def stub(tag_name: str) -> tuple[bool, bool]:
        seen.append(tag_name)
        return pair

    monkeypatch.setattr(site, "image_tags", stub)
    assert site.main(["site.py", "--image-tags", "v7.7.7"]) == 0
    captured = capsys.readouterr()
    assert captured.out == printed
    assert captured.err == ""
    assert seen == ["v7.7.7"]


def test_an_image_tags_refusal_exits_1_with_the_message_on_stderr(
    site: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert site.main(["site.py", "--image-tags", "vtest"]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("site.py: ")
    assert "is not named vN.N.N" in captured.err
    assert captured.out == ""


def test_the_image_job_asks_the_script_for_its_moving_tags(
    site: ModuleType, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    steps = yaml.safe_load(text)["jobs"]["image"]["steps"]
    checkout = steps[0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["fetch-depth"] == 0
    login = next(
        i for i, step in enumerate(steps) if "docker/login-action@" in step.get("uses", "")
    )
    moving = next(i for i, step in enumerate(steps) if step.get("id") == "moving")
    meta = next(i for i, step in enumerate(steps) if step.get("id") == "meta")
    assert login < moving < meta
    assert set(steps[moving]) == {"id", "if", "env", "run"}
    assert steps[moving]["if"] == "github.ref_type == 'tag'"
    assert steps[moving]["env"] == {"TAG": "${{ github.ref_name }}"}
    lines = [line.strip() for line in steps[moving]["run"].splitlines() if line.strip()]
    assert lines == [
        'moving=$(python3 browser/site.py --image-tags "$TAG")',
        'echo "$moving" >> "$GITHUB_OUTPUT"',
    ]
    assert steps[meta]["with"]["flavor"].strip() == "latest=false"
    assert steps[meta]["with"]["tags"].splitlines() == [
        "type=ref,event=branch,enable={{is_default_branch}}",
        "type=sha,enable={{is_default_branch}}",
        "type=semver,pattern={{version}}",
        "type=semver,pattern={{major}}.{{minor}},enable=${{ steps.moving.outputs.minor == 'true' }}",
        "type=raw,value=latest,enable=${{ steps.moving.outputs.latest == 'true' }}",
    ]
    read = set(re.findall(r"steps\.moving\.outputs\.(\w+)", text))
    assert read == {"latest", "minor"}
    monkeypatch.setattr(site, "image_tags", lambda _tag: (True, True))
    assert site.main(["site.py", "--image-tags", "v7.7.7"]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert {line.split("=", 1)[0] for line in printed} == read

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""browser/build.py: what the site holds, that it is reproducible, and what it refuses."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_FILES = {
    "index.html",
    "worker.js",
    "northplan.zip",
    "example.yaml",
    "minimal.yaml",
    "build-info.json",
}


def load_build() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "northplan_build_under_test", REPO_ROOT / "browser" / "build.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def listed(root: Path) -> list[str]:
    out = git(root, "ls-files", "-z", "engine", "report", "params")
    return sorted(name for name in out.split("\0") if name)


@pytest.fixture(scope="module")
def build_module() -> ModuleType:
    return load_build()


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory, build_module: ModuleType) -> Path:
    out = tmp_path_factory.mktemp("built") / "site"
    build_module.build(out)
    return out


def test_the_site_holds_exactly_its_files(site: Path) -> None:
    assert {path.name for path in site.iterdir()} == SITE_FILES
    assert all(path.is_file() for path in site.iterdir())


def test_the_pages_files_are_byte_copies(site: Path) -> None:
    for name in ("index.html", "worker.js"):
        assert (site / name).read_bytes() == (REPO_ROOT / "browser" / name).read_bytes()
    assert (site / "example.yaml").read_bytes() == (
        REPO_ROOT / "scenarios" / "example.yaml"
    ).read_bytes()


def test_the_minimal_scenario_is_a_byte_copy(site: Path) -> None:
    assert (site / "minimal.yaml").read_bytes() == (
        REPO_ROOT / "scenarios" / "minimal.yaml"
    ).read_bytes()


def test_the_archive_ships_the_editors_calls(site: Path) -> None:
    with zipfile.ZipFile(site / "northplan.zip") as archive:
        assert "report/editor_calls.py" in archive.namelist()


def test_the_archive_holds_the_listed_files_and_the_entry_point_only(site: Path) -> None:
    with zipfile.ZipFile(site / "northplan.zip") as archive:
        names = archive.namelist()
    assert names == sorted(names)
    assert sorted(names) == sorted([*listed(REPO_ROOT), "northplan_browser.py"])
    assert not [name for name in names if "__pycache__" in name or name.endswith(".pyc")]
    assert any(name.startswith("params/2026/") for name in names)


def test_the_archive_holds_the_working_trees_content(site: Path) -> None:
    with zipfile.ZipFile(site / "northplan.zip") as archive:
        assert (
            archive.read("northplan_browser.py")
            == (REPO_ROOT / "browser" / "northplan_browser.py").read_bytes()
        )
        assert archive.read("report/run.py") == (REPO_ROOT / "report" / "run.py").read_bytes()


def test_the_archive_has_fixed_timestamps_and_is_deflated(site: Path) -> None:
    with zipfile.ZipFile(site / "northplan.zip") as archive:
        infos = archive.infolist()
    assert {info.date_time for info in infos} == {(1980, 1, 1, 0, 0, 0)}
    assert {info.compress_type for info in infos} == {zipfile.ZIP_DEFLATED}


def test_two_builds_of_one_tree_are_byte_identical(
    site: Path, tmp_path: Path, build_module: ModuleType
) -> None:
    again = tmp_path / "again"
    build_module.build(again)
    for name in SITE_FILES:
        assert (again / name).read_bytes() == (site / name).read_bytes(), name


def test_build_info_names_the_parameter_years_and_the_commit(site: Path) -> None:
    info = json.loads((site / "build-info.json").read_text(encoding="utf-8"))
    assert info == {
        "parameter_years": ["2026"],
        "commit": git(REPO_ROOT, "rev-parse", "HEAD").strip(),
    }


def test_a_non_empty_out_dir_is_refused(tmp_path: Path, build_module: ModuleType) -> None:
    (tmp_path / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(build_module.BuildError, match="not an empty directory"):
        build_module.build(tmp_path)
    assert [path.name for path in tmp_path.iterdir()] == ["stray.txt"]


def test_an_empty_existing_out_dir_is_accepted(tmp_path: Path, build_module: ModuleType) -> None:
    build_module.build(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == SITE_FILES


def test_the_command_line_refuses_a_non_empty_out_dir_with_a_message(tmp_path: Path) -> None:
    (tmp_path / "stray.txt").write_text("x", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "browser" / "build.py"), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "not an empty directory" in completed.stderr


def make_repository(root: Path) -> None:
    """A small repository with the layout build.py reads, one commit."""
    for relative, content in {
        "engine/__init__.py": "",
        "report/__init__.py": "",
        "params/2031/federal.yaml": "a: 1\n",
        "params/notes.yaml": "b: 2\n",
        "browser/index.html": "<html></html>\n",
        "browser/worker.js": "// worker\n",
        "browser/northplan_browser.py": "# entry\n",
        "scenarios/example.yaml": "name: x\n",
        "scenarios/minimal.yaml": "name: y\n",
        "README.md": "readme\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "commit",
        "-q",
        "-m",
        "x",
    )


def test_an_untracked_file_under_engine_is_not_shipped(
    tmp_path: Path, build_module: ModuleType
) -> None:
    repository = tmp_path / "repository"
    make_repository(repository)
    (repository / "engine" / "untracked.py").write_text("x = 1\n", encoding="utf-8")
    (repository / "engine" / "__pycache__").mkdir()
    (repository / "engine" / "__pycache__" / "a.pyc").write_bytes(b"\0")
    build_module.build(tmp_path / "out", repository)
    with zipfile.ZipFile(tmp_path / "out" / "northplan.zip") as archive:
        assert archive.namelist() == [
            "engine/__init__.py",
            "northplan_browser.py",
            "params/2031/federal.yaml",
            "params/notes.yaml",
            "report/__init__.py",
        ]


def test_an_intent_to_add_file_is_shipped_from_the_working_tree(
    tmp_path: Path, build_module: ModuleType
) -> None:
    repository = tmp_path / "repository"
    make_repository(repository)
    (repository / "report" / "new.py").write_text("y = 2\n", encoding="utf-8")
    git(repository, "add", "-N", "report/new.py")
    build_module.build(tmp_path / "out", repository)
    with zipfile.ZipFile(tmp_path / "out" / "northplan.zip") as archive:
        assert archive.read("report/new.py") == b"y = 2\n"


def test_a_listed_file_missing_on_disk_is_an_error(
    tmp_path: Path, build_module: ModuleType
) -> None:
    repository = tmp_path / "repository"
    make_repository(repository)
    (repository / "engine" / "__init__.py").unlink()
    with pytest.raises(build_module.BuildError, match=r"engine/__init__\.py"):
        build_module.build(tmp_path / "out", repository)


def test_the_parameter_years_are_the_four_digit_directories_in_the_archive(
    tmp_path: Path, build_module: ModuleType
) -> None:
    repository = tmp_path / "repository"
    make_repository(repository)
    (repository / "params" / "2030").mkdir()
    (repository / "params" / "2030" / "federal.yaml").write_text("a: 1\n", encoding="utf-8")
    (repository / "params" / "12345").mkdir()
    (repository / "params" / "12345" / "x.yaml").write_text("a: 1\n", encoding="utf-8")
    (repository / "params" / "2029").mkdir()
    (repository / "params" / "2029" / "x.yaml").write_text("a: 1\n", encoding="utf-8")
    git(repository, "add", "params/2030", "params/12345")
    build_module.build(tmp_path / "out", repository)
    info = json.loads((tmp_path / "out" / "build-info.json").read_text(encoding="utf-8"))
    assert info["parameter_years"] == ["2030", "2031"]


def test_the_default_parameters_root_is_the_archives_params(site: Path, tmp_path: Path) -> None:
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(site / "northplan.zip") as archive:
        archive.extractall(unpacked)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from engine.params.loader import DEFAULT_PARAMS_ROOT; print(DEFAULT_PARAMS_ROOT)",
            str(unpacked),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert Path(completed.stdout.strip()) == unpacked / "params"
    assert (unpacked / "params" / "2026").is_dir()

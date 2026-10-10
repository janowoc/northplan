# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The package as pip installs it: a real non-editable install in a temporary directory.

The install is made once from a copy of the source, offline, with no build isolation, and
the commands run from a directory outside the repository with the install first on
``PYTHONPATH``. The install has no ``params/`` and no ``scenarios/``, which is the shape
a wheel install has; the repository's own copies are passed in explicitly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "scenarios" / "example.yaml"
PARAMS = REPO_ROOT / "params"
HINT = "pass --params DIR, the directory holding one subdirectory per tax year."


@dataclass(frozen=True)
class Install:
    target: Path
    cwd: Path
    env: dict[str, str]

    def run(self, argv: list[str], *, timeout: float = 300) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            cwd=self.cwd,
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    def script(self, name: str) -> str:
        return str(self.target / "bin" / name)


@pytest.fixture(scope="module")
def install(tmp_path_factory: pytest.TempPathFactory) -> Install:
    root = tmp_path_factory.mktemp("install")
    source = root / "source"
    source.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy(REPO_ROOT / name, source / name)
    for name in ("engine", "api", "cli", "report"):
        shutil.copytree(REPO_ROOT / name, source / name, ignore=ignore)
    target = root / "target"
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--no-cache-dir",
            "--quiet",
            "--target",
            str(target),
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, f"stdout: {built.stdout}\nstderr: {built.stderr}"
    cwd = root / "cwd"
    cwd.mkdir()
    return Install(target, cwd, {**os.environ, "PYTHONPATH": str(target)})


def test_the_install_is_what_the_subprocesses_import(install: Install) -> None:
    script = (
        "import engine, api.main, cli.main; "
        "print(engine.__file__); print(api.main.__file__); print(cli.main.__file__)"
    )
    completed = install.run([sys.executable, "-c", script])
    assert completed.returncode == 0, completed.stderr
    files = completed.stdout.splitlines()
    assert len(files) == 3
    for file in files:
        assert Path(file).resolve().is_relative_to(install.target.resolve()), file
    assert not (install.target / "params").exists()
    assert not (install.target / "scenarios").exists()


def test_the_page_ships_in_the_package(install: Install) -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "api/web"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    expected = {Path(name).relative_to("api/web") for name in tracked}
    shipped_root = install.target / "api" / "web"
    shipped = {path.relative_to(shipped_root) for path in shipped_root.rglob("*") if path.is_file()}
    assert expected
    assert shipped == expected
    for name in expected:
        assert (shipped_root / name).read_bytes() == (REPO_ROOT / "api" / "web" / name).read_bytes()


def test_simulate_without_params_fails_with_the_hint(install: Install) -> None:
    completed = install.run(
        [install.script("northplan"), "simulate", str(EXAMPLE), "--paths", "10", "--out", "x.csv"]
    )
    assert completed.returncode == 1
    assert "the params root itself does not exist" in completed.stderr
    assert f"northplan: {HINT}" in completed.stderr
    assert not (install.cwd / "x.csv").exists()


def test_simulate_with_params_writes_the_tables(install: Install) -> None:
    out = install.cwd / "ok" / "x.csv"
    out.parent.mkdir()
    completed = install.run(
        [
            install.script("northplan"),
            "simulate",
            str(EXAMPLE),
            "--paths",
            "10",
            "--params",
            str(PARAMS),
            "--out",
            str(out),
        ]
    )
    assert completed.returncode == 0, completed.stderr
    for path in (out, out.with_name("x.final.csv")):
        assert path.stat().st_size > 0


def test_serve_without_params_refuses_with_the_hint(install: Install) -> None:
    completed = install.run([install.script("northplan-serve")], timeout=60)
    assert completed.returncode == 1
    assert f"northplan-serve: {HINT}" in completed.stderr


def test_serve_with_a_missing_example_is_a_usage_error(install: Install) -> None:
    completed = install.run(
        [
            install.script("northplan-serve"),
            "--params",
            str(PARAMS),
            "--example",
            str(install.cwd / "absent.yaml"),
        ],
        timeout=60,
    )
    assert completed.returncode == 2
    assert "absent.yaml" in completed.stderr


def test_the_installed_app_serves_the_page_example_and_runs(install: Install) -> None:
    script = """
import json, sys
from pathlib import Path
from fastapi.testclient import TestClient
import api.main

example, params = Path(sys.argv[1]), Path(sys.argv[2])
client = TestClient(api.main.app, base_url="http://localhost")
out = {}
page = client.get("/")
out["page"] = [page.status_code, "<html" in page.text]
out["example_before"] = client.get("/api/example").status_code
api.main.EXAMPLE_PATH = example
served = client.get("/api/example")
out["example_after"] = [served.status_code, served.content == example.read_bytes()]
api.main.PARAMS_ROOT = params
run = client.post(
    "/api/simulate?paths=10",
    content=example.read_bytes(),
    headers={"Content-Type": "application/yaml"},
)
out["simulate"] = run.status_code
print(json.dumps(out))
"""
    completed = install.run([sys.executable, "-c", script, str(EXAMPLE), str(PARAMS)])
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "page": [200, True],
        "example_before": 404,
        "example_after": [200, True],
        "simulate": 200,
    }

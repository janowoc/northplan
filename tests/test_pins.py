# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The pins that two files must agree on, and the form of each pin."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

_FROM_LINE = re.compile(r"\s*FROM\s", re.IGNORECASE)
_FROM_PIN = re.compile(r"(?i:FROM) \S+:\S+@sha256:[0-9a-f]{64}(?: (?i:AS) \S+)?")
_ACTION_PIN = re.compile(r"[^@\s]+@[0-9a-f]{40}")


def _from_lines(dockerfile: str) -> list[str]:
    return [" ".join(line.split()) for line in dockerfile.splitlines() if _FROM_LINE.match(line)]


def _image_and_tag(from_line: str) -> tuple[str, str]:
    reference = from_line.split()[1].split("@")[0]
    image, _, tag = reference.rpartition(":")
    return image, tag


def _jobs(workflows: dict[str, str]) -> list[dict]:
    jobs: list[dict] = []
    for text in workflows.values():
        jobs.extend((yaml.safe_load(text).get("jobs") or {}).values())
    return jobs


def _steps(workflows: dict[str, str]) -> list[dict]:
    steps: list[dict] = []
    for job in _jobs(workflows):
        steps.extend(job.get("steps") or [])
    return steps


def _uses(workflows: dict[str, str]) -> list[object]:
    """Every `uses` of every step and every job (a reusable workflow)."""
    found = [s["uses"] for s in _steps(workflows) if "uses" in s]
    found.extend(j["uses"] for j in _jobs(workflows) if "uses" in j)
    return found


def digest_pins(dockerfile: str) -> list[str]:
    lines = _from_lines(dockerfile)
    problems = [
        f"FROM is not pinned by digest: {line!r}" for line in lines if not _FROM_PIN.fullmatch(line)
    ]
    if not lines:
        problems.append("the Dockerfile has no FROM line")
    return problems


def python_patch(dockerfile: str, python_version_file: str) -> list[str]:
    tags = [
        _image_and_tag(line)[1]
        for line in _from_lines(dockerfile)
        if _image_and_tag(line)[0] == "python"
    ]
    if len(tags) != 1:
        return [f"expected exactly one FROM python, found {len(tags)}"]
    suffix = "-slim-trixie"
    if not tags[0].endswith(suffix):
        return [f"the python tag {tags[0]!r} does not end in {suffix!r}"]
    patch = tags[0][: -len(suffix)]
    if python_version_file.strip() != patch:
        return [
            f".python-version holds {python_version_file.strip()!r}, "
            f"but the Dockerfile's python image is {patch!r}"
        ]
    return []


def setup_uv_version(dockerfile: str, workflows: dict[str, str]) -> list[str]:
    stages = [
        _image_and_tag(line)[1]
        for line in _from_lines(dockerfile)
        if _image_and_tag(line)[0] == "ghcr.io/astral-sh/uv"
    ]
    if len(stages) != 1:
        return [f"expected exactly one uv stage in the Dockerfile, found {len(stages)}"]
    steps = [
        s
        for s in _steps(workflows)
        if str(s.get("uses", "")).lower().startswith("astral-sh/setup-uv@")
    ]
    if not steps:
        return ["no setup-uv step in the workflows"]
    return [
        f"setup-uv version {(s.get('with') or {}).get('version')!r} is not the uv stage's {stages[0]!r}"
        for s in steps
        if (s.get("with") or {}).get("version") != stages[0]
    ]


def build_backend(pyproject: str, uv_lock: str) -> list[str]:
    versions = [
        p["version"] for p in tomllib.loads(uv_lock).get("package", []) if p["name"] == "setuptools"
    ]
    if len(versions) != 1:
        return [f"expected one setuptools in uv.lock, found {len(versions)}"]
    held = (
        tomllib.loads(pyproject).get("tool", {}).get("uv", {}).get("build-constraint-dependencies")
    )
    if held != [f"setuptools=={versions[0]}"]:
        return [
            f"build-constraint-dependencies is {held!r}, but uv.lock pins setuptools {versions[0]}"
        ]
    return []


def actions_by_commit(workflows: dict[str, str]) -> list[str]:
    uses = _uses(workflows)
    problems = []
    for u in uses:
        if not isinstance(u, str):
            problems.append(f"uses is not a string: {u!r}")
        elif not u.startswith("./") and not _ACTION_PIN.fullmatch(u):
            problems.append(f"action is not pinned by commit: {u!r}")
    if not uses:
        problems.append("no action in the workflows")
    return problems


def _repo_workflows(directory: Path = ROOT / ".github" / "workflows") -> dict[str, str]:
    paths = sorted([*directory.glob("*.yml"), *directory.glob("*.yaml")])
    return {p.name: p.read_text() for p in paths}


def _repo_dockerfile() -> str:
    return (ROOT / "Dockerfile").read_text()


_SHA = "a" * 40
_DIGEST = "@sha256:" + "0" * 64
_NO_USES = "jobs:\n  j:\n    steps:\n      - run: true\n"


def _steps_workflow(*uses: str, version: str | None = None) -> dict[str, str]:
    """A one-job workflow with one step per `uses`; `version` goes on each."""
    lines = ["jobs:", "  j:", "    steps:"]
    for u in uses:
        lines.append(f"      - uses: {u}")
        if version is not None:
            lines.extend(["        with:", f'          version: "{version}"'])
    return {"w.yml": "\n".join(lines) + "\n"}


def _uv_stage(tag: str) -> str:
    return f"FROM ghcr.io/astral-sh/uv:{tag}{_DIGEST} AS uv\n"


def test_digest_pins_hold_in_the_repository() -> None:
    assert digest_pins(_repo_dockerfile()) == []


def test_digest_pins_refuse_an_unpinned_from_in_any_spelling() -> None:
    pinned = f"FROM python:3.12-slim{_DIGEST}\n"
    assert digest_pins(pinned) == []
    assert digest_pins(f"from python:3.12-slim{_DIGEST} as uv\n") == []
    assert digest_pins(pinned + "FROM python:3.12-slim\n") == [
        "FROM is not pinned by digest: 'FROM python:3.12-slim'"
    ]
    assert digest_pins(pinned + "from python:3.12-slim\n") == [
        "FROM is not pinned by digest: 'from python:3.12-slim'"
    ]
    assert digest_pins(pinned + "  FROM\tpython:3.12-slim\n") == [
        "FROM is not pinned by digest: 'FROM python:3.12-slim'"
    ]
    assert digest_pins(pinned + f"FROM python:3.12-slim{_DIGEST} junk\n") == [
        f"FROM is not pinned by digest: 'FROM python:3.12-slim{_DIGEST} junk'"
    ]
    assert digest_pins(pinned + f"FROM python:3.12-slim{_DIGEST}0\n") == [
        f"FROM is not pinned by digest: 'FROM python:3.12-slim{_DIGEST}0'"
    ]


def test_digest_pins_refuse_a_dockerfile_with_no_from_line() -> None:
    assert digest_pins("# FROM python:3.12-slim\nFROMAGE x\n") == [
        "the Dockerfile has no FROM line"
    ]


def test_python_patch_holds_in_the_repository() -> None:
    assert python_patch(_repo_dockerfile(), (ROOT / ".python-version").read_text()) == []


def test_python_patch_refuses_each_way_the_tag_can_disagree() -> None:
    dockerfile = f"FROM python:3.12.15-slim-trixie{_DIGEST}\n"
    assert python_patch(dockerfile, "3.12.15\n") == []
    assert python_patch(dockerfile, "3.12.14\n") == [
        ".python-version holds '3.12.14', but the Dockerfile's python image is '3.12.15'"
    ]
    assert python_patch("FROM debian:13\n", "3.12.15\n") == [
        "expected exactly one FROM python, found 0"
    ]
    assert python_patch(dockerfile + dockerfile, "3.12.15\n") == [
        "expected exactly one FROM python, found 2"
    ]
    assert python_patch(f"FROM python:3.12.15-slim{_DIGEST}\n", "3.12.15\n") == [
        "the python tag '3.12.15-slim' does not end in '-slim-trixie'"
    ]


def test_setup_uv_version_holds_in_the_repository() -> None:
    assert setup_uv_version(_repo_dockerfile(), _repo_workflows()) == []


def test_setup_uv_version_refuses_each_way_it_can_disagree() -> None:
    step = f"astral-sh/setup-uv@{_SHA}"
    dockerfile = _uv_stage("0.12.23")
    assert setup_uv_version(dockerfile, _steps_workflow(step, version="0.12.23")) == []
    assert setup_uv_version(dockerfile, _steps_workflow(step, version="0.12.22")) == [
        "setup-uv version '0.12.22' is not the uv stage's '0.12.23'"
    ]
    assert setup_uv_version("FROM debian:13\n", _steps_workflow(step, version="0.12.23")) == [
        "expected exactly one uv stage in the Dockerfile, found 0"
    ]
    assert setup_uv_version(
        dockerfile + _uv_stage("0.12.23"), _steps_workflow(step, version="0.12.23")
    ) == ["expected exactly one uv stage in the Dockerfile, found 2"]
    assert setup_uv_version(dockerfile, {"w.yml": _NO_USES}) == [
        "no setup-uv step in the workflows"
    ]
    upper = _steps_workflow(f"ASTRAL-SH/setup-uv@{_SHA}", version="0.12.22")
    assert setup_uv_version(dockerfile, upper) == [
        "setup-uv version '0.12.22' is not the uv stage's '0.12.23'"
    ]


def test_build_backend_holds_in_the_repository() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert build_backend(pyproject, (ROOT / "uv.lock").read_text()) == []


def test_build_backend_refuses_each_way_it_can_disagree() -> None:
    pyproject = '[tool.uv]\nbuild-constraint-dependencies = ["setuptools==84.0.0"]\n'
    package = '[[package]]\nname = "setuptools"\nversion = "{v}"\n'
    assert build_backend(pyproject, package.format(v="84.0.0")) == []
    assert build_backend(pyproject, package.format(v="83.0.0")) == [
        "build-constraint-dependencies is ['setuptools==84.0.0'], "
        "but uv.lock pins setuptools 83.0.0"
    ]
    assert build_backend(pyproject, "") == ["expected one setuptools in uv.lock, found 0"]
    two = package.format(v="84.0.0") * 2
    assert build_backend(pyproject, two) == ["expected one setuptools in uv.lock, found 2"]
    lock = package.format(v="84.0.0")
    assert build_backend("[project]\nname = 'x'\n", lock) == [
        "build-constraint-dependencies is None, but uv.lock pins setuptools 84.0.0"
    ]


def test_actions_by_commit_hold_in_the_repository() -> None:
    assert actions_by_commit(_repo_workflows()) == []


def test_actions_by_commit_refuse_each_way_a_uses_can_float() -> None:
    assert actions_by_commit(_steps_workflow(f"actions/checkout@{_SHA}")) == []
    assert actions_by_commit(_steps_workflow("actions/checkout@v7")) == [
        "action is not pinned by commit: 'actions/checkout@v7'"
    ]
    reusable = "jobs:\n  j:\n    uses: org/repo/.github/workflows/w.yml@main\n"
    assert actions_by_commit({"w.yml": reusable}) == [
        "action is not pinned by commit: 'org/repo/.github/workflows/w.yml@main'"
    ]
    block = f"jobs:\n  j:\n    steps:\n      - uses: |\n          actions/checkout@{_SHA}\n"
    assert actions_by_commit({"w.yml": block}) == [
        f"action is not pinned by commit: 'actions/checkout@{_SHA}\\n'"
    ]
    local = "jobs:\n  j:\n    uses: ./.github/workflows/w.yml\n"
    assert actions_by_commit({"w.yml": local}) == []
    assert actions_by_commit({"w.yml": _NO_USES}) == ["no action in the workflows"]
    not_a_string = "jobs:\n  j:\n    steps:\n      - uses: 7\n"
    assert actions_by_commit({"w.yml": not_a_string}) == ["uses is not a string: 7"]


def test_the_workflow_reader_takes_yml_and_yaml(tmp_path: Path) -> None:
    (tmp_path / "a.yml").write_text("a: 1\n")
    (tmp_path / "b.yaml").write_text("b: 2\n")
    (tmp_path / "c.txt").write_text("c: 3\n")
    assert _repo_workflows(tmp_path) == {"a.yml": "a: 1\n", "b.yaml": "b: 2\n"}

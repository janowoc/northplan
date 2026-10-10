# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``api.serve.main``: its refusals, the globals it sets, and per-request parameters.

``uvicorn.run`` is replaced by a recorder, so nothing binds a port. The oracle for the
parameters directory is the default behaviour: a copy of ``params/`` must answer exactly
as the original does.
"""

from __future__ import annotations

import errno
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.main
import api.serve
from engine.params.loader import DEFAULT_PARAMS_ROOT

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "scenarios" / "example.yaml"
COUPLE = REPO_ROOT / "scenarios" / "late_life_couple.yaml"
HINT = "northplan-serve: pass --params DIR, the directory holding one subdirectory per tax year."

Lock = Callable[[Path, int], None]
DENIED = os.strerror(errno.EACCES)


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[object, str, int]] = []

    def __call__(self, app: object, *, host: str, port: int) -> None:
        self.calls.append((app, host, port))


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    record = Recorder()
    monkeypatch.setattr(api.serve.uvicorn, "run", record)
    monkeypatch.setattr(api.main, "PARAMS_ROOT", api.main.PARAMS_ROOT)
    monkeypatch.setattr(api.main, "EXAMPLE_PATH", api.main.EXAMPLE_PATH)
    return record


def _copied_params(tmp_path: Path) -> Path:
    root = tmp_path / "params"
    shutil.copytree(DEFAULT_PARAMS_ROOT, root)
    return root


def run_serve(argv: list[str]) -> int:
    """``main(argv)``, with a usage error's ``SystemExit`` turned into its code."""
    try:
        return api.serve.main(argv)
    except SystemExit as exit_:
        assert isinstance(exit_.code, int)
        return exit_.code


def test_no_flags_serves_on_loopback_8000_with_the_defaults(recorder: Recorder) -> None:
    example = api.main.EXAMPLE_PATH
    assert api.serve.main([]) == 0
    assert recorder.calls == [(api.main.app, "127.0.0.1", 8000)]
    assert api.main.PARAMS_ROOT is DEFAULT_PARAMS_ROOT
    assert example == api.main.EXAMPLE_PATH


def test_flags_set_the_params_root_example_and_port(recorder: Recorder, tmp_path: Path) -> None:
    params = tmp_path / "params"
    shutil.copytree(DEFAULT_PARAMS_ROOT, params)
    example = tmp_path / "example.yaml"
    example.write_text("x: 1\n", encoding="utf-8")
    argv = ["--params", str(params), "--example", str(example), "--port", "8123"]
    assert api.serve.main(argv) == 0
    assert params == api.main.PARAMS_ROOT
    assert example == api.main.EXAMPLE_PATH
    assert recorder.calls == [(api.main.app, "127.0.0.1", 8123)]


def test_a_given_params_directory_that_does_not_exist_refuses_to_start(
    recorder: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    absent = tmp_path / "absent"
    root, example = api.main.PARAMS_ROOT, api.main.EXAMPLE_PATH
    assert api.serve.main(["--params", str(absent)]) == 1
    stderr = capsys.readouterr().err
    assert str(absent) in stderr
    assert "pass --params" not in stderr
    assert recorder.calls == []
    assert api.main.PARAMS_ROOT is root
    assert api.main.EXAMPLE_PATH is example


def test_a_missing_default_params_directory_refuses_with_the_hint(
    recorder: Recorder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    absent = tmp_path / "absent"
    monkeypatch.setattr(api.serve, "DEFAULT_PARAMS_ROOT", absent)
    assert api.serve.main([]) == 1
    lines = capsys.readouterr().err.splitlines()
    assert str(absent) in lines[0]
    assert lines[1] == HINT
    assert recorder.calls == []


def test_a_params_path_that_is_a_file_refuses_to_start(
    recorder: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    file = tmp_path / "params"
    file.write_text("x\n", encoding="utf-8")
    assert api.serve.main(["--params", str(file)]) == 1
    assert str(file) in capsys.readouterr().err
    assert recorder.calls == []


def test_a_missing_example_is_a_usage_error(
    recorder: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    absent = tmp_path / "absent.yaml"
    root, example = api.main.PARAMS_ROOT, api.main.EXAMPLE_PATH
    assert run_serve(["--example", str(absent)]) == 2
    assert str(absent) in capsys.readouterr().err
    assert recorder.calls == []
    assert api.main.PARAMS_ROOT is root
    assert api.main.EXAMPLE_PATH is example


def test_the_params_check_runs_before_the_example_check(
    recorder: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    absent = tmp_path / "absent"
    argv = ["--params", str(absent), "--example", str(tmp_path / "absent.yaml")]
    assert api.serve.main(argv) == 1
    assert str(absent) in capsys.readouterr().err
    assert recorder.calls == []


def test_a_missing_example_leaves_the_params_root_unset(recorder: Recorder, tmp_path: Path) -> None:
    params = tmp_path / "params"
    shutil.copytree(DEFAULT_PARAMS_ROOT, params)
    root, example = api.main.PARAMS_ROOT, api.main.EXAMPLE_PATH
    argv = ["--params", str(params), "--example", str(tmp_path / "absent.yaml")]
    assert run_serve(argv) == 2
    assert api.main.PARAMS_ROOT is root
    assert api.main.EXAMPLE_PATH is example
    assert recorder.calls == []


def test_an_example_that_is_a_directory_is_a_usage_error(
    recorder: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_serve(["--example", str(tmp_path)]) == 2
    assert str(tmp_path) in capsys.readouterr().err
    assert recorder.calls == []


def test_a_missing_default_example_still_starts(
    recorder: Recorder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api.main, "EXAMPLE_PATH", tmp_path / "absent.yaml")
    assert api.serve.main([]) == 0
    assert len(recorder.calls) == 1


@pytest.mark.parametrize(
    ("port", "message"),
    [
        ("0", "'0' must be between 1 and 65535"),
        ("65536", "'65536' must be between 1 and 65535"),
        ("-1", "'-1' must be between 1 and 65535"),
        ("abc", "'abc' is not an integer"),
        ("8.5", "'8.5' is not an integer"),
    ],
)
def test_port_bounds_refuse(
    recorder: Recorder, capsys: pytest.CaptureFixture[str], port: str, message: str
) -> None:
    assert run_serve(["--port", port]) == 2
    assert recorder.calls == []
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("port", [1, 65535])
def test_port_bounds_accept(recorder: Recorder, port: int) -> None:
    assert api.serve.main(["--port", str(port)]) == 0
    assert recorder.calls == [(api.main.app, "127.0.0.1", port)]


def _assert_refused_to_start(
    recorder: Recorder,
    capsys: pytest.CaptureFixture[str],
    code: int,
    message: str,
    params_root_before: object,
) -> None:
    assert code == 1
    assert recorder.calls == []
    assert api.main.PARAMS_ROOT is params_root_before
    assert capsys.readouterr().err.splitlines() == [f"northplan-serve: error: {message}"]


def test_a_params_root_whose_parent_cannot_be_entered_refuses_to_start(
    recorder: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str], lock: Lock
) -> None:
    parent = tmp_path / "outer"
    root = parent / "params"
    shutil.copytree(DEFAULT_PARAMS_ROOT, root)
    lock(parent, 0o000)
    before = api.main.PARAMS_ROOT
    code = api.serve.main(["--params", str(root)])
    message = f"Cannot look up the parameters directory {root}: {DENIED}."
    _assert_refused_to_start(recorder, capsys, code, message, before)


def test_a_params_root_that_cannot_be_entered_or_read_refuses_to_start(
    recorder: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str], lock: Lock
) -> None:
    root = _copied_params(tmp_path)
    lock(root, 0o000)
    before = api.main.PARAMS_ROOT
    code = api.serve.main(["--params", str(root)])
    message = f"Cannot list the parameters directory {root}: {DENIED}."
    _assert_refused_to_start(recorder, capsys, code, message, before)


def test_a_params_root_that_can_be_read_but_not_entered_refuses_to_start(
    recorder: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str], lock: Lock
) -> None:
    root = _copied_params(tmp_path)
    lock(root, 0o444)
    before = api.main.PARAMS_ROOT
    code = api.serve.main(["--params", str(root)])
    message = f"Cannot look up tax year 2026 in the parameters directory {root}: {DENIED}."
    _assert_refused_to_start(recorder, capsys, code, message, before)


def test_an_unreadable_year_directory_refuses_to_start(
    recorder: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str], lock: Lock
) -> None:
    root = _copied_params(tmp_path)
    lock(root / "2026", 0o000)
    before = api.main.PARAMS_ROOT
    code = api.serve.main(["--params", str(root)])
    message = f"Cannot list the parameters directory {root / '2026'}: {DENIED}."
    _assert_refused_to_start(recorder, capsys, code, message, before)


def test_a_params_root_that_can_be_entered_but_not_read_starts(
    recorder: Recorder, tmp_path: Path, lock: Lock
) -> None:
    root = _copied_params(tmp_path)
    lock(root, 0o111)
    with pytest.raises(PermissionError):
        list(root.iterdir())
    assert api.serve.main(["--params", str(root)]) == 0
    assert len(recorder.calls) == 1
    assert root == api.main.PARAMS_ROOT


def test_an_unreadable_default_params_directory_refuses_without_the_hint(
    recorder: Recorder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    lock: Lock,
) -> None:
    root = _copied_params(tmp_path)
    lock(root, 0o000)
    monkeypatch.setattr(api.serve, "DEFAULT_PARAMS_ROOT", root)
    before = api.main.PARAMS_ROOT
    code = api.serve.main([])
    message = f"Cannot list the parameters directory {root}: {DENIED}."
    _assert_refused_to_start(recorder, capsys, code, message, before)


def test_the_unreadable_check_runs_before_the_example_check(
    recorder: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str], lock: Lock
) -> None:
    root = _copied_params(tmp_path)
    lock(root, 0o000)
    argv = ["--params", str(root), "--example", str(tmp_path / "absent.yaml")]
    assert run_serve(argv) == 1
    assert str(root) in capsys.readouterr().err
    assert recorder.calls == []


def post(client: TestClient, scenario: Path, paths: int) -> object:
    return client.post(
        f"/api/simulate?paths={paths}",
        content=scenario.read_bytes(),
        headers={"Content-Type": "application/yaml"},
    )


def test_params_root_is_read_per_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(api.main.app, base_url="http://localhost")
    absent = tmp_path / "absent"
    monkeypatch.setattr(api.main, "PARAMS_ROOT", absent)
    response = post(client, EXAMPLE, 10)
    assert response.status_code == 400  # type: ignore[attr-defined]
    assert str(absent) in response.json()["detail"]  # type: ignore[attr-defined]


@pytest.mark.parametrize("endpoint", ["simulate", "optimize"])
def test_an_unreadable_params_root_is_a_400(
    endpoint: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock: Lock
) -> None:
    root = _copied_params(tmp_path)
    lock(root, 0o000)
    monkeypatch.setattr(api.main, "PARAMS_ROOT", root)
    client = TestClient(api.main.app, base_url="http://localhost")
    query = "paths=10" + ("&objective=median_estate_after_tax" if endpoint == "optimize" else "")
    response = client.post(
        f"/api/{endpoint}?{query}",
        content=EXAMPLE.read_bytes(),
        headers={"Content-Type": "application/yaml"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == (
        f"Cannot look up tax year 2026 in the parameters directory {root}: {DENIED}."
    )


def test_an_unreadable_year_directory_is_a_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock: Lock
) -> None:
    root = _copied_params(tmp_path)
    lock(root / "2026", 0o000)
    monkeypatch.setattr(api.main, "PARAMS_ROOT", root)
    client = TestClient(api.main.app, base_url="http://localhost")
    response = post(client, EXAMPLE, 10)
    assert response.status_code == 400  # type: ignore[attr-defined]
    assert response.json()["detail"] == (  # type: ignore[attr-defined]
        f"Cannot list the parameters directory {root / '2026'}: {DENIED}."
    )


@pytest.mark.parametrize("scenario", [EXAMPLE, COUPLE], ids=lambda path: path.stem)
def test_a_copied_params_root_gives_the_same_response(
    scenario: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(api.main.app, base_url="http://localhost")
    default = post(client, scenario, 50)
    assert default.status_code == 200  # type: ignore[attr-defined]
    copy = tmp_path / "params"
    shutil.copytree(DEFAULT_PARAMS_ROOT, copy)
    monkeypatch.setattr(api.main, "PARAMS_ROOT", copy)
    copied = post(client, scenario, 50)
    assert copied.status_code == 200  # type: ignore[attr-defined]
    assert copied.json() == default.json()  # type: ignore[attr-defined]

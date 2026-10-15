# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``api.serve.main``: its refusals, the globals it sets, and per-request parameters.

``uvicorn.run`` is replaced by a recorder, so nothing binds a port. The oracle for the
parameters directory is the default behaviour: a copy of ``params/`` must answer exactly
as the original does.
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import uvicorn
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
        self.calls: list[tuple[object, str, int, bool]] = []
        self.kwargs: list[dict[str, object]] = []

    def __call__(self, app: object, *, host: str, port: int, proxy_headers: bool) -> None:
        self.calls.append((app, host, port, proxy_headers))
        self.kwargs.append({"host": host, "port": port, "proxy_headers": proxy_headers})


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    record = Recorder()
    monkeypatch.setattr(api.serve.uvicorn, "run", record)
    monkeypatch.setattr(api.main, "PARAMS_ROOT", api.main.PARAMS_ROOT)
    monkeypatch.setattr(api.main, "EXAMPLE_PATH", api.main.EXAMPLE_PATH)
    monkeypatch.setattr(api.main, "TOKEN", api.main.TOKEN)
    monkeypatch.delenv("NORTHPLAN_TOKEN", raising=False)
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
    assert recorder.calls == [(api.main.app, "127.0.0.1", 8000, False)]
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
    assert recorder.calls == [(api.main.app, "127.0.0.1", 8123, False)]


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


def _last_line(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().err.rstrip("\n").splitlines()[-1]


def test_an_example_file_of_mode_zero_is_a_usage_error(
    recorder: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str], lock: Lock
) -> None:
    path = tmp_path / "example.yaml"
    path.write_text("x: 1\n", encoding="utf-8")
    lock(path, 0o000)
    assert run_serve(["--example", str(path)]) == 2
    assert recorder.calls == []
    assert _last_line(capsys) == (
        f"northplan-serve: error: --example {str(path)!r} cannot be read: {DENIED}."
    )


def test_an_example_in_a_directory_of_mode_zero_is_a_usage_error(
    recorder: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str], lock: Lock
) -> None:
    directory = tmp_path / "d"
    directory.mkdir()
    path = directory / "example.yaml"
    path.write_text("x: 1\n", encoding="utf-8")
    lock(directory, 0o000)
    assert run_serve(["--example", str(path)]) == 2
    assert recorder.calls == []
    assert _last_line(capsys) == (
        f"northplan-serve: error: --example {str(path)!r} cannot be read: {DENIED}."
    )


def test_an_example_that_is_a_symlink_to_itself_is_a_usage_error(
    recorder: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "loop.yaml"
    path.symlink_to(path)
    assert run_serve(["--example", str(path)]) == 2
    assert recorder.calls == []
    assert _last_line(capsys) == (
        f"northplan-serve: error: --example {str(path)!r} "
        f"cannot be read: {os.strerror(errno.ELOOP)}."
    )


def test_an_example_through_a_regular_file_is_not_a_file(
    recorder: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    through = tmp_path / "f"
    through.write_text("x\n", encoding="utf-8")
    path = through / "example.yaml"
    assert run_serve(["--example", str(path)]) == 2
    assert recorder.calls == []
    assert _last_line(capsys) == f"northplan-serve: error: --example {str(path)!r} is not a file."


def test_an_example_directory_is_not_a_file(
    recorder: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_serve(["--example", str(tmp_path)]) == 2
    assert recorder.calls == []
    assert (
        _last_line(capsys) == f"northplan-serve: error: --example {str(tmp_path)!r} is not a file."
    )


def test_an_example_that_cannot_be_opened_is_a_usage_error(
    recorder: Recorder,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("boom")

    path = tmp_path / "example.yaml"
    path.write_text("x: 1\n", encoding="utf-8")
    monkeypatch.setattr(Path, "open", boom)
    assert run_serve(["--example", str(path)]) == 2
    assert recorder.calls == []
    assert _last_line(capsys) == (
        f"northplan-serve: error: --example {str(path)!r} cannot be read: boom."
    )


def test_an_absent_example_is_not_a_file(
    recorder: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "absent.yaml"
    assert run_serve(["--example", str(path)]) == 2
    assert recorder.calls == []
    assert _last_line(capsys) == f"northplan-serve: error: --example {str(path)!r} is not a file."


def test_an_example_that_is_a_dangling_symlink_is_not_a_file(
    recorder: Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "example.yaml"
    path.symlink_to(tmp_path / "absent.yaml")
    assert run_serve(["--example", str(path)]) == 2
    assert recorder.calls == []
    assert _last_line(capsys) == f"northplan-serve: error: --example {str(path)!r} is not a file."


def test_an_example_fifo_is_not_a_file_and_is_not_opened(
    recorder: Recorder,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unopened(*_args: object, **_kwargs: object) -> None:
        pytest.fail("the example was opened")

    path = tmp_path / "example.yaml"
    os.mkfifo(path)
    monkeypatch.setattr(Path, "open", unopened)
    assert run_serve(["--example", str(path)]) == 2
    assert recorder.calls == []
    assert _last_line(capsys) == f"northplan-serve: error: --example {str(path)!r} is not a file."


def test_an_example_lookup_error_without_a_reason_is_a_usage_error(
    recorder: Recorder,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "example.yaml"
    path.write_text("x: 1\n", encoding="utf-8")
    original = Path.stat

    def stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == path:
            raise OSError("boom")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    assert run_serve(["--example", str(path)]) == 2
    assert recorder.calls == []
    assert _last_line(capsys) == (
        f"northplan-serve: error: --example {str(path)!r} cannot be read: boom."
    )


def test_an_example_in_an_enterable_unlistable_directory_starts(
    recorder: Recorder, tmp_path: Path, lock: Lock
) -> None:
    directory = tmp_path / "d"
    directory.mkdir()
    path = directory / "example.yaml"
    path.write_text("x: 1\n", encoding="utf-8")
    lock(directory, 0o100)
    assert api.serve.main(["--example", str(path)]) == 0
    assert len(recorder.calls) == 1


def test_an_unreadable_default_example_still_starts(
    recorder: Recorder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock: Lock
) -> None:
    path = tmp_path / "example.yaml"
    path.write_text("x: 1\n", encoding="utf-8")
    lock(path, 0o000)
    monkeypatch.setattr(api.main, "EXAMPLE_PATH", path)
    assert api.serve.main([]) == 0
    assert len(recorder.calls) == 1


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
    assert recorder.calls == [(api.main.app, "127.0.0.1", port, False)]


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
    client = TestClient(api.main.app, base_url="http://localhost", client=("127.0.0.1", 50000))
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
    client = TestClient(api.main.app, base_url="http://localhost", client=("127.0.0.1", 50000))
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
    client = TestClient(api.main.app, base_url="http://localhost", client=("127.0.0.1", 50000))
    response = post(client, EXAMPLE, 10)
    assert response.status_code == 400  # type: ignore[attr-defined]
    assert response.json()["detail"] == (  # type: ignore[attr-defined]
        f"Cannot list the parameters directory {root / '2026'}: {DENIED}."
    )


@pytest.mark.parametrize("scenario", [EXAMPLE, COUPLE], ids=lambda path: path.stem)
def test_a_copied_params_root_gives_the_same_response(
    scenario: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(api.main.app, base_url="http://localhost", client=("127.0.0.1", 50000))
    default = post(client, scenario, 50)
    assert default.status_code == 200  # type: ignore[attr-defined]
    copy = tmp_path / "params"
    shutil.copytree(DEFAULT_PARAMS_ROOT, copy)
    monkeypatch.setattr(api.main, "PARAMS_ROOT", copy)
    copied = post(client, scenario, 50)
    assert copied.status_code == 200  # type: ignore[attr-defined]
    assert copied.json() == default.json()  # type: ignore[attr-defined]


# =============================================================================
# --host and the token
# =============================================================================

TOKEN_REFUSAL = (
    "northplan-serve: error: NORTHPLAN_TOKEN must be at least 16 characters, each a "
    "printable ASCII character other than a space. To make one: python3 -c "
    "'import secrets; print(secrets.token_urlsafe(32))'\n"
)


def _generated_line(token: str) -> str:
    return (
        f"northplan-serve: NORTHPLAN_TOKEN is not set, so this run's token is {token}; "
        "it is new on every start.\n"
    )


def test_host_is_passed_to_uvicorn(recorder: Recorder) -> None:
    assert api.serve.main(["--host", "0.0.0.0"]) == 0
    assert recorder.calls == [(api.main.app, "0.0.0.0", 8000, False)]


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "myhost.example", "[::1]"])
@pytest.mark.usefixtures("recorder")
def test_a_token_is_generated_for_a_host_other_than_loopback(
    capsys: pytest.CaptureFixture[str], host: str
) -> None:
    assert api.serve.main(["--host", host]) == 0
    token = api.main.TOKEN
    assert isinstance(token, str)
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", token)
    captured = capsys.readouterr()
    assert captured.err == _generated_line(token)
    assert captured.out == ""


@pytest.mark.usefixtures("recorder")
def test_each_start_generates_a_new_token() -> None:
    assert api.serve.main(["--host", "0.0.0.0"]) == 0
    first = api.main.TOKEN
    assert api.serve.main(["--host", "0.0.0.0"]) == 0
    assert first is not None
    assert api.main.TOKEN is not None
    assert first != api.main.TOKEN


@pytest.mark.parametrize("host", [None, "127.0.0.2", "::1", "localhost", "LOCALHOST"])
@pytest.mark.usefixtures("recorder")
def test_no_token_is_generated_for_a_loopback_host(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, host: str | None
) -> None:
    monkeypatch.setattr(api.main, "TOKEN", "stale-synthetic-token")
    assert api.serve.main([] if host is None else ["--host", host]) == 0
    assert api.main.TOKEN is None
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
@pytest.mark.usefixtures("recorder")
def test_a_set_token_is_used_and_never_written(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    host: str,
) -> None:
    value = "synthetic-token-0123"
    monkeypatch.setenv("NORTHPLAN_TOKEN", value)
    caplog.set_level(logging.DEBUG)
    assert api.serve.main(["--host", host]) == 0
    assert value == api.main.TOKEN
    captured = capsys.readouterr()
    assert value not in captured.out
    assert value not in captured.err
    assert value not in caplog.text
    assert captured.err == ""


@pytest.mark.parametrize(
    "value",
    ["", "x" * 15, "x" * 15 + " ", "x" * 15 + "\t", "x" * 16 + "\n", "x" * 15 + "\u00e9"],
    ids=["empty", "short", "space", "tab", "newline", "non-ascii"],
)
def test_a_token_breaking_the_rules_refuses_to_start(
    recorder: Recorder,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("NORTHPLAN_TOKEN", value)
    assert run_serve([]) == 2
    err = capsys.readouterr().err
    assert err == TOKEN_REFUSAL
    if value:
        assert value not in err
    assert recorder.calls == []
    assert api.main.TOKEN is None


@pytest.mark.parametrize(
    "value",
    ["x" * 16, "".join(chr(code) for code in range(ord("!"), ord("~") + 1))],
    ids=["sixteen", "every-printable"],
)
def test_a_token_at_the_bounds_is_accepted(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("NORTHPLAN_TOKEN", value)
    assert api.serve.main([]) == 0
    assert len(recorder.calls) == 1
    assert value == api.main.TOKEN


@pytest.mark.usefixtures("recorder")
def test_the_token_is_checked_before_the_params(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NORTHPLAN_TOKEN", "x" * 15)
    assert run_serve(["--params", str(tmp_path / "absent")]) == 2
    assert capsys.readouterr().err == TOKEN_REFUSAL


@pytest.mark.usefixtures("recorder")
def test_no_token_is_generated_when_the_start_fails(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert api.serve.main(["--host", "0.0.0.0", "--params", str(tmp_path / "absent")]) == 1
    assert "NORTHPLAN_TOKEN is not set" not in capsys.readouterr().err
    assert api.main.TOKEN is None


def _forwarded_status(config_kwargs: dict[str, object]) -> int:
    """Status of a LAN client claiming X-Forwarded-For 127.0.0.1, through uvicorn's stack."""
    config = uvicorn.Config(api.main.app, log_config=None, **config_kwargs)  # type: ignore[arg-type]
    config.load()
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/health",
        "raw_path": b"/health",
        "query_string": b"",
        "headers": [(b"host", b"localhost"), (b"x-forwarded-for", b"127.0.0.1")],
        "scheme": "http",
        "http_version": "1.1",
        "client": ("192.168.1.5", 40000),
        "server": ("127.0.0.1", 8000),
    }
    assert config.loaded_app is not None
    asyncio.run(config.loaded_app(scope, receive, send))
    return sent[0]["status"]


def test_forwarded_headers_cannot_make_a_lan_client_loopback(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")
    assert api.serve.main(["--host", "0.0.0.0"]) == 0
    kwargs = recorder.kwargs[0]
    monkeypatch.setattr(api.main, "TOKEN", "synthetic-token-0123")
    assert _forwarded_status(kwargs) == 401
    assert _forwarded_status({**kwargs, "proxy_headers": True}) == 200


def test_the_image_runs_northplan_serve_on_every_interface() -> None:
    dockerfile = Path(__file__).resolve().parents[2] / "Dockerfile"
    commands = [
        line
        for line in dockerfile.read_text(encoding="utf-8").splitlines()
        if line.startswith("CMD ")
    ]
    assert len(commands) == 1
    assert json.loads(commands[0][len("CMD ") :]) == ["northplan-serve", "--host", "0.0.0.0"]

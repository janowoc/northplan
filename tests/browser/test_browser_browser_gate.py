# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The browser gate has four outcomes, and none of them needs a browser to be tested."""

from __future__ import annotations

from pathlib import Path

import pytest
from browser_gate import REQUIRE_VARIABLE, require_browser
from playwright.sync_api import Error

NOT_INSTALLED = Error("BrowserType.launch: Executable doesn't exist at /nowhere/chrome")


class Launcher:
    """A fake ``launch`` that records its calls and returns or raises what it was given."""

    def __init__(self, result: object = None, raises: BaseException | None = None) -> None:
        self.result = result
        self.raises = raises
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.result


def present(tmp_path: Path) -> list[Path]:
    path = tmp_path / "present.js"
    path.write_text("", encoding="utf-8")
    return [path]


def test_files_present_returns_what_launch_returns(tmp_path: Path) -> None:
    sentinel = object()
    launcher = Launcher(result=sentinel)
    assert require_browser(launcher, files=present(tmp_path), environ={}) is sentinel
    assert launcher.calls == 1


def test_a_missing_file_and_unset_skips_without_launching(tmp_path: Path) -> None:
    launcher = Launcher()
    with pytest.raises(pytest.skip.Exception, match=r"npm ci --ignore-scripts"):
        require_browser(launcher, files=[tmp_path / "absent.js"], environ={})
    assert launcher.calls == 0


def test_a_missing_file_and_set_fails(tmp_path: Path) -> None:
    with pytest.raises(pytest.fail.Exception, match="is absent"):
        require_browser(Launcher(), files=[tmp_path / "absent.js"], environ={REQUIRE_VARIABLE: "1"})


def test_two_missing_files_are_both_named(tmp_path: Path) -> None:
    first, second = tmp_path / "first.js", tmp_path / "second.js"
    with pytest.raises(pytest.skip.Exception) as raised:
        require_browser(Launcher(), files=[first, second], environ={})
    assert f"{first} is absent and {second} is absent" in str(raised.value)


def test_a_browser_not_installed_and_unset_skips_with_the_way_to_install(tmp_path: Path) -> None:
    with pytest.raises(pytest.skip.Exception, match=r"playwright install --only-shell chromium"):
        require_browser(Launcher(raises=NOT_INSTALLED), files=present(tmp_path), environ={})


def test_a_browser_not_installed_and_set_fails_with_the_same_text(tmp_path: Path) -> None:
    with pytest.raises(pytest.fail.Exception, match=r"playwright install --only-shell chromium"):
        require_browser(
            Launcher(raises=NOT_INSTALLED),
            files=present(tmp_path),
            environ={REQUIRE_VARIABLE: "1"},
        )


def test_a_browser_not_installed_and_not_one_still_skips(tmp_path: Path) -> None:
    with pytest.raises(pytest.skip.Exception):
        require_browser(
            Launcher(raises=NOT_INSTALLED),
            files=present(tmp_path),
            environ={REQUIRE_VARIABLE: "0"},
        )


def test_any_other_playwright_error_propagates(tmp_path: Path) -> None:
    closed = Error("Target page, context or browser has been closed")
    with pytest.raises(Error, match="has been closed"):
        require_browser(
            Launcher(raises=closed), files=present(tmp_path), environ={REQUIRE_VARIABLE: "1"}
        )


def test_any_other_exception_propagates(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        require_browser(Launcher(raises=RuntimeError("boom")), files=present(tmp_path), environ={})

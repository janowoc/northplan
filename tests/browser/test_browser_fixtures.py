# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The browser fixtures leave nothing behind that a later test module trips over."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import Browser

REPO_ROOT = Path(__file__).resolve().parents[2]
API_PAGE_TEST = "tests/browser/test_browser_api_page.py::test_it_shows_the_health_and_the_example"
# This test trips over the loop Playwright leaves marked as running on the main thread,
# because it runs an event loop there with asyncio.run. A replacement must still trip over
# it, for instance by running one there too.
LOOP_TEST = "tests/api/test_api.py::test_host_check_refuses_a_missing_host"


def test_a_later_module_can_still_run_an_event_loop(chromium: Browser) -> None:  # noqa: ARG001
    environment = {**os.environ, "NORTHPLAN_REQUIRE_BROWSER": "1"}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "-p",
            "no:cacheprovider",
            "-q",
            API_PAGE_TEST,
            LOOP_TEST,
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    assert re.search(r"\b2 passed\b", completed.stdout)
    assert "skipped" not in completed.stdout

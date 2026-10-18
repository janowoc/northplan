# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The gate every test that drives a real browser goes through.

Without Chromium, or without the npm files that answer the pages' CDN requests, the test is
skipped with the way to install them; with ``NORTHPLAN_REQUIRE_BROWSER=1`` (CI) it fails
instead, so that a runner that lost the browser does not report green over page tests that
never ran.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pytest
from playwright.sync_api import Error

BROWSER_TESTS = Path(__file__).resolve().parent
REQUIRE_VARIABLE = "NORTHPLAN_REQUIRE_BROWSER"
MISSING_EXECUTABLE = "Executable doesn't exist at "


def require_browser[T](
    launch: Callable[[], T],
    *,
    files: Iterable[Path],
    environ: Mapping[str, str] | None = None,
) -> T:
    """What ``launch`` returns, when every one of ``files`` exists and the browser launches.

    Raises:
        pytest.skip.Exception: A file is missing, or the browser is not installed, and
            ``NORTHPLAN_REQUIRE_BROWSER`` is not ``1``.
        pytest.fail.Exception: The same, and it is ``1``.
        playwright.sync_api.Error: The launch fails for any other reason.
    """
    environment = os.environ if environ is None else environ
    missing = [path for path in files if not path.is_file()]
    if missing:
        reason = (
            f"{' and '.join(f'{path} is absent' for path in missing)}; run "
            f"`npm ci --ignore-scripts` in {BROWSER_TESTS} to run the browser page tests."
        )
    else:
        try:
            return launch()
        except Error as error:
            if MISSING_EXECUTABLE not in error.message:
                raise
            reason = (
                "Chromium is not installed; run `playwright install --only-shell chromium` "
                "to run the browser page tests."
            )
    if environment.get(REQUIRE_VARIABLE) == "1":
        pytest.fail(f"{REQUIRE_VARIABLE}=1 and {reason}")
    pytest.skip(reason)

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fixtures for the tests that drive the two pages in Chromium.

``chromium`` is one headless browser for each test module that uses it. Playwright's
sync API marks an asyncio loop as running on the main thread until Playwright stops, so
a test in such a module cannot call ``asyncio.run``; stopping it at the end of the
module keeps every other module clear of it. ``pyodide_page`` and ``api_page`` each give
a fresh page of that browser, with the :class:`offline.Offline` handler that it is
routed by, as ``(page, offline)``. A request the handler refuses fails its test at
teardown, unless the test cleared ``offline.refused`` after asserting it. ``form_page`` gives
a routed page of either site, with the ids ``api`` and ``pyodide``, built the way ``api_page``
and ``pyodide_page`` are and already at its site with the engine ready; ``api_form_page`` is
``api_page`` at its site. The scenario form's tests use them.

``pyodide_site`` is the site that ``browser/build.py`` writes, served from a temporary
directory under ``/northplan/`` as GitHub Pages serves a project site. Building it runs
the Node harness once on an empty corpus, to fill the cache of Pyodide's wheels in
``tests/browser/node_modules/pyodide``; on a fresh checkout that downloads them from
jsDelivr. ``api_site`` is ``api.main.app`` under uvicorn on a loopback port. Both are
``offline.Site`` values.
"""

from __future__ import annotations

import functools
import http.server
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn
from browser_gate import require_browser
from form_helpers import goto_ready
from node_gate import BROWSER_TESTS, require_node
from offline import CDN_FILES, Offline, Site
from playwright.sync_api import Browser, Page, sync_playwright

import api.main

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def chromium() -> Iterator[Browser]:
    with sync_playwright() as playwright:
        browser = require_browser(playwright.chromium.launch, files=list(CDN_FILES.values()))
        yield browser
        browser.close()


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        return None


@pytest.fixture(scope="session")
def pyodide_site(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Site]:
    node = require_node()
    root = tmp_path_factory.mktemp("pyodide-page")
    site = root / "northplan"
    built = subprocess.run(
        ["python3", str(REPO_ROOT / "browser" / "build.py"), str(site)],
        capture_output=True,
        text=True,
    )
    if built.returncode != 0:
        pytest.fail(
            f"browser/build.py exited {built.returncode}\nstdout:\n{built.stdout}\n"
            f"stderr:\n{built.stderr}"
        )
    completed = subprocess.run(
        [node, str(BROWSER_TESTS / "harness.mjs"), str(site), str(root / "primed.json")],
        input="[]",
        cwd=BROWSER_TESTS,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(root))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield Site(f"http://127.0.0.1:{server.server_address[1]}/northplan/", site)
    server.shutdown()
    thread.join()
    server.server_close()


@pytest.fixture(scope="session")
def api_site() -> Iterator[Site]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(api.main.app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()

    def stop() -> None:
        server.should_exit = True
        thread.join(30)
        sock.close()

    deadline = time.monotonic() + 30
    while not server.started:
        if not thread.is_alive():
            stop()
            pytest.fail("the API server thread died before it started")
        if time.monotonic() > deadline:
            stop()
            pytest.fail("the API server did not start within 30 seconds")
        time.sleep(0.02)
    yield Site(f"http://127.0.0.1:{sock.getsockname()[1]}/", None)
    stop()


def _routed_page(chromium: Browser, site: Site) -> Iterator[tuple[Page, Offline]]:
    context = chromium.new_context()
    offline = Offline(site.url)
    context.route("**/*", offline)
    page = context.new_page()
    yield page, offline
    context.close()
    assert offline.refused == [], offline.refused


@pytest.fixture
def pyodide_page(chromium: Browser, pyodide_site: Site) -> Iterator[tuple[Page, Offline]]:
    yield from _routed_page(chromium, pyodide_site)


@pytest.fixture
def api_page(chromium: Browser, api_site: Site) -> Iterator[tuple[Page, Offline]]:
    yield from _routed_page(chromium, api_site)


@pytest.fixture(params=["api", "pyodide"])
def form_page(request: pytest.FixtureRequest, chromium: Browser) -> Iterator[tuple[Page, Offline]]:
    """The page under test with its engine ready, once for the API page and once for the
    Pyodide page."""
    site = request.getfixturevalue(f"{request.param}_site")
    for page, offline in _routed_page(chromium, site):
        goto_ready(page, site)
        yield page, offline


@pytest.fixture
def api_form_page(api_page: tuple[Page, Offline], api_site: Site) -> tuple[Page, Offline]:
    """The API page with the example in its box."""
    goto_ready(api_page[0], api_site)
    return api_page

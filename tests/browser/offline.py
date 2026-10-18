# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What keeps a browser context offline: the pages' CDN files, answered from node_modules.

:class:`Offline` routes every request of a context. A request to the site under test goes
through; a CDN URL the pages load is answered from the npm package that pins it, the
Pyodide release from ``node_modules/pyodide``; anything else is aborted and recorded in
:attr:`Offline.refused`. The CDN URLs are built from the versions in ``package.json``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import Route

BROWSER_TESTS = Path(__file__).resolve().parent
NODE_MODULES = BROWSER_TESTS / "node_modules"
_VERSIONS = json.loads((BROWSER_TESTS / "package.json").read_text(encoding="utf-8"))["dependencies"]
CDN_FILES: dict[str, Path] = {
    f"https://cdn.jsdelivr.net/npm/alpinejs@{_VERSIONS['alpinejs']}/dist/cdn.min.js": (
        NODE_MODULES / "alpinejs" / "dist" / "cdn.min.js"
    ),
    f"https://cdn.plot.ly/plotly-{_VERSIONS['plotly.js-dist-min']}.min.js": (
        NODE_MODULES / "plotly.js-dist-min" / "plotly.min.js"
    ),
}
PYODIDE_BASE = f"https://cdn.jsdelivr.net/pyodide/v{_VERSIONS['pyodide']}/full/"
PYODIDE_FILES = NODE_MODULES / "pyodide"
CONTENT_TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".wasm": "application/wasm",
    ".json": "application/json",
    ".zip": "application/zip",
    ".whl": "application/zip",
}


@dataclass(frozen=True)
class Site:
    """A site under test: its base URL (ending in ``/``) and, for the Pyodide page, the
    directory of the built files; ``None`` for the API."""

    url: str
    directory: Path | None


class Offline:
    """A route handler that lets the site through and answers the CDN from disk.

    ``refused`` collects the URL of every request it aborted, in order; a test that
    expects a refusal clears it, and the fixtures fail a test that leaves any.
    """

    def __init__(self, site_url: str) -> None:
        self.site_url = site_url
        self.refused: list[str] = []

    def __call__(self, route: Route) -> None:
        """Continue, answer or abort ``route.request``, by the rules in the module docstring."""
        url = route.request.url
        if url.startswith(self.site_url):
            route.continue_()
            return
        path = CDN_FILES.get(url)
        if path is None and url.startswith(PYODIDE_BASE):
            rest = url[len(PYODIDE_BASE) :]
            if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*", rest):
                path = PYODIDE_FILES / rest
        if path is None or not path.is_file():
            self.refused.append(url)
            route.abort()
            return
        route.fulfill(
            status=200,
            body=path.read_bytes(),
            headers={
                "Content-Type": CONTENT_TYPES.get(path.suffix, "application/octet-stream"),
                "Access-Control-Allow-Origin": "*",
            },
        )

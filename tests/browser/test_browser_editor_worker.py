# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The Pyodide page's worker answers each message by its id and in the order it arrived: a
second worker started in the page gets an editor call, an unknown editor call and a message of
an unknown type. The first two payloads are the native entry point's; the third is the worker's
own refusal of the unknown type.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from offline import Offline, Site
from playwright.sync_api import Page

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_TIMEOUT_MS = 120_000
MESSAGES = """async (limit) => {
    const within = (promise) => Promise.race([
        promise,
        new Promise((_, reject) => setTimeout(() => reject(new Error('no answer in time')), limit)),
    ]);
    const worker = new Worker(new URL('worker.js', location.href), { type: 'module' });
    await within(new Promise((resolve, reject) => {
        worker.onmessage = (event) => {
            if (event.data.type === 'ready') {
                resolve();
            } else if (event.data.type === 'failed') {
                reject(new Error(event.data.detail));
            }
        };
    }));
    const replies = [];
    const done = new Promise((resolve) => {
        worker.onmessage = (event) => {
            replies.push(event.data);
            if (replies.length === 3) {
                resolve();
            }
        };
    });
    worker.postMessage({ id: 7, type: 'editor', call: 'schema', arguments: '{}' });
    worker.postMessage({ id: 3, type: 'editor', call: 'nope', arguments: '{}' });
    worker.postMessage({ id: 9, type: 'bogus' });
    await within(done);
    worker.terminate();
    return replies;
}"""


@pytest.fixture(scope="module")
def entry() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "northplan_browser_under_test", REPO_ROOT / "browser" / "northplan_browser.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_worker_answers_by_id_in_order(
    pyodide_page: tuple[Page, Offline], pyodide_site: Site, entry: ModuleType
) -> None:
    page, _ = pyodide_page
    page.goto(pyodide_site.url)
    replies = page.evaluate(MESSAGES, WORKER_TIMEOUT_MS)
    assert [reply["id"] for reply in replies] == [7, 3, 9]
    assert {reply["type"] for reply in replies} == {"reply"}
    assert replies[0]["payload"] == entry.editor_call("schema", "{}")
    assert replies[1]["payload"] == entry.editor_call("nope", "{}")
    assert replies[2]["payload"] == '{"status":"error","detail":"unknown message type \\"bogus\\""}'

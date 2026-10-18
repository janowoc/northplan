// SPDX-FileCopyrightText: 2026 Jan Owoc
// SPDX-License-Identifier: AGPL-3.0-or-later

// The engine's worker: loads Pyodide and the engine's archive once, then answers each
// message the page posts, in the order they arrive. A message is {id, type: "simulate",
// text} or {id, type: "editor", call, arguments}; the answer is {type: "reply", id,
// payload}, where payload is the JSON string northplan_browser.simulate or
// northplan_browser.editor_call returns, untouched. Every file is requested relative to
// this worker's own URL, except Pyodide, which comes from the pinned jsDelivr release
// below.

const PYODIDE_VERSION = "314.0.7";
const PYODIDE_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let entry = null;

async function load() {
  const { loadPyodide } = await import(`${PYODIDE_BASE}pyodide.mjs`);
  const pyodide = await loadPyodide({ indexURL: PYODIDE_BASE });
  await pyodide.loadPackage(["numpy", "pyyaml", "pydantic"]);
  const response = await fetch(new URL("northplan.zip", self.location));
  if (!response.ok) {
    throw new Error(`northplan.zip: HTTP ${response.status}`);
  }
  const archive = new Uint8Array(await response.arrayBuffer());
  pyodide.unpackArchive(archive, "zip", { extractDir: "/northplan" });
  pyodide.runPython("import sys\nsys.path.insert(0, '/northplan')\nimport northplan_browser");
  entry = pyodide.pyimport("northplan_browser");
}

const loading = load().then(
  () => self.postMessage({ type: "ready" }),
  (error) => self.postMessage({ type: "failed", detail: String(error) }),
);

self.onmessage = async (event) => {
  await loading;
  if (entry === null) {
    return;
  }
  const { id, type } = event.data;
  let payload;
  if (type === "simulate") {
    try {
      payload = entry.simulate(event.data.text, "{}");
    } catch (error) {
      payload = JSON.stringify({ status: "error", detail: String(error) });
    }
  } else if (type === "editor") {
    try {
      payload = entry.editor_call(event.data.call, event.data.arguments);
    } catch (error) {
      payload = JSON.stringify({ status: "error", body: { detail: String(error) } });
    }
  } else {
    payload = JSON.stringify({
      status: "error",
      detail: `unknown message type ${JSON.stringify(type)}`,
    });
  }
  self.postMessage({ type: "reply", id, payload });
};

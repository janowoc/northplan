// SPDX-FileCopyrightText: 2026 Jan Owoc
// SPDX-License-Identifier: AGPL-3.0-or-later

// The engine's worker: loads Pyodide and the engine's archive once, then answers each
// scenario the page posts with the JSON string northplan_browser.simulate returns,
// untouched. Every file is requested relative to this worker's own URL, except Pyodide,
// which comes from the pinned jsDelivr release below.

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
  let payload;
  try {
    payload = entry.simulate(event.data.text, "{}");
  } catch (error) {
    payload = JSON.stringify({ status: "error", detail: String(error) });
  }
  self.postMessage({ type: "result", payload });
};

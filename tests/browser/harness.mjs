// SPDX-FileCopyrightText: 2026 Jan Owoc
// SPDX-License-Identifier: AGPL-3.0-or-later

// node harness.mjs SITE_DIR OUT_FILE < corpus.json
//
// Runs the browser entry point under Pyodide in Node, as browser/worker.js runs it in a
// page. The corpus is a JSON array of {text, options}, run with simulate, or of
// {call, arguments}, run with editor_call (arguments is the JSON text, passed as it is);
// OUT_FILE receives a JSON array of the strings northplan_browser returned, untouched:
// Python's JSON must not pass through JavaScript's JSON.parse, which turns 80000.0 into
// 80000.

import { loadPyodide } from "pyodide";
import { readFileSync, writeFileSync } from "node:fs";

const [siteDir, outFile] = process.argv.slice(2);
if (!siteDir || !outFile) {
  throw new Error("usage: node harness.mjs SITE_DIR OUT_FILE < corpus.json");
}
const corpus = JSON.parse(readFileSync(0, "utf8"));

const pyodide = await loadPyodide();
await pyodide.loadPackage(["numpy", "pyyaml", "pydantic"], { messageCallback: () => {} });
pyodide.unpackArchive(new Uint8Array(readFileSync(`${siteDir}/northplan.zip`)), "zip", {
  extractDir: "/northplan",
});
pyodide.runPython("import sys\nsys.path.insert(0, '/northplan')\nimport northplan_browser");
const entry = pyodide.pyimport("northplan_browser");

const results = [];
for (const item of corpus) {
  results.push(
    "call" in item
      ? entry.editor_call(item.call, item.arguments)
      : entry.simulate(item.text, JSON.stringify(item.options)),
  );
}
writeFileSync(outFile, JSON.stringify(results));

<!-- SPDX-FileCopyrightText: 2026 Jan Owoc -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# northplan

A deterministic month-by-month Canadian household financial simulator, wrapped
in Monte Carlo, wrapped in an optimizer that compares *policies*.

Three use cases, one engine:

| Use case | What the optimizer compares |
|---|---|
| Accumulation | Contribution split across RRSP / RESP / TFSA / taxable |
| Readiness | Nothing — evaluate one policy, report the distribution |
| Decumulation | Withdrawal order and bracket ceiling; CPP, OAS and RRIF conversion ages |

One run covers a household from the start year to the second death, so the
same simulation serves all three. The objective is the after-tax estate at
death with spending met; success probability and GIS exposure are reported
alongside it.

There is exactly one simulation loop and it is over months. If a second one
appears, something has gone wrong.

## Status

The parameter files for 2026 are populated and structurally tested. The
loader, the scenario schema and loader, and all of `engine/tax/`,
`engine/benefits/`, `engine/accounts/`, `engine/core/`, `engine/mc/`,
`engine/policy/` and `engine/optimize/` are implemented, the monthly step, the
run loop and the policy search included. `cli/` runs `northplan simulate` and
`northplan optimize`, and `api/` serves both over HTTP, with a web page that
runs a scenario and draws its fan chart. Planned work is tracked as GitHub
issues; the ordered plan that built the first version is archived in
`docs/archive/roadmap-v0.1.0.md`. Known simplifications are listed in
`docs/limitations.md`, and a change that adds one adds its entry there.

Still to do: checking the engine's figures by hand, in a spreadsheet of the
deterministic path ([#21](https://github.com/janowoc/northplan/issues/21)), a
hand check of the couple scenario's death mechanics
([#56](https://github.com/janowoc/northplan/issues/56)), and golden cases for
the lines no case reaches yet
([#69](https://github.com/janowoc/northplan/issues/69)).

Scope of the first version: Alberta residents, no GIS, no Quebec. See
`CLAUDE.md`.

## Design decisions

1. **Vectorize across paths, not time.** Months are sequentially dependent;
   paths are independent. The inner loop computes month N for all `n_paths` at
   once. Every function in `engine/tax/` and `engine/benefits/` takes and
   returns NumPy arrays, using `np.clip` / `np.where` for bracket logic.
   Because broadcasting means a scalar is a valid input, golden-number tests
   exercise the exact same code path as production.
2. **Common random numbers.** Returns and the mortality draws are generated
   once from a fixed seed and reused across every policy the optimizer
   evaluates. Path 123 has the same market and the same death month under
   every policy. Different draws for different policies means the optimizer
   chases Monte Carlo noise.
3. **Optimize policies, not paths.** A policy function may only depend on
   information available at that simulated point in time. Any clairvoyant
   policy is a bug. Year to date is knowable; the year's total is not.
4. **Household is a list of persons from day one.** Pension splitting,
   survivor benefits, the RRIF spousal rollover, OAS ceasing at first death,
   and two mortality timelines all require two people.
5. **Real dollars internally, January dollars of the start year.** A scenario
   states the two years of net income for each person as filed; the builder
   restates the opening pair to real dollars before the run starts, and a
   December close then replaces each with that year's real figure in turn. One
   parameter year serves the whole run. An indexed amount is constant in real
   terms except for the erosion it suffers between adjustment dates, which is a
   constant factor per indexation schedule computed once per scenario in
   `engine/core/indexation.py`. An amount fixed in nominal terms by statute —
   the pension income amount, the CESG figures, a non-indexed DB pension —
   decays without limit and is decayed explicitly. Which amounts are on which
   schedule is declared in the parameter files, not in code. Conversion to
   nominal happens only at display.
6. **The timestep is one month.** Life events happen mid-year, benefits are
   paid monthly, and the balance owing on a tax year is paid in the filing
   month of the next one. Annual events — the assessment, contribution room,
   the RRIF minimum — are phases the monthly step invokes in the month that
   calls for them, not a second loop.
7. **Cash is an account.** Every inflow and outflow passes through a cash
   balance that pays zero real return. The policy sees the month's inflows
   and outflows before it decides what to move between cash and the other
   accounts, so it never has to guess this month's income. Tax is assessed at
   the December close on the current year's income, withholding is remitted
   monthly, and the balance settles from cash in the filing month.
8. **Stochastic mortality.** A period life table per sex gives an annual
   hazard; one uniform draw per person per path is mapped through the
   survival curve to a death month. The simulation runs every path to the
   second death, and the estate is valued after the terminal return.
9. **RESP is tracked per beneficiary**, not as one pot, in three buckets —
   contributions, grants, accumulated income — because they leave the plan
   under different rules. Grant room and withdrawal windows do not aggregate.

## Layout

```
params/2026/     hand-populated parameters, one directory per tax year
params/*.yaml    placeholder templates and drafts, unreachable by the loader
engine/          pure Python, no FastAPI. MUST NOT import from api/, cli/ or report/
  params/        YAML -> typed frozen parameter sets, no defaults
  scenario/      scenario schema and loader (pydantic)
  tax/           array-valued pure functions: brackets, federal, provincial, combined
  benefits/      CPP, OAS, DB pensions, employment income; GIS exposure metric
  accounts/      cash, RRSP, RRIF, LIRA/LIF, TFSA, RESP, taxable
  core/          state, the monthly step, timeline, indexation, mortality
  mc/            monthly draws + the loop over months
  policy/        parameterized decision rules
  optimize/      objectives and the policy search
report/          result tables shared by cli/ and api/, real or nominal; run.py, the
                 simulate pipeline api/ and browser/ share; editor.py and
                 editor_calls.py, the scenario editor's logic and calls they share
api/             FastAPI, thin. Serves api/web/ via StaticFiles; northplan-serve
  web/           plain HTML + Alpine.js + Plotly from CDN, and editor.js, the form
                 both pages share. No build step
browser/         the static page that runs the engine in the browser under
                 Pyodide; browser/build.py builds it, with api/web/editor.js
cli/             YAML scenario in, results out
scenarios/       example.yaml, late_life_couple.yaml and minimal.yaml are committed;
                 *.local.yaml is gitignored
docs/            limitations.md (every simplification), maintenance.md (pins)
  archive/       the roadmap that built v0.1.0, not updated
tests/golden/    cases whose expected values the human supplied — sacred
tests/character*/ behaviour snapshots, regenerated only on instruction
tests/params/    structural and provenance tests over the real parameter files
```

`engine/` is importable and fully testable with FastAPI absent. This is
enforced by `tests/test_layering.py`, which walks the AST of every module under
`engine/` and fails on any import of `api`, `fastapi`, `cli`, `report`,
`starlette`, or `uvicorn`.

## Parameters are never invented

No tax constant is ever inlined in a `.py` file, recalled from memory, or
estimated. Every one is loaded from YAML under `params/` that a human populated
by hand, under a comment giving the source URL and the date it was checked —
that comment is the audit record. When a requested parameter is absent, the
loader raises `MissingParameterError` rather than returning a default — there
is deliberately no way to supply one. The mortality table is a parameter. So
are the filing month and the indexation schedules. See `CLAUDE.md`.

## File formats

- **Parameters** — YAML, one directory per tax year. Every value carries a
  source comment with the date it was checked. Each file's `indexation` block
  names its schedules and lists the dollar amounts on each; an amount on no
  schedule stops the run.
- **Scenarios** — YAML, or JSON over the API, validated by `engine/scenario/`.
  One schema for regression fixtures, real households, and the API. A scenario
  carries the household (persons with birth date and sex, employment schedule,
  CPP history or amount in pay, OAS amount if already in pay, prior-year net
  income, DB pensions, account balances and room; RESP beneficiaries with plan
  state and an education schedule), a spending schedule, return assumptions
  (asset classes with real mean, volatility and the yields that fix their tax
  character; a correlation matrix; allocations per account kind), and a list
  of named policies. `scenarios/example.yaml` is the reference.
- **Results** — JSON over the API; CSV or JSON from the command line: a row per
  year, and for `optimize` a row per policy evaluated; a single-path monthly
  trace (`northplan simulate --trace-path`) for debugging. The engine steps
  monthly and aggregates to years at the boundary.
- **No database.** A scenario is a file, or the body of an API request.

## Run it in your browser

https://janowoc.github.io/northplan/ runs the latest release.
https://janowoc.github.io/northplan/main/ is the development build: it
follows the head of main, and its results can change from one commit to the
next.

The page runs the engine inside the browser: Python compiled to WebAssembly,
by Pyodide. The scenario never leaves your machine. The only requests the page
makes are for its own files and for the pinned libraries from jsDelivr and
cdn.plot.ly, none of which carry the scenario. The first visit downloads about
11 MB. A run takes roughly two to three times as long as with the local
install.

A run is limited to 25,000 paths, because the browser gives WebAssembly at most
4 GB; the local install allows 100,000. A run that still runs out of memory
says so. `optimize` is not available in the browser; use the local install.
The page simulates with the scenario's own options only: it has no controls
for the policy, the number of paths, a deterministic run or nominal dollars.

This page and the local install's both have a form for editing a scenario.
**Edit in form** opens the text in the box; **New scenario** starts from
`scenarios/minimal.yaml` and **Open example** from the example, without changing
the box. The engine's own validation checks each change; every error is listed
above the form, and shown beside the field or section it concerns when it
concerns one. **Back to YAML**, **Download YAML** and **Run** take the form's
scenario once it has no errors. The YAML the form writes keeps no comments,
anchors or flow style, and puts each section's fields in the schema's order.
Text that does not validate does not open in the form; its errors are listed
instead.

On this page, while a scenario runs, the form cannot be edited and
**Edit in form**, **New scenario** and **Open example** are off;
**Back to YAML**, **Download YAML** and **Discard form** still work. The
local install's form stays usable during a run.

## Running

```sh
docker compose up --build
```

Serves the API and static files from one container on http://localhost:8000.
`--build` rebuilds the image from the current tree. Without it, Compose reuses
the image it built last, however old. Only `params/` and `scenarios/` are
mounted into the container, so a change anywhere else takes effect only after a
rebuild.

The server is for one household. It bounds what a request can cost, so that a
mistake cannot exhaust a laptop: at most 100,000 paths, at most 1,000 candidate
policies, at most 10,000,000 paths across all of them, a body of at most 1 MiB,
and one run at a time. A browser on the server's own machine that opens
`localhost`, `127.0.0.1` or `[::1]` needs nothing more. Any other client must
give the server's token: the browser asks for a user name and a password; the
user name can be anything and the password is the token. `curl` takes it as
`-u :TOKEN`. In a container every client is another, the Docker host's own
browser included, because Docker forwards each connection from an address of its
own.

Unless `NORTHPLAN_TOKEN` is set, the server makes a new token each time it
starts and writes it to its log, on the line that begins
`northplan-serve: NORTHPLAN_TOKEN is not set`. `docker compose logs northplan`
or `docker logs` shows it; Portainer shows it in the container's logs. To keep
one token across restarts, set `NORTHPLAN_TOKEN` to at least 16 printable ASCII
characters without spaces, such as the output of
`python3 -c 'import secrets; print(secrets.token_hex(16))'`; in Portainer,
add it as an environment variable of the container. A token you set is never
written to the log.

The server speaks plain HTTP. Anywhere but on its own machine, the token and
every scenario cross the network unencrypted, readable by anyone who can watch
the traffic. Open it to other devices only on a network you trust.

Open that address to run the example scenario and see its fan chart. The API
takes a scenario as the request body, JSON or YAML, and the run's options as
query parameters: `paths`, `deterministic` and `nominal` on both endpoints,
`policy` on `/api/simulate`, and `objective` (required), `risk_aversion` and
`estate_utility_shift` on `/api/optimize`. Each does what the command line's
flag of that name does, with underscores for hyphens. Under Compose, first set
TOKEN in your shell to the server's token.

```sh
curl -s -u ":$TOKEN" -X POST 'http://localhost:8000/api/simulate?paths=200' \
  -H 'Content-Type: application/yaml' --data-binary @scenarios/example.yaml
```

`GET /api/example` returns the example scenario and `GET /api/minimal` returns
`scenarios/minimal.yaml`, a small scenario that runs. `POST /api/editor/{call}`
answers the scenario editor's calls, such as `parse` and `check`, with the
arguments as a JSON object. Those calls load each tax year's parameters the
first time they need it and keep it until the server restarts, so an edit under
`params/` reaches them only after a restart; runs read the parameters on each
request.

To run the published image without cloning the repository:

```sh
docker run --rm -p 127.0.0.1:8000:8000 ghcr.io/janowoc/northplan:latest
```

`latest` is the most recently published release, also tagged `X.Y.Z` and
`X.Y`. `main` is the most recently published build of the main branch, and
`sha-` followed by a 7-character commit hash is the build of the push to `main`
that ended at that commit. With the `127.0.0.1:` prefix only this machine can
connect. To open it to other devices on the network, publish the port on every
interface and choose the token:

```sh
docker run --rm -p 8123:8000 -e NORTHPLAN_TOKEN='your-token-here' ghcr.io/janowoc/northplan:latest
```

and open `http://<server>:8123` from the other device. The image carries its own
copies of `params/`, `scenarios/example.yaml` and `scenarios/minimal.yaml`.
Compose mounts `./params` and `./scenarios` over them, so there an edit takes
effect on the next request, without a rebuild, except that the editor's calls
keep the parameters they have already loaded until a restart; `docker run`
serves the image's copies.

The image also has the command line. Mount a directory for its output:

```sh
docker run --rm -v "$PWD:/out" --user "$(id -u):$(id -g)" ghcr.io/janowoc/northplan:latest \
  northplan simulate scenarios/example.yaml --paths 200 --out /out/example.csv
```

`scenarios/example.yaml` there is the image's copy, under `/app`. To run a
scenario of your own, put it in the mounted directory and pass it as
`/out/` followed by its name. `--user` makes the files written yours; without
it they belong to root.

Locally, without Docker:

```sh
uv sync --locked --extra dev
. .venv/bin/activate
pytest
ruff check .
ruff format .
```

This needs [uv](https://docs.astral.sh/uv/). `uv sync --locked` installs the
versions in `uv.lock` on the Python in `.python-version`, the ones the image
and CI use. [docs/maintenance.md](docs/maintenance.md) says how each pinned
version is updated.

`pytest` runs the tests on as many worker processes as `-n auto` picks, at
most eight; a larger `-n` also needs a larger `--maxprocesses`. `pytest -n 0`
runs them serially.

The browser tests also need Node, the npm packages pinned in `tests/browser`
and Chromium's headless shell:

```sh
(cd tests/browser && npm ci --ignore-scripts)
playwright install --only-shell chromium
```

Without them those tests are skipped, and the skip says what is missing. On
Linux, `playwright install-deps chromium` installs the system libraries Chromium
needs, with sudo; without those, Chromium does not start and the tests that
drive Chromium report Playwright's error. The first run downloads Pyodide's
packages from jsDelivr into `tests/browser/node_modules`; the pages themselves
are tested offline.

Run a scenario from the command line:

```sh
northplan simulate scenarios/example.yaml --paths 200 --out /tmp/example.csv
northplan optimize scenarios/example.yaml --objective median_estate_after_tax --paths 200 --out /tmp/example.optimize.csv
```

`northplan simulate --help` and `northplan optimize --help` list the options.

`northplan-serve` serves the API and the page on http://localhost:8000, as
Compose does; `--port` changes the port. `--host` changes the address it listens
on; on any other than a loopback address, a client on another machine must give
the token. An editable install finds `params/`, `scenarios/example.yaml` and
`scenarios/minimal.yaml` in the checkout. Any other install has none of them:
pass `--params DIR`, the directory holding one subdirectory per tax year, to
`northplan simulate`, `northplan optimize` and `northplan-serve`, and
`--example FILE` to `northplan-serve` for the page's example scenario. No flag
supplies `scenarios/minimal.yaml`, so such an install answers 404 at
`/api/minimal`.

## Development

Each piece of work is a GitHub issue or an open pull request. The first
version was built engine first, UI last, in the order archived in
`docs/archive/roadmap-v0.1.0.md`.

**RESP sanity check:** contributions up to the CESG maximum should beat
smaller ones whenever the plan is spent on education: a guaranteed match is
hard to beat. Contributions beyond it should lose in two ways: front-loaded,
they reach the lifetime contribution limit early and forfeit grant room later
years would have used; and whatever education does not use is wound up with
the special tax on its income. `tests/optimize/test_resp_oracle.py` checks
both. A modest excess that education does use is not penalised here: it grows
sheltered and is paid out tax-free (`docs/limitations.md` L30, L31). If the
oracle fails, the model is wrong.

## Disclaimer

northplan is a personal modelling tool. Nothing it produces, including
projections, tax estimates, withdrawal strategies or any other output, is
financial, tax or legal advice, and you should not rely on it as such. Consult
a qualified professional before acting on any result.

**Tax and benefit figures.** All tax and benefit parameters (rates, brackets,
credits, thresholds and amounts) are 2026 figures entered by hand. They cover
Alberta only; other provinces and territories are not modelled. They have not
been independently verified and may contain errors or omissions. They will also
go out of date as legislation and indexation change. Check any figure that
matters against the Canada Revenue Agency, the Government of Alberta or another
official source.

**Simplifications.** The model simplifies many tax rules and real-world
conditions. These simplifications are listed in
[docs/limitations.md](docs/limitations.md). Read it before interpreting any
results.

**No warranty.** This software is provided "as is", without warranty of any
kind, express or implied. The author accepts no liability for any loss arising
from its use. The full terms are in the [License](#license) section and in
sections 15 (Disclaimer of Warranty) and 16 (Limitation of Liability) of the
GNU Affero General Public License v3 in [LICENSE](LICENSE).

## License

AGPL-3.0-or-later. The full text is in `LICENSE`.

Copyright (C) 2026 Jan Owoc.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>.

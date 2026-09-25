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
loader, the scenario schema and loader, all of `engine/tax/`, all of
`engine/benefits/`, and all of `engine/core/` and `engine/mc/` apart from the
monthly step and the run loop are implemented, and so is `engine/accounts/`
apart from `rrif.spousal_rollover`. The step, the loop, the policies and the
optimizer are still stubs carrying their specification as a docstring. The
build order is `docs/roadmap.md`; each step is a GitHub issue. Known
simplifications are listed in `docs/limitations.md`, and a change that adds
one adds its entry there.

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
5. **Real dollars internally, January dollars of the start year.** The
   exception is the two years of net income a scenario states for each
   person, taken as filed; a December close replaces each with a real
   figure in turn. One parameter year serves the whole run. An indexed
   amount is constant in real terms except for the erosion it suffers
   between adjustment dates, which is a constant factor per indexation
   schedule computed once per scenario in `engine/core/indexation.py`. An
   amount fixed in nominal terms by statute — the pension income amount,
   the CESG figures, a non-indexed DB pension — decays without limit and is
   decayed explicitly. Which amounts are on which schedule is declared in
   the parameter files, not in code. Conversion to nominal happens only at
   display.
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
engine/          pure Python, no FastAPI. MUST NOT import from api/ or cli/
  params/        YAML -> typed frozen parameter sets, no defaults
  scenario/      scenario schema and loader (pydantic)
  tax/           array-valued pure functions: brackets, federal, provincial, combined
  benefits/      CPP, OAS, DB pensions, employment income; GIS exposure metric
  accounts/      cash, RRSP, RRIF, LIRA/LIF, TFSA, RESP, taxable
  core/          state, the monthly step, timeline, indexation, mortality
  mc/            monthly draws + the loop over months
  policy/        parameterized decision rules
  optimize/      objectives and the policy search
api/             FastAPI, thin. Serves web/ via StaticFiles
web/             plain HTML + Alpine.js + Plotly from CDN. No build step
cli/             YAML scenario in, results out
scenarios/       example.yaml is committed; *.local.yaml is gitignored
docs/            roadmap.md (build order), limitations.md (every simplification)
tests/golden/    cases whose expected values the human supplied — sacred
tests/character*/ behaviour snapshots, regenerated only on instruction
tests/params/    structural and provenance tests over the real parameter files
```

`engine/` is importable and fully testable with FastAPI absent. This is
enforced by `tests/test_layering.py`, which walks the AST of every module under
`engine/` and fails on any import of `api`, `fastapi`, or `cli`.

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
- **Scenarios** — YAML, validated by `engine/scenario/`. One schema for
  regression fixtures, real households, and the API. A scenario carries the
  household (persons with birth date and sex, employment schedule, CPP history
  or amount in pay, OAS amount if already in pay, prior-year net income, DB
  pensions, account balances and room; RESP beneficiaries with plan state and
  an education schedule), a spending schedule, return assumptions (asset
  classes with real mean, volatility and the yields that fix their tax
  character; a correlation matrix; allocations per account kind), and a list
  of named policies. `scenarios/example.yaml` is the reference.
- **Results** — JSON over the API; CSV row-per-year for export; a single-path
  monthly trace for debugging. The engine steps monthly and aggregates to
  years at the boundary.
- **No database.** Scenarios are files on a mounted volume.

## Running

```sh
docker compose up
```

Serves the API and static files from one container on http://localhost:8000.

Locally, without Docker:

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
```

## Build order

Engine first, UI last. The ordered list of steps, each a GitHub issue with
its owner and success criteria, is `docs/roadmap.md`.

**Early sanity check:** once RESP exists, the optimizer should discover
unprompted that contributions up to the CESG maximum dominate nearly everything
else, and that contributions beyond it do not. A guaranteed 20% match is hard
to beat; a plan that winds up with a penalty is easy to beat. If it does not
find both, the model is wrong.

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

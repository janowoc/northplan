# northplan

A deterministic month-by-month Canadian household financial simulator, wrapped
in Monte Carlo, wrapped in an optimizer that searches over *policy parameters*.

Three use cases, one engine:

| Use case | Free variables the optimizer searches |
|---|---|
| Accumulation | Contribution split across RRSP / RESP / TFSA / taxable |
| Readiness | None — evaluate a fixed policy, report the distribution |
| Decumulation | Withdrawal order and thresholds; CPP / OAS start ages |

There is exactly one simulation loop and it is over months. If a second one
appears, something has gone wrong.

## Status

Scaffolding. No tax logic, benefit logic, or simulation logic is implemented.
Every module under `engine/` outside `engine/params/` raises
`NotImplementedError`. Every file under `params/` is an empty stub awaiting
hand-verified values.

## Design decisions

1. **Vectorize across paths, not time.** Months are sequentially dependent;
   paths are independent. The inner loop computes month N for all `n_paths` at
   once. Every function in `engine/tax/` and `engine/benefits/` takes and
   returns NumPy arrays, using `np.clip` / `np.where` for bracket logic.
   Because broadcasting means a scalar is a valid input, golden-number tests
   exercise the exact same code path as production.
2. **Common random numbers.** The return matrix is generated once from a fixed
   seed and reused across every policy the optimizer evaluates. Different draws
   for different policies means the optimizer chases Monte Carlo noise.
3. **Optimize policies, not paths.** A policy function may only depend on
   information available at that simulated point in time. Any clairvoyant
   policy is a bug. On a monthly timeline that includes the current year's
   eventual total income: year to date is knowable, the year's total is not.
4. **Household is a list of persons from day one.** Pension splitting, survivor
   benefits, the RRIF spousal rollover, OAS ceasing at first death, and two
   mortality timelines all require two people.
5. **Real dollars internally.** CPI-indexed brackets and benefits stay
   constant in real terms — but at a constant slightly below their published
   real value, because indexation is periodic and lags the inflation it
   compensates for. That shortfall is computed once per scenario in
   `engine/core/indexation.py`; the oscillation around it is not modelled,
   being bounded, mean-zero, and smaller than the error in the inflation
   assumption. Conversion to nominal happens only at display. Non-indexed DB
   pensions are the case that genuinely cannot be flattened: their real decay
   grows without limit and is applied explicitly.
6. **The timestep is one month.** Life events happen mid-year, benefits are
   paid monthly, OAS and GIS index quarterly, CPP and the brackets index each
   January, and the balance owing on a tax year is paid in the filing month of
   the next one. Annual events — the tax assessment, contribution room, the
   RRIF minimum — are phases the monthly step invokes in the month that calls
   for them, not a second loop.
7. **RESP is tracked per beneficiary**, not as one pot. Grant room and
   withdrawal windows do not aggregate.

## Layout

```
params/2026/     hand-populated tax parameters, one directory per tax year
engine/          pure Python, no FastAPI. MUST NOT import from api/ or cli/
  params/        YAML -> typed frozen dataclasses
  tax/           array-valued pure functions
  benefits/      CPP, OAS, GIS — same shape as tax/
  accounts/      RRSP, RRIF, TFSA, RESP, LIRA, taxable
  core/          (state, month, policy) -> state; timeline and indexation
  mc/            monthly return generation + path loop
  policy/        parameterized decision rules
  optimize/      searches policy space
api/             FastAPI, thin. Serves web/ via StaticFiles
web/             plain HTML + Alpine.js + Plotly from CDN. No build step
cli/             YAML scenario in, results out
scenarios/       *.local.yaml is gitignored
tests/golden/    hand-verified expected values — sacred
tests/character*/ behaviour snapshots
```

`engine/` is importable and fully testable with FastAPI absent. This is
enforced by `tests/test_layering.py`, which walks the AST of every module under
`engine/` and fails on any import of `api`, `fastapi`, or `cli`.

## Parameters are never invented

No tax constant is ever inlined in a `.py` file, recalled from memory, or
estimated. Every one is loaded from YAML under `params/` that a human populated
by hand, under a comment giving the source URL and the date it was checked —
that comment is the audit record. When a requested parameter is
absent, the loader raises `MissingParameterError` rather than returning a
default — there is deliberately no way to supply one. See `CLAUDE.md`.

## File formats

- **Parameters** — YAML, one directory per tax year. Every value carries an
  inline comment with its source URL and the date it was checked.
- **Scenarios** — YAML. Regression fixtures and real user scenarios use the
  same schema; they are literally the same format.
- **Results** — JSON over the API; CSV row-per-year for export. The engine
  steps monthly and aggregates to years at the boundary; sub-annual detail is
  inspected on a single path, not carried for all of them.
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

Engine first, UI last.

1. `params/` + `tax/` + golden tests against TaxTips.ca. No simulation yet.
2. `benefits/` — CPP adjustment factors per month, monthly payment amounts,
   OAS clawback on prior-year net income over the benefit period, the constant
   real cost of indexation lag, GIS if applicable.
3. `core/` deterministic, single path, zero volatility — hand-checked in a
   spreadsheet with a row per month. Check the month boundaries first: January
   grants room and fixes the RRIF minimum, December assesses, and the filing
   month pays the prior year's balance.
4. `mc/` — verify that sigma=0 reproduces step 3 exactly, and that twelve
   monthly draws compound to the specified annual distribution.
5. `policy/` + `optimize/` — brute-force grid search first.
6. `cli/` with YAML scenarios.
7. `api/` + `web/`.

**Early sanity check:** once RESP exists, the optimizer should discover
unprompted that contributions up to the CESG maximum dominate nearly everything
else. A guaranteed 20% match is hard to beat. If it does not find that, the
model is wrong.

## License

AGPL-3.0. See `LICENSE`.

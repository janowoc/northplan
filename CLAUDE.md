# Working rules

## Tax and benefit constants — the most important rule in this repo
- NEVER invent, recall, infer, or estimate a tax parameter. Not a bracket
  edge, rate, threshold, credit amount, RRIF factor, or adjustment
  percentage. Every one comes from a file under `params/` that the human
  populated by hand.
- If a needed parameter is missing from `params/`, STOP and ask. Do not
  guess, do not use a placeholder, do not "use a reasonable value for now".
  Report the exact key name and file you expected it under.
- Never inline a numeric tax constant in a `.py` file. If you need one,
  it goes in YAML and gets loaded.
- Files under `params/` are edited only by the human. That includes renaming
  a key, deleting a key, and adding a routing entry to an `indexation` block.
- A mortality table is a parameter. So is a filing month, a benefit-year start
  month, and an indexation schedule.

## Tests
- NEVER edit an expected value in a test to make it pass. If the
  implementation and the expectation disagree, report the discrepancy and
  stop. Expected values are ground truth supplied by the human.
- Never delete or skip a failing test.
- Golden tests in `tests/golden/` are sacred. Their cases live in YAML files
  the human writes; an agent may add the harness and may never add a case.
  Characterization snapshots in `tests/characterization/` may be regenerated
  only when I explicitly say so.
- A test may use a made-up bracket table or life table to exercise arithmetic,
  as long as it is obviously synthetic and never presented as a real value.

## Scope of the first version (decided 2026-09-04)
- Alberta residents only. No GIS or Allowances. No Quebec. No residence
  history for OAS or CPP.
- Every simplification is recorded in `docs/limitations.md` in the form
  "in reality X; we model Y", with the direction of the error. A change that
  cuts a corner adds its entry there in the same commit. A change that removes
  a simplification deletes the entry. Cite entries from code as `L12`.
- The sequence of work is `docs/roadmap.md`, and each step is a GitHub issue.
  Do not start work that is not an issue.

## Conventions the engine holds everywhere
- Real dollars are January dollars of the scenario's start year. One
  parameter year serves the whole run. Indexed amounts are constant in real
  terms except for a per-schedule erosion factor; unindexed amounts decay.
  Which is which is declared in each parameter file's `indexation` block and
  resolved once per scenario in `engine/core/indexation.py`, never at a call
  site.
- The simulation starts on 1 January of the start year and runs every path to
  the second death. There is no separate horizon.
- Cash is an account. Every inflow and every outflow passes through it. A
  policy moves money between cash and the other accounts, and decides after
  it has seen the month's inflows and outflows.
- Tax is assessed at the December close on the current year's income,
  including the OAS repayment and the pension-splitting election. Withholding
  is remitted monthly and the balance settles from cash in the filing month.
- The timestep is one month and there is exactly one loop over time, in
  `engine/mc/simulate.py`. Annual events are phases the step invokes.
- Vectorize across paths. Every state array is `(n_paths,)` and read-only.
  Per-person and per-beneficiary state are tuples of frozen dataclasses.
- A policy reads only the opening state, the month context, and the
  parameters. Year to date is knowable; the year's total is not.
- `engine/` never imports `api`, `fastapi`, `cli`, `starlette`, or `uvicorn`.

## Scope of a change
- One logical change per commit. Conventional commit messages.
- Do not refactor code outside the module you were asked to change.
- Do not add dependencies without asking.

## Issues
- You may read and comment on GitHub issues via `gh`.
- You may NEVER close an issue. Only the human closes issues, after
  verifying the output.

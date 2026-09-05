# Roadmap

The ordered sequence of work toward a first end-to-end version. Each step is a
GitHub issue with an owner, its dependencies, and success criteria a reviewer
can check. Engine first, UI last. Human issues are parameter sourcing and
verification; agent issues are code. Issue numbers are filled in as they are
created.

Decisions behind this order are recorded in `docs/limitations.md` (what is
simplified) and `CLAUDE.md` (the conventions). This file is the index; the
issues hold the specifications.

| # | Step | Owner | Depends on | Done when |
|---|---|---|---|---|
| 1 | Date the four undated parameter files | human | — | provenance test warns nothing |
| 2 | Housekeeping: provenance ratchet, packaging, rate-units test | agent | 1 | suite passes with no warnings |
| 3 | Golden test harness under `tests/golden/` | agent | — | YAML cases become parametrized tests; zero cases pass |
| 4 | Mortality template and its structural tests | agent | — | `params/mortality-template.yaml` exists; tests fail on a bad table |
| 5 | Life table from Statistics Canada 13-10-0114-01 | human | 4 | `params/2026/mortality.yaml` loads clean |
| 6 | New parameters, deletions, and key renames | human | 1 | every v1 key exists and is dated |
| 7 | Indexation routing lists in every parameter file | human | 6 | every dollar amount on exactly one schedule |
| 8 | Indexation module and the real-terms parameter view | agent | 7 | dollar amounts readable only deflated |
| 9 | Bracket arithmetic under the YAML convention | agent | — | boundary tests pass; docstrings say "upper" |
| 10 | Scenario schema, loader, and `scenarios/example.yaml` | agent | — | example loads; every rule has a failing fixture |
| 11 | Typed immutable state and the initial-state builder | agent | 10 | walker test finds no writeable array |
| 12 | Timeline arithmetic and mortality | agent | 4, 11 | death months geometric on a synthetic table |
| 13 | Monthly return draws and mortality uniforms | agent | — | twelve months compound to the annual spec |
| 14 | Tax engine: federal, Alberta, household assessment, withholding | agent | 6, 8, 9, 11 | structural tests pass; no undeflated dollar read |
| 15 | Golden tax cases from an external calculator | human | 3, 14 | cases pass or bugs filed |
| 16 | Benefits and income: CPP, OAS, DB pension, employment, GIS metric | agent | 6, 8, 11, 14 | structural tests pass; refusal gone |
| 17 | Golden benefit cases from published tables | human | 3, 16 | cases pass or bugs filed |
| 18 | Account mechanics including RESP buckets and wind-up | agent | 6, 8, 11 | every annual limit year-to-date aware |
| 19 | The month step and the loop to second death | agent | 12, 13, 14, 16, 18 | cash identity holds every month |
| 20 | Contribution and withdrawal policies, elections, grid | agent | 19 | bracket filled once per year; no clairvoyance |
| 21 | Spreadsheet verification of the deterministic path | human | 20 | snapshot authorised |
| 22 | Objectives, search, and the RESP oracle test | agent | 20 | oracle test passes |
| 23 | Command line | agent | 22 | example runs end to end; `ARG001` ignore removed |
| 24 | API and web page | agent | 23 | chart renders for the example |
| 25 | Derive golden tolerance from a declared rounding | agent | 3 | no numeric tolerance field remains |

Parallel tracks: 1→2, 3, 4→5, 9, 10→11, 13 can all start at once. The human
track is 1, 5, 6, 7, then 15, 17, 21. Everything in the engine funnels into
19.

Issue 25 is out of sequence: it came out of the review of 3 and belongs
immediately after it, before 15 and 17 write the first real cases against the
format it changes.

Parameters the human supplies along the way, by issue: 6 adds the pension
splitting share and eligibility age, EI rate and maximum, CPP base rate and
survivor parameters, CESG cessation age, the RESP penalty rate, and the Alberta
dividend credit rate; 5 adds the life table; 15 and 17 add expected values.

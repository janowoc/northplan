<!-- SPDX-FileCopyrightText: 2026 Jan Owoc -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

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
| 19 | The month step and the loop to second death | agent | 12, 13, 14, 16, 18, 29 | cash identity holds every month |
| 20 | Contribution and withdrawal policies, elections, grid | agent | 19 | bracket filled once per year; no clairvoyance |
| 21 | Spreadsheet verification of the deterministic path | human | 20 | snapshot authorised |
| 22 | Objectives, search, and the RESP oracle test | agent | 20 | oracle test passes |
| 23 | Command line | agent | 22 | example runs end to end; `ARG001` ignore removed |
| 24 | API and web page | agent | 23 | chart renders for the example |
| 25 | Derive golden tolerance from a declared rounding | agent | 3 | no numeric tolerance field remains |
| 26 | Copyright and licence headers on every source file | agent | — | every source file carries an SPDX header; ruff and a test enforce it |
| 27 | Shape check for mortality tables: fall to a trough, then rise | agent | 4, 5 | a transposed q(x) fails; the boxed warning in both files is narrowed |
| 28 | Extend the placeholder-marker check to the province template | agent | — | an unmarked number in either template fails |
| 29 | `Assumptions` to covariance, and attainability checked at load | agent | 10, 13 | an unrealisable correlation is refused at load, naming the classes |

Parallel tracks: 1→2, 3, 4→5, 9, 10→11, 13 can all start at once. The human
track is 1, 5, 6, 7, then 15, 17, 21. Everything in the engine funnels into
19.

Issue 25 is out of sequence: it came out of the review of 3 and belongs
immediately after it, before 15 and 17 write the first real cases against the
format it changes.

Issue 26 is housekeeping and depends on nothing, but every file it touches is
a file some other issue will also touch, so it is cheapest done between two
pieces of work rather than alongside one. It carries a narrow authorisation to
add the two header lines — and only those — to files under `params/`.

Issue 27 came out of reviewing the table issue 5 landed. It is the only
automated guard on a hand transcription of 222 numbers, so it belongs before
the engine reads the table in 12 rather than after. Like 26 it carries a narrow
authorisation under `params/`: rewriting one boxed comment in the template and
in `params/2026/mortality.yaml`, and nothing else.

Issue 28 came out of reviewing issue 6, which added a block to the province
template with two unmarked placeholder numbers. The marker count is what tells
a human how much of a template is left to fill, so an uncounted placeholder is
the failure the workflow exists to prevent. The equivalent test already exists
for the mortality template; this generalizes it. It touches no file under
`params/` and needs no authorisation there.

Issue 29 came out of reviewing issue 13. The moment matching that turns annual
assumptions into a monthly distribution can take a correlation matrix that is
perfectly valid — symmetric, unit diagonal, non-negative eigenvalues — and
produce one no lognormal realises; two classes at 5% volatility with a
correlation of -1 are enough. `engine/mc/returns.py` refuses those inputs, but
only at draw time, so the error lands a long way from the scenario file that
caused it. Moving the check to load needs something that builds a covariance
matrix out of `Assumptions`, and nothing does. It sits before 19 rather than
after because 19 would otherwise build that piece inline, and it carries a
layering decision — where the shared algebra lives, so that `engine/scenario/`
need not import `engine/mc/` — that does not belong inside the month step.
Both ends of the interface are already fixed, by 10 and by 13, so the builder
is a segment between two pinned endpoints rather than a speculative design.

Parameters the human supplies along the way, by issue: 6 adds the pension
splitting share and eligibility age, EI rate and maximum, CPP base rate and
survivor parameters, CESG cessation age, the RESP penalty rate, and the Alberta
dividend credit rate; 5 adds the life table; 15 and 17 add expected values.

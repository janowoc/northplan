<!-- SPDX-FileCopyrightText: 2026 Jan Owoc -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

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

## Conventions the engine holds everywhere
- Real dollars are January dollars of the scenario's start year. The two
  years of prior net income a scenario states for each person are stated as
  filed and restated to real dollars when the opening state is built, each
  valued at the middle of the year it was earned in. One parameter year
  serves the whole run. Indexed amounts are constant in real terms except
  for a per-schedule erosion factor; unindexed amounts decay.
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
- `engine/` never imports `api`, `fastapi`, `cli`, `report`, `starlette`, or `uvicorn`.

## Docstrings
- A docstring states the contract: what the function takes and in what
  units, what it returns, what it raises, the shape convention, and the one
  non-obvious decision a caller must know to use it correctly. A module
  docstring says what the module owns and its invariants.
- The rationale, the alternatives rejected, and the review history go in the
  commit message, not the docstring. A reader who wants the argument runs
  `git log -p`. As a rule of thumb a function docstring fits in twenty-five
  lines and a module docstring in forty.

## Scope of a change
- One logical change per commit. Conventional commit messages.
- Do not refactor code outside the module you were asked to change.
- Do not add dependencies without asking.
- Run `ruff format` on each whole file you touch. The tree is fully formatted
  and `tests/test_formatting.py` holds it there. ruff is pinned in
  `pyproject.toml`; a version bump is a commit of its own that reformats the
  tree.

## Doing the work on an issue or a pull request
- Each piece of work is a GitHub issue or an open pull request. Do not start
  work that is neither.
- When I ask for work on an issue or a pull request, the default is that you
  dispatch the `implementer` agent to write it and then the `verifier` agent
  to review it. Do it yourself only when I say so. The patches come out
  better this way: the main thread stays free to hold the design and to argue
  with the review, instead of also being the hand that types.
- This is a standing authorisation that overrides the general instruction not
  to reach for subagents. It covers these two agents, on an issue or a pull
  request, and nothing else.
- Resolve the design gaps BEFORE dispatching. The implementer does not design
  and must not improvise: an issue that leaves a field list, a sentinel, a
  signature, or a default open is a decision for the two of us, and the
  implementer receives it already made, in writing. Report those decisions to
  me as a numbered list.
- Before dispatching, probe the design as it will be briefed, not only the
  code it replaces, in a scratch script against the inputs it could break. A
  fix brief after a review is probed the same way before it goes back.
- Where a change must match an existing behaviour in all but one respect, the
  brief names that behaviour as the oracle and asks for a differential test
  over a corpus of inputs. The brief's test list follows the design's
  branches: each skip and each raise gets a named test.
- A hole that predates the issue, found in review or while probing, is
  handled by where it lies:
  - Inside the area the issue is changing (a missing test, an unchecked edge
    case in the module or behaviour under work): fold it into the issue by
    default.
  - Outside that area, whether or not the same problem also occurs inside it:
    a new issue by default, folded in only when I say so. Instances inside the
    area are still fixed under the rule above; one new issue covers every
    instance outside it.
  Either way, the hole and what was done with it go in the numbered list of
  decisions reported to me.
- A verifier's NIT or test-gap finding does not by itself start a new round.
  It is batched into the last round or filed. A statement that is wrong is not
  a nit.
- The brief carries the issue's comment content the agent needs. A subagent
  reads what you hand it, not the issue.
- The verifier runs after the implementer has finished, never alongside it. A
  review of a half-written tree reports the race rather than the code.
- Neither agent commits and neither closes an issue. You commit after the
  review, once I have seen it.
- Name the issue in the commit message: a `Refs #12` line of its own, after
  the body and before the attribution lines. `Refs`, never `Closes`, `Fixes`,
  or `Resolves` — those close the issue when the commit reaches the default
  branch, and closing an issue is mine alone. A commit that serves two issues
  names both.
- A pull request is worked as itself and needs no issue. The `verifier` agent
  reviews its diff before I merge it. Unless it comes from a fork, a fix it
  needs goes on its branch, and the fix's commit carries no `Refs` line.
- A pull request from a fork is read and never run: no checkout, no test run
  and no approval of its workflows until I have read it and say so. Nothing is
  pushed to its branch; a change it needs is asked for in a review comment.

## Issues and pull requests
- You may read and comment on GitHub issues and pull requests via `gh`. Show
  me the text of anything you post there first (a comment, an issue, a pull
  request or a review), and post it only once I approve it; if I ask for a
  change, show me the new text.
- Read an issue's comments, not just its description:
  `gh issue view <n> --comments`. Plain `gh issue view` prints the body and a
  `comments: <count>` header line, never the comment bodies. Amendments and
  decisions from review land in comments, so an issue read without them is the
  issue as it stood before the review that changed it.
- A new issue says where it came from in its own body. Nothing in the
  repository registers it. `docs/archive/roadmap-v0.1.0.md` is the ordered
  plan that built the first version and is not updated.
- A new issue gets labels, and they go in the text you show me: `agent`
  and/or `human` for who does the work; one of `engine`, `params`, `tests`,
  `web` and `documentation` for each area it touches; and `bug` for a defect
  or `enhancement` for something new. A label the scheme lacks is created
  only once I agree.
- You may NEVER close an issue. Only the human closes issues, after
  verifying the output.
- You may NEVER merge a pull request. Only the human merges, after seeing the
  review.

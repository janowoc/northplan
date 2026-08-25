# Working rules

## Tax and benefit constants — the most important rule in this repo
- NEVER invent, recall, infer, or estimate a tax parameter. Not a bracket
  edge, rate, threshold, credit amount, RRIF factor, or adjustment
  percentage. Every one comes from a file under `params/` that the human
  populated by hand.
- If a needed parameter is missing from `params/`, STOP and ask. Do not
  guess, do not use a placeholder, do not "use a reasonable value for now".
- Never inline a numeric tax constant in a `.py` file. If you need one,
  it goes in YAML and gets loaded.

## Tests
- NEVER edit an expected value in a test to make it pass. If the
  implementation and the expectation disagree, report the discrepancy and
  stop. Expected values are ground truth supplied by the human.
- Never delete or skip a failing test.
- Golden tests in `tests/golden/` are sacred. Characterization snapshots in
  `tests/characterization/` may be regenerated only when I explicitly say so.

## Scope
- One logical change per commit. Conventional commit messages.
- Do not refactor code outside the module you were asked to change.
- Do not add dependencies without asking.

## Issues
- You may read and comment on GitHub issues via `gh`.
- You may NEVER close an issue. Only the human closes issues, after
  verifying the output.

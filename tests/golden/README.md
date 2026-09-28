<!-- SPDX-FileCopyrightText: 2026 Jan Owoc -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Golden tests

A golden test asserts that an engine function returns a specific, known-correct
number for a specific input, where "known-correct" means someone checked it
against an authority outside this codebase — a government calculator, a
published bracket table, a T1 line from a real return — and wrote down what
they checked it against and when. That is a different kind of evidence than
anything a test can derive from the code or the parameter files themselves,
which is why golden cases live apart from the structural and provenance
checks under `tests/params/`: those ask whether a file is internally
consistent, this asks whether the engine's arithmetic agrees with the world.

Cases are written by the human, by hand, from that external source. An agent
may add the harness that turns a case file into a test — the files in this
directory other than `cases/*.yaml` — but must never add, edit, or delete a
case, and must never edit an expected value to make a failing test pass. A
golden test that fails means the implementation is wrong until a human says
otherwise; it does not mean the test's expected value needs adjusting. If a
case looks wrong, the right response is to say so and stop, not to fix it.

## Why `source` and `checked` are mandatory

A number with no record of where it came from is indistinguishable from a
number someone guessed, and this repository's whole discipline around tax
parameters exists to make guessed numbers impossible to mistake for checked
ones. Every case therefore carries `source` (the calculator, page, or
publication it was checked against, as a non-empty string — a bare or
blank `source:` is treated exactly like a missing one) and `checked` (the
date it was checked, as a bare ISO date, not a timestamp, and not a date in
the future — the same requirement `tests/params/test_param_provenance.py`
places on a parameter file's source comments, for the same reason: a check
that has not happened yet is worse than no claim of one). A case missing
either field, or setting either to something that is not really a value,
fails at collection, before any test runs, with a message naming the
offending file and case — see `conftest.discover_cases`. There is no case in
this repository, real or synthetic, that gets to skip this.

## File format

A case file is YAML with a `target` (a dotted path to the engine function
under test) and a list of `cases`. Each case has a `name` — non-empty, and
unique within its file, because a blank or duplicated name breaks the
`<file>::<name>` id a failure is grepped back to — keyword arguments under
`inputs`, an expected result under `expected`, and its `source` and `checked`
provenance. `expected` is either a single number under `value`, compared
against the function's whole return value, or a mapping of output name to
number, looked up on the result. A named output is only ever looked up on a
mapping, a dataclass instance, or a NamedTuple, by its declared field — never
by an unrestricted `getattr`, which would resolve attributes like `imag` or
`ndim` on an ordinary float or numpy scalar and let a named-output case pass
against a bare number forever without checking anything. A result that is
just a scalar must be addressed with `value:` instead. `value` itself means
the whole result and cannot appear alongside a named output in the same
`expected` — a case that mixed them would have `value`'s meaning silently
change from "the whole result" to "the output named `value`" the moment a
second key was added, so the harness rejects the combination at discovery
instead. `expected` itself must be non-empty, and every number in it must be
finite: a case with nothing under `expected` asserts nothing and would
otherwise report green having checked no number at all, and a case expecting
`inf` would report green off an overflowing target via `inf == approx(inf)`,
so both are rejected at discovery. A key repeated within one mapping anywhere
in a case file is refused at discovery, naming the file, line, and key — it
is not silently overwritten with the last value, which is what plain YAML
would do.

## Parameter inputs: `params`, `real_params`, `real_params_year`

Three input names are special, and a case names exactly one of them —
naming two of them fails at discovery, naming both keys; naming all three
fails too, naming all three.

- `params: {year, file}` resolves to the real, hand-populated parameter set
  — `load_year(year, ...)[file]` — for a target that reads one file at its
  published, undeflated values.
- `real_params: {year, file, inflation}` resolves to that same file's
  deflated real-terms view (`engine.core.indexation.RealParamSet`) —
  `real_year(load_year(year, ...), inflation)[file]`.
- `real_params_year: {year, inflation}` — one file shallower, with no `file`
  key — resolves to the *whole* tax year's real-terms view
  (`engine.core.indexation.RealParamYear`) —
  `real_year(load_year(year, ...), inflation)` — for a target that reaches
  more than one parameter set itself, such as both `federal` and a province.

All three resolve under the keyword `params`, because an engine function
names its primary parameter-set argument `params`. `inflation` is an int or a
float, a bare fraction — `0.02` for 2%, never `2` — and there is no default
for it on either real-terms kind: the human states it explicitly in every
case, `0.0` for a source that is a nominal calculator. A bool and a
non-finite value (`.nan`, `.inf`, `-.inf`) are rejected at discovery. The
range check on the inflation rate itself — at or below -1 — is not done at
discovery: it is `engine.core.indexation`'s own check, raised at run time,
so that rule has a single owner.

## Target shims: `tests/golden/targets.py`

`target` in a case file is a dotted path to any importable callable, so a
case can also name a function in `tests/golden/targets.py` —
`golden.targets.<name>` — rather than an engine function directly. A shim
exists for exactly one reason: the real target's argument cannot be written
as a YAML mapping, because it is a constructed engine type rather than a
number, a string, or one of the three parameter-input kinds above.
`engine.tax.combined.person_assessment` is the first example: its `ledger`
argument is an `engine.core.state.IncomeLedger`, a frozen dataclass of
fifteen arrays, which a case file has no way to build.

A shim is pure construction and nothing else: it takes plain keyword
arguments, builds the engine value the real target wants, and calls that
target unchanged. It contains no arithmetic operator and no conditional —
checkable at a glance, since the whole point of a shim is that a reviewer
can see it computes nothing. Every per-path numeric argument becomes a
one-path `(1,)` array, since every carried engine array is `(n_paths,)`,
via `np.asarray([value], dtype=np.float64)`. There is no type check on that
conversion, because a shim contains no conditional to perform one: numpy's
own conversion decides which mistake raises and which one converts
silently, and this list is not exhaustive. Raises: a non-numeric string, a
mapping, a date. Converts silently: a quoted numeric string such as
`"60000"` becomes `60000.0`; a bool becomes `1.0` or `0.0`; a YAML null
becomes `NaN`. A null usually makes the compared output `NaN` too and fails
the case loudly — but a null in a field that never reaches the compared
output passes silently, e.g. `remitted` on `person_assessment`, which the
engine function never reads, or `cpp_base_contributions` or `ei_premiums`
in a case whose `expected` compares only `net_income`. A null
`age_at_end_of_year` is subtler still: every use of age in `engine/tax/` is
a comparison, and `NaN >= x` is `False`, so the result is a finite number
computed as if the person were under the age threshold — not `NaN` — and it
too can pass silently. The case author writes plain unquoted numbers, same
as with any other target.

A shim is added only when a YAML mapping genuinely cannot express an
argument the real target needs — it is not a place to make an ordinary call
site more convenient. A case for a function that needs one names
`golden.targets.<name>`; every other case names the engine function it is
testing directly.

## Tolerance is derived from a declared `rounding`, not written by hand

Comparison uses `pytest.approx(abs=tolerance)`, but a case never writes
`tolerance` itself. Instead it may declare `rounding` — a statement of *why*
the source is imprecise — and the harness derives the tolerance from it. The
premise: a case needs slack because the source is imprecise, not because the
engine is approximately right, and the ways a source is imprecise are few and
enumerable. There is deliberately no numeric field to widen; loosening a case
means changing a claim about the source, which a reviewer can check against
the `source` URL sitting in the same case.

`rounding` is one of four members, each with a fixed, derived tolerance:

| `rounding` | Tolerance | Why |
|---|---|---|
| `source_rounds_to_cent` | 0.01 | Half the last displayed digit is 0.005; 0.01 leaves room for float representation |
| `source_rounds_to_dollar` | 0.50 | Half the last displayed digit |
| `source_rounds_to_ten_dollars` | 5.00 | Half the last displayed digit |
| `monthly_cent_times_twelve` | 0.06 | A monthly figure rounded to the cent, multiplied by 12, compounds ±0.005 twelve times |

Omitting `rounding` means `source_rounds_to_cent`, so the harness's original
default tolerance of 0.01 is preserved and most cases write nothing. An
unrecognized value — including a number, a list, `None`, or a bare
`rounding:` key, which parses to `None` — fails at discovery naming the file,
the case, the offending value, and the four allowed members. A case that
still sets the old `tolerance:` field is rejected too, with a message that
names `rounding` as its replacement.

These four numbers describe how a source *publishes* a figure, never the tax
system itself — no bracket, rate, threshold, or credit lives among them —
which is why they are fixed in `conftest.py` (`ROUNDING_TOLERANCES`) rather
than under `params/`, which `CLAUDE.md` reserves for hand-populated tax
parameters. There is deliberately no fifth, numeric escape hatch: a source
whose imprecision none of the four members describes gets a new member added
to that table, in a diff someone reviews, not a bespoke tolerance a case
writer can quietly inflate. If real cases later prove this too strict,
reinstating a numeric escape with a mandatory written justification is a
separate, deliberate change — not something to work around here.

A case whose `rounding` is not the default carries a short suffix in its test
id — `(dollar)`, `(ten_dollars)`, or `(monthly_cent_x12)` — for example:

```
tests/golden/test_cases.py::test_golden_case[federal::basic rate on 100k (dollar)]
```

so that loosening a case changes its id in the diff, and every run's output
shows which cases run loose. A case at the default cent rounding keeps the
plain `<file stem>::<case name>` id.

`inputs` must be a mapping with string keys — a YAML list where a mapping was
meant (`inputs: [x: 7.0]` instead of `inputs: {x: 7.0}`) is a small, easy
slip and is caught here rather than dying inside the target call with no
file or case named.

An input named `params` must be a mapping with exactly its keys — `year` (an
int) and `file` (a string) — no more, no fewer, so a typo one level into
`inputs` (`provice` for `province`) is caught at discovery exactly as loudly
as a typo at the case level. The same exact-keys rule holds for
`real_params` (`year`, `file`, `inflation`) and `real_params_year` (`year`,
`inflation`). See "Parameter inputs" above for what each of the three
resolves to, why all three are handed to the target under the keyword
`params`, and the rules around `inflation`.

A worked example, using obviously synthetic numbers rather than a real tax
value — this is not a shape any real case in this repository should take,
only an illustration of the fields:

```yaml
target: some.module.doubles_its_input
cases:
  - name: seven doubled
    source: "arithmetic, checked by hand"
    checked: 2026-09-05
    inputs:
      x: 7.0
    expected:
      value: 14.0
    rounding: source_rounds_to_cent   # optional; this is the default
```

## An empty `cases/` directory is a valid state

`cases/` holding nothing but a `.gitkeep` is a valid state: the harness runs
without a single case. `pytest tests/golden` collects and passes with zero
golden tests in that state, and
`pytest --collect-only -q -m golden` lists nothing. What is *not* valid is a
case file present in `cases/` that yields zero cases — an empty or missing
`cases:` list, or a copy-paste mistake — and that is treated as a broken
file, not an empty one, and fails at collection naming the file, exactly like
a missing `source` or `checked` would.

Discovery walks `cases/` recursively, so a case in a subdirectory (e.g.
`cases/federal/brackets.yaml`, once there are enough cases to want one) is
found, not skipped. The flip side is symmetric: *any* file anywhere under
`cases/`, at any depth, that is not a `.yaml` or `.yml` case file (matched
case-sensitively — `c.YAML` does not count) and not `.gitkeep` — a stray
editor backup, a case file under the wrong extension, a document left in the
wrong place — fails at collection naming that file. Nothing under `cases/`
is ever silently invisible to the harness; it either becomes a test or it is
a collection error.

# Outstanding work

Things known to be missing or provisional, recorded when they were found so
that they surface at the moment they become relevant rather than being
rediscovered. Grouped by the build order in `README.md`, because most of these
are cheap to deal with while the surrounding module is open and expensive to
retrofit afterwards.

Each item says what is missing, why it matters, and where it lands. Nothing
here is a parameter value — where a value is needed, the item says which source
to check, never what the answer is.

---

## 1. `params/` + `tax/`

### Pension income splitting has no statutory maximum share

`engine/tax/combined.py::optimal_pension_split` is specified as returning a
fraction "in `[0, statutory maximum]`", and no such key exists in
`federal.yaml`. The function cannot be written without it.

This is the largest single omission in the tax layer. Splitting eligible
pension income with a lower-earning spouse is among the biggest levers in
Canadian decumulation, and an optimizer that cannot see it will systematically
misprice every drawdown order it evaluates for a couple. It is not a rounding
matter.

Needs: the maximum transferable share, and the definition of which income is
eligible — the latter is a rule rather than a number and belongs in a comment
in `federal.yaml` next to the share.

### EI premiums cannot be computed

`federal.yaml` holds `contribution_credit.ei_maximum_annual` and nothing else.
That is enough to value the credit and not enough to work out what a person
actually paid.

`cpp.yaml` keeps the CPP contribution rates on the stated grounds that
contributions "reduce cash available to save". EI premiums do exactly the same
and there is no rate or maximum insurable earnings to derive them from. Either
add both — probably a small `ei:` block in `federal.yaml`, since EI does not
otherwise justify a file — or state explicitly that EI is folded into net
employment income as a scenario input and the household's contributions are
assumed to be at the maximum.

### The enhanced CPP contribution is a deduction and is not modelled

`contribution_credit.cpp_maximum_annual` is correctly the *base* CPP portion
only: the enhanced portion is deducted from income rather than credited against
tax. There is no parameter for the deduction and nothing consumes one, so
contributions are currently credited but not deducted.

Deal with this when `engine/tax/federal.py::taxable_income` is written, not
before — it needs a deduction path to attach to.

### Unindexed amounts need routing to `unindexed_real_factor`

`federal.yaml` now carries `indexation.pension_income_amount` with an empty
`adjustment_months`, meaning never adjusted. `engine/core/indexation.py`
already has both halves — `real_factor` for indexed amounts,
`unindexed_real_factor` for amounts fixed in nominal terms — but nothing routes
between them yet, and `real_factor` rejects a non-positive adjustment count, so
a caller that gets this wrong fails loudly rather than silently. Good; it still
has to be written.

`cpp.yaml`'s `contributions.first_tier.basic_exemption_annual` is the same
case — a round figure fixed in nominal terms — and has no schedule entry at
all. Add one when the contribution side is implemented.

The distinction matters over a thirty-year plan: an indexed amount's real loss
is bounded by one indexation cycle, an unindexed one's grows without limit.

### The dividend tax credit is an orphan

`federal.investment_income.eligible_dividend_credit_rate` is read by nothing.
`engine/accounts/taxable.py` applies the gross-up and the inclusion rate; no
module applies the credit, and there is no provincial counterpart in
`ab.yaml` — which there will need to be, since the provincial DTC is a separate
rate.

The base is pinned in a comment in `federal.yaml` (6/11 of the gross-up amount,
not of the dividend and not of the grossed-up dividend) because secondary
sources quote it both ways and the two differ by roughly 3.6x. Read that
comment before wiring it up.

### The basic personal amount clawback is not modelled

Noted in a comment in `federal.yaml`. The BPA is reduced for income in the top
bracket, and the file records the full amount only. Fine while households are
below that threshold; wrong and in the flattering direction above it. Decide
whether to model it or to state it as a documented limitation.

### Golden tests against an external calculator

`tests/params/test_param_file_structure.py` asserts shape and internal
agreement without knowing a tax value, which is deliberate — a value copied
into a test is a third place to keep in sync. It therefore cannot catch a wrong
*isolated scalar*, one that no other value in the repository stands in a fixed
relation to. That escape was demonstrated during mutation testing: an OAS
deferral rate of `0.06` where `0.006` was meant passes every structural check.

The parameters in that gap, and so the ones a calculator comparison buys the
most on:

- `oas.deferral.increment_rate_per_month`
- `cpp.start_adjustment.early_rate_per_month` and `late_rate_per_month`
- every `cpi_lag_months`
- the credit amounts: basic personal, age, pension income
- `rrif.withholding.rates`
- `resp.grant.match_rate`, `annual_room`, `annual_maximum`, `lifetime_maximum`
- `tfsa.room.annual_amount`
- `rrsp.room.accrual_rate` and `annual_dollar_limit`
- `oas.recovery_tax.rate` and `threshold_annual`
- `federal.investment_income.*`

Compare against a full-return calculator rather than a per-value lookup: a
lookup is a third transcription of the same number, while a computed return
exercises the interaction between them.

### Four parameter files have no dated sources

`tests/params/test_param_provenance.py` warns every run. `ab`, `cpp`, `federal`
and `oas` carry source URLs but no check dates; `resp`, `rrif` and `tfsa` are
clean. The convention is `# YYYY-MM-DD https://...` above each group of values,
date first so that `grep -h '^# 20' params/2026/*.yaml | sort` lists every
check in date order.

When all seven are covered, change the two `warnings.warn` calls in that file
to `pytest.fail` so the ratchet holds. The docstring says so too.

### Stale references in `README.md`

`tests/golden/` appears in the directory tree at line 77 and in build step 1,
and the directory no longer exists. Decide where calculator-derived tests will
live and fix both, or drop the references.

`federal.yaml`'s `# Read by:` header omits `engine/tax/provincial.py`, which
reads `contribution_credit.*` for the Canada-wide maxima. Those headers are how
a reader finds the consumers of a file, so an incomplete one is worse than
none.

---

## 2. `benefits/` — CPP, OAS, GIS

### The GIS tripwire has no caller

`engine/benefits/gis.py::check_within_scope` is implemented and tested and
nothing calls it, so the refusal never fires and an unmodelled GIS is silently
treated as zero for the households that depend on it most.

It wants a call **once per benefit year**, against projected testable income
for that year, before any policy decision reads the result. Calling it monthly
would compare a month's income against an annual threshold and refuse every
household.

Do this while the OAS benefit-year boundary is being written — the two share
`benefit_year.start_month` and the same income-year mapping.

### Implementing GIS proper

The schema is written up at `params/gis_not_implemented.yaml`, including the
three ways the headline maxima mislead. It sits outside any year directory so
the loader cannot reach it and the placeholder guard does not scan it.

To implement: populate it, move it under `params/{year}/`, flip
`gis.modelled` to `true`, and replace `check_within_scope` with the real
calculation. `engine/benefits/allowance.py` does not exist yet and the
Allowance and Allowance for the Survivor belong with it.

Note the file's own warning that the reduction is computed against a different
income base from the OAS recovery tax, and that entitlement is recalculated per
benefit year — so a large registered withdrawal cuts the supplement a year
later, not immediately. That delayed consequence is visible on a monthly
timeline and is exactly what the decumulation optimizer should learn to avoid.

### OAS is guaranteed never to decrease

Recorded in a comment in `oas.yaml` and flagged there as belonging in the
engine rather than in parameters. Quarterly indexation can in principle produce
a downward adjustment; the statute floors it. Implement in
`engine/benefits/oas.py` alongside the indexation factor.

### Residence history is not modelled, for either program

Full OAS requires 40 years of residence after 18, with a partial pension at
1/40th per year from 10 years, plus a separate 20-year rule for non-residents.
`oas.yaml`'s `pension.age_bands` assume 40+.

CPP is the same shape: `engine/benefits/cpp.py` takes `contributory_history` as
a caller-supplied fraction in `[0, 1]` rather than deriving it from an earnings
history.

Both are deliberate, both are documented in their parameter files, and both
overstate the benefit for anyone who does not qualify fully. If either is ever
derived, the parameters belong with the new module rather than sitting unread
in these files.

### Enhanced CPP is invisible to later cohorts

A single `contributory_history` fraction cannot express that enhanced CPP
raises the replacement rate for later cohorts, so a household retiring in the
2060s is modelled on today's benefit shape. In real dollars that understates
CPP, which is the safe direction, but it is a real limitation and should be
stated in any output that projects far out.

---

## 3. `core/` and `accounts/`

### Alberta's LIF maximum is a greater-of rule the signature cannot express

`ab.yaml` sets `lif.maximum_is_greater_of_factor_and_prior_year_return: true`,
sourced. `engine/accounts/lira.py::maximum_withdrawal(opening_balance,
age_at_start_of_year, params)` has the balance and the age, which covers the
prescribed factor and not the other branch.

The change:

- add a fourth argument, `prior_year_investment_earnings`. **Dollars, not a
  rate** — name it so, or someone will pass `0.06`.
- it must be the growth *credited to the account*, excluding withdrawals and
  transfers. Computing it as closing minus opening balance is wrong, because
  last year's withdrawals reduced the balance, and the error inflates the
  ceiling in exactly the years the household drew the most. This is where the
  bug will be.
- state has to carry it. `LockedInTerms` is frozen static configuration and is
  the wrong home; this needs a per-account field accumulated during the year,
  rolled at `close_year` and read by `open_year` when the ceiling is fixed.
- the first year of a LIF has no prior year. Passing `0.0` is arithmetically
  safe since `max(factor_amount, 0)` is the factor amount — but confirm that is
  what the rule says rather than taking the convenience as the answer.

Keep the argument required even where the flag is false, so every
jurisdiction's call site looks identical. Optional invites omission and then a
silent under-cap in Alberta only.

### Age-keyed tables are indexed by string, not int

`engine/params/loader.py::_freeze` casts every mapping key to `str` so that
dotted-path lookup works uniformly. An age table written `71:` in YAML arrives
as `"71"`, so `table[71]` raises `KeyError` against a table that visibly
contains 71.

Affects `rrif.minimum_factors.by_age` and every province's
`lif.maximum_factors.by_age`. Use `table[str(age)]`. Asserted and explained in
`tests/params/test_param_file_structure.py` so the contract is written down
outside the loader's implementation.

### Federally regulated pensions have no jurisdiction file

`ParamYear.jurisdiction()` raises `ParamFileMissingError` for a code with no
file, and federally regulated pensions are a jurisdiction in their own right
with their own LIF rules. A household holding one currently stops the run,
which is correct behaviour and not a working feature.

Needs a file of its own — not a province's table borrowed. Decide the code
(`federal` collides with the tax file; something like `pbsa` may be clearer)
before writing it.

### A second province

`tests/params/test_param_file_structure.py::test_every_province_file_has_the_same_shape`
is vacuous with one province and becomes load-bearing with two.
`params/province-template.yaml` is the starting point; `engine/tax/provincial.py`
looks up identical paths in every province file rather than branching on the
code, so adding one should be a copy of the template plus sourcing, with no new
code.

---

## 4. RESP

### CESG has no cessation age

`grant` carries the match rate, annual room, and the annual and lifetime
maxima, and nothing that stops it. `Beneficiary.birth_year` is documented as
driving grant eligibility, but `grant_on_contribution` takes no age argument at
all, so as specified the model pays CESG into a beneficiary's twenties.

Needs both a parameter and a signature change. Check the exact rule at the
source — the cut-off year, and the separate condition that applies at 16 and 17
requiring prior contributions. Decide whether to model that second rule or to
record it as out of scope the way residence history is.

### Year-to-date windows must reset in January

Two of them, and they are separate: grant received this calendar year, and
contributions made this calendar year (which is what the enhanced tier's
eligible window is measured against). `engine/core/step.py::open_year` names
both. Enforced per month rather than per year, twelve contributions each
collect a full year's grant.

### The EAP initial-window cap

`eap.initial_window_months` is deliberately fractional — 13 weeks is not 3
months, and rounding up extends a statutory cap by a fortnight in the student's
favour. `resp.py::withdraw` compares it against months since enrolment. Not
wired up yet.

---

## Undecided, and fine to stay that way

These are recorded so they read as decisions rather than oversights.

- **TFSA over-contribution penalty.** The block exists in `tfsa.yaml` with a
  comment saying it may never be modelled: a plan that deliberately
  over-contributes is not a plan the optimizer should be searching. Delete the
  block or implement it; leaving it is also a choice.
- **RRSP over-contribution cushion.** No parameter, same reasoning available.
- **Installment payments.** `federal.yaml` notes that installments are due in
  March, June, September and December, and that the engine assumes a balance
  owing is paid in full in the filing month. That assumption is worth revisiting
  once the tax ledger carries real balances.

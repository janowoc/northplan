<!-- SPDX-FileCopyrightText: 2026 Jan Owoc -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Known limitations

The canonical list of every place the model knowingly departs from the rules.
Each entry says what the rule is, what the model does instead, which way the
error runs, and where in the code the simplification lives. A change that cuts
a corner adds its entry here in the same commit. A change that removes a
simplification deletes the entry.

Entries are grouped by topic and numbered so that a comment in code or a test
can cite one (`# limitations.md L12`). Numbers are never reused.

The direction of error is given from the household's point of view:
*optimistic* means the model makes the plan look better than it is,
*conservative* means worse.

---

## Scope

**L1. Province.** In reality thirteen jurisdictions levy personal income tax,
several with structures Alberta lacks (Ontario's surtax and health premium,
Quebec's separate return, Manitoba's frozen brackets). We model Alberta
residents only; any other province code stops the run. Lives in
`params/2026/ab.yaml`, `engine/tax/provincial.py`. Adding a province is a copy
of `params/province-template.yaml` plus sourcing, with no new code.

**L2. GIS and the Allowances.** In reality low-income OAS pensioners receive a
non-taxable supplement reduced at fifty cents per dollar of testable income,
tested on the prior year's income over a July-to-June benefit year and on a
base that matches no line of the return; a spouse aged 60 to 64 may receive an
Allowance. We do not model either. Instead the engine reports the fraction of
path-years in which a living pensioner's testable income sits inside the band
where GIS would apply, so a plan that leans on unmodelled income is visible
rather than silent. The testable income is approximated as line 23600 less the
OAS received, in the current calendar year. Of the five keys
`params/gis_not_implemented.yaml` lists under `gis.income_base`, that meets
three — the OAS pension is excluded by the subtraction, GIS itself vacuously
since none is modelled, and couples are tested on combined income when the
caller sums the household — and misses two, the employment-income exemption
and the prior-year timing. The GIS exclusion is vacuous rather than
structural: a modelled GIS would arrive at line 14600 and sit inside line
23600, so it would have to be removed to satisfy that key — and that is where
L14 stops being able to equate line 26000 with line 23600. Direction:
conservative on income for low-income households, and optimistic on the cost
of registered drawdown, since it hides the effective marginal rate that makes
early drawdown attractive to them. The exposure indicator carries its own
error, in path-years reported rather than in dollars. The missed
employment-income exemption under-reports: a household with employment income
is tested on more than GIS would test it on, so it falls in the band less
often. The missed timing over-reports for the common shape, a household whose
income falls after retirement: the current year is lower than the prior year
GIS would test, so it falls in the band more often. For a household whose
income is rising, the timing under-reports too. The band thresholds are the
published cut-offs for a single pensioner and for a couple who both receive
OAS; there they carry only L5's error, which sets them a little high, so the
indicator over-reports slightly. A couple in which only one person receives
OAS has a higher published cut-off, whether or not the spouse receives the
Allowance, but is tested against the both-OAS figure, so the indicator
under-reports for it.
Lives in `engine/benefits/gis.py`, `params/2026/oas.yaml` under `gis`.

**L3. Quebec, other provinces' locked-in rules, and federally regulated
pensions.** A LIF is governed by the jurisdiction its pension was registered
in. We carry that jurisdiction per account and stop the run for any
jurisdiction without a parameter file, which today means every one except
Alberta. Lives in `engine/core/state.py::LiraState` and `LifState`,
`engine/params/loader.py::ParamYear.jurisdiction`.

## Timeline and dollars

**L4. Start date.** In reality a plan is drawn up on any date and the current
tax year is partly complete. We start every simulation on 1 January of the
scenario's start year with balances as at that date, and model no income
earned earlier in that year. Lives in `engine/scenario/`.

Nor do we model the return for the year before the start year: in reality
its balance owing or refund settles in the start year's filing month; we
take it as already reflected in the opening cash, so that filing month
settles nothing. Direction: optimistic where a balance was owing,
conservative where a refund was due.

**L5. Real dollars are January dollars of the start year.** All amounts in a
scenario are stated in the purchasing power of January of the start year. A
person's two years of prior net income are the one exception to that statement,
not to the rule: a scenario states them as filed, and the builder restates each
to January dollars of the start year, valuing the figure at the middle of the
calendar year it was earned in, at the scenario's inflation rate
(`engine.core.indexation.as_filed_to_real_factor`). One parameter year serves
the whole run. Indexed amounts are constant in real terms except for the
within-cycle erosion an amount suffers between its adjustment dates, applied as
a constant factor per indexation schedule. Amounts fixed in nominal terms by
statute decay without limit. The routing is declared in each parameter file's
`indexation` block. Lives in `engine/core/indexation.py`. Not modelled: the
quarter-to-quarter oscillation around the erosion mean (bounded, mean zero),
and the rounding of the TFSA limit to five-hundred-dollar steps (treated as
annually indexed).

Nor is the CPI lag. In reality an adjustment is computed from a price window
that closed some months before it takes effect, so the step back up restores an
earlier window's inflation rather than the months just past and an indexed
amount never fully catches up; we model the adjustment as if it restored the
cycle just ended. Every indexed bracket, credit, and benefit is therefore a
little too high in real terms, which understates tax and overstates income —
the direction that flatters the plan, and the same direction as the oscillation
term above.

Annual amounts are valued at the year's mean. In reality a nominal annual
amount — a bracket edge, a credit, a cap — is measured against the year's
nominal income, each month's dollars at that month's price level; we measure
the year's real income against the amount deflated by its mean over the year.
For income spread evenly across the months the model's threshold is higher
than the true one by a second-order amount, a few parts in a hundred thousand
at ordinary inflation rates. Income that arrives late, such as a RRIF minimum
forced out in December, meets a threshold that is too high by more; income
that arrives early errs the other way. Too high is the direction that flatters
the plan: it understates tax against a bracket edge or a credit, and lets more
through a cap than the law would. For a household drawing monthly the error is
close to zero.

**L6. Inflation is a constant scenario assumption.** It is used for the
erosion factor, for the decay of unindexed amounts, and for restating a
person's two years of prior net income from as filed to real dollars at
build (L5). Returns are real. There is no stochastic inflation.

**L57. Erosion of nominal state balances.** In reality a contribution room,
a lifetime total, or an adjusted cost base fixed in nominal terms loses real
value continuously as prices rise. We apply one year's decay once, each
January. Direction: within a year the modelled real value is too high, by at
most one year's inflation and on average half of it; the sign of the effect
on the household depends on the field — too much room is optimistic, too high
an ACB is optimistic on tax. Lives in each account module's `erode_nominal`,
with the factor from `engine/core/indexation.py::nominal_carry_factor`.

**L61. Nominal output is valued at the year end.** In reality a year's
spending leaves across its months and an estate passes in the month of the
second death; tax is a figure assessed on the year, not a payment on a date.
With `--nominal` on the command line, or `nominal=true` on the API, each is
converted to nominal dollars at 31 December of the year it falls in, by
`(1 + inflation)` raised to the years
from the start year's January. Direction: nominal spending and estates are
overstated, by at most one year's inflation and about half of it on average;
nominal tax is stated at its assessment date, the December close. Year-end
balances are exact, and no simulated number changes. Lives in
`report/tables.py`.

**L7. Future legislated changes.** In reality the CPP enhancement phases in to
2065 and the TFSA limit steps. We use the start year's parameters for every
year. Direction: conservative on CPP for cohorts retiring after roughly 2040.

## Mortality

**L8. Life table.** In reality mortality improves over time and people with
retirement savings live longer than the population average. We use a
Statistics Canada period life table for the general population, per sex, with
no improvement scale. Direction: conservative on longevity, therefore
optimistic on estate and on the sufficiency of savings. Lives in
`params/2026/mortality.yaml`, `engine/core/mortality.py`.

**L45. One national life table for every province.** In reality mortality
differs by province, and Statistics Canada publishes a separate table for each
one in 13-10-0114-01. We load a single Canada-wide table and apply it to every
household regardless of where they live. For Alberta, the only province
modelled (L1), the national table is the lighter of the two: 13-10-0114-01
gives e(65) as 20.85 for Canada against 20.70 for Alberta, both sexes,
2022/2024. We therefore credit an Albertan with roughly two extra months at 65
and overstate longevity. Direction: conservative on estate and on the
sufficiency of savings — the opposite way round from L8, which this partly
offsets rather than compounds. Retirement ages are what the comparison turns
on, not e(0), which carries working-age mortality the model never reaches.

The gap is accepted deliberately rather than pending. Closing it means
transcribing a table per jurisdiction by hand, and two months of life
expectancy is not where the error in a household projection lives: age at death
in the family and smoking status would each move it further than geography
does, and the model takes neither as an input. Lives in
`params/2026/mortality.yaml`, whose `GEOGRAPHY` comment records the choice.

**L9. Monthly hazard.** The table gives annual death probabilities. We assume
a constant force of mortality within each year of age to derive a monthly
hazard, and draw one uniform per person per path mapped through the survival
curve to a death month. Deaths of spouses are independent.

**L10. Terminal age.** Every path dies by the table's terminal age, where the
probability is one. The simulation runs each path to the second death; there
is no separate horizon. The draws are sized to whole years, so the run always
reaches the December close of the year of the last second death, and that
year has a row. `engine/scenario/lifespan.py::check_lifespan` refuses rather
than simulates a scenario the table cannot represent: a person already past
the terminal age at the opening, or a household no one in which can be alive
at the run's first December close.

## Income tax

**L11. Credits modelled.** Basic personal amount, age amount, pension income
amount, CPP and EI contribution credits, and the dividend tax credit, federal
and Alberta. Not modelled: the spousal amount (material for a single-earner
couple below the pension-splitting age; conservative on tax for them), the
Canada employment amount, disability, medical, tuition, and donation credits.
Lives in `engine/tax/federal.py`, `engine/tax/provincial.py`.

**L12. Basic personal amount clawback.** In reality the federal amount is
reduced for net income across the fourth bracket. We use the full amount at
every income. Direction: optimistic on tax above that bracket, by at most a
few hundred dollars a year.

**L13. Alternative minimum tax.** Not modelled. Bites only on very large
capital gains or donations in one year.

**L14. Net income and taxable income.** In reality several deductions separate
total income from net income, and a further Division C computation separates
net income from taxable income; we model only two deductions from total
income — RRSP contributions and the enhanced CPP contribution — and the one
Division C computation described in L17. Line 23600 is line 23400 less line
23500, the social benefits repayment: the parts repaid of the OAS pension
(line 11300), of EI and other benefits (line 11900), and of net federal
supplements (line 14600). Of those three only the OAS repayment can be
non-zero here — EI is carried as premiums paid and never as benefits
received (L39), and net federal supplements are GIS and the Allowances,
which are not modelled (L2) — so the two lines differ by the OAS repayment
alone, and only for a person who owes one. Line 26000 equals line 23600
except in a person's year of death, when the ITA 111(2) deduction (L17)
reduces it further; that is the only Division C computation modelled. The
repayment is tested against line 23400; the tax brackets are applied to line
26000, and the age amount and the RESP enhanced-grant rate — the latter
through `PersonState.net_income_two_years_prior`, a line 23600 figure like
its sibling `PersonState.prior_year_net_income` — to line 23600. Gross OAS
enters both 23400 and 23600 as income, as the rules require; what separates
them is the repayment deducted at line 23500, not a different treatment of
the pension itself. Direction: conservative on tax — the omitted deductions
overstate taxable income, so the engine's tax comes out too high outside a
death year; the one Division C deduction we do model (L17) only offsets this
for a death year's own net capital loss. Lives in `engine/tax/federal.py`,
`engine/tax/combined.py`, `engine/benefits/employment.py`,
`params/2026/cpp.yaml`.

**L15. Pension income splitting.** Modelled to the federal rule: DB pension
income at any age, RRIF and LIF income from the year the transferor is 65 at
year end, up to the statutory share, elected once at the December close to
minimise combined household tax including OAS repayments. Not modelled: CPP
pension sharing, spousal RRSPs. A payment out of a RRIF (a LIF included)
received as a consequence of a spouse's death is qualified pension income
(ITA 118(7)), eligible below 65 for both the credit and the split. We count
as qualified the share of each RRIF/LIF withdrawal that the account's
inherited fraction gives (L41), at any age. In reality a successor annuitant
holds the inherited plan apart and may draw on it first; we attribute every
withdrawal pro rata. Direction: conservative. An RRSP or LIRA rolled over at
a death and converted later counts as the survivor's own. In reality its later
payments come out of the survivor's own plan after an ITA 60(l) transfer,
which as we read 118(7) are not received as a consequence of the death, so
this matches the law; were they to qualify, the direction would be
conservative. In reality a plan a successor annuitant already held at the
scenario's start is still inherited; we count it as the survivor's own, since
a scenario cannot state it. Direction: conservative.

**L51. Pension split search resolution.** In reality the transferred amount is
any amount up to the statutory share, and the tax-minimising choice is a point
on a piecewise-linear frontier. We evaluate transfer fractions on a grid of
0.05 in each direction, plus the maximum share itself, and take the best of
them per path. Direction: conservative on tax — the grid's best is never
better than the true optimum, so household tax is overstated, by at most the
curvature of the objective across one grid step. Lives in
`engine/tax/combined.py::household_assessment`.

**L16. Withholding.** RRSP withdrawals and RRIF withdrawals above the minimum
are withheld at the banded rates in `params/2026/rrif.yaml` and remitted in
the month. Employment and DB pension income are withheld at an approximation
of the payroll tables: the engine's own combined tax on twelve times the
month's amount, divided by twelve. CPP and OAS are paid gross, which is the
default in reality. The balance settles from cash in the filing month; a
refund arrives the same way. Not modelled: instalments, and the July-to-June
OAS recovery withholding, which is a refundable prepayment of the repayment
assessed on the return. Lives in `engine/tax/withholding.py`, called from
`engine/core/step.py`.

Three further approximations inside the payroll estimate, and they do not all
run the same way. It is computed per income source, so a person with both
employment income and a DB pension is given the full basic personal amount and
the full age amount against each: direction, under-withholds. Against that, it
carries no pension income amount, since it is told only whether the source is
employment: direction, over-withholds on a DB pension. And it is computed on
gross annualised income with no deductions of any kind — no RRSP deduction, no
enhanced CPP contribution, no second-tier CPP contribution, since only the
first-tier base contribution is derived and that one is a credit rather than a
deduction: direction, over-withholds, the same way as the second and so adding
to it rather than cancelling it. None of the three changes the assessment, and
so none changes lifetime tax; all three change when the cash moves, which is
what a policy reading the cash balance between the accrual month and the filing
month sees.

**L56. Withholding on a RRIF or LIF minimum.** In reality withholding applies
to the excess of a payment over the year's minimum amount (ITR 103(6); see
also ITA 153(1)(l), 146.3(1), CRA guide T4079), and the source gives no
ordering within the year. We treat the year's **first** dollars as satisfying
the minimum, so each later month's excess is banded separately rather than
as one annual payment. Direction: splitting one payment into twelve lowers
the band each falls in, so the model **under-withholds**; cash timing only,
the December assessment is unchanged. Same family of error as L16. Lives in
`engine/accounts/rrif.py::withdraw` and `engine/accounts/lif.py::withdraw`.

**L58. Withholding on the cash floor.** In reality every RRSP, RRIF and LIF
withdrawal is withheld at source. When the step force-withdraws to restore
negative cash to zero (L38), it withholds nothing: the withdrawal is income
of the year, and its tax settles with the balance in the filing month.
Direction: under-withholds; cash timing only, the December assessment is
unchanged. Same family of error as L16. Lives in `engine/core/step.py`.

**L17. Investment income detail.** Interest, eligible dividends, and capital
gains are modelled with the inclusion rate, gross-up, and both dividend tax
credits. In reality a net capital loss for the year is carried back three
years or forward indefinitely against capital gains; we offset losses against
gains of the same year only, and otherwise drop a net loss. In the year of
death (and the preceding year), ITA 111(2) additionally allows that year's
own net capital loss as a deduction from taxable income against any income —
confirmed against the source
(https://laws-lois.justice.gc.ca/eng/acts/i-3.3/section-111.html). We apply
this to the year of death only, for that year's own net loss (there are no
carried losses in this model), both on the terminal return
(`engine/core/step.py::resolve_deaths`) and at the December close of a
person's death year (`close_year`); it changes neither net income (line
23400) nor the OAS repayment, both of which are computed before it applies.
The preceding-year leg is not modelled. Direction: conservative on tax and on
the OAS repayment in general; conservative specifically for the un-modelled
preceding-year leg. Not modelled: non-eligible dividends, foreign withholding
tax, return of capital, the superficial loss rule. Lives in
`engine/tax/federal.py::death_year_capital_loss_deduction`,
`engine/tax/combined.py::person_assessment`.

## Public pensions

**L18. CPP amount.** In reality the pension is computed from a full earnings
history with drop-out provisions and the post-2019 enhancement. We take
either a fraction of the maximum supplied by the scenario, or the amount
already in pay. Direction: on the CPP pension, depends on the fraction
supplied; conservative for younger cohorts (L7). Lives in
`engine/benefits/cpp.py`.

**L19. CPP survivor pension.** In reality the survivor formula differs above
and below age 65 and is capped when combined with the survivor's own pension.
We apply the 65-and-over formula at every age: a fixed share of the deceased's
age-65 base pension, with the combined total capped. For a person already in
pay the base pension is taken as the amount in pay. Direction: either way on
the survivor pension for survivors under 65, small.

**L20. Other CPP benefits.** Not modelled: the post-retirement benefit for
contributions made while receiving CPP, the disability pension, the death
benefit, credit splitting.

**L21. CPP contributions.** In reality a worker contributes on employment
income until 70, except that one aged 65 to 70 who is receiving their CPP
pension may elect to stop. We take contributions on all employment income at
every age. Direction: conservative on cash for anyone who would elect to
stop, or who works past 70. The post-retirement benefit those contributions
earn is not modelled (L20), which is conservative on pension. Lives in
`engine/benefits/employment.py`.

**L22. OAS residence.** In reality full OAS needs forty years of residence
after 18. We assume every person qualifies in full. Direction: optimistic on
OAS for anyone who does not. Lives in `params/2026/oas.yaml`.

**L23. OAS repayment.** Assessed at the December close on the current year's
net income before the repayment (line 23400), capped at the OAS received that
year, and settled with the balance owing. See L16 for the withholding that is
not modelled.

**L24. OAS never decreases.** In reality a quarterly adjustment is floored at
zero. Under constant real modelling the rule has no effect and is not coded.

**L52. Amounts already in pay are not eroded.** In reality a CPP or OAS
pension in pay, and a DB pension indexed to prices, is fixed in nominal terms
between adjustments and loses real value until the next one, as a published
maximum does. We pay the amount the scenario states, in January dollars of
the start year, unchanged every month apart from the OAS age-band step-up;
the erosion factor reaches only amounts read from `params/` (L5). Direction:
optimistic on pension income, by one erosion factor per schedule, larger for
CPP (adjusted annually) than for OAS (quarterly). Lives in
`engine/benefits/cpp.py::pension_monthly`, `oas.py::gross_pension_monthly`,
`pension.py::db_pension_monthly`.

**L54. A benefit start already past when the run opens.** In reality an
unstarted CPP or OAS pension can be backdated a limited time, with a back
payment. We refuse an elected start age below the person's age in whole
years at the run's opening (1 January of `start_year`), along with any start
age outside the window in `params/`, any CPP in pay for someone younger than
its earliest start age, and any OAS in pay for someone not older than its
earliest start age, since OAS is first paid the month after it. An election
equal to the person's age in whole years at the opening is first paid at the
opening, with no back payment; an OAS election reached in the opening month
is the exception (below). CPP starts at the person's age in months at the
opening. OAS, which is first paid in the month after its start (OAS Act
s.8(1)), starts at their age in months in the month before the opening, so
that it too is first paid at the opening; an OAS election whose age in
months is reached in the opening month itself has not passed: it is first
paid the month after the opening, like any other OAS election. Either start
is capped at the latest start age. A person whose age in whole years at the
opening is above the latest start age, and whose benefit is not yet in pay,
has no valid election; the scenario must state that benefit as in pay.
Direction: neither optimistic nor conservative for a refused scenario, which
produces no wrong number — the cost is expressiveness. Either way for an
election first paid at the opening: it forgoes any back payment but pays the
adjustment for the later start for life, and which is worth more depends on
survival. Lives in `engine/scenario/start_ages.py::check_start_ages`, and in
`engine/benefits/cpp.py::pension_monthly` and
`oas.py::gross_pension_monthly` for the start at the opening.

## Registered accounts

**L25. RRSP room.** In reality room is eighteen percent of prior-year earned
income to the dollar limit, less any pension adjustment from a workplace
plan. We take employment income as earned income and use no pension
adjustment. Direction: optimistic on room for anyone accruing a DB pension
while working. Not modelled: the over-contribution cushion, spousal RRSPs,
the Home Buyers' and Lifelong Learning plans, contributions in the first sixty
days of the next year. Lives in `engine/accounts/rrsp.py`.

**L26. RRIF conversion.** Modelled as one optional partial conversion at a
policy-chosen age, and full conversion at the statutory age. An election at
the statutory age itself converts the whole balance, whatever its fraction,
and one above it is refused by
`engine/scenario/start_ages.py::check_start_ages`. Not modelled: the
younger-spouse election for the minimum, and any one-year reduction of the
minimum. Lives in `engine/accounts/rrif.py`.

**L27. LIF maximum.** In reality Alberta's maximum is the greater of the
factor result and the prior year's investment return. We apply the factor
result only. Direction: conservative on the LIF maximum: the modelled
ceiling is never higher than the rule allows. Not modelled: the one-time
unlocking transfer, small-balance and hardship unlocking. Lives in
`engine/accounts/lif.py`.

**L55. LIRA conversion timing.** In reality a LIRA may be converted to a LIF
at any time from the unlocking age, and the latest is 31 December of the year
the holder reaches the Income Tax Act's RRSP maturity age. We convert at the
deadline only, in the December close of that year. Direction: conservative —
the money stays locked longer than it need be, and the LIF's minimum and
maximum both start later. Lives in `engine/accounts/lira.py::must_convert`.

**L47. One locked-in account per person.** In reality a person may hold
several LIRAs or LIFs, from several employers, registered in different
jurisdictions, and they do not merge: each is drawn under the maximum table of
the jurisdiction its own originating pension was registered in. We model at
most one LIRA and one LIF per person, sharing one jurisdiction. Direction:
neutral on the LIF maximum for a household whose locked-in money is all from
one jurisdiction, which is the common case; where it is not, the ceiling on
the whole balance comes from whichever jurisdiction the scenario names, and
the error runs in either direction depending on which table is the looser.
Lives in `engine/scenario/schema.py::Accounts.lira` and `Accounts.lif`,
`engine/core/state.py::LiraState` and `LifState`, read by
`engine/accounts/lira.py` and `engine/accounts/lif.py`.

The narrowing arrived with the scenario schema rather than with the account
code, which is why it is recorded here rather than as a gap in `lira.py`: a
second account cannot be *stated* in a scenario, so no code below could act on
one.

**L28. TFSA.** Over-contribution is impossible by construction, so the penalty
is not modelled. Not modelled: room accrued before the start year is a scenario
input rather than computed from history.

**L29. FHSA.** Not modelled.

## RESP

**L30. Student's tax.** In reality grants and growth are taxable to the
student when paid out. We pay them tax-free, on the grounds that a student
drawing a typical plan over four years stays below the basic personal amounts.
Direction: optimistic on the student's tax when the student has other income.

**L31. Withdrawal shape.** The plan pays the scheduled monthly education cost,
or the plan's whole value if less; the payment draws grant and income in
proportion to their shares while income is positive, and from grant alone
when income is zero or negative; contributions are drawn only once grant and
income are exhausted. Not modelled: the cap on assistance payments in the
first thirteen weeks of study, which the scheduled monthly draw stays under.

**L32. Wind-up.** At the end of the education window any unused grant is
repaid and vanishes; any accumulated income is taxable income of the
subscriber, or of the living spouse where the subscriber has died (L41),
assessed with the special tax, and goes to cash net of that tax withheld at
source (L60); and any contributions go to cash tax-free. Not modelled: rolling
accumulated income into an RRSP with room, the ten-year and age-21 conditions
on the wind-up, plan lifetime limits.

**L60. Withholding on an accumulated income payment.** In reality the RESP
provider withholds tax when it pays an accumulated income payment (Form T1171,
*Tax Withholding Waiver on Accumulated Income Payments From RESPs*, exists to
waive it), but we do not know the formula it withholds by. We assume it
withholds exactly the special tax the return assesses on the payment (T1172):
`aip.penalty_rate` in `params/2026/resp.yaml`, 20%, of the accumulated income.
It is remitted in the wind-up month for the person the income is credited to
(L41). Direction: unknown, because the real withholding may be higher or lower
than the special tax. Cash timing only: the December assessment and lifetime
tax are unchanged. Same family of error as L16, but unlike L16 its sign cannot
be stated. Lives in `engine/core/step.py::_phase8_transfers`.

**L33. Grants.** The basic grant with carry-forward room, the annual and
lifetime caps, the enhanced tier on household net income from two years
before, and cessation at the end of the year the beneficiary turns 17 are
modelled. We assume the conditions for grants at 16 and 17 are met. Not
modelled: the Canada Learning Bond, provincial grants.

*Family income.* In reality the additional grant rate follows the primary
caregiver's CCB adjusted income, which includes a cohabiting spouse's, and the
source addresses no death. We model the sum of line 23600 across every person
in the household, including a person who has died, whose net income for the
year before the last calendar year they were alive in — a full year's
income, not the one on their terminal return — keeps counting. Direction:
counting a deceased spouse's income overstates family income and so lowers the
rate; conservative.

*Stale income against this year's edges.* In reality the comparison is between
a past year's nominal income and this year's published, nominal cut-offs. We
compare a real-dollar income with a cut-off the engine has deflated, so the
income runs high by roughly two years of inflation. Direction: conservative — a
household crosses a cut-off slightly early — and small: at most one rate step
on the eligible window per beneficiary per year.

**L34. RESP at death.** The plan is excluded from the estate calculation and is
assumed to pass to the beneficiary intact. At the second death of the
household the plan leaves the model with the beneficiary: its
contributions, grants and income buckets are zeroed and the education cost
is no longer paid. Lives in `engine/core/step.py::resolve_deaths`.

## Taxable account and returns

**L35. Return model.** Monthly returns are independent lognormal draws
calibrated so that twelve compounded reproduce the annual real mean and
covariance given in the scenario. Not modelled: fat tails, mean reversion,
regime dependence, sequence effects beyond what the draws produce.

**L36. Tax character of returns.** Each asset class carries fixed yields for
interest, eligible dividends, and distributed capital gains as fractions of
balance per year; price change is total return less yields. Distributions are
reinvested and raise the adjusted cost base. Withdrawals realise a
proportional share of the embedded gain. Direction: neutral on tax in
expectation. Lives in `engine/accounts/taxable.py`.

**L37. Allocation.** Each account holds fixed weights over asset classes and
is rebalanced monthly at no cost. No glide path.

**L38. Cash.** Cash pays zero real return. Negative cash at a month end is not
allowed: the step force-withdraws in the policy's order, and if nothing
remains spending is cut and the path is flagged depleted. In reality each
spouse holds their own cash; we hold one household balance. Exact while cash
pays nothing, since no income arises to attribute. Direction: neutral on tax.
The attribution lost when pooled cash funds one spouse's account is L50. Lives
in `engine/core/state.py::HouseholdState.cash`.

**L50. Spousal attribution.** In reality income and gains on property one
spouse funds in the other's name are attributed back to the spouse who
supplied the money. Cash is one household balance (L38), so the model does
not know whose money funded a contribution, and taxes a taxable account's
income and gains to its holder. A policy can therefore shift investment
income to the lower-income spouse for free. Direction: optimistic on tax for
couples with unequal incomes. Lives in
`engine/core/state.py::HouseholdState.cash`.

## Household, spending, death

**L39. Employment income.** A real step schedule per person. CPP and EI
contributions are taken on it. Not modelled: self-employment and the employer
share, bonuses, leave, unemployment.

**L40. Spending.** A real step schedule for the household, reduced by a
survivor share from the month the first death takes effect (the first month
the deceased is not alive), plus education cost per beneficiary. No other
age-related change. Lives in `engine/core/step.py`.

**L41. First death.** Every registered account (RRSP, RRIF, LIRA, LIF) rolls
to the survivor's account of the same kind, tax-deferred; a LIF stays locked
in. A RRIF or LIF rolls as if the survivor were named successor annuitant, so
the inherited share of its payments is qualified pension income (L15). In
reality the plan may instead be transferred to the survivor's own, whose later
payments may not qualify. Direction: optimistic for such a household. A
successor annuitant's plan also stays apart from the survivor's own; we merge
the two and carry the inherited share of the balance (L15 gives the direction
of that). The survivor's own RRIF/LIF **minimum** for the year of death is
unchanged, already fixed in January from their own opening balance; the
survivor's LIF **maximum** for the year, by contrast, carries the deceased's
unused annual maximum forward on top of their own. The
deceased's unmet RRIF/LIF minimum for the year of death is not forced out —
in reality it is paid, or continues to a successor annuitant, direction
optimistic and small. The TFSA passes to the survivor as successor holder;
taxable holdings pass at cost (no deemed disposition on the rollover
itself). DB pensions pay the survivor share excluding any bridge — L59
covers a pension not yet in pay at the member's death. CPP pays the
survivor pension per L19, recomputed each month against the survivor's own
CPP. OAS stops. Pension splitting stops after the year of the first death; in
that year the maximum split is pro-rated by the months married (ITA 60.03(1);
T1032 line 18), unless the second death falls in the same year, when every
person is assessed alone (L42). In reality the age-65 tests on a person who
died in the year use their age at death (T1032); we use their age on 31
December. The two differ only for a death in the year of the 65th birthday,
before the birthday, where RRIF/LIF income then counts as eligible pension
income one year early and a receiving spouse's pension credit escapes the
under-65 cap (L15). Direction: optimistic, small. A contribution intended
for a dead person is capped at zero rather than reaching their account. An
RESP wind-up's accumulated income is credited to the living spouse when the
subscriber has died, rather than to the subscriber. Lives in
`engine/core/step.py::resolve_deaths`, `_phase8_transfers`,
`engine/tax/combined.py::household_assessment`, and the six rollover
functions in `engine/accounts/{rrsp,rrif,lira,lif,tfsa,taxable}.py`.

**L42. Second death.** The terminal return brings the full registered balance
(RRSP, RRIF, LIRA, LIF) and the deemed capital gain on taxable holdings into
income with that year's income, taxed as a single person — every person is
assessed alone on the terminal return, never split, even in a household of
two. In reality, where both die in the same year, the two final returns may
still split; we split nothing that year. Direction: conservative. The whole
registered balance enters net income as an amount deemed received at death,
never as pension income, so no part of it earns the pension income amount at
any age. This matches the law: CRA reports the deemed RRIF amount (T4RIF box
18) on line 13000 and excludes the deemed RRSP amount (T4RSP box 34) from the
pension income amount; a LIF is a RRIF and a LIRA an RRSP for this purpose. A
death taking effect in January is taxed that year on an empty ledger, as if it
had happened on 1 January. The estate is what remains after that tax and any
balance owing, floored at zero: in reality the heirs owe nothing beyond the
estate's assets; the shortfall is recorded nowhere. The estate then leaves the
household entirely — every balance is zeroed, and every later row reads zero.
The terminal tax is not part of
`engine.mc.simulate.SimulationResult.tax_assessed`, which holds December
assessments only. The after-tax net worth reported for a living household
(`YearRecord.after_tax_net_worth`) uses the same arithmetic, hypothetically,
as if every person died on 31 December with no rollover, and is floored at
zero the same way; `YearRecord.net_worth` is not floored. Not modelled: probate
(a flat fee in Alberta), charitable bequests, graduated-rate estates. Lives in
`engine/core/step.py::resolve_deaths` and `close_year`.

**L59. DB pension, death before it starts.** In reality a pre-retirement
death usually pays the spouse a commuted-value lump sum; we pay the survivor
share from the pension's start month as if the pension had been deferred.
Direction: either way. Lives in
`engine/benefits/pension.py::survivor_pension_monthly`.

**L43. Depletion.** A path is depleted from the first month in which cash
cannot be restored to zero. A locked-in balance capped by the LIF maximum may
remain in a depleted path.

When every non-RESP account is exhausted and cash is still short by more
than the month's spending — a tax settlement or an education cost the plan
no longer covers — spending achieved falls by the month's spending only, and
the rest of the shortfall is forgiven. Direction: optimistic, on depleted
paths only. Lives in `engine/core/step.py`.

**L48. No birth after the run opens.** In reality a household can plan for a
child not yet born — an RESP beneficiary a scenario states as "assume a child
born in 2030" is the case we would plausibly want. We refuse any person or
beneficiary whose birth date falls after 1 January of `start_year`: every age
the engine derives assumes the birth already happened by the run's opening,
and a birth after it would be a negative age with no meaning to give it.
Direction: neither optimistic nor conservative — a refused scenario produces
no wrong number at all, so the cost is expressiveness, not accuracy. Lives in
`engine/scenario/schema.py::Scenario._check_no_one_is_born_after_the_run_opens`.

## Policies and search

**L44. Policies.** The first version offers one contribution rule (fixed split
with spill) and one withdrawal rule (fixed order, fill to a bracket edge,
optional pension-credit fill). The bracket fill draws RRIF then RRSP every
month, whether or not cash is short, toward a federal bracket edge. It
measures the edge against year-to-date net income (no pension split, no OAS
repayment, since neither is known before December) plus a projection of the
rest of the year: the year to date's run rate, registered withdrawals
excluded, carried over the months still to come, and the LIF minimum not yet
drawn. What remains under the edge is spread evenly over the months left in
the year; a policy may turn the fill off entirely. In reality the bracket
fill's own brackets apply to taxable income after the pension-splitting
election and the OAS repayment deduction; we measure it against year-to-date
net income with neither applied, since neither is known before December.
Direction: the bracket fill under-fills for a person who transfers pension
income away and for one who repays OAS -- both lower taxable income below what
year-to-date net income shows -- and it can over-fill for the person receiving
a pension-split transfer, whose taxable income ends up above it. In reality
the year's income is whatever the rest of the year brings; we project it from
the run rate so far. Direction: income that arrives later than its run rate
implies -- a pension that starts mid-year, investment income, which posts
after each month's decision, and any gain December's own draws realise after
its fill -- lands on top of a fill already taken, so the year can end above
the edge by that much; income that falls off during the year makes the early
months under-fill, which December's fill, measured with nothing left to
project, makes up. The pension-credit fill acts only from the year a person
is `eligible_pension_income.rrif_minimum_age_years` at year end. In reality an
under-65 survivor's withdrawals from a plan inherited from the deceased earn
the credit too (L15); we never fill toward it below that age. Direction:
conservative. The contribution spill pours through `spill_order` person 0
before person 1, so an unlimited taxable spill concentrates investment income
on person 0 rather than sharing it the way a real couple would; this is
pessimistic on tax exactly where person 0 carries the higher marginal rate.
Benefit start ages and the RRIF conversion election are elections the same
policy makes. The optimizer evaluates a named list of policies, with optional
grid expansion, against common random numbers. Not modelled: dynamic rules
that respond to market state, spending rules that vary with wealth.

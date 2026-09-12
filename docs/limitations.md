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
and a spouse aged 60 to 64 may receive an Allowance. We do not model either.
Instead the engine reports the fraction of path-years in which a living
pensioner's testable income sits inside the band where GIS would apply, so a
plan that leans on unmodelled income is visible rather than silent. Direction:
optimistic on income for low-income households, and it hides the effective
marginal rate that makes early registered drawdown attractive to them. Lives
in `engine/benefits/gis.py`, `params/2026/oas.yaml` under `gis`.

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

**L5. Real dollars are January dollars of the start year.** All amounts in a
scenario are stated in the purchasing power of January of the start year,
except a person's prior-year net income, which is taken as filed; one
parameter year serves the whole run. Indexed amounts are constant in real
terms except for the within-cycle erosion an amount suffers between its
adjustment dates, applied as a constant factor per indexation schedule.
Amounts fixed in nominal terms by statute decay without limit. The routing is
declared in each parameter file's `indexation` block. Lives in
`engine/core/indexation.py`. Not modelled: the quarter-to-quarter oscillation
around the erosion mean (bounded, mean zero), and the rounding of the TFSA
limit to five-hundred-dollar steps (treated as annually indexed).

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

**L6. Inflation is a constant scenario assumption.** It is used only for the
erosion factor and for the decay of unindexed amounts. Returns are real. There
is no stochastic inflation.

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
is no separate horizon.

## Income tax

**L11. Credits modelled.** Basic personal amount, age amount, pension income
amount, CPP and EI contribution credits, and the dividend tax credit, federal
and Alberta. Not modelled: the spousal amount (material for a single-earner
couple below the pension-splitting age; optimistic on tax for them), the
Canada employment amount, disability, medical, tuition, and donation credits.
Lives in `engine/tax/federal.py`, `engine/tax/provincial.py`.

**L12. Basic personal amount clawback.** In reality the federal amount is
reduced for net income across the fourth bracket. We use the full amount at
every income. Direction: optimistic above that bracket by at most a few
hundred dollars a year.

**L13. Alternative minimum tax.** Not modelled. Bites only on very large
capital gains or donations in one year.

**L14. Net income and taxable income.** In reality the two differ by several
deductions. We take RRSP contributions and the enhanced CPP contribution as
the only deductions, so the two coincide, and net income for the OAS
repayment and the age amount includes OAS itself as the rules require.

**L49. One net income figure, not two.** In reality the OAS repayment is tested
against line 23400, net income *before* adjustments, while the age amount and
the RESP enhanced-grant rate use line 23600, which subtracts line 23500 — the
social benefits repayment: EI benefits repaid, the OAS repayment itself, and
net federal supplements. We compute one figure per person per year, total
income less deductions, and test everything against it. That figure is line
23400: nothing is ever subtracted back out of it. Of the three components of
line 23500 only the OAS repayment can be non-zero in this engine — EI is
carried as premiums paid and never as benefits received (L39), and net federal
supplements are GIS and the Allowances, which are not modelled (L2) — so the
two lines differ by the repayment alone, and only for a person who owes one.
Direction: pessimistic for that person, whose age amount and enhanced-grant
rate are tested against an income higher than the statutory one by the amount
of the repayment. Taking line 23600 as the single figure instead would be the
worse error, since the repayment would then reduce the base it is computed
from. Lives in `engine/tax/federal.py`.

**L15. Pension income splitting.** Modelled to the federal rule: DB pension
income at any age, RRIF and LIF income from the year the transferor is 65 at
year end, up to the statutory share, elected once at the December close to
minimise combined household tax including OAS repayments. Not modelled: CPP
pension sharing, spousal RRSPs.

**L51. Pension split search resolution.** In reality the transferred amount is
any amount up to the statutory share, and the tax-minimising choice is a point
on a piecewise-linear frontier. We evaluate transfer fractions on a grid of
0.05 in each direction, plus the maximum share itself, and take the best of
them per path. Direction: conservative — the grid's best is never better than
the true optimum, so household tax is overstated, by at most the curvature of
the objective across one grid step. Lives in
`engine/tax/combined.py::household_assessment`.

**L16. Withholding.** RRSP withdrawals and RRIF withdrawals above the minimum
are withheld at the banded rates in `params/2026/rrif.yaml` and remitted in
the month. Employment and DB pension income are withheld at an approximation
of the payroll tables: the engine's own combined tax on twelve times the
month's amount, divided by twelve. CPP and OAS are paid gross, which is the
default in reality. The balance settles from cash in the filing month; a
refund arrives the same way. Not modelled: instalments, and the July-to-June
OAS recovery withholding, which is a refundable prepayment of the repayment
assessed on the return. Lives in `engine/tax/withholding.py`, to be called by
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

**L17. Investment income detail.** Interest, eligible dividends, and capital
gains are modelled with the inclusion rate, gross-up, and both dividend tax
credits. Not modelled: non-eligible dividends, foreign withholding tax, return
of capital, capital losses and their carry-forward, the superficial loss rule.

## Public pensions

**L18. CPP amount.** In reality the pension is computed from a full earnings
history with drop-out provisions and the post-2019 enhancement. We take
either a fraction of the maximum supplied by the scenario, or the amount
already in pay. Direction: depends on the fraction supplied; conservative for
younger cohorts (L7). Lives in `engine/benefits/cpp.py`.

**L19. CPP survivor pension.** In reality the survivor formula differs above
and below age 65 and is capped when combined with the survivor's own pension.
We apply the 65-and-over formula at every age: a fixed share of the deceased's
age-65 base pension, with the combined total capped. For a person already in
pay the base pension is taken as the amount in pay. Direction: either way for
survivors under 65, small.

**L20. Other CPP benefits.** Not modelled: the post-retirement benefit for
contributions made while receiving CPP, the disability pension, the death
benefit, credit splitting.

**L21. CPP contributions after pension start.** In reality a working
pensioner under 70 keeps contributing unless they opt out. We stop
contributions at pension start. Direction: slightly optimistic on cash,
slightly conservative on pension.

**L22. OAS residence.** In reality full OAS needs forty years of residence
after 18. We assume every person qualifies in full. Direction: optimistic for
anyone who does not. Lives in `params/2026/oas.yaml`.

**L23. OAS repayment.** Assessed at the December close on the current year's
net income, capped at the OAS received that year, and settled with the balance
owing. See L16 for the withholding that is not modelled.

**L24. OAS never decreases.** In reality a quarterly adjustment is floored at
zero. Under constant real modelling the rule has no effect and is not coded.

## Registered accounts

**L25. RRSP room.** In reality room is eighteen percent of prior-year earned
income to the dollar limit, less any pension adjustment from a workplace
plan. We take employment income as earned income and use no pension
adjustment. Direction: optimistic on room for anyone accruing a DB pension
while working. Not modelled: the over-contribution cushion, spousal RRSPs,
the Home Buyers' and Lifelong Learning plans, contributions in the first sixty
days of the next year. Lives in `engine/accounts/rrsp.py`.

**L26. RRIF conversion.** Modelled as one optional partial conversion at a
policy-chosen age, and full conversion at the statutory age. Not modelled: the
younger-spouse election for the minimum, and any one-year reduction of the
minimum. Lives in `engine/accounts/rrif.py`.

**L27. LIF maximum.** In reality Alberta's maximum is the greater of the
factor result and the prior year's investment return. We apply the factor
result only. Direction: conservative, the ceiling is never higher than the
rule allows. Not modelled: the one-time unlocking transfer, small-balance and
hardship unlocking. Lives in `engine/accounts/lira.py`.

**L47. One locked-in account per person.** In reality a person may hold
several LIRAs or LIFs, from several employers, registered in different
jurisdictions, and they do not merge: each is drawn under the maximum table of
the jurisdiction its own originating pension was registered in. We model at
most one LIRA and one LIF per person, sharing one jurisdiction. Direction:
neutral for a household whose locked-in money is all from one jurisdiction,
which is the common case; where it is not, the ceiling on the whole balance
comes from whichever jurisdiction the scenario names, and the error runs in
either direction depending on which table is the looser. Lives in
`engine/scenario/schema.py::Accounts.lira` and `Accounts.lif`,
`engine/core/state.py::LiraState` and `LifState`, read by
`engine/accounts/lira.py`.

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
Direction: optimistic when the student has other income.

**L31. Withdrawal shape.** The plan pays the scheduled education cost evenly
over the window, grants and growth first, then contributions. Not modelled:
the cap on assistance payments in the first thirteen weeks of study, which an
even draw stays under.

**L32. Wind-up.** At the end of the education window any unused grant is
repaid and vanishes, any accumulated income goes to cash as the subscriber's
taxable income plus the penalty tax, and any contributions go to cash
tax-free. Not modelled: rolling accumulated income into an RRSP with room, the
ten-year and age-21 conditions on the wind-up, plan lifetime limits.

**L33. Grants.** The basic grant with carry-forward room, the annual and
lifetime caps, the enhanced tier on prior-year household net income, and
cessation at the end of the year the beneficiary turns 17 are modelled. We
assume the conditions for grants at 16 and 17 are met. Not modelled: the
Canada Learning Bond, provincial grants.

**L34. RESP at death.** The plan is excluded from the estate calculation and is
assumed to pass to the beneficiary intact.

## Taxable account and returns

**L35. Return model.** Monthly returns are independent lognormal draws
calibrated so that twelve compounded reproduce the annual real mean and
covariance given in the scenario. Not modelled: fat tails, mean reversion,
regime dependence, sequence effects beyond what the draws produce.

**L36. Tax character of returns.** Each asset class carries fixed yields for
interest, eligible dividends, and distributed capital gains as fractions of
balance per year; price change is total return less yields. Distributions are
reinvested and raise the adjusted cost base. Withdrawals realise a
proportional share of the embedded gain. Direction: neutral in expectation.
Lives in `engine/accounts/taxable.py`.

**L37. Allocation.** Each account holds fixed weights over asset classes and
is rebalanced monthly at no cost. No glide path.

**L38. Cash.** Cash pays zero real return. Negative cash at a month end is not
allowed: the step force-withdraws in the policy's order, and if nothing
remains spending is cut and the path is flagged depleted. In reality each
spouse holds their own cash; we hold one household balance. Exact while cash
pays nothing, since no income arises to attribute. Direction: neutral. The
attribution lost when pooled cash funds one spouse's account is L50. Lives
in `engine/core/state.py::HouseholdState.cash`.

**L50. Spousal attribution.** In reality income and gains on property one
spouse funds in the other's name are attributed back to the spouse who
supplied the money. Cash is one household balance (L38), so the model does
not know whose money funded a contribution, and taxes a taxable account's
income and gains to its holder. A policy can therefore shift investment
income to the lower-income spouse for free. Direction: optimistic for
couples with unequal incomes. Lives in
`engine/core/state.py::HouseholdState.cash`.

## Household, spending, death

**L39. Employment income.** A real step schedule per person. CPP and EI
contributions are taken on it. Not modelled: self-employment and the employer
share, bonuses, leave, unemployment.

**L40. Spending.** A real step schedule for the household, reduced by a
survivor share from the month after the first death, plus education cost per
beneficiary. No other age-related change.

**L41. First death.** Every registered account rolls to the survivor's account
of the same kind, a LIF staying locked in; the TFSA passes as successor
holder; taxable holdings pass at cost; DB pensions pay the survivor share; CPP
pays the survivor pension; OAS stops; pension splitting stops.

**L42. Second death.** The terminal return brings the full registered balance
and the deemed capital gain on taxable holdings into income with that year's
income, taxed as a single person. The estate is what remains after that tax
and any balance owing. Not modelled: probate (a flat fee in Alberta),
charitable bequests, graduated-rate estates.

**L43. Depletion.** A path is depleted from the first month in which cash
cannot be restored to zero. A locked-in balance capped by the LIF maximum may
remain in a depleted path.

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

**L44. Policies.** The first version offers one contribution rule (fixed
split with spill) and one withdrawal rule (fixed order, fill to a bracket
edge, optional pension-credit fill), plus benefit start ages and the RRIF
conversion election. The optimizer evaluates a named list of policies, with
optional grid expansion, against common random numbers. Not modelled:
dynamic rules that respond to market state, spending rules that vary with
wealth.

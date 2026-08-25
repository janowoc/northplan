"""northplan simulation engine.

Pure Python. This package must never import ``api``, ``fastapi``, or ``cli``;
``tests/test_layering.py`` enforces that by walking the AST of every module
here.

Conventions that hold everywhere below this package
---------------------------------------------------

**The timestep is one month.** The simulation advances month by month, and
every state carries the ``(year, month)`` it is the opening position for, with
``month`` in ``1..12`` for January through December. A monthly step is not a
refinement of an annual one; it is what the domain actually looks like. People
retire, die, turn 65, and start a pension in a particular month. CPP's start
adjustment is defined per month away from 65. OAS, GIS, CPP, and DB pensions
are paid monthly. Withdrawals and contributions happen through the year, not on
31 December.

Annual things still exist, and they are *events on the monthly timeline* rather
than a second loop:

- Income tax is assessed on a calendar year. It is computed once, at the
  December close, from year-to-date income accumulated over the twelve monthly
  steps.
- The resulting balance is *paid* in the filing month of the following year,
  which is later than the year it relates to. Cash and accrual differ, and
  ``engine.core.state.TaxLedger`` is where that difference is carried.
- RRIF and LIF minimums and maximums are annual quantities fixed in January
  and satisfied across the months that follow.
- Contribution room is granted in January.

There is exactly one loop, and it is over months. ``advance_month`` invokes
year-opening and year-closing *phases* when the month calls for them. Those are
phases inside the single loop, not loops of their own. If a second function in
this package starts to look like a loop over time, that is a bug — say so
rather than writing it.

**Real dollars, internally, always.** Every dollar amount that crosses a
function boundary inside ``engine/`` is in real (constant purchasing power)
dollars, expressed in the base year of the scenario. Consequences:

- CPI-indexed tax brackets, credits, and benefit thresholds are *constant* in
  real terms from one indexation date to the next. They are loaded once per tax
  year and reused. Do not index them forward; that would double-count
  inflation.
- **Indexation is periodic and lagged, and that costs a constant.** A benefit
  is fixed in nominal terms between adjustment dates, and each adjustment is
  computed from a CPI window that closed some months earlier, so an indexed
  benefit sits permanently a little below its published real value. That
  shortfall is a *constant*: ``engine.core.indexation`` computes it once per
  benefit per scenario and it is applied unchanged every month. The oscillation
  around it is deliberately not modelled — it is bounded, mean-zero, and
  smaller than the uncertainty in the inflation assumption it depends on. What
  is modelled is the level, because the level is permanent and always
  optimistic. The adjustment frequency and the lag are statutory rules and come
  from ``params/``; the inflation rate they act on is a scenario input.
- Non-indexed amounts decay fastest of all, and must decay explicitly. A DB
  pension with no indexation loses real value every month, and the code that
  models it must apply that decay by hand and say so in a comment. Silence
  here is a bug.
- Return assumptions are real returns. Contribution and withdrawal amounts are
  real amounts.
- Conversion to nominal dollars happens exactly once, at display, in ``api/``
  or ``cli/``. No function in ``engine/`` returns a nominal figure.

**Shapes.** The simulation is vectorized across paths, not across time. Months
are sequentially dependent so the outer loop steps through them one at a time;
the paths within a month are independent, so the inner computation handles all
of them at once. Every function in ``tax/`` and ``benefits/`` takes and returns
NumPy arrays of shape ``(n_paths,)``, using ``np.clip`` and ``np.where`` for
bracket logic rather than Python branching. Because NumPy broadcasts, a scalar
is a valid input: a golden-number test calling a function with a single float
exercises the identical code path as production. One implementation, two use
sites.

**Common random numbers.** The return matrix is generated once, from a fixed
seed, and reused across every policy the optimizer evaluates. Redrawing per
policy would make the optimizer chase Monte Carlo noise instead of signal. With
a monthly timestep the draws are monthly, ``(n_months, ...)``; the seed still
fixes them for the whole search.

**No clairvoyance.** A policy function may read only what is knowable at that
simulated month — current balances, current age, income accumulated so far this
year, the current year's brackets, realized history. It may never read a future
return, a future balance, a later month's income, or a terminal value. A
clairvoyant policy produces an answer that cannot be acted on. The monthly
timestep widens the surface for this bug: "how much room is left under the
bracket edge this year" is knowable, "what this year's total income will turn
out to be" is not.

**Household is a list of persons**, from day one, even when only one person is
modelled. Pension splitting, survivor benefits, the RRIF spousal rollover, OAS
ceasing at first death, and two mortality timelines all need two people.

**No invented parameters.** No tax, benefit, or account constant is ever a
literal in a ``.py`` file under this package. Every one is loaded from
``params/`` through ``engine.params.loader``, which raises
``MissingParameterError`` rather than substituting a default. This covers
calendar rules too: the filing month, the month a benefit year starts, and the
months a benefit is indexed in are statutory and are loaded, not typed in.
"""

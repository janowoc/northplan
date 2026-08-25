"""Government benefits: CPP, OAS, GIS.

Same shape as ``engine/tax/``: array-valued pure functions over ``(n_paths,)``,
every constant loaded from ``params/``.

Three conventions in this package are easy to get wrong and are stated once
here.

**Monthly is the native unit.** CPP, OAS, and GIS are published as monthly
amounts, paid as monthly amounts, and now simulated in monthly steps, so the
functions here return *monthly* amounts. A function that returns an annual
figure says so in its name, ends in ``_annual``, and exists only for golden
tests against published annual totals. The conversion happens once, at that
boundary, never twice. On the previous annual timestep the default ran the
other way; anything that still multiplies a benefit by twelve inside the step
is now wrong by a factor of twelve.

**Benefit year is not calendar year, and neither is the income year.** The OAS
recovery tax and GIS are assessed over a benefit period running from a
statutory month of year Y to the month before it in year Y+1, against net
income from an *earlier* calendar year. A function that takes a year documents
which year it means, and the mapping between them is
``engine.core.timeline.benefit_year_income_year`` — not something a caller
works out for itself.

**Indexation costs a constant, and the constant is not one.** OAS and GIS are
adjusted quarterly; CPP is adjusted annually. Between adjustments the amount is
fixed in nominal terms, and each adjustment is computed from a CPI window that
closed some months earlier, so an indexed benefit sits permanently a little
below its published real value. The functions here return the published
amount; ``engine.core.indexation.real_factor`` gives the constant it is
multiplied by, and the step applies it.

That factor does not vary by month — it is computed once per benefit per
scenario. The within-cycle oscillation is not modelled, on purpose: it is
bounded, it averages to nothing, and it is smaller than the error in the
inflation assumption. The level is what is kept, because the level is permanent
and always flatters the plan.

Two applications of that factor, or none, are both bugs, and only one of them
is visible.
"""

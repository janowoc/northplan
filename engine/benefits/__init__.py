# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

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

**The OAS repayment is assessed on the calendar year.** It is a line on the
return: computed once at the December close on that year's net income,
including the OAS received in the same year, and capped at it. OAS is paid
gross monthly. The benefit-year block in ``params/<year>/oas.yaml`` —
``benefit_year.start_month`` and ``benefit_year.income_year_offset`` — is read
by no module here; it belongs to a future GIS implementation, which is assessed
over a benefit period rather than a calendar year.

**Indexation costs a constant, and the constant is not one.** OAS and GIS are
adjusted quarterly; CPP is adjusted annually. Between adjustments the amount is
fixed in nominal terms, so across the cycle an indexed benefit averages a
little below its published real value. The functions here return the published
amount; ``engine.core.indexation.erosion_factor`` gives the constant it is
multiplied by, and the step applies it.

That factor does not vary by month — it is computed once per schedule per
scenario. The within-cycle oscillation is not modelled, on purpose: it is
bounded, it averages to nothing, and it is smaller than the error in the
inflation assumption. Neither is the CPI lag (L5). The level is what is kept,
because the level is permanent and always flatters the plan.

Two applications of that factor, or none, are both bugs, and only one of them
is visible.
"""

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Government benefits and employment income: CPP, OAS, GIS, DB pensions, employment.

Same shape as ``engine/tax/``: array-valued pure functions over ``(n_paths,)``,
every constant loaded from ``params/``.

- ``cpp.py``: the CPP retirement pension, its start-age adjustment, and the
  survivor pension.
- ``oas.py``: the OAS pension, its age-band step-up and deferral. The
  repayment is assessed elsewhere, in ``engine.tax.combined``.
- ``gis.py``: not a GIS calculation — an indicator of the fraction of
  path-years in which testable income sits in the band an unmodelled GIS
  would matter for.
- ``pension.py``: defined-benefit pensions, a scenario input rather than a
  parameter, including their bridge. The survivor share is applied by the
  step at first death (L41), not by this module.
- ``employment.py``: employment income and the CPP and EI contributions it
  generates.

**Monthly is the native unit.** These are published or stated as monthly
amounts, paid monthly, and simulated in monthly steps. A function returning
an annual figure says so, ends in ``_annual``; apart from
``gis.band_threshold_annual``, which serves the GIS indicator, these exist
only for golden tests against published annual totals.

**The OAS repayment is assessed on the calendar year**, at the December close,
on that year's net income before the repayment (line 23400), including the OAS
received in it, capped at it. OAS is paid gross monthly. ``benefit_year`` in
``params/<year>/oas.yaml`` is read by no module here; it belongs to a future
GIS implementation, assessed over a benefit period rather than a calendar
year.

**Indexation is already applied by the time a dollar reaches here.** Every
dollar arrives through :class:`~engine.core.indexation.RealParamSet`, which
has already applied its schedule's erosion factor; nothing here multiplies by
it again. An amount already in pay — CPP, OAS, or a DB pension's
``monthly_amount`` — comes from the scenario, not from ``params/``, and
carries no erosion factor at all (L52). It is held constant, except that OAS
in pay steps up by the band ratio as current age crosses a threshold. A DB
pension's amount decays under ``engine.core.indexation.unindexed_factor``,
applied by hand in ``pension.py``, only when it is not indexed.
"""

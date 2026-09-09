# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Array-valued pure functions for Canadian income tax.

Every function here takes and returns NumPy arrays of shape ``(n_paths,)`` and
is pure: same inputs, same outputs, no state. Bracket logic uses ``np.clip``
and ``np.where`` rather than Python branching, so that a scalar input works by
broadcasting and a golden-number test exercises the identical code path as a
100,000-path Monte Carlo run.

**These functions are annual, and that has not changed.** The simulation steps
monthly, but income tax is assessed on a calendar year, so everything in this
package takes a full year's figures and is called once per simulated year, from
the year-end close in ``engine/core/step.py``. Nothing here is called twelve
times a year and nothing here takes a month.

Three consequences worth stating, because the monthly loop around this package
makes each of them possible:

- The income these functions receive is the year-to-date total accumulated over
  twelve monthly steps, from an ``IncomeLedger``. Passing a single month's
  income to a progressive bracket function produces a number roughly a twelfth
  the size at a much lower marginal rate — plausible, and wrong.
- Tax assessed is not tax paid. What these functions return is an assessment;
  the cash leaves in the following year's filing month, less whatever was
  withheld along the way. That timing lives in the step, not here.
- Brackets and credits are indexed once a year, in January. A tax year has one
  set of them, which is why a ``ParamYear`` is the right granularity. This
  package is **not** exempt from the indexation decay: a bracket edge fixed in
  nominal terms for twelve months averages below its January real value exactly
  as a benefit does, and the annual schedule is the widest cycle there is, so
  the erosion is larger here than for anything paid quarterly. The amounts
  arrive already deflated, through ``engine.core.indexation.RealParamSet``.

All dollar amounts in and out are real dollars (see ``engine/__init__.py``).
All parameters come from ``params/`` via ``engine.params.loader``; there is
never a numeric tax constant in this package.
"""

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Array-valued pure functions for Canadian income tax.

Every function here takes and returns NumPy arrays of shape ``(n_paths,)`` and
is pure: same inputs, same outputs, no state. Bracket logic uses ``np.clip``
and ``np.where`` rather than Python branching, so a scalar input works by
broadcasting and a golden-number test exercises the same code path as a
100,000-path Monte Carlo run.

The assessment is annual: ``combined`` is to be called once per simulated
year, from the year-end close in ``engine/core/step.py``, on income
accumulated over that year's twelve monthly steps. What it returns is an
assessment, not a cash flow: the cash leaves in the following year's filing
month, less withholding. Every income figure that reaches ``federal`` or
``provincial`` is annual rather than monthly; a monthly caller scales to a
year first, and what it hands over may be an annualised approximation rather
than the calendar year's income (L16).

``withholding`` is the exception, and is monthly on both counts.
``registered_withholding`` takes one withdrawal and the month index of that
withdrawal; its band is measured against that single withdrawal and never
against a year's, so a year's total handed to it takes the top band on the
whole amount where several small withdrawals each take the lowest.
``payroll_withholding_monthly`` takes one month's amount and is called every
month. What they return is a prepayment, reconciled against the assessment at
the December close.

Everything in the package reaches a dollar the same way: brackets, credits,
and bands arrive already indexed and deflated to real dollars, through
``engine.core.indexation.RealParamSet``; there is never a numeric tax
constant here.
"""

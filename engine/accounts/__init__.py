"""Registered and taxable account mechanics.

Each module models one account type: contribution room, growth, withdrawal, and
the tax character of what comes out. Balances are real dollars, shape
``(n_paths,)``.

Ordering is decided once, in ``engine/core/step.py``, and every account obeys
it. Three consequences of a monthly timestep that every module here has to
respect:

- **Growth is one month.** ``base.grow`` applies a monthly real return.
  Applying an annual return in a monthly step overstates growth twelvefold and
  produces numbers that still look like money.
- **Annual limits are fixed in January and drawn down over the year.** The RRIF
  minimum, the LIF maximum, and every contribution room figure are annual
  quantities. They are established by ``open_year`` from the balance on
  1 January — before that year's growth — and what remains of them is tracked
  across the months. An account module must never recompute an annual limit
  mid-year from a mid-year balance.
- **A "per year" cap is not a "per month" cap.** A withdrawal function that
  clamps each month against the annual maximum permits twelve times the
  maximum. Every annual bound is checked against the year-to-date total, not
  against this month's amount.
"""

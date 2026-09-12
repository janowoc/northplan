# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Array-valued pure functions for Canadian income tax.

Every function here takes and returns NumPy arrays of shape ``(n_paths,)`` and
is pure: same inputs, same outputs, no state. Bracket logic uses ``np.clip``
and ``np.where`` rather than Python branching, so a scalar input works by
broadcasting and a golden-number test exercises the same code path as a
100,000-path Monte Carlo run.

Everything here is annual: each function takes a full year's figures and is
called once per simulated year, from the year-end close in
``engine/core/step.py`` — never once a month. The income received must be the
year-to-date total from an ``IncomeLedger``, not a single month's income.
What these functions return is an assessment, not a cash flow: the cash
leaves in the following year's filing month, less withholding. Brackets and
credits arrive already indexed and deflated to real dollars, through
``engine.core.indexation.RealParamSet``; there is never a numeric tax
constant in this package.
"""

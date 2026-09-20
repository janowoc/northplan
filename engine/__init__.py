# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""northplan simulation engine.

Pure Python. This package must never import ``api``, ``fastapi``, or ``cli``;
``tests/test_layering.py`` enforces that by walking the AST of every module
here.

Invariants that hold everywhere below this package:

- The timestep is one month; ``(year, month)`` with ``month`` in ``1..12``.
- There is exactly one loop over time, in ``engine.mc.simulate``. Annual events
  (tax assessment, RRIF/LIF minimums, contribution room) are phases on that
  monthly timeline, not a second loop.
- Every dollar amount crossing a function boundary is real (constant
  purchasing power) dollars in the scenario's base year, except the opening
  ``PersonState.prior_year_net_income`` and
  ``PersonState.net_income_two_years_prior``, taken as filed until the
  December closes that replace them with real figures.
- Indexed amounts are constant in real terms except for a per-schedule
  erosion factor (``engine.core.indexation.erosion_factor``) covering the
  within-cycle erosion between adjustment dates; the CPI lag is separate,
  not modelled, and biases every indexed amount upward (L5). Non-indexed
  amounts must decay explicitly.
- A dollar amount reaches the engine through ``RealParamSet``, never
  ``ParamSet``.
- Conversion to nominal happens exactly once, at display, outside ``engine/``.
- The simulation is vectorized across paths, not time: every function in
  ``tax/`` and ``benefits/`` takes and returns arrays of shape ``(n_paths,)``.
- The return matrix is drawn once from a fixed seed and reused across every
  policy the optimizer evaluates (common random numbers).
- A policy function reads only what is knowable at that simulated
  month — never a future return, balance, or this year's total income.
- Household is a list of persons, always, even for one person.
- No tax, benefit, or account constant is ever a literal in a ``.py`` file
  under this package; every one loads from ``params/`` via
  ``engine.params.loader``, which raises ``MissingParameterError`` rather
  than substituting a default.
"""

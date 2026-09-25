# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The deterministic monthly step and the state it passes between months.

``step.advance_month(state, month_returns, policy, market, real_params)``
returns the next month's state. It is the body of the one simulation loop,
which lives in ``engine.mc.simulate`` and runs every path to the second death
(L10); the optimizer (``engine/optimize/``) drives Monte Carlo across
policies. Neither the loop nor the optimizer reimplements the month.

Annual events — the tax assessment, contribution room, the RRIF minimum, the
balance owing coming due in the filing month — happen as phases invoked by
``advance_month`` in the months that call for them. They are not a second loop
and must not become one.

``timeline`` owns month and age arithmetic. ``indexation`` owns the real-terms
decay that periodic indexation produces once the timestep is shorter than the
indexation period, and the real-terms parameter view that is the only route
from a dollar in ``params/`` to the engine. ``mortality`` owns the survival
curve and the per-path death draw. ``state`` is what passes between months.

All amounts are real dollars, except the opening
``PersonState.prior_year_net_income`` and
``PersonState.net_income_two_years_prior``, which are the scenario's figures
as filed. All arrays are shape ``(n_paths,)``.
"""

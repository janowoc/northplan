# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Monte Carlo: return generation and the loop over months.

The return matrix and mortality draws are generated once, from a fixed seed,
and reused unchanged across every policy the optimizer evaluates (common
random numbers).

Draws are monthly on their first axis. Scenario assumptions arrive annual
and are converted to monthly exactly once, in
``moments.monthly_log_moments``, which ``returns.generate`` calls.
Downstream code receives monthly figures and must never convert again.

Results are recorded annually even though the loop is monthly — see
``SimulationResult`` for why.
"""

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Monte Carlo: return generation and the loop over months.

**Common random numbers.** The return matrix and the mortality draws are
generated once, from a fixed seed, and reused unchanged across every policy the
optimizer evaluates. Two policies must be compared on the same futures. Drawing
fresh numbers per policy makes the difference between two policies mostly
sampling noise, and the optimizer will happily maximise that noise.

Vectorization is across paths. The loop here is over months, because months are
sequentially dependent; within a month, all paths are computed at once.

Draws are monthly on their first axis. Scenario assumptions arrive annual and
are converted to monthly exactly once, in ``returns.generate``. Downstream code
receives monthly figures and must never convert again.

Results are recorded annually even though the loop is monthly — see
``SimulationResult`` for why.
"""

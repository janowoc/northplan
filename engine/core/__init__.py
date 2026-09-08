# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The deterministic monthly step: ``(state, month, policy) -> state``.

There is exactly one simulation loop in this repository and it lives here. It
is a loop over *months*. Monte Carlo (``engine/mc/``) drives this step across
the horizon for all paths at once; the optimizer (``engine/optimize/``) drives
Monte Carlo across policies. Neither reimplements the month.

Annual events — the tax assessment, contribution room, the RRIF minimum, the
balance owing coming due in the filing month — happen as phases invoked by
``advance_month`` in the months that call for them. They are not a second loop
and must not become one.

``timeline`` owns month and age arithmetic. ``indexation`` owns the real-terms
decay that periodic indexation produces once the timestep is shorter than the
indexation period, and the real-terms parameter view that is the only route
from a dollar in ``params/`` to the engine. ``state`` is what passes between
months.

All amounts are real dollars. All arrays are shape ``(n_paths,)``.
"""

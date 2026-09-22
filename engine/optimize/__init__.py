# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Search over policy parameters.

The outermost of the three layers: the optimizer proposes a policy, Monte
Carlo evaluates it across paths, and the monthly step runs each month. The
optimizer searches *policy parameters*, never paths and never per-path
decisions.

Every candidate policy is evaluated against the same
:class:`~engine.mc.returns.RandomDraws` (common random numbers); draws are
never regenerated per candidate. ``search`` evaluates an already-expanded
list of candidates exhaustively; any cleverer method must be validated
against it.
"""

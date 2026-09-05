# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Parameterized decision rules — what the optimizer searches over.

A policy is a small set of numbers plus the rules that turn state into
decisions. The optimizer searches the numbers; the rules are fixed code. A
policy is asked to decide once a *month*.

**The constraint that makes any of this meaningful:** a policy function may
read only what is knowable at that simulated moment — current balances, current
ages, income recognised so far this year, realized history, and the current
year's parameters. It may never read a future return, a future balance, a
terminal value, or any array sliced past the current month index.

A policy that peeks produces a plan nobody can follow, and it will look
excellent. That failure mode is silent, which is why the verifier checks for it
specifically. The monthly timestep gives it a new disguise: "this year's
income" is knowable in December and unknowable in January, and a policy that
uses the year's total in any earlier month is clairvoyant even though it never
touches a future array. Year to date is the only income figure a policy may
read.

**The other monthly hazard is scale.** Contribution room, withdrawal ceilings,
grant maxima, and LIF maxima are annual quantities. Applied per month without
tracking what the year has already used, each of them permits twelve times what
it should, and the run that results is internally consistent and wrong.
"""

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Parameterized decision rules — what the optimizer searches over.

A policy is a small set of numbers plus the rules that turn state into
decisions. The optimizer searches the numbers; the rules are fixed code. A
policy is asked to decide once a month.

A policy function may read only what is knowable at that simulated
moment — current balances, ages, realized history, and this year's
parameters — never a future return or balance, and never this year's total
income before December: year to date is the only income figure it may read.

Annual quantities — contribution room, withdrawal ceilings, grant maxima,
LIF maxima — must be checked against the year-to-date total, not applied
per month against the annual figure.
"""

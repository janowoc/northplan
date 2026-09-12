# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registered and taxable account mechanics.

Each module models one account type: contribution room, growth, withdrawal, and
the tax character of what comes out. Balances are real dollars, shape
``(n_paths,)``.

Annual quantities — the RRIF minimum, the LIF maximum, contribution room — are
fixed in January by ``open_year`` from the balance on 1 January, before that
year's growth, and drawn down against the year-to-date total as the months
pass. An account module must never recompute one mid-year from a mid-year
balance.
"""

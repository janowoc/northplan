"""Registered and taxable account mechanics.

Each module models one account type: contribution room, growth, withdrawal, and
the tax character of what comes out. Balances are real dollars, shape
``(n_paths,)``.

Ordering within a year is decided once, in ``engine/core/step.py``, and every
account obeys it. In particular the RRIF minimum is computed from the balance
at the *start* of the year, before that year's growth is applied — an account
module must never apply growth itself and then report a minimum.
"""

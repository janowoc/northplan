# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The golden test harness.

This directory is a package — rather than a loose collection of test modules —
so that a case file can name a target function by a dotted path that resolves
inside the test tree itself. ``test_harness.py`` defines its own tiny,
obviously synthetic functions and exercises the harness against them by
importing them the same way a real case does: ``golden.test_harness.<name>``.
Without an ``__init__.py`` here, pytest would import that module as a bare
top-level ``test_harness`` with no stable dotted path for a case file to name.
"""

from __future__ import annotations

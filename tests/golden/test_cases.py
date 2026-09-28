# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Discover and run every human-written case under ``tests/golden/cases/``.

All the interesting logic — parsing a case file, validating that ``source``
and ``checked`` are present, resolving ``params``, ``real_params``, and
``real_params_year`` inputs, comparing a result — lives in ``conftest.py``
and is exercised directly by ``test_harness.py``. This module's only job is
to point that machinery at the real cases directory and turn what it finds
into one parametrized test per case.

``pyproject.toml`` marks an empty ``parametrize`` list as a collection failure
(``empty_parameter_set_mark = "fail_at_collect"``), because a suite whose
cases come from files on disk should not go quietly green when it finds none.
That guard is right for a directory that is supposed to hold cases and
doesn't — see ``conftest.discover_cases`` for the file-present-but-empty
failure it exists to catch. It is wrong for the bootstrap state, where
``cases/`` legitimately holds nothing but ``.gitkeep`` and a green, empty
collection is exactly what issue #3 asks for. The parametrized test below is
therefore only *defined* when discovery finds at least one case; with none,
this module defines no golden test at all, and there is no empty
``parametrize`` for the guard to trip over. Do not "fix" this back into a bare
``@pytest.mark.parametrize`` — that reintroduces the collection failure
whenever ``cases/`` holds no case.

This exemption is specific to this file, because an empty ``cases/`` is a
state this suite allows deliberately and the others do not. The disk-driven
parametrizations in ``tests/test_layering.py``,
``tests/params/test_param_provenance.py``, and
``tests/params/test_param_file_structure.py`` all read from files that are
expected to exist and be non-empty by the time the suite runs; they have no
equivalent bootstrap state, so they must keep failing at collection if they
ever go empty. Do not generalize the ``if CASES:`` idiom to those.
"""

from __future__ import annotations

import pytest

from .conftest import CASES_DIR, GoldenCase, discover_cases, run_case

CASES = discover_cases(CASES_DIR)

if CASES:

    @pytest.mark.golden
    @pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
    def test_golden_case(case: GoldenCase) -> None:
        run_case(case)

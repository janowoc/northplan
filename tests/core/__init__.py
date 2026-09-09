# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``engine.core`` tests: a package, not a loose collection of modules.

``test_state.py`` and ``test_build.py`` both need the shape/dtype/
immutability walker ``conftest.py`` defines, and reach it with a *relative*
import: ``from .conftest import walk``. Never ``from tests.core.conftest
import walk`` — that absolute form is reachable only because ``pythonpath =
["."]`` puts the repository root, and therefore the ``tests`` namespace
package, on the path, and it imports ``conftest.py`` a second time under a
second name. Two module objects for one file is exactly the failure this
``__init__.py`` exists to prevent — it gives pytest's own rootless import of
every file in this directory a stable package to belong to, ``core``, so
that a relative import inside the package and pytest's own import resolve to
the same module object rather than two. See ``tests/golden/__init__.py`` for
the same fix for the same reason, predating this one.
"""

from __future__ import annotations

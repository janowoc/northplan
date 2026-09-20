# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``engine.accounts`` tests: a package, not a loose collection of modules.

Gives pytest's rootless import of every file in this directory a stable
package to belong to, ``accounts``, for the same reason
``tests/core/__init__.py`` exists — see that file's docstring for the failure
this avoids.
"""

from __future__ import annotations

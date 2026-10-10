# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""HTTP layer. Thin by construction.

Translates a request into engine calls and the engine's results back to JSON, and
serves ``api/web/`` as static files. No financial logic lives here, and no table
logic either: the tables, and the conversion to nominal dollars when the client
asks for it, come from :mod:`report.tables`, the same code the command line writes
its files with.
"""

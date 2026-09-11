# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""HTTP layer. Thin by construction.

Translates JSON to engine calls and engine results back to JSON, and serves
``web/`` as static files. No financial logic lives here.

This is also where real dollars become nominal, if the client asks for nominal:
the engine never converts a figure to nominal, so the conversion happens here,
once, on the way out.
"""

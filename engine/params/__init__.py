# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Loading of hand-verified tax and benefit parameters from ``params/``.

This package is the only sanctioned route from a YAML file to a number used by
the engine. It exists to make the "never invent a parameter" rule enforceable
rather than aspirational: a lookup either finds a value a human wrote down, or
raises. There is no third outcome and no default.
"""

from engine.params.loader import (
    MalformedParamFileError,
    MissingParameterError,
    ParamError,
    ParamFileMissingError,
    ParamSet,
    ParamYear,
    ParamYearMissingError,
    load_year,
)

__all__ = [
    "MalformedParamFileError",
    "MissingParameterError",
    "ParamError",
    "ParamFileMissingError",
    "ParamSet",
    "ParamYear",
    "ParamYearMissingError",
    "load_year",
]

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``northplan`` command line interface.

Reads a scenario YAML file, runs it, writes results as CSV (row per year) or
JSON. Real-to-nominal conversion, if requested, happens here — the engine never
returns nominal figures.

The engine steps monthly and reports per year, so a row here is a year: net
worth at 31 December, the year's total spending, and the tax assessed on that
year rather than the cash paid during it. A scenario may open in any month; if
it does, its first row covers a short tax year, and the export says so rather
than pretending it is a full one.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        A parser with the ``simulate`` and ``optimize`` subcommands.
    """
    raise NotImplementedError


def run_scenario(scenario_path: Path, output_path: Path, nominal: bool) -> int:
    """Run one scenario file and write its results.

    Args:
        scenario_path: Scenario YAML. May be a ``*.local.yaml`` file, which is
            gitignored.
        output_path: Destination. ``.csv`` writes a row per year, aggregated
            from the twelve simulated months; ``.json`` writes the API response
            shape.
        nominal: Convert real dollars to nominal on the way out. The conversion
            is to the year end each row reports, not to the middle of the year
            — the choice matters once amounts are accrued monthly, so it is
            stated rather than left to whichever line is written first.

    Returns:
        Process exit code.
    """
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())

"""``northplan`` command line interface.

Reads a scenario YAML file, runs it, writes results as CSV (row per year) or
JSON. Real-to-nominal conversion, if requested, happens here — the engine never
returns nominal figures.
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
        output_path: Destination. ``.csv`` writes a row per year; ``.json``
            writes the API response shape.
        nominal: Convert real dollars to nominal on the way out.

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

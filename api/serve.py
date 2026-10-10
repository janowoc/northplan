# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``northplan-serve`` command: serve the API and the web page.

Runs :data:`api.main.app` with uvicorn on 127.0.0.1 only. There is no host option,
because the server has no authentication. ``--params`` and ``--example`` set
``api.main.PARAMS_ROOT`` and ``api.main.EXAMPLE_PATH``; a flag not given leaves the
module's own default. The parameters directory is checked before anything starts.

Exit codes: 0 after the server stops on Ctrl-C; 1 on a parameters directory that does
not exist, is not a directory, or is refused by
:func:`~engine.params.loader.check_params_root`; 2 on a usage error, with the message on
stderr; 3, from uvicorn, when the server cannot start, for instance because the port is
in use. On SIGTERM the process ends by the signal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

import api.main
from engine.params.loader import (
    DEFAULT_PARAMS_ROOT,
    ParamDirectoryUnreadableError,
    check_params_root,
)


def _port(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError(f"{text!r} must be between 1 and 65535")
    return value


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 after the server stops on Ctrl-C; on SIGTERM the process ends by the signal
        instead. 1, after a message on stderr and with nothing started, when the
        parameters directory (``--params``, else the engine's default) does not exist, is
        not a directory, or is refused by :func:`~engine.params.loader.check_params_root`;
        the message adds a line saying to pass ``--params`` only when the directory does
        not exist (or is not a directory) and the default was used.

    Raises:
        SystemExit: With code 2, on a usage error (argparse's own, a port outside 1 to
            65535, or an ``--example`` that is not a file). With code 3, from uvicorn,
            when the server cannot start, for instance because the port is in use.
    """
    parser = argparse.ArgumentParser(
        prog="northplan-serve",
        description="Serve the northplan API and web page on 127.0.0.1.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        metavar="DIR",
        help="parameters directory, holding one subdirectory per tax year "
        "(default: params/ beside the source tree)",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=8000,
        metavar="N",
        help="port on 127.0.0.1 (default: 8000)",
    )
    parser.add_argument(
        "--example",
        type=Path,
        default=None,
        metavar="FILE",
        help="scenario file served at /api/example "
        "(default: scenarios/example.yaml beside the source tree)",
    )
    args = parser.parse_args(argv)

    root = args.params if args.params is not None else DEFAULT_PARAMS_ROOT
    try:
        check_params_root(root)
    except ParamDirectoryUnreadableError as error:
        print(f"northplan-serve: error: {error}", file=sys.stderr)
        return 1
    if not Path(root).is_dir():
        print(
            f"northplan-serve: error: the parameters directory {root} does not exist "
            "or is not a directory.",
            file=sys.stderr,
        )
        if args.params is None:
            print(
                "northplan-serve: pass --params DIR, the directory holding one "
                "subdirectory per tax year.",
                file=sys.stderr,
            )
        return 1
    if args.example is not None and not args.example.is_file():
        parser.error(f"--example {str(args.example)!r} is not a file.")

    if args.params is not None:
        api.main.PARAMS_ROOT = args.params
    if args.example is not None:
        api.main.EXAMPLE_PATH = args.example
    uvicorn.run(api.main.app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

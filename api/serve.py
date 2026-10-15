# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``northplan-serve`` command: serve the API and the web page.

Runs :data:`api.main.app` with uvicorn on ``--host`` (default 127.0.0.1) with proxy
headers off, so the peer :mod:`api.main` checks is the socket's. ``NORTHPLAN_TOKEN``,
when set, becomes ``api.main.TOKEN`` and must be at least 16 printable ASCII characters
other than a space; when it is unset and ``--host`` is not a loopback address or
``localhost``, a token is generated and written once to stderr, after every check has
passed. ``--params`` and ``--example`` set ``api.main.PARAMS_ROOT`` and
``api.main.EXAMPLE_PATH``; a flag not given leaves the module's own default. The
parameters directory is checked before anything starts.

Exit codes: 0 after the server stops on Ctrl-C; 1 on a parameters directory that does
not exist, is not a directory, or is refused by
:func:`~engine.params.loader.check_params_root`; 2 on a usage error, with the message on
stderr, including a ``NORTHPLAN_TOKEN`` that breaks the rules, checked right after the
command line is parsed, before the parameters directory; 3, from uvicorn, when the
server cannot start, for instance because the port is in use. On SIGTERM the process
ends by the signal.
"""

from __future__ import annotations

import argparse
import os
import secrets
import stat
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


def _check_example(parser: argparse.ArgumentParser, path: Path) -> None:
    """Return if ``path`` is a regular file that can be opened for reading; else exit 2.

    Exits through ``parser.error``: "is not a file" when the lookup of ``path`` fails with
    ENOENT or ENOTDIR or finds something other than a regular file; "cannot be read" when
    the lookup fails with any other ``OSError`` or the open does, with the operating
    system's reason, or the exception's own text when it carries none. Reads nothing from
    the file.
    """
    try:
        mode = path.stat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        parser.error(f"--example {str(path)!r} is not a file.")
    except OSError as exc:
        parser.error(f"--example {str(path)!r} cannot be read: {exc.strerror or exc}.")
    if not stat.S_ISREG(mode):
        parser.error(f"--example {str(path)!r} is not a file.")
    try:
        with path.open("rb"):
            pass
    except OSError as exc:
        parser.error(f"--example {str(path)!r} cannot be read: {exc.strerror or exc}.")


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
            65535, a ``NORTHPLAN_TOKEN`` that breaks the rules, checked right after
            the command line is parsed, before the parameters directory, or an
            ``--example`` that is not a regular file or cannot be read). With code 3,
            from uvicorn, when the server cannot start, for instance because the port
            is in use.
    """
    parser = argparse.ArgumentParser(
        prog="northplan-serve",
        description="Serve the northplan API and web page.",
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
        help="port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        metavar="ADDRESS",
        help="address to listen on (default: 127.0.0.1); on any other than a loopback "
        "address, a client on another machine must give the token",
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

    token = os.environ.get("NORTHPLAN_TOKEN")
    if token is not None and not (len(token) >= 16 and all("!" <= c <= "~" for c in token)):
        parser.exit(
            2,
            "northplan-serve: error: NORTHPLAN_TOKEN must be at least 16 characters, each a "
            "printable ASCII character other than a space. To make one: python3 -c "
            "'import secrets; print(secrets.token_urlsafe(32))'\n",
        )

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
    if args.example is not None:
        _check_example(parser, args.example)

    if args.params is not None:
        api.main.PARAMS_ROOT = args.params
    if args.example is not None:
        api.main.EXAMPLE_PATH = args.example
    if token is None and not (
        args.host.lower() == "localhost" or api.main.is_loopback_address(args.host)
    ):
        token = secrets.token_urlsafe(32)
        print(
            f"northplan-serve: NORTHPLAN_TOKEN is not set, so this run's token is {token}; "
            "it is new on every start.",
            file=sys.stderr,
        )
    api.main.TOKEN = token
    uvicorn.run(api.main.app, host=args.host, port=args.port, proxy_headers=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

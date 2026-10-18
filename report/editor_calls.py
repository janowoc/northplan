# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The scenario editor's calls as JSON text, answered alike for the page's worker and the API.

A call has a name in ``CALLS`` and takes its arguments as the text of a JSON object; it is
answered with an HTTP status and the text of a JSON body. The body is the call's result for
200, and ``{"detail": str}`` otherwise, with ``"references"``, the places that block a refused
edit, added for 409. ``browser/northplan_browser.py`` and ``POST /api/editor/{call}`` pass both
through unchanged.

Invariants:
    * The arguments are read with the pure-Python JSON scanner, which raises RecursionError on
      a nesting too deep to read; under Pyodide the C scanner overflows the WebAssembly stack
      instead.
    * The arguments hold no NaN, no Infinity, no number written with a fraction or an exponent
      that is too large for a float and no object that gives a key twice; the body is ASCII and
      holds no NaN or infinity either.
"""

from __future__ import annotations

import json
import json.scanner
import math
from pathlib import Path
from typing import Any, Final

from report import editor

__all__ = ["CALLS", "answer"]

CALLS: Final[dict[str, tuple[str, ...]]] = {
    "parse": ("text",),
    "check": ("mapping",),
    "schema": (),
    "references": ("mapping", "kind", "name"),
    "rename": ("mapping", "kind", "old", "new"),
    "remove": ("mapping", "kind", "name"),
    "add-asset-class": ("mapping", "name"),
    "move-asset-class": ("mapping", "name", "index"),
    "grid-paths": ("mapping",),
}

_KINDS: Final = ("person", "asset-class")


def _run(call: str, a: dict[str, Any], root: Path) -> Any:
    """The result of the :mod:`report.editor` function ``call`` names, on the arguments ``a``."""
    match call:
        case "parse":
            return editor.parse(a["text"], params_root=root)
        case "check":
            return editor.check(a["mapping"], params_root=root)
        case "schema":
            return editor.schema()
        case "references":
            return editor.references(a["mapping"], a["kind"], a["name"])
        case "rename":
            return editor.rename(a["mapping"], a["kind"], a["old"], a["new"])
        case "remove":
            return editor.remove(a["mapping"], a["kind"], a["name"])
        case "add-asset-class":
            return editor.add_asset_class(a["mapping"], a["name"])
        case "move-asset-class":
            return editor.move_asset_class(a["mapping"], a["name"], a["index"])
        case "grid-paths":
            return editor.grid_paths(a["mapping"])
        case _:
            raise AssertionError(call)


def _constant(name: str) -> Any:
    raise ValueError(f"{name} is not a number JSON allows")


def _finite(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text} is too large for a float")
    return value


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for key, value in pairs:
        if key in found:
            raise ValueError(f"the key {key!r} is given twice in one object")
        found[key] = value
    return found


def _loads(text: str) -> Any:
    decoder = json.JSONDecoder(
        parse_float=_finite, parse_constant=_constant, object_pairs_hook=_pairs
    )
    decoder.scan_once = json.scanner.py_make_scanner(decoder)
    return decoder.decode(text)


def _detail(detail: str) -> str:
    return json.dumps({"detail": detail})


def _type_problem(name: str, value: Any) -> str | None:
    if name == "mapping":
        return None
    if name == "kind":
        if value in _KINDS:
            return None
        return f"kind must be 'person' or 'asset-class', not {value!r}."
    if name == "index":
        return None if type(value) is int else f"index must be an integer, not {value!r}."
    return None if type(value) is str else f"{name} must be a string, not {value!r}."


def answer(call: str, arguments: str, *, params_root: Path) -> tuple[int, str]:
    """Answer the editor call ``call`` on ``arguments``, the text of a JSON object.

    Args:
        call: A name in ``CALLS``.
        arguments: The call's arguments by name, each required and no other allowed: ``text``,
            ``name``, ``old`` and ``new`` are strings, ``kind`` is ``"person"`` or
            ``"asset-class"``, ``index`` is an integer and ``mapping`` is any JSON value.
        params_root: The directory ``parse`` and ``check`` load parameters from.

    Returns:
        ``(status, body)``: 200 and the result of the :mod:`report.editor` function of that
        name; 404 for an unknown call, before the arguments are read; 422 when the arguments
        cannot be read, are not an object, name an unknown argument, lack one or give one the
        wrong type, checked in that order and each argument in the call's order, or when the
        mapping is nested too deeply to work on; 409, with ``references``, for an
        :class:`~report.editor.EditRefusedError`.

    Raises:
        RoutedParameterError: An engine bug, passed through, as is any exception the call
            raises other than EditRefusedError and RecursionError.
        ValueError: A result holds a NaN or an infinity, which the body cannot carry.
    """
    if call not in CALLS:
        return 404, _detail(f"no editor call named {call!r}; the calls are {sorted(CALLS)!r}.")
    names = CALLS[call]
    try:
        value = _loads(arguments)
    except RecursionError:
        return 422, _detail("the arguments are nested too deeply to read.")
    except ValueError as error:
        return 422, _detail(f"the arguments are not valid JSON: {error}.")
    if not isinstance(value, dict):
        return 422, _detail(f"the arguments must be a JSON object, not {type(value).__name__}.")
    unknown = sorted(set(value) - set(names))
    if unknown:
        first = unknown[0]
        return 422, _detail(f"unknown argument {first!r}; {call} takes {list(names)!r}.")
    for name in names:
        if name not in value:
            return 422, _detail(f"{call} needs the argument {name!r}.")
    for name in names:
        problem = _type_problem(name, value[name])
        if problem is not None:
            return 422, _detail(problem)
    try:
        result = _run(call, value, params_root)
    except editor.EditRefusedError as error:
        return 409, json.dumps({"detail": str(error), "references": error.references})
    except RecursionError:
        return 422, _detail("the mapping is nested too deeply to work on.")
    return 200, json.dumps(result, allow_nan=False)

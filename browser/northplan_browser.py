# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The Python entry points of the browser page: run a scenario or answer an editor call, each as
one JSON string.

The page's worker calls :func:`simulate` under Pyodide; the archive ships this file at its
top level. It owns the options check, the path cap and the mapping of every outcome to a
JSON string. :func:`editor_call` wraps :func:`report.editor_calls.answer` in an envelope that
carries the status. The scenario's syntax is always YAML, as the textarea holds it.
"""

from __future__ import annotations

import json
from typing import Final

from engine.params.loader import DEFAULT_PARAMS_ROOT
from report.editor_calls import answer
from report.run import PATHS_WITH_DETERMINISTIC, RunRefusedError, simulate_document

PATH_CAP: Final = 25_000
OUT_OF_MEMORY: Final[str] = (
    "the run ran out of the browser's memory: WebAssembly gives a page at most 4 GB. "
    "Lower n_paths, or use the local install."
)

_OPTION_KEYS = frozenset({"policy", "paths", "deterministic", "nominal"})


def _checked_options(options: str) -> tuple[str | None, int | None, bool, bool]:
    """The (policy, paths, deterministic, nominal) of the JSON object ``options``.

    Raises:
        ValueError: ``options`` is not a JSON object, names a key other than ``policy``,
            ``paths``, ``deterministic`` and ``nominal``, or gives one a value of the wrong
            type; ``paths`` must be an int of at least 1 (a bool is not one).
    """
    parsed = json.loads(options)
    if not isinstance(parsed, dict):
        raise ValueError(f"options must be a JSON object, not {type(parsed).__name__}")
    unknown = sorted(set(parsed) - _OPTION_KEYS)
    if unknown:
        raise ValueError(f"unknown option {unknown[0]!r}; the options are {sorted(_OPTION_KEYS)!r}")
    policy = parsed.get("policy")
    paths = parsed.get("paths")
    deterministic = parsed.get("deterministic", False)
    nominal = parsed.get("nominal", False)
    if policy is not None and type(policy) is not str:
        raise ValueError(f"policy must be a string or null, not {policy!r}")
    if paths is not None and (type(paths) is not int or paths < 1):
        raise ValueError(f"paths must be an integer of at least 1 or null, not {paths!r}")
    if type(deterministic) is not bool:
        raise ValueError(f"deterministic must be true or false, not {deterministic!r}")
    if type(nominal) is not bool:
        raise ValueError(f"nominal must be true or false, not {nominal!r}")
    return policy, paths, deterministic, nominal


def simulate(text: str, options: str) -> str:
    """Run the scenario ``text`` (YAML) and return one JSON string.

    ``options`` is a JSON object with the optional keys ``policy`` (str or null), ``paths``
    (int or null), ``deterministic`` (bool) and ``nominal`` (bool).

    Returns:
        ``{"document": <the simulate document>}`` on success; otherwise
        ``{"status": <int or str>, "detail": <str>}``, where the status is 422 or 400 for a
        refused run, ``"memory"`` when the run exhausted memory, and ``"error"`` for any
        other exception, including a bad ``options`` (``ValueError``) and a
        ``RoutedParameterError``, with ``"<ExceptionName>: <message>"`` as the detail.
    """
    try:
        policy, paths, deterministic, nominal = _checked_options(options)
        if deterministic and paths is not None:
            raise RunRefusedError(422, PATHS_WITH_DETERMINISTIC)
        document = simulate_document(
            text,
            "yaml",
            policy=policy,
            paths=paths,
            deterministic=deterministic,
            nominal=nominal,
            params_root=DEFAULT_PARAMS_ROOT,
            path_cap=PATH_CAP,
        )
        return json.dumps({"document": document})
    except RunRefusedError as error:
        return json.dumps({"status": error.status, "detail": error.detail})
    except MemoryError:
        return json.dumps({"status": "memory", "detail": OUT_OF_MEMORY})
    except Exception as error:
        return json.dumps({"status": "error", "detail": f"{type(error).__name__}: {error}"})


def editor_call(call: str, arguments: str) -> str:
    """Answer the editor call ``call`` on ``arguments``, a JSON object's text, as one JSON string.

    Returns:
        ``{"status": <int>, "body": <JSON>}``, the status and body
        :func:`report.editor_calls.answer` gives, with the body's text written in unchanged; or
        ``{"status": "error", "body": {"detail": <str>}}`` for any exception, with
        ``"<ExceptionName>: <message>"`` as the detail.
    """
    try:
        status, body = answer(call, arguments, params_root=DEFAULT_PARAMS_ROOT)
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        return json.dumps({"status": "error", "body": {"detail": detail}})
    return f'{{"status": {status}, "body": {body}}}'

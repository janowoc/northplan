# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Scenario YAML or JSON text, or a YAML file -> a validated Scenario.

Three entry points, and they are for different callers.
:meth:`Scenario.model_validate` takes a mapping that is already in memory — a
test fixture, a scenario built in code. :func:`load_scenario` takes a path, and
is what a human's file goes through. :func:`scenario_from_text` takes text
already in memory (a request body), in YAML or JSON, with the same refusals as
a file; :func:`load_scenario` is the file path to it.

The difference is not only where the bytes come from. A YAML *file* can express
one thing a mapping cannot: the same key twice. ``yaml.safe_load`` keeps the
last of a repeated key and says nothing, so::

    asset_classes:
      equity: {real_mean: 0.05, vol: 0.16, ...}
      bonds:  {real_mean: 0.01, vol: 0.05, ...}
      equity: {real_mean: 0.02, vol: 0.04, ...}

parses to two classes with the third silently overwriting the first, and every
correlation and allocation below it now means something else. The schema cannot
catch that: by the time it sees a dict the duplicate is gone. So the parse here
rejects it, which is also the only place the scenario's "asset class names are
unique" rule can be enforced at all.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import yaml
from pydantic import ValidationError

from engine.params.loader import DuplicateYamlKeyError, parse_yaml
from engine.scenario.schema import Scenario

__all__ = [
    "MAX_DOCUMENT_VALUES",
    "DuplicateKeyError",
    "InvalidScenarioError",
    "MalformedScenarioFileError",
    "ScenarioError",
    "ScenarioFileMissingError",
    "load_scenario",
    "scenario_from_text",
]

# How many values a scenario document may hold once each alias is expanded; the
# committed example holds 140. It bounds how often aliases repeat a container, not
# merge keys or the length of a string, so it is no defence against a hostile
# document.
MAX_DOCUMENT_VALUES = 100_000


class ScenarioError(Exception):
    """Base class for failures loading a scenario or checking it against its params."""


class ScenarioFileMissingError(ScenarioError):
    """No file exists at the path given."""


class MalformedScenarioFileError(ScenarioError):
    """The text is not YAML or JSON, or its top level is not a mapping.

    Also raised when a document refers to itself through a YAML alias, holds more
    than :data:`MAX_DOCUMENT_VALUES` values once its aliases are expanded, or is JSON
    that Python refuses (an integer too long to convert, nesting too deep), and when
    a file is not UTF-8.

    Unlike a parameter file, an *empty* scenario file is malformed rather than
    a stub awaiting values. A parameter file with no values is the expected
    state of a tax year nobody has transcribed yet; a scenario with no values
    describes no household.
    """


class DuplicateKeyError(MalformedScenarioFileError):
    """The same key appears twice in one YAML mapping or JSON object.

    Accepted by both parsers and almost never intended. The parser would keep one
    and discard the rest without a word.
    """


class InvalidScenarioError(ScenarioError):
    """The text parsed but does not describe a valid scenario.

    Wraps the underlying :class:`pydantic.ValidationError`, which stays
    reachable as ``__cause__``, and adds the one thing it cannot know: which
    source the fields it names are in (a file path, or the request body). A run may
    load several scenarios, and a report of ``household.persons.0.birth_month``
    without its source sends the reader looking through all of them.
    """


def load_scenario(path: str | Path) -> Scenario:
    """Read, parse, and validate the scenario file at ``path``.

    Args:
        path: Path to a scenario YAML file. May be a ``*.local.yaml`` file,
            which is gitignored, so a real household's figures need never be
            committed.

    Returns:
        The validated, deeply immutable :class:`Scenario`.

    Raises:
        ScenarioFileMissingError: No file at ``path``.
        MalformedScenarioFileError: Not valid UTF-8, not parseable as YAML,
            top level is not a mapping, a mapping repeats a key, the document
            refers to itself through a YAML alias, or it holds more than
            ``MAX_DOCUMENT_VALUES`` values counting each value an alias repeats.
        InvalidScenarioError: Parsed, but a validation rule rejects it. The
            message names the offending field and this file.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ScenarioFileMissingError(f"No scenario file at {source}.") from exc
    except IsADirectoryError as exc:
        raise ScenarioFileMissingError(f"{source} is a directory, not a scenario file.") from exc
    except UnicodeDecodeError as exc:
        raise MalformedScenarioFileError(
            f"{source} is not valid UTF-8: {exc}. Save it as UTF-8."
        ) from exc
    return scenario_from_text(text, str(source), syntax="yaml")


def _unique_json_object(source: str, pairs: list[tuple[str, object]]) -> dict[str, object]:
    """``object_pairs_hook`` that refuses a repeated key rather than keeping the last."""
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise DuplicateKeyError(
                f"{source}: key {key!r} appears more than once in this object. JSON parsers "
                "differ on which value they keep, so one of them would be discarded silently."
            )
        out[key] = value
    return out


_DONE = object()


class _CyclicDocumentError(Exception):
    """A parsed document contains a container that contains itself."""


def _expanded_size(value: object, limit: int) -> int:
    """The number of values ``value`` holds with every YAML alias expanded.

    Containers are ``dict`` (its values), ``list``, ``tuple``, ``set`` and
    ``frozenset``; anything else is a leaf counting 1. A container counts 1 plus
    its children, and counts again each time it is reached. Iterative, so depth
    cannot exhaust the recursion limit.

    Returns:
        The count, or some number greater than ``limit`` once the count of a
        container still being counted passes it. Each distinct container is counted
        once, so the work is linear in the document as parsed.

    Raises:
        _CyclicDocumentError: A container is reached again while it is still
            being counted.
    """

    def children(node: object) -> list[object] | None:
        if isinstance(node, dict):
            return list(node.values())
        if isinstance(node, (list, tuple, set, frozenset)):
            return list(node)
        return None

    sizes: dict[int, int] = {}
    open_ids: set[int] = set()
    nodes: list[object] = []
    pending: list[Iterator[object]] = []
    counts: list[int] = []

    def enter(node: object) -> int | None:
        """The node's count if known at once; else None, with its frame pushed."""
        key = id(node)
        if key in sizes:
            return sizes[key]
        if key in open_ids:
            raise _CyclicDocumentError
        kids = children(node)
        if kids is None:
            return 1
        open_ids.add(key)
        nodes.append(node)
        pending.append(iter(kids))
        counts.append(1)
        return None

    known = enter(value)
    if known is not None:
        return known
    total = 0
    while nodes:
        child = next(pending[-1], _DONE)
        if child is not _DONE:
            size = enter(child)
            if size is None:
                continue
        else:
            node = nodes.pop()
            pending.pop()
            size = counts.pop()
            open_ids.discard(id(node))
            sizes[id(node)] = size
            if not nodes:
                total = size
                break
        counts[-1] += size
        if counts[-1] > limit:
            return counts[-1]
    return total


def scenario_from_text(text: str, source: str, *, syntax: Literal["yaml", "json"]) -> Scenario:
    """Parse and validate scenario ``text`` already in memory.

    Args:
        text: The scenario, as YAML or JSON.
        source: The label messages name: a file path, or ``"request body"``.
        syntax: ``"yaml"`` or ``"json"``.

    Returns:
        The validated, deeply immutable :class:`Scenario`.

    Raises:
        ValueError: ``syntax`` is neither ``"yaml"`` nor ``"json"``.
        MalformedScenarioFileError: Not parseable in ``syntax`` (for JSON, including
            text Python refuses: an integer over the digit limit, nesting too deep),
            or the top level is not a mapping, or the document refers to itself
            through a YAML alias, or it holds more than ``MAX_DOCUMENT_VALUES``
            values counting each value an alias repeats.
        DuplicateKeyError: A mapping repeats a key (a subclass of the above).
        InvalidScenarioError: Parsed, but a validation rule rejects it.

    JSON goes through :func:`json.loads`, never the YAML parser: YAML 1.1 reads
    ``1e5`` as a string, which JSON defines as a number.
    """
    if syntax not in ("yaml", "json"):
        raise ValueError(f"scenario_from_text: syntax must be 'yaml' or 'json', not {syntax!r}.")

    values: object
    if syntax == "yaml":
        try:
            values = parse_yaml(text)
        except DuplicateYamlKeyError as exc:
            raise DuplicateKeyError(f"{source}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise MalformedScenarioFileError(f"{source} is not valid YAML: {exc}") from exc
    else:
        try:
            values = json.loads(
                text, object_pairs_hook=functools.partial(_unique_json_object, source)
            )
        except (ValueError, RecursionError) as exc:
            raise MalformedScenarioFileError(f"{source} is not valid JSON: {exc}") from exc

    try:
        size = _expanded_size(values, MAX_DOCUMENT_VALUES)
    except _CyclicDocumentError as exc:
        raise MalformedScenarioFileError(
            f"{source}: the document refers to itself through a YAML alias, "
            "so it has no finite value."
        ) from exc
    if size > MAX_DOCUMENT_VALUES:
        raise MalformedScenarioFileError(
            f"{source}: the document holds more than {MAX_DOCUMENT_VALUES:,} values, "
            "counting each value an alias repeats; a scenario holds a few hundred."
        )

    if not isinstance(values, dict):
        raise MalformedScenarioFileError(
            f"{source}: a scenario file is a mapping of top-level keys, and "
            f"this parses to {type(values).__name__}."
        )

    try:
        return Scenario.model_validate(values)
    except ValidationError as exc:
        raise InvalidScenarioError(f"{source} is not a valid scenario:\n{exc}") from exc

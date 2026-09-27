# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Scenario YAML file -> a validated :class:`~engine.scenario.schema.Scenario`.

Two entry points, and they are for two different callers.
:meth:`Scenario.model_validate` takes a mapping that is already in memory — a
request body, a test fixture, a scenario built in code. :func:`load_scenario`
takes a path, and is what a human's file goes through.

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

from pathlib import Path

import yaml
from pydantic import ValidationError

from engine.params.loader import DuplicateYamlKeyError, parse_yaml
from engine.scenario.schema import Scenario

__all__ = [
    "DuplicateKeyError",
    "InvalidScenarioError",
    "MalformedScenarioFileError",
    "ScenarioError",
    "ScenarioFileMissingError",
    "load_scenario",
]


class ScenarioError(Exception):
    """Base class for failures loading a scenario or checking it against its params."""


class ScenarioFileMissingError(ScenarioError):
    """No file exists at the path given."""


class MalformedScenarioFileError(ScenarioError):
    """The file is not YAML, or its top level is not a mapping.

    Unlike a parameter file, an *empty* scenario file is malformed rather than
    a stub awaiting values. A parameter file with no values is the expected
    state of a tax year nobody has transcribed yet; a scenario with no values
    describes no household.
    """


class DuplicateKeyError(MalformedScenarioFileError):
    """The same key appears twice in one mapping.

    Legal YAML and almost never intended. The parser would keep the last and
    discard the rest without a word.
    """


class InvalidScenarioError(ScenarioError):
    """The file parsed but does not describe a valid scenario.

    Wraps the underlying :class:`pydantic.ValidationError`, which stays
    reachable as ``__cause__``, and adds the one thing it cannot know: which
    file the fields it names are in. A run may load several scenarios, and a
    report of ``household.persons.0.birth_month`` without a filename sends the
    reader looking through all of them.
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
            top level is not a mapping, or a mapping repeats a key.
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

    try:
        values = parse_yaml(text)
    except DuplicateYamlKeyError as exc:
        raise DuplicateKeyError(f"{source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise MalformedScenarioFileError(f"{source} is not valid YAML: {exc}") from exc

    if not isinstance(values, dict):
        raise MalformedScenarioFileError(
            f"{source}: a scenario file is a mapping of top-level keys, and "
            f"this parses to {type(values).__name__}."
        )

    try:
        return Scenario.model_validate(values)
    except ValidationError as exc:
        raise InvalidScenarioError(f"{source} is not a valid scenario:\n{exc}") from exc

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The scenario editor's logic: scenario text to a form's data and back, and the edits a form
cannot do cell by cell.

The editor's document is a *mapping*: what ``json.loads`` would give for a scenario. Its keys
are those the text wrote (an absent key stays absent; no default is filled in), the keys of
each model come in schema order, and each leaf is the value the validated ``Scenario`` holds,
except that an int the text gave for a float field stays that int. Everything crossing this
module's boundary is JSON data: dict, list, str, int, float, bool, None.

Invariants:
    * :func:`parse` opens a mapping only for text that passes everything a run checks first,
      start-age, lifespan and ceiling-bracket refusals included, and whose mapping keys are all
      text (an ``!!omap`` or ``!!pairs`` key aside).
    * :func:`check` validates YAML, never the mapping: the text :func:`emit` writes for it,
      then, when that passes, the text :func:`emit` writes for the mapping :func:`parse` would
      open from that text. No errors means exactly the returned text passes everything a run
      checks first.
    * An edit works on a deep copy and returns it; its argument is never changed. Every
      reader of a mapping tolerates a half-edited one and contributes nothing for a missing
      key, a wrong type or an index out of range.
    * Comments, each model's key order, flow style, anchors, aliases and merge keys are not
      kept: each model's keys are written in schema order with unknown keys after them, other
      keys in the order given, in block style except that an empty list or mapping is written
      ``[]`` or ``{}``, and each shared value is written out where it is used.

Parameter years are loaded once per (year, root) for the life of the process; a change under
``params/`` takes effect on a restart. Imports the engine; never imports ``api``, ``cli``, or
FastAPI.
"""

from __future__ import annotations

import copy
import functools
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Union

import pydantic
import yaml
from pydantic import BaseModel

from engine.core.indexation import RoutedParameterError
from engine.mc.prepare import check_scenario
from engine.params.loader import ParamError, ParamYear, load_year, parse_yaml
from engine.scenario.load import (
    MAX_REPORTED_ERRORS,
    InvalidScenarioError,
    ScenarioError,
    scenario_from_text,
    validation_message,
)
from engine.scenario.schema import (
    CONTRIBUTION_KINDS,
    DEFAULT_ALLOCATION,
    INVESTABLE_KINDS,
    WITHDRAWAL_KINDS,
    AssetClass,
    PolicySpec,
    Scenario,
    resolve_policy_path,
)

__all__ = [
    "FLATTENED",
    "EditRefusedError",
    "add_asset_class",
    "check",
    "emit",
    "grid_paths",
    "move_asset_class",
    "parse",
    "references",
    "remove",
    "rename",
    "schema",
]

Kind = Literal["person", "asset-class"]

FLATTENED: Final[str] = (
    "The text uses YAML anchors, aliases or merge keys. The form shows each value "
    "where it is used, and the YAML it writes repeats it there."
)

# Where a person's id is a key, relative to a policy; the grid names the same paths.
_PERSON_KEYED: Final[tuple[tuple[str, str], ...]] = (
    ("elections", "cpp_start_age_years"),
    ("elections", "oas_start_age_years"),
)

# What each ``x-northplan-*`` keyword means to the form:
#   exactly-one  the model takes exactly one of these fields;
#   defines      this value, or these keys, name a person or an asset class;
#   keys         this object's keys are names of that kind;
#   ref          this string is a name of that kind;
#   widget       the form draws this field with its own widget.
_KEYWORDS: Final[tuple[tuple[tuple[str, ...], dict[str, Any]], ...]] = (
    (
        ("$defs", "CppEntitlement"),
        {"x-northplan-exactly-one": ["contributory_history", "in_pay_monthly"]},
    ),
    (("$defs", "Person", "properties", "id"), {"x-northplan-defines": "person"}),
    (
        ("$defs", "ElectionsSpec", "properties", "cpp_start_age_years"),
        {"x-northplan-keys": "person"},
    ),
    (
        ("$defs", "ElectionsSpec", "properties", "oas_start_age_years"),
        {"x-northplan-keys": "person"},
    ),
    (("$defs", "Resp", "properties", "subscriber"), {"x-northplan-ref": "person"}),
    (
        ("$defs", "Assumptions", "properties", "asset_classes"),
        {"x-northplan-defines": "asset-class"},
    ),
    (
        ("$defs", "Assumptions", "properties", "correlation"),
        {"x-northplan-widget": "correlation"},
    ),
    (
        ("$defs", "Assumptions", "properties", "allocations"),
        {"propertyNames": {"enum": sorted(INVESTABLE_KINDS | {DEFAULT_ALLOCATION})}},
    ),
    (
        ("$defs", "Assumptions", "properties", "allocations", "additionalProperties"),
        {"x-northplan-keys": "asset-class"},
    ),
    (
        ("$defs", "ContributionRule", "properties", "weights"),
        {"propertyNames": {"enum": sorted(CONTRIBUTION_KINDS)}},
    ),
    (("$defs", "ContributionRule", "properties", "spill_order"), {"uniqueItems": True}),
    (
        ("$defs", "ContributionRule", "properties", "spill_order", "items"),
        {"enum": sorted(CONTRIBUTION_KINDS)},
    ),
    (("$defs", "WithdrawalRule", "properties", "order"), {"uniqueItems": True}),
    (
        ("$defs", "WithdrawalRule", "properties", "order", "items"),
        {"enum": sorted(WITHDRAWAL_KINDS)},
    ),
    (("properties", "grid"), {"x-northplan-widget": "grid"}),
)

_MISSING: Final = object()


class EditRefusedError(Exception):
    """An edit the mapping cannot take: ``references`` lists the places that block it."""

    def __init__(self, message: str, references: list[list[str | int]] | None = None) -> None:
        super().__init__(message)
        self.references: list[list[str | int]] = [] if references is None else references


# --- errors ---------------------------------------------------------------------------


def _load_year_uncached(year: int, root: Path) -> ParamYear:
    return load_year(year, root=root)


_load_year = functools.cache(_load_year_uncached)


def _capped(found: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``found`` cut to ``MAX_REPORTED_ERRORS``, with a trailer counting those cut."""
    if len(found) > MAX_REPORTED_ERRORS:
        hidden = len(found) - MAX_REPORTED_ERRORS
        found = found[:MAX_REPORTED_ERRORS]
        found.append(
            {
                "loc": [],
                "message": f"... and {hidden:,} more "
                f"{'error' if hidden == 1 else 'errors'}, not shown.",
            }
        )
    return found


def _errors(
    text: str, source: str, params_root: Path
) -> tuple[Scenario | None, list[dict[str, Any]]]:
    """Everything a run checks before it builds anything: the scenario, then its run."""
    try:
        scenario = scenario_from_text(text, source, syntax="yaml")
    except InvalidScenarioError as error:
        cause = error.__cause__
        assert isinstance(cause, pydantic.ValidationError)
        return None, _capped(
            [{"loc": list(e["loc"]), "message": e["msg"]} for e in cause.errors(include_url=False)]
        )
    except ScenarioError as error:
        return None, [{"loc": [], "message": str(error)}]
    try:
        check_scenario(
            scenario, paths=None, load_params=functools.partial(_load_year, root=params_root)
        )
    except pydantic.ValidationError as error:
        return scenario, [{"loc": ["grid"], "message": validation_message(error)}]
    except RoutedParameterError:
        raise
    except (ScenarioError, ParamError) as error:
        return scenario, [{"loc": [], "message": str(error)}]
    return scenario, []


# --- the mapping ----------------------------------------------------------------------


def _typed(raw: Any, value: Any) -> Any:
    """The parsed text ``raw`` and the validated ``value`` at the same place, as JSON data."""
    if isinstance(raw, dict) and isinstance(value, BaseModel):
        return {key: _typed(item, getattr(value, key)) for key, item in raw.items()}
    if isinstance(raw, dict) and isinstance(value, Mapping):
        return {key: _typed(item, value[key]) for key, item in raw.items()}
    if isinstance(raw, list) and isinstance(value, tuple):
        return [_typed(r, v) for r, v in zip(raw, value, strict=True)]
    if type(raw) is int and type(value) is float and raw == value:
        return raw
    if isinstance(value, tuple):
        return list(value)
    return value


def _strip(annotation: Any) -> Any:
    """``annotation`` without ``Annotated`` layers, and a union narrowed to its first
    model, mapping or tuple."""
    while typing.get_origin(annotation) is Annotated:
        annotation = typing.get_args(annotation)[0]
    if typing.get_origin(annotation) in (Union, types.UnionType):
        for argument in typing.get_args(annotation):
            argument = _strip(argument)
            if _is_model(argument) or typing.get_origin(argument) in (Mapping, tuple):
                return argument
    return annotation


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _ordered(value: Any, annotation: Any) -> Any:
    """``value`` with each model's known keys in schema order, then any unknown keys."""
    annotation = _strip(annotation)
    if _is_model(annotation) and isinstance(value, dict):
        fields = annotation.model_fields
        known = {
            key: _ordered(value[key], fields[key].annotation) for key in fields if key in value
        }
        return known | {key: item for key, item in value.items() if key not in fields}
    origin = typing.get_origin(annotation)
    arguments = typing.get_args(annotation)
    if origin is Mapping and isinstance(value, dict) and len(arguments) == 2:
        return {key: _ordered(item, arguments[1]) for key, item in value.items()}
    if origin is tuple and isinstance(value, list) and arguments:
        return [_ordered(item, arguments[0]) for item in value]
    return value


def _uses_aliases(text: str) -> bool:
    for event in yaml.parse(text, Loader=yaml.SafeLoader):
        if getattr(event, "anchor", None) is not None:
            return True
        if isinstance(event, yaml.ScalarEvent) and (
            (event.value == "<<" and event.implicit[0]) or event.tag == "tag:yaml.org,2002:merge"
        ):
            return True
    return False


def _opened(text: str, scenario: Scenario) -> Any:
    """The mapping :func:`parse` opens from ``text``, which validated as ``scenario``."""
    return _ordered(_typed(parse_yaml(text), scenario), Scenario)


def parse(text: str, *, params_root: Path) -> dict[str, Any]:
    """Scenario ``text`` as the editor's mapping, or the errors that stop a form opening.

    Args:
        text: The scenario, as YAML.
        params_root: The directory parameters are loaded from.

    Returns:
        ``{"mapping": ..., "errors": [...], "notices": [...]}``. Any error leaves ``mapping``
        None and ``notices`` empty; the errors are found as :func:`check` finds them, but on
        ``text`` itself, whose messages call it "the text". Otherwise ``notices`` is
        ``[FLATTENED]`` when the text uses an anchor, an alias or a merge key, else empty.
    """
    scenario, errors = _errors(text, "the text", params_root)
    if errors or scenario is None:
        return {"mapping": None, "errors": errors, "notices": []}
    return {
        "mapping": _opened(text, scenario),
        "errors": [],
        "notices": [FLATTENED] if _uses_aliases(text) else [],
    }


class _Dumper(yaml.SafeDumper):
    """``yaml.SafeDumper`` that writes a value used twice out each time, never as an alias, and
    double-quotes a string holding U+0085, which a plain or single-quoted scalar reads back as
    a line break."""

    def ignore_aliases(self, data: Any) -> bool:  # noqa: ARG002
        return True


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    if "\x85" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_Dumper.add_representer(str, _represent_str)


def emit(mapping: Any) -> str:
    """The YAML for ``mapping``: block style except an empty list or mapping (``[]``, ``{}``),
    keys of each model in schema order, unknown keys last, a shared object written out each
    time it is used.

    Raises:
        yaml.representer.RepresenterError: ``mapping`` holds a value ``yaml.SafeDumper``
            cannot represent.
        RecursionError: ``mapping`` is nested too deep to write.
    """
    return yaml.dump(
        _ordered(mapping, Scenario),
        Dumper=_Dumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )


def check(mapping: Any, *, params_root: Path) -> dict[str, Any]:
    """Emit ``mapping`` and check the text as a run checks a scenario first.

    When that text passes, it is written again from the mapping :func:`parse` would open from
    it, and the new text is checked in its place. Text that pydantic read as a number is then
    written as that number: ``"1_000"`` given for an int field is written ``1000``.

    Returns:
        ``{"yaml": text, "errors": [...]}``; each error is ``{"loc": [...], "message": str}``.
        The text is what is validated, so empty ``errors`` means exactly ``yaml`` passes. When
        the first text has errors, ``yaml`` is that text.

    Raises:
        yaml.representer.RepresenterError: ``mapping`` holds a value ``yaml.SafeDumper``
            cannot represent.
        RecursionError: ``mapping`` is nested too deep to write.
        RoutedParameterError: An engine bug, passed through.
    """
    text = emit(mapping)
    scenario, errors = _errors(text, "the form", params_root)
    if errors or scenario is None:
        return {"yaml": text, "errors": errors}
    typed = emit(_opened(text, scenario))
    return {"yaml": typed, "errors": _errors(typed, "the form", params_root)[1]}


# --- the schema -----------------------------------------------------------------------


def schema() -> dict[str, Any]:
    """The scenario's JSON Schema with the form's ``x-northplan-*`` keywords added.

    A fresh dict each call. Class docstrings (the root's and each definition's
    ``description``) are dropped; property descriptions stay.

    Raises:
        KeyError: A node a keyword is added to no longer exists in the schema.
    """
    document = Scenario.model_json_schema()
    document.pop("description", None)
    for definition in document.get("$defs", {}).values():
        definition.pop("description", None)
    for path, keywords in _KEYWORDS:
        node = document
        for segment in path:
            node = node[segment]
        node.update(copy.deepcopy(keywords))
    return document


# --- references and edits -------------------------------------------------------------


def _dig(value: Any, *path: str | int) -> Any:
    """The value at ``path`` in nested dicts and lists, or ``_MISSING``."""
    for step in path:
        in_dict = isinstance(value, dict) and isinstance(step, str) and step in value
        in_list = (
            isinstance(value, list)
            and isinstance(step, int)
            and not isinstance(step, bool)
            and 0 <= step < len(value)
        )
        if in_dict or in_list:
            value = value[step]
        else:
            return _MISSING
    return value


def _list(value: Any, *path: str | int) -> list[Any]:
    found = _dig(value, *path)
    return found if isinstance(found, list) else []


def _dict(value: Any, *path: str | int) -> dict[Any, Any]:
    found = _dig(value, *path)
    return found if isinstance(found, dict) else {}


def _places(count: int) -> str:
    return "1 place" if count == 1 else f"{count} places"


def _check_kind(kind: str) -> None:
    if kind not in ("person", "asset-class"):
        raise ValueError(f"kind must be 'person' or 'asset-class', not {kind!r}")


def _grid_matches(key: Any, name: str) -> tuple[str, ...] | None:
    """The path of ``_PERSON_KEYED`` that grid ``key`` names ``name`` under, if any."""
    if not isinstance(key, str):
        return None
    segments = key.split(".")
    for path in _PERSON_KEYED:
        if segments[:3] == [*path, name]:
            return path
    return None


def references(mapping: Any, kind: Kind, name: str) -> list[list[str | int]]:
    """Every place in ``mapping`` that names ``name`` as a ``kind``, as locs.

    A person is named as a key of a policy's CPP or OAS start-age election, as a beneficiary's
    RESP subscriber, and in a grid key. An asset class is named in an account's allocation,
    whatever its weight. Reads a malformed mapping without raising.

    Raises:
        ValueError: ``kind`` is neither ``"person"`` nor ``"asset-class"``.
    """
    _check_kind(kind)
    found: list[list[str | int]] = []
    if kind == "person":
        for i, policy in enumerate(_list(mapping, "policies")):
            for path in _PERSON_KEYED:
                if name in _dict(policy, *path):
                    found.append(["policies", i, *path, name])
        for j, beneficiary in enumerate(_list(mapping, "household", "beneficiaries")):
            if _dig(beneficiary, "resp", "subscriber") == name:
                found.append(["household", "beneficiaries", j, "resp", "subscriber"])
        for key in _dict(mapping, "grid"):
            if _grid_matches(key, name) is not None:
                found.append(["grid", key])
    else:
        for account, weights in _dict(mapping, "assumptions", "allocations").items():
            if isinstance(weights, dict) and name in weights:
                found.append(["assumptions", "allocations", account, name])
    return found


def _person_count(mapping: Any, person_id: str) -> int:
    return sum(
        1
        for person in _list(mapping, "household", "persons")
        if isinstance(person, dict) and person.get("id") == person_id
    )


def _check_one_person(mapping: Any, person_id: str) -> None:
    count = _person_count(mapping, person_id)
    if count == 0:
        raise EditRefusedError(f"household.persons: no person has id {person_id!r}.")
    if count > 1:
        raise EditRefusedError(f"household.persons: more than one person has id {person_id!r}.")


def _rekey(source: dict[Any, Any], old: str, new: str) -> dict[Any, Any]:
    """``source`` with key ``old`` renamed ``new`` in the same position."""
    return {(new if key == old else key): item for key, item in source.items()}


def _check_classes(mapping: Any) -> dict[Any, Any]:
    classes = _dig(mapping, "assumptions", "asset_classes")
    if not isinstance(classes, dict):
        raise EditRefusedError("assumptions.asset_classes is not a mapping of classes.")
    return classes


def _check_correlation(mapping: Any, count: int) -> list[list[Any]]:
    matrix = _dig(mapping, "assumptions", "correlation")
    if (
        not isinstance(matrix, list)
        or len(matrix) != count
        or not all(isinstance(row, list) and len(row) == count for row in matrix)
    ):
        raise EditRefusedError(
            f"assumptions.correlation is not a {count}x{count} matrix for the {count} asset "
            "classes; correct it first."
        )
    return matrix


def rename(mapping: Any, kind: Kind, old: str, new: str) -> Any:
    """A copy of ``mapping`` with the ``kind`` named ``old`` renamed ``new`` everywhere.

    A person's id changes with its election keys, RESP subscribers and grid keys; an asset
    class's key changes in ``asset_classes`` and every allocation, each keeping its position
    (``correlation`` has no names). ``new == old`` returns the copy unchanged.

    Raises:
        EditRefusedError: ``old`` is missing or ambiguous, ``assumptions.asset_classes`` is not
            a mapping, or ``new`` is already in use or already named somewhere (renaming would
            merge those places with its own); the last case lists the places in ``references``.
        ValueError: ``kind`` is neither ``"person"`` nor ``"asset-class"``.
    """
    _check_kind(kind)
    result = copy.deepcopy(mapping)
    if kind == "person":
        _check_one_person(mapping, old)
        if new == old:
            return result
        used = [
            item.get("id")
            for group in ("persons", "beneficiaries")
            for item in _list(mapping, "household", group)
            if isinstance(item, dict)
        ]
        if new in used:
            raise EditRefusedError(f"household: id {new!r} is already in use.")
        blocking = references(mapping, "person", new)
        if blocking:
            raise EditRefusedError(
                f"person {new!r} is already named in {_places(len(blocking))}; renaming "
                f"{old!r} to it would merge those with its own.",
                blocking,
            )
        for person in _list(result, "household", "persons"):
            if isinstance(person, dict) and person.get("id") == old:
                person["id"] = new
        for policy in _list(result, "policies"):
            for path in _PERSON_KEYED:
                holder = _dict(policy, path[0])
                if old in _dict(policy, *path):
                    holder[path[1]] = _rekey(holder[path[1]], old, new)
        for beneficiary in _list(result, "household", "beneficiaries"):
            resp = _dig(beneficiary, "resp")
            if isinstance(resp, dict) and resp.get("subscriber") == old:
                resp["subscriber"] = new
        grid = _dict(result, "grid")
        if grid:
            rebuilt = {}
            for key, values in grid.items():
                if _grid_matches(key, old) is not None:
                    segments = key.split(".")
                    segments[2] = new
                    key = ".".join(segments)
                rebuilt[key] = values
            result["grid"] = rebuilt
        return result
    classes = _check_classes(mapping)
    if old not in classes:
        raise EditRefusedError(f"assumptions.asset_classes: no class named {old!r}.")
    if new == old:
        return result
    if new in classes:
        raise EditRefusedError(f"assumptions.asset_classes: a class named {new!r} already exists.")
    blocking = references(mapping, "asset-class", new)
    if blocking:
        raise EditRefusedError(
            f"asset class {new!r} is already named in {_places(len(blocking))}; renaming "
            f"{old!r} to it would merge those with its own.",
            blocking,
        )
    assumptions = result["assumptions"]
    assumptions["asset_classes"] = _rekey(assumptions["asset_classes"], old, new)
    allocations = assumptions.get("allocations")
    if isinstance(allocations, dict):
        for account, weights in list(allocations.items()):
            if isinstance(weights, dict) and old in weights:
                allocations[account] = _rekey(weights, old, new)
    return result


def remove(mapping: Any, kind: Kind, name: str) -> Any:
    """A copy of ``mapping`` without the person, or the asset class and its ``correlation``
    row and column, named ``name``.

    Raises:
        EditRefusedError: ``name`` is missing or ambiguous, ``assumptions.asset_classes`` is
            not a mapping, the name is still referenced (the places are in ``references``), or
            ``correlation`` is not an n-by-n matrix.
        ValueError: ``kind`` is neither ``"person"`` nor ``"asset-class"``.
    """
    _check_kind(kind)
    result = copy.deepcopy(mapping)
    if kind == "person":
        _check_one_person(mapping, name)
        blocking = references(mapping, "person", name)
        if blocking:
            raise EditRefusedError(
                f"person {name!r} is still named in {_places(len(blocking))}; remove those first.",
                blocking,
            )
        persons = _list(result, "household", "persons")
        persons[:] = [p for p in persons if not (isinstance(p, dict) and p.get("id") == name)]
        return result
    classes = _check_classes(mapping)
    if name not in classes:
        raise EditRefusedError(f"assumptions.asset_classes: no class named {name!r}.")
    blocking = references(mapping, "asset-class", name)
    if blocking:
        raise EditRefusedError(
            f"asset class {name!r} is still named in {_places(len(blocking))}; remove those first.",
            blocking,
        )
    _check_correlation(mapping, len(classes))
    position = list(classes).index(name)
    assumptions = result["assumptions"]
    del assumptions["asset_classes"][name]
    assumptions["correlation"] = [
        [item for j, item in enumerate(row) if j != position]
        for i, row in enumerate(assumptions["correlation"])
        if i != position
    ]
    return result


def add_asset_class(mapping: Any, name: str) -> Any:
    """A copy of ``mapping`` with asset class ``name`` appended: every field 0.0, and a
    ``correlation`` row and column of zeros with 1.0 on the diagonal.

    Raises:
        EditRefusedError: ``asset_classes`` is not a mapping, ``name`` exists or is already
            named in an allocation, or ``correlation`` is not an n-by-n matrix.
    """
    classes = _check_classes(mapping)
    if name in classes:
        raise EditRefusedError(f"assumptions.asset_classes: a class named {name!r} already exists.")
    blocking = references(mapping, "asset-class", name)
    if blocking:
        raise EditRefusedError(
            f"asset class {name!r} is already named in {_places(len(blocking))}; "
            "remove those first.",
            blocking,
        )
    count = len(classes)
    _check_correlation(mapping, count)
    result = copy.deepcopy(mapping)
    assumptions = result["assumptions"]
    assumptions["asset_classes"][name] = {field: 0.0 for field in AssetClass.model_fields}
    for row in assumptions["correlation"]:
        row.append(0.0)
    assumptions["correlation"].append([0.0] * count + [1.0])
    return result


def move_asset_class(mapping: Any, name: str, index: int) -> Any:
    """A copy of ``mapping`` with asset class ``name`` moved to position ``index``, and the
    rows and columns of ``correlation`` permuted to follow.

    Raises:
        EditRefusedError: ``asset_classes`` is not a mapping, ``name`` is missing, ``index``
            is not an integer in ``0 <= index < n``, or ``correlation`` is not n-by-n.
    """
    classes = _check_classes(mapping)
    if name not in classes:
        raise EditRefusedError(f"assumptions.asset_classes: no class named {name!r}.")
    count = len(classes)
    if type(index) is not int or not 0 <= index < count:
        raise EditRefusedError(f"index must be an integer from 0 to {count - 1}, not {index!r}.")
    matrix = _check_correlation(mapping, count)
    names = list(classes)
    order = [n for n in names if n != name]
    order.insert(index, name)
    permutation = [names.index(n) for n in order]
    result = copy.deepcopy(mapping)
    assumptions = result["assumptions"]
    assumptions["asset_classes"] = {n: assumptions["asset_classes"][n] for n in order}
    assumptions["correlation"] = [[matrix[p][q] for q in permutation] for p in permutation]
    return result


# --- grid paths -----------------------------------------------------------------------


def _numeric_paths(value: Any, prefix: str = "") -> list[str]:
    """Dotted paths to the numbers inside a validated model or mapping, in order, except under
    a mapping key containing a dot."""
    if isinstance(value, BaseModel):
        children = [(key, getattr(value, key)) for key in type(value).model_fields]
    elif isinstance(value, Mapping):
        children = [(key, child) for key, child in value.items() if "." not in str(key)]
    elif isinstance(value, int | float) and not isinstance(value, bool):
        return [prefix]
    else:
        return []
    found: list[str] = []
    for key, child in children:
        found.extend(_numeric_paths(child, f"{prefix}.{key}" if prefix else str(key)))
    return found


def _resolves(spec: PolicySpec, path: str) -> bool:
    """Whether ``resolve_policy_path`` accepts ``path`` on ``spec``."""
    try:
        resolve_policy_path(spec, path)
    except ValueError:
        return False
    return True


def grid_paths(mapping: Any) -> list[str]:
    """The dotted paths, relative to a policy, that ``resolve_policy_path`` resolves to a number
    on every policy. A mapping key containing a dot is skipped: no dotted path can name it.

    Returns:
        The first policy's numeric paths, in its order, that ``resolve_policy_path`` accepts on
        every policy; ``[]`` when ``policies`` is not a non-empty list or any entry fails
        ``PolicySpec`` validation.
    """
    policies = _dig(mapping, "policies")
    if not isinstance(policies, list) or not policies:
        return []
    try:
        specs = [PolicySpec.model_validate(entry) for entry in policies]
    except pydantic.ValidationError:
        return []
    return [path for path in _numeric_paths(specs[0]) if all(_resolves(s, path) for s in specs)]

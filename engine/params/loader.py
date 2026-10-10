# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""YAML parameter files -> typed, frozen, defaultless parameter objects.

The guardrail behind the most important rule in this repository: no tax or
benefit constant is ever invented, recalled, or estimated. Every number the
engine uses is read from a file under ``params/`` that a human populated by
hand, under a comment naming the source it was checked against and the date it
was checked.

The design choice that does the actual work here is a negative one: **no lookup
in this module accepts a default.** ``ParamSet.get`` has no ``default``
parameter, there is no ``get_or``, and nothing returns ``None`` for an absent
key. A missing parameter raises :class:`MissingParameterError` naming the file
and the key path.

Everything returned is deeply immutable. Parameter objects are loaded once and
shared across every Monte Carlo path and every policy the optimizer evaluates;
a mutation anywhere would silently corrupt an entire run.

Conventions the files themselves must follow (see the header in any file under
``params/``): amounts are annual dollars unless the key says otherwise, rates
are bare fractions rather than percentages or basis points, and dollar amounts
are the nominal figures as published for that tax year.

A mapping or an ordered map (``!!omap``) that repeats a key is refused: YAML
forbids it, but PyYAML accepts it and keeps the last value without a word.
:func:`parse_yaml` is the one parser for hand-edited
YAML, so the refusal is the same everywhere it is used; a merge key (``<<``)
is not a repeat. Two keys that read the same once stored as text (``65`` and
``"65"``) are refused too, because every key is stored as text. A merged mapping
keeps one pair per key, as construction would, so merge keys cannot multiply a
document's size.
"""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import yaml

__all__ = [
    "DEFAULT_PARAMS_ROOT",
    "DuplicateYamlKeyError",
    "MalformedParamFileError",
    "MissingParameterError",
    "ParamDirectoryUnreadableError",
    "ParamError",
    "ParamFileMissingError",
    "ParamSet",
    "ParamYear",
    "ParamYearMissingError",
    "YamlConstructionError",
    "check_params_root",
    "load_year",
    "parse_yaml",
]

#: The YAML 1.1 value-key tag, which PyYAML's resolver gives a plain ``=``
#: key. SafeLoader loads such a key as its text.
_VALUE_TAG: Final[str] = "tag:yaml.org,2002:value"

#: The scalar tags whose keys the check builds early. A value-tagged key is
#: compared by its text; any other tag (merge, a collection tag, an unknown
#: tag) is left to SafeLoader's own construction. They are YAML tags, not tax
#: constants.
_SCALAR_KEY_TAGS: Final[frozenset[str]] = frozenset(
    f"tag:yaml.org,2002:{suffix}"
    for suffix in ("str", "int", "float", "bool", "null", "binary", "timestamp")
)

#: Repository-root ``params/`` directory. Resolved from this file's location so
#: that it works regardless of the process working directory.
DEFAULT_PARAMS_ROOT: Final[Path] = Path(__file__).resolve().parents[2] / "params"

#: Separator for nested lookups: ``brackets.federal.rates``.
PATH_SEP: Final[str] = "."


class ParamError(Exception):
    """Base class for every failure raised by the parameter loader."""


class ParamYearMissingError(ParamError):
    """No parameter directory exists for the requested tax year."""


class ParamDirectoryUnreadableError(ParamError):
    """A parameters directory, or one on the way to it, cannot be looked up, entered or listed."""


class ParamFileMissingError(ParamError):
    """A parameter set was requested but its YAML file does not exist or cannot be read."""


class MalformedParamFileError(ParamError):
    """A parameter file exists but cannot be read as one YAML mapping.

    An empty file is *not* malformed — a stub awaiting hand-entered values is
    the expected state of a new tax year. A file that is invalid YAML, whose
    top level is a list or a scalar, that repeats a key within a mapping, or
    whose mapping holds two keys that are the same once read as text, is
    malformed.
    """


class MissingParameterError(ParamError):
    """A requested parameter is absent from the file that should hold it.

    This is the loud stop that the "never invent a parameter" rule depends on.
    The correct response is to put the value in the YAML file, by hand, under a
    comment giving the source it was checked against and the date. It is never
    to supply a fallback here.
    """


class DuplicateYamlKeyError(ValueError):
    """A mapping or an ordered map repeats a key.

    YAML forbids it, and PyYAML accepts it by keeping the last value.
    """


class YamlConstructionError(yaml.YAMLError):
    """PyYAML raised something other than a YAML error while building the document.

    Typically on a malformed scalar (``!!int abc``, ``rate: !!float`` with no
    value, an out-of-range unquoted date). The original exception is this
    one's ``__cause__``.
    """


class _UniqueKeyLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that refuses a mapping node that repeats a scalar key.

    Checked when the node is composed, before merge flattening rewrites nodes
    in place. Only a key whose tag is in ``_SCALAR_KEY_TAGS`` is built and
    compared; a value-tagged key (``=``) is compared by its text. Every other
    key, including a merge key, is left to SafeLoader's own construction.
    An ordered map (``!!omap``) that repeats a key is refused the same way. After
    merge flattening a mapping keeps one pair per key, with keys compared as
    construction compares them (``1`` and ``1.0`` are one key).
    """

    def compose_mapping_node(self, anchor: Any) -> yaml.MappingNode:
        node = super().compose_mapping_node(anchor)
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if not isinstance(key_node, yaml.ScalarNode):
                continue
            if key_node.tag == _VALUE_TAG:
                key = key_node.value
            elif key_node.tag in _SCALAR_KEY_TAGS:
                key = self.construct_object(key_node)
            else:
                continue
            if key in seen:
                raise DuplicateYamlKeyError(
                    f"line {key_node.start_mark.line + 1}, column "
                    f"{key_node.start_mark.column + 1}: key {key!r} appears more "
                    "than once in this mapping. YAML forbids it, and PyYAML would keep "
                    "the last, so one of the values would be discarded silently."
                )
            seen.add(key)
        return node

    def flatten_mapping(self, node: yaml.MappingNode) -> None:
        """Merge as PyYAML does, then keep one pair per key.

        A key keeps the position of its first occurrence and the value of its
        last, which is the dict construction would build. Without this, a merge
        list is multiplied by its fan-out at each level of nesting.
        """
        super().flatten_mapping(node)
        index: dict[Any, int] = {}
        pairs: list[tuple[yaml.Node, yaml.Node]] = []
        for key_node, value_node in node.value:
            identity: Any
            if isinstance(key_node, yaml.ScalarNode):
                if key_node.tag in _SCALAR_KEY_TAGS:
                    identity = ("k", self.construct_object(key_node))
                else:
                    identity = ("t", key_node.tag, key_node.value)
            else:
                identity = ("i", id(key_node))
            if identity in index:
                pairs[index[identity]] = (pairs[index[identity]][0], value_node)
            else:
                index[identity] = len(pairs)
                pairs.append((key_node, value_node))
        if len(pairs) != len(node.value):
            node.value = pairs

    def construct_yaml_omap(self, node: yaml.Node) -> Iterator[Any]:
        """Refuse a repeated key in an ordered map, then build it as SafeLoader does."""
        if isinstance(node, yaml.SequenceNode):
            seen: set[Any] = set()
            for entry in node.value:
                if not isinstance(entry, yaml.MappingNode) or len(entry.value) != 1:
                    continue
                key_node = entry.value[0][0]
                if not isinstance(key_node, yaml.ScalarNode):
                    continue
                if key_node.tag == _VALUE_TAG:
                    key = key_node.value
                elif key_node.tag in _SCALAR_KEY_TAGS:
                    key = self.construct_object(key_node)
                else:
                    continue
                if key in seen:
                    raise DuplicateYamlKeyError(
                        f"line {key_node.start_mark.line + 1}, column "
                        f"{key_node.start_mark.column + 1}: key {key!r} appears more "
                        "than once in this ordered map. YAML forbids it, and PyYAML would "
                        "keep the last, so one of the values would be discarded silently."
                    )
                seen.add(key)
        yield from super().construct_yaml_omap(node)


_UniqueKeyLoader.add_constructor("tag:yaml.org,2002:omap", _UniqueKeyLoader.construct_yaml_omap)


def parse_yaml(text: str) -> Any:
    """Parse ``text`` as YAML, refusing a mapping that repeats a key.

    Args:
        text: YAML document text.

    Returns:
        Whatever ``yaml.safe_load(text)`` would return for the same text.

    Raises:
        DuplicateYamlKeyError: If one mapping, or one ordered map (``!!omap``), in
            ``text`` repeats a key. Keys
            are compared as built, so ``1``, ``1.0`` and ``true`` are one key.
            A merge key (``<<``) is never counted, even when a mapping has
            two. A ``ValueError``, not a :class:`yaml.YAMLError`.
        YamlConstructionError: If PyYAML raises anything other than a
            :class:`yaml.YAMLError` while building the document, such as a
            malformed scalar or nesting too deep to build. The original
            exception is its ``__cause__``.
        yaml.YAMLError: Any other YAML parse failure, unchanged.

    A malformed key or a repeated key in a nested mapping may be reported
    ahead of a syntax error later in the text, and a repeat made through an
    alias key is reported at the alias's anchor. The message does not name a
    file; a caller that has one prefixes it.
    """
    try:
        return yaml.load(text, Loader=_UniqueKeyLoader)
    except (DuplicateYamlKeyError, yaml.YAMLError):
        raise
    except Exception as exc:
        raise YamlConstructionError(
            f"the document could not be built: {type(exc).__name__}: {exc}"
        ) from exc


def _freeze(value: Any, *, where: str = "") -> Any:
    """Recursively convert loaded YAML into deeply immutable structures.

    Mappings become read-only views, sequences become tuples, scalars pass
    through. Strings and bytes are scalars, not sequences, for this purpose.

    Mapping keys are stored as ``str``. Two keys of one mapping that produce
    the same text (``65`` and ``"65"``) raise :class:`MalformedParamFileError`
    rather than silently discarding one of the two values.

    Args:
        where: Dotted path of the mapping being frozen, used only for the
            collision message above. The top level is ``""``.

    Raises:
        MalformedParamFileError: If two keys of one mapping read the same
            once stored as text. The message gives the key path, not the
            file — construction is over by then, so it cannot give a line
            number either; the caller that has a file prefixes it.
    """
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        originals: dict[str, Any] = {}
        location = f"at {where}" if where else "at the top level"
        for k, v in value.items():
            key = str(k)
            if key in frozen:
                raise MalformedParamFileError(
                    f"{originals[key]!r} and {k!r} both read as {key!r} once "
                    f"stored as text, in the mapping {location}. Keys are "
                    "stored as text, so one of the two values would be "
                    "discarded silently."
                )
            originals[key] = k
            child_where = f"{where}{PATH_SEP}{key}" if where else key
            frozen[key] = _freeze(v, where=child_where)
        return MappingProxyType(frozen)
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Sequence):
        return tuple(
            _freeze(v, where=f"{where}{PATH_SEP}{i}" if where else str(i))
            for i, v in enumerate(value)
        )
    return value


@dataclass(frozen=True, slots=True)
class ParamSet:
    """One parameter file — e.g. ``params/2026/federal.yaml`` — read-only.

    Attributes:
        name: The file's stem, e.g. ``"federal"``.
        year: The tax year the file belongs to.
        source: Absolute path to the file, quoted in every error message so a
            failure points at the file a human has to edit.
        values: Deeply immutable mapping of the file's contents. Empty for a
            stub file that has not been populated yet.
    """

    name: str
    year: int
    source: Path
    values: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def get(self, path: str) -> Any:
        """Return the value at a dotted ``path``, or raise.

        There is deliberately no ``default`` argument. A caller that wants to
        proceed without a hand-verified parameter is a caller that is about to
        produce a wrong number.

        Args:
            path: Dot-separated key path, e.g. ``"brackets.rates"``. A path
                with no separator is a top-level key.

        Returns:
            The value, deeply immutable: a scalar, a tuple, or a read-only
            mapping.

        Raises:
            MissingParameterError: If any segment of the path is absent, or if
                an intermediate segment is a scalar and cannot be traversed.
        """
        cursor: Any = self.values
        walked: list[str] = []
        for segment in path.split(PATH_SEP):
            if not isinstance(cursor, Mapping):
                raise MissingParameterError(
                    f"{self._where(path)}: {PATH_SEP.join(walked) or '<root>'} is a "
                    f"{type(cursor).__name__}, not a mapping, so {segment!r} cannot be "
                    f"looked up under it."
                )
            if segment not in cursor:
                raise MissingParameterError(
                    f"{self._where(path)}: {segment!r} is not present"
                    f"{' under ' + PATH_SEP.join(walked) if walked else ''}. "
                    f"{self._remedy()}"
                )
            cursor = cursor[segment]
            walked.append(segment)
        return cursor

    def number(self, path: str) -> float:
        """Return the value at ``path`` as a float, or raise.

        Booleans are rejected: YAML's ``yes``/``no`` parse as ``bool``, which
        would otherwise silently become ``1.0`` in a calculation.

        Raises:
            MissingParameterError: If the parameter is absent.
            MalformedParamFileError: If the value is present but not numeric.
        """
        value = self.get(path)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MalformedParamFileError(
                f"{self._where(path)}: expected a number, found {type(value).__name__} ({value!r})."
            )
        return float(value)

    def sequence(self, path: str) -> tuple[Any, ...]:
        """Return the value at ``path`` as a tuple, or raise.

        Used for ordered parameter tables — bracket edges, rates, RRIF factors
        by age. A scalar is not silently wrapped in a one-element tuple: a
        table that should have many entries and has one is a data error worth
        surfacing.

        Raises:
            MissingParameterError: If the parameter is absent.
            MalformedParamFileError: If the value is present but not a sequence.
        """
        value = self.get(path)
        if not isinstance(value, tuple):
            raise MalformedParamFileError(
                f"{self._where(path)}: expected a list, found {type(value).__name__} ({value!r})."
            )
        return value

    def numbers(self, path: str) -> tuple[float, ...]:
        """Return the value at ``path`` as a tuple of floats, or raise.

        Raises:
            MissingParameterError: If the parameter is absent.
            MalformedParamFileError: If the value is not a list, or if any
                element is not numeric.
        """
        values = self.sequence(path)
        out: list[float] = []
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MalformedParamFileError(
                    f"{self._where(path)}: element {index} is "
                    f"{type(value).__name__} ({value!r}), expected a number."
                )
            out.append(float(value))
        return tuple(out)

    def has(self, path: str) -> bool:
        """Whether ``path`` is present.

        For diagnostics and for tests that assert on the loader's own
        behaviour. Not an escape hatch: engine code that branches on ``has``
        to pick a fallback value is violating the parameter rule just as surely
        as one that inlines a constant.
        """
        try:
            self.get(path)
        except MissingParameterError:
            return False
        return True

    def is_stub(self) -> bool:
        """Whether the file exists but holds no values yet."""
        return not self.values

    def _where(self, path: str) -> str:
        return f"{self.source}[{path}]"

    def _remedy(self) -> str:
        stub = " The file is an empty stub." if self.is_stub() else ""
        return (
            f"{stub} Add it to {self.source.name} by hand, under a comment giving "
            f"the source it was checked against and the date. Never substitute an "
            f"estimated or remembered value."
        )


@dataclass(frozen=True, slots=True)
class ParamYear:
    """Every parameter file for one tax year.

    Attributes:
        year: The tax year, e.g. ``2026``.
        root: The directory the files were read from.
        sets: Read-only mapping of file stem to :class:`ParamSet`.
    """

    year: int
    root: Path
    sets: Mapping[str, ParamSet]

    def __getitem__(self, name: str) -> ParamSet:
        """Return the named parameter set, or raise.

        Raises:
            ParamFileMissingError: If no such file exists for this year.
        """
        try:
            return self.sets[name]
        except KeyError:
            available = ", ".join(sorted(self.sets)) or "none"
            raise ParamFileMissingError(
                f"No parameter file {name!r} for tax year {self.year} in {self.root}. "
                f"Available: {available}."
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self.sets

    def names(self) -> tuple[str, ...]:
        """Names of the parameter sets loaded for this year, sorted."""
        return tuple(sorted(self.sets))

    @property
    def federal(self) -> ParamSet:
        """Federal income tax parameters."""
        return self["federal"]

    @property
    def cpp(self) -> ParamSet:
        """Canada Pension Plan parameters."""
        return self["cpp"]

    @property
    def oas(self) -> ParamSet:
        """Old Age Security parameters."""
        return self["oas"]

    @property
    def rrif(self) -> ParamSet:
        """RRSP and RRIF parameters.

        One file, because they are one program at two stages of life. The file
        keeps them under symmetric ``rrsp:`` and ``rrif:`` keys, so a RRIF
        lookup through this property reads ``rrif.rrif.minimum_factors...``.
        The doubled segment is deliberate, not a bug.
        """
        return self["rrif"]

    @property
    def tfsa(self) -> ParamSet:
        """TFSA contribution room and recontribution rules."""
        return self["tfsa"]

    @property
    def resp(self) -> ParamSet:
        """RESP contribution, grant, and Educational Assistance Payment rules."""
        return self["resp"]

    def province(self, code: str) -> ParamSet:
        """Provincial income tax parameters for a two-letter province code.

        This is the province of *residence*: the jurisdiction whose income tax
        the household pays. For a locked-in account it is usually the wrong
        lookup — see :meth:`jurisdiction`.

        Args:
            code: Province code, case-insensitive, e.g. ``"ab"``.

        Raises:
            ParamFileMissingError: If that province has no file for this year.
        """
        return self[code.lower()]

    def jurisdiction(self, code: str) -> ParamSet:
        """Pension parameters for the jurisdiction a locked-in account is registered in.

        Same files as :meth:`province`, reached deliberately by a different
        name. A LIF is governed by the pension legislation of the jurisdiction
        its originating pension was registered under, which is not necessarily
        where the holder now lives: someone resident in Alberta may hold an
        Ontario-registered LIF, draw it under Ontario's maximum, and file
        Alberta income tax. Reading the LIF maximum out of
        ``province(household.province)`` gets that household wrong, silently
        and with no test to catch it.

        The two methods return the same object when the household never moved,
        which is the common case and the reason the bug is easy to ship.

        Args:
            code: Jurisdiction code, case-insensitive, e.g. ``"on"``.

        Raises:
            ParamFileMissingError: If that jurisdiction has no file for this
                year. Federally regulated pensions are a jurisdiction in their
                own right and will need their own file; there is no such file
                today and this raises rather than falling back to a province.
        """
        return self[code.lower()]


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise MalformedParamFileError(
            f"{path} is not valid UTF-8: {exc}. Save it as UTF-8."
        ) from exc
    except OSError as exc:
        raise ParamFileMissingError(f"Cannot read parameter file {path}: {exc}") from exc

    try:
        raw = parse_yaml(text)
    except DuplicateYamlKeyError as exc:
        raise MalformedParamFileError(f"{path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise MalformedParamFileError(f"{path} is not valid YAML: {exc}") from exc

    if raw is None:
        # A stub: comment header only, no values entered yet. Expected state
        # for a tax year a human has not populated. Not an error.
        return MappingProxyType({})
    if not isinstance(raw, Mapping):
        raise MalformedParamFileError(
            f"{path} must contain a YAML mapping at its top level, found {type(raw).__name__}."
        )
    try:
        return _freeze(raw)
    except MalformedParamFileError as exc:
        raise MalformedParamFileError(f"{path}: {exc}") from exc


#: The errors ``Path.is_dir()`` reads as "no such directory" in Python 3.12, rather
#: than raising. Any other ``OSError`` from a lookup is raised to the caller.
_ABSENT_ERRNOS: Final[frozenset[int]] = frozenset(
    {errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP}
)


def _lookup(path: Path) -> os.stat_result | None:
    """``path.stat()``, or None on a ``ValueError`` or on ENOENT, ENOTDIR, EBADF or ELOOP.

    Any other error is raised.
    """
    try:
        return path.stat()
    except ValueError:
        return None
    except OSError as exc:
        if exc.errno in _ABSENT_ERRNOS:
            return None
        raise


def _is_directory(path: Path) -> bool:
    found = _lookup(path)
    return found is not None and stat.S_ISDIR(found.st_mode)


def _reason(exc: OSError) -> str:
    return exc.strerror or str(exc)


def _available_years(root: Path) -> str:
    try:
        if _is_directory(root):
            entries = list(root.iterdir())
        elif _lookup(root) is not None:
            return "none — the params root is not a directory"
        else:
            return "none — the params root itself does not exist"
    except OSError as exc:
        return f"unknown — {root} cannot be listed: {_reason(exc)}"
    years = []
    for entry in entries:
        try:
            if _is_directory(entry):
                years.append(entry.name)
        except OSError:
            pass  # an entry the loader cannot look up is not an available year
    return ", ".join(sorted(years)) or "none"


def load_year(year: int, root: Path | str = DEFAULT_PARAMS_ROOT) -> ParamYear:
    """Load every parameter file for one tax year.

    Reads ``{root}/{year}/*.yaml``. Files that exist but are empty load as stub
    :class:`ParamSet` objects, so that a lookup against them fails with a
    :class:`MissingParameterError` naming the file — a more useful failure than
    the file simply being absent.

    Args:
        year: Tax year, e.g. ``2026``.
        root: Directory holding the per-year subdirectories. Defaults to the
            repository's ``params/``.

    Returns:
        A frozen :class:`ParamYear`.

    Raises:
        ParamYearMissingError: If the lookup of the year's directory fails with a
            ``ValueError`` or with ENOENT, ENOTDIR, EBADF or ELOOP, or finds something
            other than a directory.
        ParamDirectoryUnreadableError: If the lookup of the year's directory fails with
            an ``OSError`` other than ENOENT, ENOTDIR, EBADF or ELOOP (the message names
            ``root``), or the listing of that directory fails (the message names it).
        ParamFileMissingError: If a file in the directory exists but the
            operating system cannot open or read it.
        MalformedParamFileError: If any file in the directory is not valid
            UTF-8, is not valid YAML, is not a YAML mapping, repeats a key
            within a mapping, or holds two keys in one mapping that are the
            same once read as text.
    """
    root = Path(root)
    year_dir = root / str(year)
    try:
        is_year_dir = _is_directory(year_dir)
    except OSError as exc:
        raise ParamDirectoryUnreadableError(
            f"Cannot look up tax year {year} in the parameters directory {root}: {_reason(exc)}."
        ) from exc
    if not is_year_dir:
        raise ParamYearMissingError(
            f"No parameters for tax year {year}: {year_dir} does not exist. "
            f"Available years: {_available_years(root)}."
        )
    try:
        entries = list(year_dir.iterdir())
    except OSError as exc:
        raise ParamDirectoryUnreadableError(
            f"Cannot list the parameters directory {year_dir}: {_reason(exc)}."
        ) from exc

    sets: dict[str, ParamSet] = {}
    for path in sorted(entry for entry in entries if entry.name.endswith(".yaml")):
        sets[path.stem] = ParamSet(
            name=path.stem,
            year=year,
            source=path.resolve(),
            values=_read_yaml(path),
        )

    return ParamYear(year=year, root=year_dir.resolve(), sets=MappingProxyType(sets))


def check_params_root(root: Path | str) -> None:
    """Raise if a lookup or listing of ``root``, or of a year entry in it, is refused.

    For the start of a long-running process, so that a refusal comes at once rather than
    on every run. A year entry is an entry of ``root`` whose name is ASCII digits; when
    ``root`` can be listed, each is looked up, and listed if it is a directory. Opens no
    parameter file, so a year directory that can be listed but not entered passes. A
    lookup that fails with a ``ValueError`` or with ENOENT, ENOTDIR, EBADF or ELOOP finds
    nothing; it is not a refusal.

    ``root`` passes when it can be listed and the lookup of every year entry in it finds
    a directory that can be listed, something else, or nothing, including when it holds
    no year entry. It also passes when its own lookup finds nothing or something other
    than a directory: :func:`load_year` reports it per year. And it passes when it cannot
    be listed but can be entered, an exception to the summary: :func:`load_year` loads
    from it, and its years cannot be enumerated, so none is checked.

    Raises:
        ParamDirectoryUnreadableError: If the lookup of ``root`` fails with an
            ``OSError`` other than ENOENT, ENOTDIR, EBADF or ELOOP; if ``root`` is a
            directory that can be neither listed nor entered; or if ``root`` can be
            listed and the lookup of a year entry in it fails with such an error or finds
            a directory that cannot be listed.
    """
    root = Path(root)
    try:
        if not _is_directory(root):
            return
    except OSError as exc:
        raise ParamDirectoryUnreadableError(
            f"Cannot look up the parameters directory {root}: {_reason(exc)}."
        ) from exc
    try:
        names = [entry.name for entry in root.iterdir()]
    except OSError as listing:
        if not os.access(root, os.X_OK):
            raise ParamDirectoryUnreadableError(
                f"Cannot list the parameters directory {root}: {_reason(listing)}."
            ) from listing
        return
    for name in sorted(names):
        if not (name.isascii() and name.isdigit()):
            continue
        year_dir = root / name
        try:
            if not _is_directory(year_dir):
                continue
        except OSError as exc:
            raise ParamDirectoryUnreadableError(
                f"Cannot look up tax year {name} in the parameters directory {root}: "
                f"{_reason(exc)}."
            ) from exc
        try:
            list(year_dir.iterdir())
        except OSError as exc:
            raise ParamDirectoryUnreadableError(
                f"Cannot list the parameters directory {year_dir}: {_reason(exc)}."
            ) from exc

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
and the key path. That turns "we don't have this number yet" from a silent
plausible-looking result into a loud stop, which is the only way the rule
survives contact with an agent under time pressure.

Everything returned is deeply immutable. Parameter objects are loaded once and
shared across every Monte Carlo path and every policy the optimizer evaluates;
a mutation anywhere would silently corrupt an entire run.

Conventions the files themselves must follow (see the header in any file under
``params/``): amounts are annual dollars unless the key says otherwise, rates
are bare fractions rather than percentages or basis points, and dollar amounts
are the nominal figures as published for that tax year.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import yaml

__all__ = [
    "DEFAULT_PARAMS_ROOT",
    "MalformedParamFileError",
    "MissingParameterError",
    "ParamError",
    "ParamFileMissingError",
    "ParamSet",
    "ParamYear",
    "ParamYearMissingError",
    "load_year",
]

#: Repository-root ``params/`` directory. Resolved from this file's location so
#: that it works regardless of the process working directory.
DEFAULT_PARAMS_ROOT: Final[Path] = Path(__file__).resolve().parents[2] / "params"

#: Separator for nested lookups: ``brackets.federal.rates``.
PATH_SEP: Final[str] = "."


class ParamError(Exception):
    """Base class for every failure raised by the parameter loader."""


class ParamYearMissingError(ParamError):
    """No parameter directory exists for the requested tax year."""


class ParamFileMissingError(ParamError):
    """A parameter set was requested but its YAML file does not exist."""


class MalformedParamFileError(ParamError):
    """A parameter file exists but is not a YAML mapping.

    An empty file is *not* malformed — a stub awaiting hand-entered values is
    the expected state of a new tax year. A file whose top level is a list, a
    scalar, or invalid YAML is malformed.
    """


class MissingParameterError(ParamError):
    """A requested parameter is absent from the file that should hold it.

    This is the loud stop that the "never invent a parameter" rule depends on.
    The correct response is to put the value in the YAML file, by hand, under a
    comment giving the source it was checked against and the date. It is never
    to supply a fallback here.
    """


def _freeze(value: Any) -> Any:
    """Recursively convert loaded YAML into deeply immutable structures.

    Mappings become read-only views, sequences become tuples, scalars pass
    through. Strings and bytes are scalars, not sequences, for this purpose.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Sequence):
        return tuple(_freeze(v) for v in value)
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

        Booleans are rejected: YAML's ``yes``/``no`` parse as ``bool``, and a
        bool silently becoming ``1.0`` in a tax calculation is exactly the kind
        of quiet wrongness this module exists to prevent.

        Raises:
            MissingParameterError: If the parameter is absent.
            MalformedParamFileError: If the value is present but not numeric.
        """
        value = self.get(path)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MalformedParamFileError(
                f"{self._where(path)}: expected a number, found "
                f"{type(value).__name__} ({value!r})."
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
                f"{self._where(path)}: expected a list, found "
                f"{type(value).__name__} ({value!r})."
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
        The doubled segment is deliberate: the alternative was a file named for
        one program holding two, or an asymmetry between the halves.
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
    except OSError as exc:
        raise ParamFileMissingError(f"Cannot read parameter file {path}: {exc}") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise MalformedParamFileError(f"{path} is not valid YAML: {exc}") from exc

    if raw is None:
        # A stub: comment header only, no values entered yet. Expected state
        # for a tax year a human has not populated. Not an error.
        return MappingProxyType({})
    if not isinstance(raw, Mapping):
        raise MalformedParamFileError(
            f"{path} must contain a YAML mapping at its top level, found "
            f"{type(raw).__name__}."
        )
    return _freeze(raw)


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
        ParamYearMissingError: If no directory exists for that year.
        MalformedParamFileError: If any file in the directory is not a YAML
            mapping.
    """
    root = Path(root)
    year_dir = root / str(year)
    if not year_dir.is_dir():
        available = (
            ", ".join(sorted(p.name for p in root.iterdir() if p.is_dir()))
            if root.is_dir()
            else "none — the params root itself does not exist"
        )
        raise ParamYearMissingError(
            f"No parameters for tax year {year}: {year_dir} does not exist. "
            f"Available years: {available or 'none'}."
        )

    sets: dict[str, ParamSet] = {}
    for path in sorted(year_dir.glob("*.yaml")):
        sets[path.stem] = ParamSet(
            name=path.stem,
            year=year,
            source=path.resolve(),
            values=_read_yaml(path),
        )

    return ParamYear(year=year, root=year_dir.resolve(), sets=MappingProxyType(sets))

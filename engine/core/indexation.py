# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The constant real-terms cost of periodic indexation, and the only route to a dollar.

An engine that works in real dollars and steps annually can treat a fully indexed benefit as
constant. An engine that steps monthly cannot: a nominally fixed amount's real value falls
between adjustment dates and steps back up on each one, averaging below its published real
value, permanently, in the direction that flatters the plan.

This module models that mean and nothing else:

- Every function here is independent of the month; the erosion factor for a schedule is one
  number per scenario, computed once.
- The oscillation around that mean is dropped: it is bounded, mean-zero, and smaller than the
  uncertainty in the inflation assumption it depends on.
- The lag between the window an adjustment is computed from and the date it takes effect is also
  not modelled, but it is not mean-zero: it overstates every indexed amount by a little, in the
  direction that flatters the plan (``docs/limitations.md`` L5).

Unindexed amounts are different: their real loss is not bounded by one cycle but grows without
limit over a retirement, the largest real-terms effect in the model. :func:`unindexed_factor` is
therefore the one function here that depends on time.

The real-terms view
--------------------

:class:`RealParamSet` wraps a :class:`~engine.params.loader.ParamSet` with the scenario's
inflation rate and is the **only** supported way for engine code to reach a dollar amount. Its
guarantee is negative, like the loader's: the raw accessors it inherits the names of refuse a
routed path, so a dollar cannot be read undeflated by forgetting the right method;
:meth:`RealParamSet.amount` is the one that cannot return an undeflated figure.

Which amounts sit on which schedule is declared in each parameter file's ``indexation`` block,
by hand, with a source. An amount whose rule has not been recorded is not indexed "annually by
default" — it is unrouted, and the run stops.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from engine.params.loader import (
    PATH_SEP,
    MalformedParamFileError,
    MissingParameterError,
    ParamError,
    ParamSet,
    ParamYear,
)

__all__ = [
    "INDEXATION_BLOCK",
    "WILDCARD",
    "RealParamSet",
    "RealParamYear",
    "RoutedParameterError",
    "UnroutedParameterError",
    "as_filed_to_real_factor",
    "erosion_factor",
    "nominal_carry_factor",
    "real_year",
    "schedule",
    "unindexed_factor",
]

#: Top-level key in a parameter file holding its indexation schedules.
INDEXATION_BLOCK: Final[str] = "indexation"

#: Stands for one list index in an ``applies_to`` path, so that a routing entry
#: can name every element of a table — ``pension.age_bands.*.maximum_monthly``
#: — without the file having to be rewritten when the table grows a band.
WILDCARD: Final[str] = "*"

#: Months in a year. The only reason a bare integer appears in the arithmetic
#: below: it is a property of the calendar, not a tax parameter.
MONTHS_PER_YEAR: Final[int] = 12


class RoutedParameterError(ParamError):
    """A dollar amount was read through an accessor that does not deflate it.

    Raised by the pass-through accessors on :class:`RealParamSet` for a path
    that some schedule routes. The remedy is :meth:`RealParamSet.amount` or
    :meth:`RealParamSet.amounts` for a monthly quantity, or
    :meth:`RealParamSet.annual_amount` or :meth:`RealParamSet.annual_amounts`
    for an annual one; there is deliberately no flag to suppress this, because
    the whole value of the real-terms view is that forgetting to deflate is
    impossible rather than merely discouraged.
    """


class UnroutedParameterError(ParamError):
    """A dollar amount was read that no schedule routes.

    An amount whose indexation rule has not been recorded is not indexed
    "annually by default" and is not unindexed by default either — it is
    unknown, and guessing would put an invented parameter into the model by the
    back door. The remedy is to add the path to an ``applies_to`` list in the
    file, by hand, under a source comment.
    """


def erosion_factor(inflation_rate: float, adjustments_per_year: int) -> float:
    """Mean real value of an amount adjusted ``adjustments_per_year`` times a year.

    With ``P = 12 // adjustments_per_year`` months between adjustments, the amount holds its
    nominal value for ``m = 0 .. P-1`` months, worth ``(1 + inflation_rate) ** (-m / 12)`` in
    real terms each month; this returns the mean over the cycle. Takes no month or year: an
    indexed benefit is a real-dollar constant, just not its published amount. Compute once per
    schedule per scenario.

    Args:
        inflation_rate: Assumed annual inflation as a bare fraction.
        adjustments_per_year: Times a year the amount is adjusted — four for quarterly, one for
            annual — from ``params/`` via :func:`schedule`.

    Returns:
        A multiplier in ``(0, 1]``; exactly 1 at zero inflation or at ``adjustments_per_year ==
        12``, since a monthly-adjusted amount never spends a month eroding.

    Raises:
        ValueError: If ``adjustments_per_year`` is not a positive divisor of twelve, or
            ``inflation_rate`` is at or below -1.
    """
    if adjustments_per_year < 1 or MONTHS_PER_YEAR % adjustments_per_year:
        raise ValueError(
            f"adjustments_per_year must be a positive divisor of {MONTHS_PER_YEAR}, "
            f"got {adjustments_per_year!r}. An amount adjusted on any other cadence "
            f"cannot be described by a whole number of months between adjustments."
        )
    _check_inflation(inflation_rate)
    period = MONTHS_PER_YEAR // adjustments_per_year
    total = sum((1 + inflation_rate) ** (-month / MONTHS_PER_YEAR) for month in range(period))
    return total / period


def unindexed_factor(inflation_rate: float, month_index: int) -> float:
    """Real value of an amount fixed in nominal terms since the start of the run.

    Unlike :func:`erosion_factor` this depends on time, and must: an indexed
    amount's real loss is bounded by one adjustment cycle, while an unindexed
    amount's grows without limit. Approximating this away would be a different
    kind of decision from approximating away the within-cycle oscillation.

    Args:
        inflation_rate: Assumed annual inflation as a bare fraction.
        month_index: Months since January of the scenario's start year, which
            is where the amount's nominal value was fixed. Zero in January of
            the start year.

    Returns:
        A multiplier in ``(0, 1]``, falling monotonically in ``month_index``.

    Raises:
        ValueError: If ``month_index`` is negative, or if ``inflation_rate`` is
            at or below -1.
    """
    if month_index < 0:
        raise ValueError(
            f"month_index counts forward from January of the start year and cannot be "
            f"negative, got {month_index!r}."
        )
    _check_inflation(inflation_rate)
    return (1 + inflation_rate) ** (-month_index / MONTHS_PER_YEAR)


def nominal_carry_factor(inflation_rate: float) -> float:
    """One January's decay for a balance the *state* carries, fixed in nominal terms.

    For a dollar figure a parameter file routes as unindexed, ``RealParamSet`` applies
    :func:`unindexed_factor` on read. A state balance has no ``indexation`` block to route it, so
    a caller carrying one across a January is the one that must know it is nominal and owes it
    this factor; ``tests/core/test_state_nominal_or_real.py`` is where that classification, field
    by field, is made explicit.

    Takes the inflation rate only, no month index: the ratio between two consecutive Januaries
    does not depend on which two.

    Args:
        inflation_rate: Assumed annual inflation as a bare fraction.

    Returns:
        A multiplier, exactly 1 at zero inflation, below 1 while prices rise, above 1 under
        deflation.

    Raises:
        ValueError: If ``inflation_rate`` is at or below -1.
    """
    return unindexed_factor(inflation_rate, MONTHS_PER_YEAR)


def as_filed_to_real_factor(inflation_rate: float, years_before_start: int) -> float:
    """The multiplier that restates an as-filed net-income figure to January of the start year.

    A scenario states a person's net income for a calendar year as it was filed: nominal
    dollars of that year, valued at the year's mean (L5's "annual amounts are valued at the
    year's mean"). It is the one scenario figure not already stated in real dollars, so
    :func:`build_initial_state <engine.core.build.build_initial_state>` restates it once, at
    build. ``years_before_start == 1`` is the prior year, valued at its mean, half a year
    before the start year's January — hence the exponent ``years_before_start - 0.5``:
    ``0.5`` for the prior year, ``1.5`` for the year before that. Write the half-year as
    ``0.5``; it is a convention, not a tax constant.

    Args:
        inflation_rate: Assumed annual inflation as a bare fraction.
        years_before_start: How many calendar years before the scenario's start year the figure
            was filed for. At least 1: the filed figures are always for a year strictly before
            the start year.

    Returns:
        A multiplier, exactly 1 at zero inflation.

    Raises:
        ValueError: If ``years_before_start`` is less than 1, or ``inflation_rate`` is at or
            below -1.
    """
    if years_before_start < 1:
        raise ValueError(
            f"years_before_start must be at least 1: the filed figures are always for a year "
            f"strictly before the scenario's start year, got {years_before_start!r}."
        )
    _check_inflation(inflation_rate)
    return (1 + inflation_rate) ** (years_before_start - 0.5)


def _check_inflation(inflation_rate: float) -> None:
    """Reject a rate that would make ``(1 + rate)`` zero or negative.

    A negative base raised to a fractional power is a complex number in Python,
    not an error, so without this a nonsense assumption would propagate as a
    complex "factor" and surface much later as an unreadable failure.
    """
    if inflation_rate <= -1:
        raise ValueError(
            f"inflation_rate must be greater than -1, got {inflation_rate!r}: at or "
            f"below -1 the price level is zero or negative and the factor is undefined."
        )


def _check_january_index(january_month_index: int) -> None:
    """Reject a month index that is not January of some tax year.

    Guards :meth:`RealParamSet.annual_amount` and
    :meth:`RealParamSet.annual_amounts` against a caller passing December, or
    any other month.
    """
    if january_month_index < 0 or january_month_index % MONTHS_PER_YEAR:
        raise ValueError(
            f"january_month_index must be January of the tax year being assessed, "
            f"{MONTHS_PER_YEAR} * (year - start_year) for a non-negative integer year "
            f"offset, got {january_month_index!r}."
        )


def schedule(name: str, params: ParamSet) -> tuple[int, tuple[str, ...]]:
    """Look up one indexation schedule by the name ``params/`` gives it.

    ``params/`` records the *months* an amount is adjusted in, because that is
    the verifiable statutory fact and it is what a human checks against a
    source. This returns their count, because with the oscillation dropped the
    count is all the model uses — which month of the quarter an adjustment
    lands in no longer changes any result.

    Args:
        name: The schedule, as its file names it, e.g. ``"brackets_and_credits"``.
        params: The parameter set that declares it.

    Returns:
        ``(adjustments_per_year, applies_to)``. Zero adjustments means the
        amounts are fixed in nominal terms and decay under
        :func:`unindexed_factor`.

    Raises:
        MissingParameterError: If the schedule, its ``adjustment_months``, or
            its ``applies_to`` is absent. An amount whose indexation rule has
            not been verified is not indexed "annually by default" — it is
            unknown, and the run stops.
        MalformedParamFileError: If either key is present but not a list.
    """
    where = f"{INDEXATION_BLOCK}{PATH_SEP}{name}"
    months = params.sequence(f"{where}{PATH_SEP}adjustment_months")
    applies_to = params.sequence(f"{where}{PATH_SEP}applies_to")
    for index, entry in enumerate(applies_to):
        if not isinstance(entry, str):
            raise MalformedParamFileError(
                f"{params.source}[{where}.applies_to]: element {index} is "
                f"{type(entry).__name__} ({entry!r}), expected a dotted path."
            )
    return len(months), tuple(applies_to)


def _segments(path: str) -> tuple[str, ...]:
    return tuple(path.split(PATH_SEP))


def _matches(pattern: str, path: str) -> bool:
    """Whether ``path`` is the routed ``pattern`` or one concrete case of it.

    Both directions matter. A caller reading a whole table passes the pattern
    itself (``pension.age_bands.*.maximum_monthly``); a caller that has already
    resolved one element passes a concrete path (``...age_bands.0...``). If
    only the first were treated as routed, the second would slip through the
    raw accessors undeflated, which is the exact failure the view exists to
    make impossible.
    """
    left, right = _segments(pattern), _segments(path)
    if len(left) != len(right):
        return False
    return all(a in (WILDCARD, b) for a, b in zip(left, right, strict=True))


def _routes(params: ParamSet) -> Mapping[str, int]:
    """Every routed path in a file, mapped to its schedule's adjustments per year.

    Raises:
        MalformedParamFileError: If a path is routed by two schedules. The
            structural test forbids it, but picking one silently here would
            turn a file the human can fix into a wrong number nobody sees.
    """
    if not params.has(INDEXATION_BLOCK):
        return {}
    block = params.get(INDEXATION_BLOCK)
    if not isinstance(block, Mapping):
        raise MalformedParamFileError(
            f"{params.source}[{INDEXATION_BLOCK}]: expected a mapping of schedule name "
            f"to schedule, found {type(block).__name__}."
        )
    routes: dict[str, int] = {}
    owner: dict[str, str] = {}
    for name in block:
        adjustments, applies_to = schedule(name, params)
        for path in applies_to:
            if path in routes:
                raise MalformedParamFileError(
                    f"{params.source}: {path!r} is routed by both {owner[path]!r} and "
                    f"{name!r}. An amount sits on exactly one schedule."
                )
            routes[path] = adjustments
            owner[path] = name
    return routes


def _expand(values: Any, pattern: str) -> tuple[tuple[str, Any], ...]:
    """Every ``(concrete path, value)`` a ``*`` pattern names, in table order.

    Walks the tree itself rather than resolving to dotted paths and handing
    them back to the loader, because the loader cannot traverse a list: its
    ``get`` reaches values through mapping keys only, so
    ``pension.age_bands.0.maximum_monthly`` is not a path it can read. The
    concrete path comes back alongside each value anyway, so an element that
    turns out not to be a number can be named in the error.
    """
    resolved: list[list[str]] = [[]]
    cursors: list[Any] = [values]
    for segment in _segments(pattern):
        next_resolved: list[list[str]] = []
        next_cursors: list[Any] = []
        for prefix, cursor in zip(resolved, cursors, strict=True):
            if segment == WILDCARD:
                if not isinstance(cursor, tuple):
                    continue
                for index, item in enumerate(cursor):
                    next_resolved.append([*prefix, str(index)])
                    next_cursors.append(item)
            elif isinstance(cursor, Mapping) and segment in cursor:
                next_resolved.append([*prefix, segment])
                next_cursors.append(cursor[segment])
        resolved, cursors = next_resolved, next_cursors
    return tuple(
        (PATH_SEP.join(parts), value) for parts, value in zip(resolved, cursors, strict=True)
    )


@dataclass(frozen=True, slots=True)
class RealParamSet:
    """One parameter file, read in real dollars of the scenario's start month.

    Wraps a :class:`~engine.params.loader.ParamSet` with the scenario's
    inflation rate and applies each amount's schedule on the way out. The
    pass-through accessors keep the loader's names so that a call site reads
    the same, and refuse routed paths so that keeping the same name cannot mean
    keeping the old, undeflated behaviour.

    Attributes:
        raw: The undeflated file. Public because tests and diagnostics
            legitimately need the published figure; engine code that reaches
            through it to a dollar amount is doing by hand the thing this class
            exists to prevent.
        inflation_rate: Assumed annual inflation as a bare fraction. A scenario
            assumption, not a tax parameter.
        routes: Routed path to its schedule's adjustments per year, built once
            from the file's ``indexation`` block.
    """

    raw: ParamSet
    inflation_rate: float
    routes: Mapping[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _check_inflation(self.inflation_rate)
        object.__setattr__(self, "routes", _routes(self.raw))

    @property
    def name(self) -> str:
        """The file's stem, e.g. ``"federal"``."""
        return self.raw.name

    @property
    def year(self) -> int:
        """The tax year the file belongs to."""
        return self.raw.year

    def amount(self, path: str, month_index: int) -> float:
        """Return one dollar amount in real terms, or raise.

        For an annual quantity use :meth:`annual_amount`.

        Args:
            path: Dotted path to a scalar. A path holding a ``*`` names many
                amounts and belongs to :meth:`amounts`.
            month_index: Months since January of the scenario's start year.
                Ignored for an indexed schedule, whose factor is constant, and
                honoured for an unindexed one, whose decay is not.

        Raises:
            UnroutedParameterError: If no schedule routes ``path``.
            MissingParameterError: If the parameter is absent.
            MalformedParamFileError: If it is present but not numeric.
            ValueError: If ``path`` holds a wildcard.
        """
        if WILDCARD in _segments(path):
            raise ValueError(
                f"{path!r} names every element of a table, not one amount. Use amounts() for it."
            )
        return self.raw.number(path) * self._factor(path, month_index)

    def amounts(self, path: str, month_index: int) -> tuple[float, ...]:
        """Return a table of dollar amounts in real terms, or raise.

        For an annual quantity use :meth:`annual_amount`.

        Covers both shapes a routed table takes: a list-valued parameter such
        as ``brackets.edges_annual``, and a ``*`` path naming one scalar per
        element of a list of mappings, such as
        ``pension.age_bands.*.maximum_monthly``. In table order either way.

        Raises:
            UnroutedParameterError: If no schedule routes ``path``.
            MissingParameterError: If the parameter is absent, or if a ``*``
                pattern matches nothing at all. An empty table is not a table
                of no amounts — it is a pattern that has stopped naming the
                file it was written for.
            MalformedParamFileError: If any element is not numeric.
        """
        factor = self._factor(path, month_index)
        return self._table(path, factor)

    def annual_amount(self, path: str, january_month_index: int) -> float:
        """Return one annual dollar amount in real terms, or raise.

        Indexed: ``raw * erosion_factor(rate, k)``. Unindexed: ``raw *
        unindexed_factor(rate, january_month_index) * erosion_factor(rate, 1)``.

        Args:
            path: Dotted path to a scalar annual amount. A path holding a
                ``*`` names every element of a table and belongs to
                :meth:`annual_amounts`.
            january_month_index: Month index of January of the tax year being
                assessed, ``12 * (year - start_year)``.

        Raises:
            ValueError: If ``january_month_index`` is not a non-negative
                multiple of twelve, or if ``path`` holds a wildcard.
            UnroutedParameterError: If no schedule routes ``path``.
            MissingParameterError: If the parameter is absent.
            MalformedParamFileError: If it is present but not numeric.
        """
        _check_january_index(january_month_index)
        if WILDCARD in _segments(path):
            raise ValueError(
                f"{path!r} names every element of a table, not one amount. "
                f"Use annual_amounts() for it."
            )
        return self.raw.number(path) * self._annual_factor(path, january_month_index)

    def annual_amounts(self, path: str, january_month_index: int) -> tuple[float, ...]:
        """Return a table of annual dollar amounts in real terms, or raise.

        Covers both shapes a routed table takes, exactly as :meth:`amounts`
        does, but on the annual rule: ``raw * erosion_factor(rate, k)`` per
        element when indexed, ``raw * unindexed_factor(rate,
        january_month_index) * erosion_factor(rate, 1)`` per element when not.

        Args:
            path: Dotted path, possibly holding a ``*``.
            january_month_index: Month index of January of the tax year being
                assessed, ``12 * (year - start_year)``.

        Raises:
            ValueError: If ``january_month_index`` is not a non-negative
                multiple of twelve.
            UnroutedParameterError: If no schedule routes ``path``.
            MissingParameterError: If the parameter is absent, or if a ``*``
                pattern matches nothing at all.
            MalformedParamFileError: If any element is not numeric.
        """
        _check_january_index(january_month_index)
        factor = self._annual_factor(path, january_month_index)
        return self._table(path, factor)

    def _table(self, path: str, factor: float) -> tuple[float, ...]:
        """Expand a routed table ``path`` to a tuple of amounts, in table order.

        Shared by :meth:`amounts` and :meth:`annual_amounts`, which differ
        only in how ``factor`` was computed; this does not know which.

        Raises:
            MissingParameterError: If the parameter is absent, or if a ``*``
                pattern matches nothing at all.
            MalformedParamFileError: If any element is not numeric.
        """
        if WILDCARD not in _segments(path):
            return tuple(value * factor for value in self.raw.numbers(path))

        expanded = _expand(self.raw.values, path)
        if not expanded:
            raise MissingParameterError(
                f"{self.raw.source}[{path}]: this pattern matches nothing in the file. "
                f"Either the table it names is absent, or the pattern in the "
                f"indexation block no longer matches the shape of the file."
            )
        out: list[float] = []
        for concrete, value in expanded:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MalformedParamFileError(
                    f"{self.raw.source}[{concrete}]: expected a number, found "
                    f"{type(value).__name__} ({value!r})."
                )
            out.append(float(value) * factor)
        return tuple(out)

    def get(self, path: str) -> Any:
        """Raw :meth:`~engine.params.loader.ParamSet.get`, refusing routed paths."""
        self._refuse_if_routed(path, "get")
        return self.raw.get(path)

    def number(self, path: str) -> float:
        """Raw :meth:`~engine.params.loader.ParamSet.number`, refusing routed paths."""
        self._refuse_if_routed(path, "number")
        return self.raw.number(path)

    def numbers(self, path: str) -> tuple[float, ...]:
        """Raw :meth:`~engine.params.loader.ParamSet.numbers`, refusing routed paths."""
        self._refuse_if_routed(path, "numbers")
        return self.raw.numbers(path)

    def sequence(self, path: str) -> tuple[Any, ...]:
        """Raw :meth:`~engine.params.loader.ParamSet.sequence`, refusing routed paths."""
        self._refuse_if_routed(path, "sequence")
        return self.raw.sequence(path)

    def has(self, path: str) -> bool:
        """Raw :meth:`~engine.params.loader.ParamSet.has`, refusing routed paths.

        A presence check on a routed path is refused rather than answered
        ``True``, because the only thing a caller does with the answer is read
        the value, and this is the last point at which that read can be
        redirected to :meth:`amount`.
        """
        self._refuse_if_routed(path, "has")
        return self.raw.has(path)

    def is_stub(self) -> bool:
        """Whether the file exists but holds no values yet."""
        return self.raw.is_stub()

    def _adjustments(self, path: str) -> int:
        for pattern, adjustments in self.routes.items():
            if _matches(pattern, path):
                return adjustments
        raise UnroutedParameterError(
            f"{self.raw.source}[{path}]: no indexation schedule routes this path, so "
            f"there is no way to state it in real dollars. Add it to an applies_to "
            f"list in {self.raw.source.name} by hand, under a source comment. It is "
            f"never correct to assume a cadence here."
        )

    def _factor(self, path: str, month_index: int) -> float:
        adjustments = self._adjustments(path)
        if adjustments == 0:
            return unindexed_factor(self.inflation_rate, month_index)
        return erosion_factor(self.inflation_rate, adjustments)

    def _annual_factor(self, path: str, january_month_index: int) -> float:
        adjustments = self._adjustments(path)
        if adjustments == 0:
            return unindexed_factor(self.inflation_rate, january_month_index) * erosion_factor(
                self.inflation_rate, 1
            )
        return erosion_factor(self.inflation_rate, adjustments)

    def _refuse_if_routed(self, path: str, accessor: str) -> None:
        for pattern in self.routes:
            if _matches(pattern, path):
                raise RoutedParameterError(
                    f"{self.raw.source}[{path}]: this is a dollar amount on the "
                    f"{pattern!r} indexation schedule, and .{accessor}() would return "
                    f"it undeflated. Use .amount()/.amounts() for a monthly quantity "
                    f"or .annual_amount()/.annual_amounts() for an annual one."
                )


@dataclass(frozen=True, slots=True)
class RealParamYear:
    """Every parameter file for one tax year, read in real dollars.

    Mirrors :class:`~engine.params.loader.ParamYear` accessor for accessor, so
    that switching a call site to the real view is a change of the object it
    was handed rather than a rewrite. Every accessor returns a
    :class:`RealParamSet`.

    Attributes:
        raw: The undeflated year.
        inflation_rate: Assumed annual inflation as a bare fraction.
    """

    raw: ParamYear
    inflation_rate: float
    _sets: dict[str, RealParamSet] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _check_inflation(self.inflation_rate)
        object.__setattr__(self, "_sets", {})

    @property
    def year(self) -> int:
        """The tax year, e.g. ``2026``."""
        return self.raw.year

    def __getitem__(self, name: str) -> RealParamSet:
        """Return the named parameter set in real dollars, or raise.

        Raises:
            ParamFileMissingError: If no such file exists for this year.
        """
        if name not in self._sets:
            self._sets[name] = RealParamSet(self.raw[name], self.inflation_rate)
        return self._sets[name]

    def __contains__(self, name: object) -> bool:
        return name in self.raw

    def names(self) -> tuple[str, ...]:
        """Names of the parameter sets loaded for this year, sorted."""
        return self.raw.names()

    @property
    def federal(self) -> RealParamSet:
        """Federal income tax parameters."""
        return self["federal"]

    @property
    def cpp(self) -> RealParamSet:
        """Canada Pension Plan parameters."""
        return self["cpp"]

    @property
    def oas(self) -> RealParamSet:
        """Old Age Security parameters."""
        return self["oas"]

    @property
    def rrif(self) -> RealParamSet:
        """RRSP and RRIF parameters."""
        return self["rrif"]

    @property
    def tfsa(self) -> RealParamSet:
        """TFSA contribution room and recontribution rules."""
        return self["tfsa"]

    @property
    def resp(self) -> RealParamSet:
        """RESP contribution, grant, and Educational Assistance Payment rules."""
        return self["resp"]

    def province(self, code: str) -> RealParamSet:
        """Provincial income tax parameters for the province of residence."""
        return self[code.lower()]

    def jurisdiction(self, code: str) -> RealParamSet:
        """Pension parameters for the jurisdiction a locked-in account is registered in.

        Not interchangeable with :meth:`province`; see
        :meth:`engine.params.loader.ParamYear.jurisdiction`.
        """
        return self[code.lower()]


def real_year(params: ParamYear, inflation_rate: float) -> RealParamYear:
    """Wrap a loaded tax year in the scenario's real-terms view.

    Args:
        params: The year as loaded from disk.
        inflation_rate: Assumed annual inflation as a bare fraction.

    Raises:
        ValueError: If ``inflation_rate`` is at or below -1.
    """
    return RealParamYear(params, inflation_rate)

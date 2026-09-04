"""Structural invariants over the real parameter files under ``params/``.

Not golden values. Nothing here asserts that a bracket edge is 58523 — that is
what the inline source URL and check date next to the value are for, and a
second copy of the number in a test would only be a third place to keep in
sync. What these assert is the *shape* the engine relies on, and the internal
agreements between values that a transcription slip breaks:

    a rate table one entry longer than its edge table; a terminal age that
    actually names the last row of the table above it; a factor table with no
    gap in the middle; a rate written as 15 where the loader expects 0.15.

Every check here is derivable from the files themselves. A test that needed to
know a tax value to run would be inventing a parameter, which is the thing this
repository most forbids.

These run against ``DEFAULT_PARAMS_ROOT``, not a fixture. That is deliberate:
their whole purpose is to fail on the files that ship, and a mistake in the
2026 files is invisible to a test that builds its own YAML in ``tmp_path``.

Two of these have already earned their keep. ``terminal_age_years`` was 90 in
``rrif.yaml`` against a table running to 95, silently capping a 95-year-old's
minimum withdrawal at the age-90 factor; and ``increment_rate_per_month`` was
0.06 in ``oas.yaml`` against a comment reading "6% per month", inflating
deferred OAS roughly six-fold.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import pairwise
from typing import Any

import pytest

from engine.params.loader import DEFAULT_PARAMS_ROOT, ParamSet, ParamYear, load_year

#: Parameter sets that are federal programs. Anything else in a year directory
#: is a province or pension jurisdiction and must satisfy the province contract.
FEDERAL_SETS = frozenset({"cpp", "federal", "oas", "resp", "rrif", "tfsa"})

#: Every parameter set holding a progressive rate table, and the path to it.
BRACKET_TABLES = "brackets.edges_annual", "brackets.rates"

#: Age-keyed factor tables: (set predicate, table path, terminal age path).
AGE_TABLES = (
    ("rrif", "rrif.minimum_factors.by_age", "rrif.minimum_factors.terminal_age_years"),
    (None, "lif.maximum_factors.by_age", "lif.maximum_factors.terminal_age_years"),
)


def _years() -> list[int]:
    return sorted(
        int(path.name)
        for path in DEFAULT_PARAMS_ROOT.iterdir()
        if path.is_dir() and path.name.isdigit()
    )


def _leaves(node: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Every scalar in a parameter tree, with its dotted path."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield from _leaves(value, f"{prefix}{key}.")
    elif isinstance(node, (tuple, list)):
        for index, value in enumerate(node):
            yield from _leaves(value, f"{prefix}{index}.")
    else:
        yield prefix.rstrip("."), node


def _stem(path: str) -> str:
    """The key a value belongs to, ignoring list indices.

    ``_leaves`` numbers sequence elements, so a rate inside ``brackets.rates``
    arrives as ``brackets.rates.0``. Naming checks have to look past the index
    or every list-valued parameter escapes them silently — which is exactly how
    an entire bracket table written in percentages slipped past the units check
    the first time this file was run.
    """
    while path.rsplit(".", 1)[-1].isdigit() and "." in path:
        path = path.rsplit(".", 1)[0]
    return path.rsplit(".", 1)[-1]


YEARS = _years()


@pytest.fixture(scope="module", params=YEARS, ids=[str(year) for year in YEARS])
def year(request: pytest.FixtureRequest) -> ParamYear:
    return load_year(request.param, DEFAULT_PARAMS_ROOT)


def _sets_with(year: ParamYear, path: str) -> list[tuple[str, ParamSet]]:
    return [(name, year[name]) for name in year.names() if year[name].has(path)]


# --- Rate tables ------------------------------------------------------------


def test_bracket_tables_have_one_more_rate_than_edge(year: ParamYear) -> None:
    """``len(rates) == len(edges_annual) + 1``, the invariant every file states.

    The last rate is the unbounded top bracket and has no edge. A file with as
    many rates as edges has either lost the top bracket or gained a spurious
    edge, and the branch-free clipping in ``engine.tax.federal.gross_tax``
    would read off the end or silently drop the top rate rather than raising.
    """
    tables = _sets_with(year, "brackets.rates")
    assert tables, "No parameter set defines a rate table."
    for name, params in tables:
        edges = params.numbers("brackets.edges_annual")
        rates = params.numbers("brackets.rates")
        assert len(rates) == len(edges) + 1, (
            f"{name}: {len(rates)} rates against {len(edges)} edges. A flat-rate "
            f"jurisdiction has an empty edge list and one rate, which is valid; "
            f"any other mismatch is a transcription error."
        )


def test_bracket_edges_strictly_ascend(year: ParamYear) -> None:
    """Edges ascend and are positive, so clipping into them is well defined."""
    for name, params in _sets_with(year, "brackets.edges_annual"):
        edges = params.numbers("brackets.edges_annual")
        assert all(edge > 0 for edge in edges), f"{name}: non-positive bracket edge in {edges}."
        assert all(lower < upper for lower, upper in pairwise(edges)), (
            f"{name}: bracket edges are not strictly ascending: {edges}."
        )


def test_bracket_rates_do_not_decrease(year: ParamYear) -> None:
    """A progressive table never steps down.

    Non-decreasing rather than strictly increasing: two adjacent equal rates
    are legal, if pointless. A rate that *falls* as income rises is a swapped
    pair, and would make marginal tax non-monotonic — which the optimizer would
    find and exploit.
    """
    for name, params in _sets_with(year, "brackets.rates"):
        rates = params.numbers("brackets.rates")
        assert all(lower <= upper for lower, upper in pairwise(rates)), (
            f"{name}: bracket rates decrease somewhere in {rates}."
        )


def test_credit_valuation_rate_is_not_above_the_lowest_bracket_rate(year: ParamYear) -> None:
    """Credits are valued at the lowest rate, never above it.

    ``valuation_rate`` is deliberately stored separately from ``rates[0]``
    because the two are distinct legal rules that need not move together. This
    checks the one relationship that does hold regardless: a non-refundable
    credit is not worth more per dollar than the bottom bracket.
    """
    for name, params in _sets_with(year, "credits.valuation_rate"):
        valuation = params.number("credits.valuation_rate")
        lowest = params.numbers("brackets.rates")[0]
        assert valuation <= lowest, (
            f"{name}: credits valued at {valuation} against a lowest bracket rate "
            f"of {lowest}."
        )


# --- Age-keyed factor tables ------------------------------------------------


@pytest.mark.parametrize(("only_set", "table_path", "terminal_path"), AGE_TABLES)
def test_age_factor_tables_are_contiguous_and_monotonic(
    year: ParamYear, only_set: str | None, table_path: str, terminal_path: str
) -> None:
    """One row per age, no gaps, and factors that never fall.

    A missing age in the middle of a prescribed table is the failure this
    catches: the lookup would fall through to whatever the implementation does
    with an absent key, and the household would draw a factor belonging to a
    different age for one year of the run.

    Note the key type. ``_freeze`` casts every mapping key to ``str`` so that
    dotted-path lookup works uniformly, so an age table written ``71:`` in YAML
    arrives as ``"71"``. Consumers must index it with ``str(age)``; an ``int``
    age raises ``KeyError`` against a table that plainly contains that age.
    This asserts the keys are digit strings so the contract is written down
    somewhere other than the loader's implementation.
    """
    candidates = _sets_with(year, table_path)
    if only_set is not None:
        candidates = [(name, params) for name, params in candidates if name == only_set]
    assert candidates, f"No parameter set defines {table_path}."

    for name, params in candidates:
        table = params.get(table_path)
        assert all(key.isdigit() for key in table), (
            f"{name}.{table_path}: ages must be whole-number keys, got {sorted(table)}."
        )
        ages = sorted(int(key) for key in table)
        assert ages == list(range(ages[0], ages[-1] + 1)), (
            f"{name}.{table_path}: ages {ages[0]}..{ages[-1]} have gaps."
        )
        factors = [float(table[str(age)]) for age in ages]
        assert all(0 < factor <= 1 for factor in factors), (
            f"{name}.{table_path}: factors outside (0, 1] — a percentage written "
            f"where a fraction was expected? {factors}"
        )
        assert all(lower <= upper for lower, upper in pairwise(factors)), (
            f"{name}.{table_path}: factors decrease with age."
        )


@pytest.mark.parametrize(("only_set", "table_path", "terminal_path"), AGE_TABLES)
def test_terminal_age_names_the_last_row_of_its_table(
    year: ParamYear, only_set: str | None, table_path: str, terminal_path: str
) -> None:
    """``terminal_age_years`` is the highest age in the table above it.

    Set below the last row, every row past it becomes unreachable and the
    factor is frozen early — which understates the RRIF minimum and the LIF
    maximum in exactly the late years where they are largest. Set above it, the
    lookup runs off the end of the table.
    """
    candidates = _sets_with(year, table_path)
    if only_set is not None:
        candidates = [(name, params) for name, params in candidates if name == only_set]

    for name, params in candidates:
        highest = max(int(key) for key in params.get(table_path))
        terminal = params.number(terminal_path)
        assert terminal == highest, (
            f"{name}: {terminal_path} is {terminal:g} but the table runs to {highest}. "
            f"Rows above {terminal:g} are unreachable."
        )


def test_lif_tables_start_at_the_unlocking_age(year: ParamYear) -> None:
    """The first age with a maximum is the first age a withdrawal is permitted.

    Two independently sourced values that have to agree: a table starting above
    the unlocking age leaves permitted years with no ceiling, and one starting
    below it prices years in which nothing may be withdrawn.
    """
    for name, params in _sets_with(year, "lif.maximum_factors.by_age"):
        lowest = min(int(key) for key in params.get("lif.maximum_factors.by_age"))
        assert lowest == params.number("lif.unlocking_age_years"), (
            f"{name}: LIF table starts at {lowest} but unlocking_age_years is "
            f"{params.number('lif.unlocking_age_years'):g}."
        )


def test_lif_has_maximum_is_a_boolean_and_agrees_with_the_table(year: ParamYear) -> None:
    """"No maximum" and "no data" stay distinguishable.

    ``has_maximum`` exists so that an uncapped jurisdiction is expressible
    without the loader raising on an absent table. That only works if the flag
    is a real boolean and matches whether a table is present.
    """
    for name, params in _sets_with(year, "lif.has_maximum"):
        has_maximum = params.get("lif.has_maximum")
        assert isinstance(has_maximum, bool), (
            f"{name}: lif.has_maximum is {has_maximum!r}, not a boolean."
        )
        assert has_maximum == params.has("lif.maximum_factors.by_age"), (
            f"{name}: lif.has_maximum is {has_maximum} but a maximum table is "
            f"{'absent' if has_maximum else 'present'}."
        )


# --- Cross-value agreements -------------------------------------------------


def test_cpp_annual_maximum_is_twelve_times_the_monthly(year: ParamYear) -> None:
    """The two published CPP maxima agree.

    Both are transcribed from the source rather than derived, precisely so that
    neither is computed in a ``.py`` file. Holding them side by side means a
    slip in either one shows up here instead of as a factor-of-twelve error in
    a benefit stream.
    """
    cpp = year["cpp"]
    monthly = cpp.number("pension.maximum_monthly_at_standard_age")
    annual = cpp.number("pension.maximum_annual_at_standard_age")
    assert annual == pytest.approx(monthly * 12, abs=0.01), (
        f"cpp: {annual} annual against {monthly} monthly (12x = {monthly * 12})."
    )


def test_oas_deferral_window_matches_the_start_age_range(year: ParamYear) -> None:
    """``maximum_months`` is the span between the earliest and latest start ages."""
    oas = year["oas"]
    span = oas.number("start_age.latest_months") - oas.number("start_age.earliest_months")
    assert oas.number("deferral.maximum_months") == span, (
        f"oas: deferral.maximum_months is {oas.number('deferral.maximum_months'):g} "
        f"but the start-age range spans {span:g} months."
    )


def test_oas_age_bands_ascend_from_the_earliest_start_age(year: ParamYear) -> None:
    """Bands ascend, and the first begins where entitlement begins.

    A gap between the earliest start age and the first band is an age range
    with no maximum defined for it.
    """
    oas = year["oas"]
    bands = oas.sequence("pension.age_bands")
    starts = [float(band["from_age_months"]) for band in bands]
    assert starts[0] == oas.number("start_age.earliest_months"), (
        f"oas: first age band starts at {starts[0]:g}, entitlement at "
        f"{oas.number('start_age.earliest_months'):g}."
    )
    assert all(lower < upper for lower, upper in pairwise(starts)), (
        f"oas: age bands are not ascending: {starts}."
    )
    assert all(float(band["maximum_monthly"]) > 0 for band in bands), (
        "oas: an age band has a non-positive maximum."
    )


def test_resp_enhanced_grant_table_is_a_descending_rate_table(year: ParamYear) -> None:
    """One more rate than cut-off, cut-offs ascending, and rates that never rise.

    The direction is the point. This is the one rate table in the repository
    laid out like a bracket table but running the other way — the highest match
    goes to the lowest income — so the ordering check that protects the tax
    tables would pass a table entered backwards here.
    """
    resp = year["resp"]
    edges = resp.numbers("grant.enhanced.income_edges_annual")
    rates = resp.numbers("grant.enhanced.match_rates")
    assert len(rates) == len(edges) + 1, (
        f"resp: {len(rates)} enhanced match rates against {len(edges)} cut-offs."
    )
    assert all(lower < upper for lower, upper in pairwise(edges)), (
        f"resp: enhanced income cut-offs are not ascending: {edges}."
    )
    assert all(upper <= lower for lower, upper in pairwise(rates)), (
        f"resp: enhanced match rates must not rise with income: {rates}. "
        f"Entered in tax-bracket order by mistake?"
    )


# --- Calendar rules ---------------------------------------------------------


def test_every_month_field_is_a_real_month(year: ParamYear) -> None:
    """Any key naming a single month is in 1..12.

    Months are the one field a placeholder cannot make look absurd, so they get
    a check of their own rather than relying on a reviewer's eye.

    ``*_per_month`` is excluded: those are rates *per* month, not months.
    """
    for name in year.names():
        for path, value in _leaves(year[name].values):
            leaf = path.rsplit(".", 1)[-1]
            if leaf.endswith("_month") and not leaf.endswith("_per_month"):
                assert isinstance(value, int) and 1 <= value <= 12, (
                    f"{name}.{path} is {value!r}, not a month in 1..12."
                )


def test_indexation_schedules_are_well_formed(year: ParamYear) -> None:
    """Adjustment months are real, distinct, and divide the year evenly.

    ``engine.core.indexation.schedule`` returns the *count* of these months and
    ``real_factor`` requires it to divide twelve. An empty list is valid and
    means the amount is never adjusted — a different rule from the key being
    absent, and routed to ``unindexed_real_factor`` instead.
    """
    for name in year.names():
        schedules = year[name].get("indexation") if year[name].has("indexation") else {}
        for schedule_name, schedule in schedules.items():
            where = f"{name}.indexation.{schedule_name}"
            months = schedule["adjustment_months"]
            assert all(isinstance(month, int) and 1 <= month <= 12 for month in months), (
                f"{where}.adjustment_months holds a non-month: {months}."
            )
            assert len(set(months)) == len(months), (
                f"{where}.adjustment_months repeats a month: {months}."
            )
            assert not months or 12 % len(months) == 0, (
                f"{where} adjusts {len(months)} times a year, which does not divide 12."
            )
            assert schedule["cpi_lag_months"] >= 0, (
                f"{where}.cpi_lag_months is negative."
            )


# --- Units ------------------------------------------------------------------


def test_every_rate_is_a_fraction_not_a_percentage(year: ParamYear) -> None:
    """Anything named a rate is in [0, 1].

    The single most likely units error in these files, and the one that is
    hardest to see: 15 and 0.15 both look like reasonable things to write next
    to the word "rate", and only one of them is what every consumer expects.
    """
    for name in year.names():
        for path, value in _leaves(year[name].values):
            leaf = _stem(path)
            if leaf in {"rate", "rates"} or leaf.endswith(("_rate", "_rates")):
                assert isinstance(value, (int, float)), f"{name}.{path} is not numeric."
                assert 0 <= value <= 1, (
                    f"{name}.{path} is {value}, outside [0, 1]. Rates are bare "
                    f"fractions here — 0.15, never 15 or 1500."
                )


def test_period_suffixed_amounts_are_positive(year: ParamYear) -> None:
    """A key that names a period names a dollar amount, and dollars here are positive.

    Catches a sign slip and a value that never got filled in. Offsets and lags
    may legitimately be negative or zero and do not carry these suffixes.
    """
    for name in year.names():
        for path, value in _leaves(year[name].values):
            leaf = _stem(path)
            if leaf.endswith(("_annual", "_monthly")):
                assert isinstance(value, (int, float)), f"{name}.{path} is not numeric."
                assert value > 0, f"{name}.{path} is {value}, not a positive amount."


# --- The province contract --------------------------------------------------


def test_every_province_file_has_the_same_shape(year: ParamYear) -> None:
    """Provinces are interchangeable to the engine, so their key sets must match.

    ``engine.tax.provincial`` looks up the same paths in every province file
    rather than branching on the code, which is what makes adding a province a
    copy of the template instead of new code. With one province this is
    vacuous; it starts doing work the moment there is a second, which is
    exactly when the omission would otherwise be found at runtime.
    """
    provinces = [name for name in year.names() if name not in FEDERAL_SETS]
    assert provinces, "No province file in this parameter year."
    reference = provinces[0]
    expected = {path for path, _ in _leaves(year[reference].values)}
    for name in provinces[1:]:
        actual = {path for path, _ in _leaves(year[name].values)}
        assert actual == expected, (
            f"{name} and {reference} do not define the same keys. "
            f"Only in {name}: {sorted(actual - expected)}. "
            f"Missing from {name}: {sorted(expected - actual)}."
        )


def test_province_files_are_not_federal_programs(year: ParamYear) -> None:
    """Every non-federal set in a year directory is a usable province.

    A draft or a template left in a year directory would be loaded as a
    jurisdiction and could be reached through ``ParamYear.province``. This
    asserts that anything the loader finds there is a real one.
    """
    for name in year.names():
        if name in FEDERAL_SETS:
            continue
        assert year[name].has("brackets.rates"), (
            f"{name!r} is in a year directory but defines no rate table, so it is "
            f"not a province. Drafts and templates belong at the params/ root, "
            f"where load_year does not reach them."
        )

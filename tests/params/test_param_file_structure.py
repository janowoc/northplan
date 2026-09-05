# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

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

The mortality block at the end is the exception, and only because it has no
shipped file to run against yet. Its checks run over the live year like every
other, but they also run over the template at the ``params/`` root and over
deliberately broken copies of it in ``tmp_path`` — the copies exercise the
checks themselves, which is a different job from checking the files.

Two of these have already earned their keep. ``terminal_age_years`` was 90 in
``rrif.yaml`` against a table running to 95, silently capping a 95-year-old's
minimum withdrawal at the age-90 factor; and ``increment_rate_per_month`` was
0.06 in ``oas.yaml`` against a comment reading "6% per month", inflating
deferred OAS roughly six-fold.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable, Iterator, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import pytest
import yaml

from engine.params.loader import DEFAULT_PARAMS_ROOT, ParamSet, ParamYear, load_year

#: Everything in a year directory that is NOT a province or pension
#: jurisdiction. Anything outside this set must satisfy the province contract
#: below. Most members are federal programs, which is where the name came from;
#: ``mortality`` is not a program at all but a national statistical table, and
#: it belongs here because the only question this set answers is "is this file
#: a jurisdiction?". The name stays as it is — renaming it would reach across
#: files for no behavioural gain.
FEDERAL_SETS = frozenset({"cpp", "federal", "mortality", "oas", "resp", "rrif", "tfsa"})

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

    A rate expressed per month is the same units trap as any other rate, so a
    stem ending ``_rate_per_month`` is included alongside the plain ``_rate``
    and ``_rates`` forms. Without it the CPP start adjustments, the OAS
    deferral increment and the TFSA over-contribution penalty were all
    unchecked.

    Note what this bound does and does not prove. It catches a per-month rate
    written in whole percent — ``6`` where ``0.06`` was meant — because that
    leaves [0, 1]. It does not catch a decimal shift that stays inside the
    range: ``increment_rate_per_month`` was once 0.06 against a comment
    reading "6% per month", and 0.06 and 0.6 both pass here. Bounding a
    per-month rate more tightly is a judgement about plausible magnitudes
    rather than a units check, and does not belong in this test.
    """
    for name in year.names():
        for path, value in _leaves(year[name].values):
            leaf = _stem(path)
            if leaf in {"rate", "rates"} or leaf.endswith(
                ("_rate", "_rates", "_rate_per_month")
            ):
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


# --- Mortality tables -------------------------------------------------------
#
# ``mortality`` is a life table rather than a program: ``q_x`` maps an age to
# the probability of dying before the next birthday, one table per sex. The
# checks below are written as free functions rather than as tests so that the
# same four assertions can be pointed at a live year directory, at the pristine
# template, and at deliberately broken copies of it.
#
# DELIBERATELY ABSENT: a monotonicity check. ``q_x`` does NOT increase with
# age. Infant mortality exceeds childhood mortality, so a real table falls from
# age 0 to roughly age 10 and only rises after that. The non-decreasing
# assertion in ``test_age_factor_tables_are_contiguous_and_monotonic``, which
# is correct for the RRIF and LIF factor tables, would reject the true life
# table. The same warning is in the header of
# ``params/mortality-template.yaml``, because whoever is about to add the check
# will have read only one of the two.

#: The template a real ``params/YYYY/mortality.yaml`` is copied from. It lives
#: at the ``params/`` root, where ``load_year`` cannot reach it.
MORTALITY_TEMPLATE: Final[Path] = DEFAULT_PARAMS_ROOT / "mortality-template.yaml"

#: Scaffolding marker left on every unverified line of a draft parameter file.
#: Spelled by concatenation so that this file does not itself trip the scan.
MARKER: Final[str] = "PLACE" + "HOLDER"

#: The comment carried by the two rows that are true by construction rather
#: than unverified, byte-identical in both sex tables so a text test can key on
#: it.
TERMINAL_ROW_COMMENT: Final[str] = (
    "# certain death at the terminal age, by construction — not a placeholder"
)

#: A line that assigns a number to a key, in a parameter file's raw text.
_ASSIGNS_A_NUMBER = re.compile(r"^\s*\w+:\s*[-+.0-9]")

#: A line that opens one of the two sex tables.
_OPENS_A_SEX_TABLE = re.compile(r"^\s*([fm]):\s*$")

#: A row inside a sex table: an age key with a value.
_IS_AN_AGE_ROW = re.compile(r"^\s+(\d+):\s*(\S+)")

#: ``terminal_age_years`` as the template's own text writes it.
_TERMINAL_AGE_LINE = re.compile(r"^terminal_age_years:\s*(\d+)")


def _terminal_age_in_template(lines: list[str]) -> int:
    """The template's own ``terminal_age_years``, read out of its text.

    Read rather than restated. The row count per sex is one more than this
    number, and writing that count as a literal here would put a value derived
    from a file under ``params/`` into a ``.py`` file — and would invite
    revising the literal, rather than the file, on the day the human's source
    turns out to end at a different age.
    """
    for line in lines:
        found = _TERMINAL_AGE_LINE.match(line)
        if found:
            return int(found.group(1))
    raise AssertionError(f"{MORTALITY_TEMPLATE} declares no terminal_age_years.")


def _check_q_x_has_exactly_the_two_sex_tables(name: str, params: ParamSet) -> None:
    """``q_x`` holds one table per sex the engine will ask for, and no others.

    A third key is a table nothing in the engine will ever read. The other
    checks here do walk it — they iterate whatever ``q_x`` holds — so its rows
    are validated and then ignored, which is the worst of both: the file looks
    thoroughly checked and a third of it is inert. A missing key is louder:
    one spouse's mortality resolves to ``MissingParameterError`` partway
    through a run, after the scenario has already been accepted.
    """
    tables = set(params.get("q_x"))
    assert tables == {"f", "m"}, (
        f"{name}.q_x holds {sorted(tables)}. It must hold exactly ['f', 'm']: an "
        f"extra table is never looked up, and a missing one raises "
        f"MissingParameterError mid-run for the spouse it belongs to."
    )


def _check_ages_are_contiguous_digit_strings_up_to_the_terminal_age(
    name: str, params: ParamSet
) -> None:
    """One row per age, 0 through ``terminal_age_years``, keyed by digit strings.

    The gap in the middle is the failure that matters. Depending on how the
    consumer indexes, a missing age either raises or compounds a hazard
    belonging to a different age for a whole year of the survival curve — and
    the second is invisible in the output.

    Note the key type. ``_freeze`` casts every mapping key to ``str``, so an
    age written ``71:`` in YAML arrives as ``"71"`` and consumers must index
    with ``str(age)``. Asserting the keys are digit strings writes that
    contract down somewhere other than the loader's implementation.
    """
    terminal = int(params.number("terminal_age_years"))
    for sex, table in params.get("q_x").items():
        path = f"{name}.q_x.{sex}"
        assert all(key.isdigit() for key in table), (
            f"{path}: ages must be whole-number keys, got {sorted(table)}."
        )
        ages = sorted(int(key) for key in table)
        expected = list(range(0, terminal + 1))
        assert ages == expected, (
            f"{path}: ages must run 0..{terminal} with no gaps. "
            f"Missing: {sorted(set(expected) - set(ages))}. "
            f"Unexpected: {sorted(set(ages) - set(expected))}."
        )


def _check_every_death_probability_is_in_the_unit_interval(name: str, params: ParamSet) -> None:
    """Every ``q`` is a number in ``(0, 1]``.

    Zero is excluded at the bottom, not included: a ``q`` of exactly 0 is an
    age at which nobody can die, and a run that reaches it has one year of
    guaranteed survival written into the table rather than into a modelling
    decision. Above 1 is the units error — a probability transcribed in percent
    or per mille. Under the monthly hazard the template states,
    ``1 - (1 - q) ** (1/12)``, a ``q`` above 1 raises a negative base to a
    fractional power: NumPy returns ``nan``, so every ``draw < hazard``
    comparison is False and *nobody* dies at that age. The survival curve goes
    flat, not up, and nothing raises.
    """
    for sex, table in params.get("q_x").items():
        for age, q in table.items():
            path = f"{name}.q_x.{sex}.{age}"
            assert not isinstance(q, bool) and isinstance(q, (int, float)), (
                f"{path} is {q!r} ({type(q).__name__}), not a number."
            )
            assert 0 < q <= 1, (
                f"{path} is {q}, outside (0, 1]. Probabilities here are bare "
                f"fractions — 0.01111, never 1.111 percent and never 0."
            )


def _check_only_the_terminal_row_is_certain_death(name: str, params: ParamSet) -> None:
    """``q`` is 1 at ``terminal_age_years`` and nowhere else.

    Below the terminal age a ``q`` of 1 truncates every path at that age with
    nothing in the output to say why — the household simply dies young in every
    scenario. At the terminal age anything below 1 leaves paths alive past the
    end of the table, which is the state ``docs/limitations.md`` L10 forbids:
    the simulation runs to the second death and has no separate horizon to stop
    it.
    """
    terminal = int(params.number("terminal_age_years"))
    for sex, table in params.get("q_x").items():
        path = f"{name}.q_x.{sex}"
        assert str(terminal) in table, (
            f"{path} has no row at the terminal age {terminal}, so no path is "
            f"ever certain to die."
        )
        last = table[str(terminal)]
        assert not isinstance(last, bool) and last == 1.0, (
            f"{path}.{terminal} is {last!r}, not 1.0. The terminal row is death "
            f"by construction; paths surviving it run off the end of the table."
        )
        early = {
            age: q for age, q in table.items() if age != str(terminal) and q == 1.0
        }
        assert not early, (
            f"{path}: q is 1.0 below the terminal age {terminal} at {sorted(early)}. "
            f"Every path would die at that age, silently."
        )


def _check_the_two_sex_tables_are_not_identical(name: str, params: ParamSet) -> None:
    """The ``f`` and ``m`` tables are not the same column pasted twice.

    The one check here that is a heuristic rather than an invariant, and the
    one that catches the likeliest way to get this file wrong. The source is
    filtered to a sex before its column is copied; filtering once and pasting
    twice produces a file in which every value is a real published ``q(x)``,
    every other check passes, the marker scan passes, and the provenance test
    passes — and half the household is modelled with the wrong sex's
    mortality. Nothing downstream looks wrong: the run just loses the sex
    differential in joint survival, which moves the second death and with it
    the estate and the whole drawdown horizon.

    A hundred-odd independently published probabilities do not coincide at
    every age, so equality means duplication rather than coincidence. The
    template ships the two tables with different repdigits precisely so this
    check can exist.
    """
    tables = params.get("q_x")
    if "f" not in tables or "m" not in tables:
        return  # A missing table is the previous check's failure, not this one's.
    assert dict(tables["f"]) != dict(tables["m"]), (
        f"{name}.q_x: the f and m tables are identical at every age. Two "
        f"independently published columns do not coincide everywhere — this is "
        f"one sex's column pasted into both."
    )


#: Every structural check applying to a parameter set named ``mortality``.
MORTALITY_CHECKS: tuple[Callable[[str, ParamSet], None], ...] = (
    _check_q_x_has_exactly_the_two_sex_tables,
    _check_ages_are_contiguous_digit_strings_up_to_the_terminal_age,
    _check_every_death_probability_is_in_the_unit_interval,
    _check_only_the_terminal_row_is_certain_death,
    _check_the_two_sex_tables_are_not_identical,
)

MORTALITY_CHECK_IDS = [check.__name__.lstrip("_") for check in MORTALITY_CHECKS]


@pytest.mark.parametrize("check", MORTALITY_CHECKS, ids=MORTALITY_CHECK_IDS)
def test_live_mortality_tables_satisfy_every_structural_check(
    year: ParamYear, check: Callable[[str, ParamSet], None]
) -> None:
    """Every shipped ``mortality`` set passes every check above.

    The loop is empty today: no year directory holds a mortality file yet, and
    issue 5 is what puts one there. The parametrization is over a static
    four-tuple and is never empty, so ``empty_parameter_set_mark`` is not in
    play and this collects as four passing tests either way.

    This is deliberately NOT written with the ``if CASES:`` guard that
    ``tests/golden/test_cases.py`` uses. That guard exists because its
    parametrize argument is read off disk and can genuinely be empty; here the
    argument is a literal, and copying the guard would only hide the day a
    check starts failing on the real file.

    ISSUE 5 OWES THIS TEST A GUARD. Every other collection-driven test in this
    module asserts it found something — ``assert tables``, ``assert
    candidates``, ``assert provinces``. This one cannot yet, because there is
    no mortality file to find. From the moment there is one, its absence must
    fail rather than pass silently: deleting ``params/2026/mortality.yaml``
    would otherwise turn four named mortality tests green while checking
    nothing. Landing the file, adding ``"mortality"`` to ``EXPECTED_2026_SETS``
    in ``test_loader.py``, and adding ``assert "mortality" in year.names()``
    here are one change, not three.
    """
    for name in year.names():
        if name == "mortality":
            check(name, year[name])


# --- The mortality template -------------------------------------------------


@pytest.fixture
def pristine_mortality_set(tmp_path: Path) -> ParamSet:
    """The committed template, copied byte for byte into a fixture year directory.

    ``shutil.copyfile`` rather than a YAML round trip, so that a syntax error
    in the committed template surfaces here rather than being normalised away
    by a load and dump.
    """
    year_dir = tmp_path / "2030"
    year_dir.mkdir()
    shutil.copyfile(MORTALITY_TEMPLATE, year_dir / "mortality.yaml")
    return load_year(2030, tmp_path)["mortality"]


def _mortality_set_broken_by(tmp_path: Path, mutate: Callable[[dict], None]) -> ParamSet:
    """Load the template, apply ``mutate`` to the plain dict, and reload it as a set."""
    raw = yaml.safe_load(MORTALITY_TEMPLATE.read_text(encoding="utf-8"))
    mutate(raw)
    year_dir = tmp_path / "2030"
    year_dir.mkdir()
    (year_dir / "mortality.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_year(2030, tmp_path)["mortality"]


#: ``(check, description, mutate)``: a break to the template, and the one check
#: that must notice it. A mutation may legitimately trip more than the check
#: named — a third sex table also breaks contiguity — so the test asserts only
#: that the named check fires.
MORTALITY_BREAKAGES: tuple[tuple[Callable[[str, ParamSet], None], str, Callable[[dict], None]], ...] = (
    (
        _check_q_x_has_exactly_the_two_sex_tables,
        "a third sex table",
        lambda raw: raw["q_x"].update({"x": dict(raw["q_x"]["f"])}),
    ),
    (
        _check_q_x_has_exactly_the_two_sex_tables,
        "the male table missing entirely",
        lambda raw: raw["q_x"].pop("m"),
    ),
    (
        _check_ages_are_contiguous_digit_strings_up_to_the_terminal_age,
        "a gap at age 50",
        lambda raw: raw["q_x"]["f"].pop(50),
    ),
    (
        _check_every_death_probability_is_in_the_unit_interval,
        "a probability written in percent",
        lambda raw: raw["q_x"]["m"].update({3: 1.5}),
    ),
    (
        _check_every_death_probability_is_in_the_unit_interval,
        "an age of immortality",
        lambda raw: raw["q_x"]["f"].update({7: 0.0}),
    ),
    (
        _check_only_the_terminal_row_is_certain_death,
        "certain death at 40",
        lambda raw: raw["q_x"]["m"].update({40: 1.0}),
    ),
    (
        _check_only_the_terminal_row_is_certain_death,
        "survival past the terminal age",
        lambda raw: raw["q_x"]["f"].update({raw["terminal_age_years"]: 0.9}),
    ),
    # The three below pin assertions that no other breakage reaches. Without
    # them each could be deleted and the whole suite would still pass, which is
    # the same "never seen to fail" problem the in-step guard exists to catch,
    # one level down. The terminal-row-presence guard is the one with teeth:
    # remove it and a table missing that row raises KeyError on the next line,
    # which `pytest.raises(AssertionError)` would not catch.
    (
        _check_only_the_terminal_row_is_certain_death,
        "no row at the terminal age at all",
        lambda raw: raw["q_x"]["f"].pop(raw["terminal_age_years"]),
    ),
    (
        _check_every_death_probability_is_in_the_unit_interval,
        "a probability written as a YAML boolean",
        lambda raw: raw["q_x"]["m"].update({5: True}),
    ),
    (
        _check_ages_are_contiguous_digit_strings_up_to_the_terminal_age,
        "an age key that is not a whole number",
        lambda raw: raw["q_x"]["f"].update({"7x": 0.11111}),
    ),
    (
        _check_the_two_sex_tables_are_not_identical,
        "one sex's column pasted into both tables",
        lambda raw: raw["q_x"].update({"m": dict(raw["q_x"]["f"])}),
    ),
)


@pytest.mark.parametrize("check", MORTALITY_CHECKS, ids=MORTALITY_CHECK_IDS)
def test_the_pristine_template_satisfies_every_structural_check(
    pristine_mortality_set: ParamSet, check: Callable[[str, ParamSet], None]
) -> None:
    """The template starts from a shape the whole suite already accepts.

    This is what makes issue 5 safe. The human filling in real values starts
    from a layout that has already been proved acceptable, so a layout failure
    afterwards is one they introduced while filling it in rather than one they
    inherited. It says nothing about the edited copy: a deleted row, a mistyped
    age key, or a changed ``terminal_age_years`` are all still layout failures,
    and the live test above is what catches them once issue 5 lands.
    """
    check("mortality", pristine_mortality_set)


@pytest.mark.parametrize(
    ("check", "description", "mutate"),
    MORTALITY_BREAKAGES,
    ids=[description for _, description, _ in MORTALITY_BREAKAGES],
)
def test_a_broken_mortality_table_fails_the_check_that_covers_it(
    tmp_path: Path,
    check: Callable[[str, ParamSet], None],
    description: str,
    mutate: Callable[[dict], None],
) -> None:
    """Each check actually fires on the damage it is there to catch.

    Only the named check is asserted. A mutation often trips others too, and
    pinning down which would make this table brittle for no gain.
    """
    broken = _mortality_set_broken_by(tmp_path, mutate)
    with pytest.raises(AssertionError):
        check("mortality", broken)


def test_every_mortality_check_has_a_breakage_exercising_it() -> None:
    """No check goes unexercised.

    A fifth entry in ``MORTALITY_CHECKS`` with nothing in
    ``MORTALITY_BREAKAGES`` pointed at it is a check that has never been seen
    to fail, which is indistinguishable from a check that cannot fail. This
    keeps the two tables in step.

    Note the granularity: this proves every *check* is exercised, not every
    assertion inside one. The same argument does apply one level down, and the
    breakage table covers each assertion individually today — but nothing here
    enforces that, so an assertion added inside an existing check arrives
    unexercised and this test stays green.
    """
    exercised = {check for check, _, _ in MORTALITY_BREAKAGES}
    assert exercised == set(MORTALITY_CHECKS), (
        f"Checks with no breakage: "
        f"{sorted(c.__name__ for c in set(MORTALITY_CHECKS) - exercised)}. "
        f"Breakages naming a check that is not registered: "
        f"{sorted(c.__name__ for c in exercised - set(MORTALITY_CHECKS))}."
    )


def test_every_unverified_line_of_the_template_carries_the_marker() -> None:
    """Every number in the template is marked, except the two that are not guesses.

    ``grep -c PLACEHOLDER``, less the header banner, is the count of work
    remaining. The banner qualifier is not a quibble: three lines of the header
    prose contain the word while describing the convention, so the raw grep
    count never falls below three while the banner is present, and a human
    copying this file for issue 5 deletes the banner along with the last marker.
    The property this test asserts is the one the count depends on — that no
    line assigning a number escaped the marker when the file was written. The
    two terminal rows are the sole exception: they are 1.0 by construction
    rather than by transcription, so they carry a fixed comment instead and
    must NOT carry the marker.

    Lines are parsed rather than counted. A hard-coded total would have to be
    revised every time a sentence is added to the header, and the first person
    to hit that would revise the number rather than the file.
    """
    lines = MORTALITY_TEMPLATE.read_text(encoding="utf-8").splitlines()

    assignments = [line for line in lines if _ASSIGNS_A_NUMBER.match(line)]
    assert assignments, f"{MORTALITY_TEMPLATE} assigns no numbers at all."

    terminal_rows = [line for line in assignments if line.rstrip().endswith(TERMINAL_ROW_COMMENT)]
    assert len(terminal_rows) == 2, (
        f"Expected exactly two rows carrying {TERMINAL_ROW_COMMENT!r}, one per "
        f"sex, found {len(terminal_rows)}: {terminal_rows}."
    )
    for line in terminal_rows:
        assert MARKER not in line, (
            f"The terminal row is 1.0 by construction and must not be marked "
            f"unverified: {line!r}"
        )
        value = line.split(":", 1)[1].split("#", 1)[0].strip()
        assert value == "1.0", f"A terminal row holds {value!r}, not 1.0: {line!r}"

    unmarked = [line for line in assignments if MARKER not in line]
    assert unmarked == terminal_rows, (
        f"Numeric lines in {MORTALITY_TEMPLATE.name} with no marker, which "
        f"grep would count as finished work: "
        f"{[line for line in unmarked if line not in terminal_rows]}"
    )

    rows_per_sex: dict[str, int] = {}
    sex: str | None = None
    for line in lines:
        opened = _OPENS_A_SEX_TABLE.match(line)
        if opened:
            sex = opened.group(1)
            rows_per_sex[sex] = 0
        elif sex is not None and _IS_AN_AGE_ROW.match(line):
            rows_per_sex[sex] += 1
    terminal = int(_terminal_age_in_template(lines))
    expected_rows = terminal + 1
    assert rows_per_sex == {"f": expected_rows, "m": expected_rows}, (
        f"The template must write out every age 0..{terminal} for both sexes so "
        f"the human replaces numbers rather than typing keys, found {rows_per_sex}."
    )

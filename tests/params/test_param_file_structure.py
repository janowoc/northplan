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
    monthly = cpp.number("pension.maximum_at_standard_age_monthly")
    annual = cpp.number("pension.maximum_at_standard_age_annual")
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
# same six checks can be pointed at a live year directory and at deliberately
# broken copies of the template. Five of the six are also pointed at the
# pristine template; see ``PRISTINE_TEMPLATE_EXEMPT`` for the sixth.
#
# NOT a blanket monotonicity check. ``q_x`` does NOT increase with age
# everywhere: infant mortality exceeds childhood mortality, so a real table
# falls from age 0 to roughly age 10 and only rises after that. The
# non-decreasing assertion in ``test_age_factor_tables_are_contiguous_and_
# monotonic``, which is correct for the RRIF and LIF factor tables, would
# reject the true life table (issue 27). What a test may assume instead is the
# fall-to-a-single-trough-then-rise shape that
# ``_check_the_table_falls_to_a_single_trough_then_rises`` asserts below. The
# same, narrower warning is in the header of ``params/mortality-template.yaml``
# and ``params/2026/mortality.yaml``.

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
    "# set by the terminal-age convention (L10), and the source agrees — not a placeholder"
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


#: The trough of a real human life table falls in early childhood. A trough at
#: age 0 means the column is reversed or the infant row is wrong; a trough
#: above 15 means the adult rise starts too late — either way a transcription
#: error, not a modelling choice, so the bound is a property of the shape
#: itself rather than a tax parameter and belongs here rather than in
#: ``params/``.
_SHAPE_TROUGH_BOUND: Final[range] = range(1, 16)  # ages 1..15 inclusive


def _check_the_table_falls_to_a_single_trough_then_rises(name: str, params: ParamSet) -> None:
    """``q(x)`` falls through infancy and childhood to one trough, then rises.

    A real life table is not monotonic in age (see the boxed warning in
    ``params/mortality-template.yaml``), but it is not shapeless either:
    mortality falls from infancy to a single low point in early childhood and
    rises from there to the terminal age. Hand transcription of a
    two-hundred-plus-row table is the error-prone step in issue 5, and this is
    the only automated guard on the order of its values.

    WHAT THIS DOES NOT SEE. The check is purely ordinal, so every error that
    preserves the order of the column survives it: the whole column pasted one
    row off, the two sex columns swapped, an adult block scaled by a constant,
    a value replaced by anything between its two neighbours, and any single
    digit slip small enough to stay inside the envelope — including one on
    ``q(0)``, which no ordinal rule can reach. A green suite is not a proof of
    transcription. Catching those needs a magnitude anchor, which is a separate
    decision and not this check.

    The trough is the *last* age achieving the minimum, not the first: both
    published tables carry a tie plateau at the bottom (issue 27's evidence:
    ages 8-9 for ``f``, 7-10 for ``m``), so the descent must tolerate equality
    on the way down or a multi-age plateau would fail for the wrong reason.

    The ascent tolerates equality too, but only up to the top of
    ``_SHAPE_TROUGH_BOUND``. Around the trough a five-decimal table steps by
    one unit in the last published decimal — the shipped ``f`` column does
    exactly that at ages 9-13, 23-24 and 25-26 — and one unit of rounding
    difference in another geography or reference period would tie two adjacent
    rows and reject a correctly transcribed file. That is the same fragility
    issue 27 cited when it rejected a hard-coded start age, and it costs
    nothing to remove: a transposition produces a strict *decrease*, which
    ``>=`` still catches. Above the bound, q is large enough that a tie is
    itself worth stopping on.
    """
    for sex, table in params.get("q_x").items():
        path = f"{name}.q_x.{sex}"
        if not table or not all(str(key).isdigit() for key in table):
            # An empty or non-numerically-keyed table is the contiguity check's
            # failure, not this one's. Falling through would raise ValueError
            # out of min() or int(), which pytest.raises(AssertionError) does
            # not catch, and would bury that check's clear message under a
            # traceback from this one.
            continue

        ages = sorted(int(key) for key in table)
        values = [float(table[str(age)]) for age in ages]
        minimum = min(values)
        trough_index = max(index for index, value in enumerate(values) if value == minimum)
        trough_age = ages[trough_index]

        # The bound goes first. A mistyped adult row low enough to become the
        # new global minimum moves the trough to itself, and then the descent
        # loop blames the first pair above the childhood plateau — two rows
        # that are perfectly correct. Reporting the trough first points at the
        # damage instead of at the collateral.
        assert trough_age in _SHAPE_TROUGH_BOUND, (
            f"{path}: trough at age {trough_age}, outside "
            f"{_SHAPE_TROUGH_BOUND.start}..{_SHAPE_TROUGH_BOUND.stop - 1}. A trough "
            f"at {_SHAPE_TROUGH_BOUND.start - 1} means the column is reversed or "
            f"the infant row is wrong; a trough above "
            f"{_SHAPE_TROUGH_BOUND.stop - 1} means the adult rise is broken, or one "
            f"adult row was mistyped below the childhood minimum."
        )

        # Both loops name the offending pair rather than the range it lies in.
        # This check exists to help whoever transcribed two hundred rows by
        # hand find the one they mistyped, and "somewhere between age 10 and
        # age 110" sends them back through the whole table.
        for index in range(1, trough_index + 1):
            assert values[index] <= values[index - 1], (
                f"{path}: q rises from {values[index - 1]} at age {ages[index - 1]} "
                f"to {values[index]} at age {ages[index]}, before the trough at age "
                f"{trough_age}. A real table only falls or holds flat on the way down."
            )

        for index in range(trough_index + 1, len(values)):
            ties_allowed = ages[index] in _SHAPE_TROUGH_BOUND
            if ties_allowed:
                assert values[index] >= values[index - 1], (
                    f"{path}: q falls from {values[index - 1]} at age "
                    f"{ages[index - 1]} to {values[index]} at age {ages[index]}, after "
                    f"the trough at age {trough_age}. Adjacent rows may tie this close "
                    f"to the trough, but they may not go backwards."
                )
            else:
                assert values[index] > values[index - 1], (
                    f"{path}: q does not rise from {values[index - 1]} at age "
                    f"{ages[index - 1]} to {values[index]} at age {ages[index]}, after "
                    f"the trough at age {trough_age}."
                )


#: Every structural check applying to a parameter set named ``mortality``.
MORTALITY_CHECKS: tuple[Callable[[str, ParamSet], None], ...] = (
    _check_q_x_has_exactly_the_two_sex_tables,
    _check_ages_are_contiguous_digit_strings_up_to_the_terminal_age,
    _check_every_death_probability_is_in_the_unit_interval,
    _check_only_the_terminal_row_is_certain_death,
    _check_the_two_sex_tables_are_not_identical,
    _check_the_table_falls_to_a_single_trough_then_rises,
)

MORTALITY_CHECK_IDS = [check.__name__.lstrip("_") for check in MORTALITY_CHECKS]


@pytest.mark.parametrize("check", MORTALITY_CHECKS, ids=MORTALITY_CHECK_IDS)
def test_live_mortality_tables_satisfy_every_structural_check(
    year: ParamYear, check: Callable[[str, ParamSet], None]
) -> None:
    """Every shipped ``mortality`` set passes every check above.

    The guard on the first line is the point of the test as much as the loop
    is. Every other collection-driven test in this module asserts it found
    something — ``assert tables``, ``assert candidates``, ``assert
    provinces`` — because without that, deleting the file under test turns
    each named test green while checking nothing. That risk became real when
    issue 5 landed ``params/2026/mortality.yaml``; before it there was nothing
    to find and the guard could not be written.

    This is deliberately NOT written with the ``if CASES:`` guard that
    ``tests/golden/test_cases.py`` uses. That guard exists because its
    parametrize argument is read off disk and can genuinely be empty; here the
    argument is a literal, and copying the guard would only hide the day a
    check starts failing on the real file.
    """
    assert "mortality" in year.names(), (
        "No mortality set in this parameter year, so every check below ran "
        "against nothing. params/2026/mortality.yaml is what issue 5 landed."
    )
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


def _mortality_set_with(tmp_path: Path, mutate: Callable[[dict], None]) -> ParamSet:
    """Load the template, apply ``mutate`` to the plain dict, and reload it as a set.

    Named for what it does rather than for the breakages that are most of its
    callers: one caller passes a mutation that damages nothing, to prove the
    synthetic column the shape breakages are built from is sound to begin with.
    """
    raw = yaml.safe_load(MORTALITY_TEMPLATE.read_text(encoding="utf-8"))
    mutate(raw)
    year_dir = tmp_path / "2030"
    year_dir.mkdir()
    (year_dir / "mortality.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_year(2030, tmp_path)["mortality"]


#: Age at which the synthetic shape column below reaches its trough, and the
#: age it runs to. Both are made up for this test module; see
#: ``_synthetic_shape_column``.
_SYNTHETIC_TROUGH: Final[int] = 8
_SYNTHETIC_TERMINAL: Final[int] = 40


def _synthetic_shape_column(
    terminal: int = _SYNTHETIC_TERMINAL, trough: int = _SYNTHETIC_TROUGH
) -> dict[int, float]:
    """An obviously synthetic q(x) column, shaped like a real one but made up.

    Falls in round 0.01 steps from 0.50 at age 0 to a trough, then rises in
    round 0.01 steps back up, with the terminal row forced to 1.0 by the same
    convention the real tables use (L10). Every value is a fixed step chosen
    for readability and round enough that nobody could mistake it for a
    published probability — CLAUDE.md permits exactly this kind of made-up
    table to exercise arithmetic.

    Used only to build the three breakages for
    ``_check_the_table_falls_to_a_single_trough_then_rises`` below, and to
    prove — before those breakages are trusted — that the unbroken column
    passes the check it is meant to break.
    """
    column: dict[int, float] = {age: round(0.50 - 0.01 * age, 2) for age in range(trough + 1)}
    floor = column[trough]
    for age in range(trough + 1, terminal):
        column[age] = round(floor + 0.01 * (age - trough), 2)
    column[terminal] = 1.0
    return column


def _transposed(column: dict[int, float], age_a: int, age_b: int) -> dict[int, float]:
    """A copy of ``column`` with the rows at ``age_a`` and ``age_b`` swapped.

    The shape this repository's issue 27 calls "one interior value
    transposed" — the paste-the-wrong-row error a human transcribing 222
    numbers by hand is prone to make.
    """
    swapped = dict(column)
    swapped[age_a], swapped[age_b] = swapped[age_b], swapped[age_a]
    return swapped


def _replace_q_x_with(raw: dict, column: dict[int, float]) -> None:
    """Replace both sex tables in ``raw`` with ``column``, in place.

    Both sexes get the same synthetic column: the sex-identity check is not
    the one under test here, so making the two tables differ would only add
    noise. ``terminal_age_years`` moves with the table so the two agree, even
    though the shape check itself never reads that field.
    """
    raw["q_x"] = {"f": dict(column), "m": dict(column)}
    raw["terminal_age_years"] = max(column)


#: ``(check, description, mutate, expected)``: a break to the template, the one
#: check that must notice it, and a fragment of the message it must produce. A
#: mutation may legitimately trip more than the check named — a third sex table
#: also breaks contiguity — so the test asserts only that the named check fires.
#:
#: ``expected`` is what stops a check's several assertions from covering for
#: one another. Four breakages point at the shape check alone, and without a
#: message to match, moving ``_SYNTHETIC_TROUGH`` would silently make two of
#: them trip the same assertion while the suite stayed green — leaving another
#: assertion pinned by nothing, which is the "never seen to fail" problem
#: ``test_every_mortality_check_has_a_breakage_exercising_it`` exists to
#: prevent, one level down. Matched as a literal, not a pattern.
MORTALITY_BREAKAGES: tuple[
    tuple[Callable[[str, ParamSet], None], str, Callable[[dict], None], str], ...
] = (
    (
        _check_q_x_has_exactly_the_two_sex_tables,
        "a third sex table",
        lambda raw: raw["q_x"].update({"x": dict(raw["q_x"]["f"])}),
        "holds ['f', 'm', 'x']",
    ),
    (
        _check_q_x_has_exactly_the_two_sex_tables,
        "the male table missing entirely",
        lambda raw: raw["q_x"].pop("m"),
        "holds ['f']. It must hold",
    ),
    (
        _check_ages_are_contiguous_digit_strings_up_to_the_terminal_age,
        "a gap at age 50",
        lambda raw: raw["q_x"]["f"].pop(50),
        "no gaps. Missing: [50]",
    ),
    (
        _check_every_death_probability_is_in_the_unit_interval,
        "a probability written in percent",
        lambda raw: raw["q_x"]["m"].update({3: 1.5}),
        "m.3 is 1.5, outside (0, 1]",
    ),
    (
        _check_every_death_probability_is_in_the_unit_interval,
        "an age of immortality",
        lambda raw: raw["q_x"]["f"].update({7: 0.0}),
        "f.7 is 0.0, outside (0, 1]",
    ),
    (
        _check_only_the_terminal_row_is_certain_death,
        "certain death at 40",
        lambda raw: raw["q_x"]["m"].update({40: 1.0}),
        "q is 1.0 below the terminal age 110 at ['40']",
    ),
    (
        _check_only_the_terminal_row_is_certain_death,
        "survival past the terminal age",
        lambda raw: raw["q_x"]["f"].update({raw["terminal_age_years"]: 0.9}),
        "f.110 is 0.9, not 1.0",
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
        "has no row at the terminal age 110",
    ),
    (
        _check_every_death_probability_is_in_the_unit_interval,
        "a probability written as a YAML boolean",
        lambda raw: raw["q_x"]["m"].update({5: True}),
        "m.5 is True (bool), not a number",
    ),
    (
        _check_ages_are_contiguous_digit_strings_up_to_the_terminal_age,
        "an age key that is not a whole number",
        lambda raw: raw["q_x"]["f"].update({"7x": 0.11111}),
        "ages must be whole-number keys",
    ),
    (
        _check_the_two_sex_tables_are_not_identical,
        "one sex's column pasted into both tables",
        lambda raw: raw["q_x"].update({"m": dict(raw["q_x"]["f"])}),
        "identical at every age",
    ),
    # These three replace q_x outright with a synthetic column rather than
    # mutating the template's repdigits — the template's values are shapeless
    # by construction, so the shape check would already be failing before any
    # of these mutations, and every one of them would pass for the wrong
    # reason. See _synthetic_shape_column.
    (
        _check_the_table_falls_to_a_single_trough_then_rises,
        "an interior transposition breaks the descent before the trough",
        lambda raw: _replace_q_x_with(raw, _transposed(_synthetic_shape_column(), 3, 4)),
        "before the trough at age 8",
    ),
    (
        _check_the_table_falls_to_a_single_trough_then_rises,
        "an interior transposition breaks the ascent after the trough",
        lambda raw: _replace_q_x_with(raw, _transposed(_synthetic_shape_column(), 15, 16)),
        "does not rise from 0.5 at age 15",
    ),
    (
        _check_the_table_falls_to_a_single_trough_then_rises,
        "the ascent goes backwards inside the tie-tolerant stretch",
        lambda raw: _replace_q_x_with(raw, _transposed(_synthetic_shape_column(), 10, 11)),
        "may tie this close to the trough",
    ),
    (
        _check_the_table_falls_to_a_single_trough_then_rises,
        "the trough sits outside the 1..15 bound",
        lambda raw: _replace_q_x_with(raw, _synthetic_shape_column(terminal=50, trough=20)),
        "trough at age 20, outside 1..15",
    ),
)


#: Checks a shapeless placeholder table cannot satisfy by construction. The
#: template ships repdigit q(x) values (0.11111 for every f row, 0.22222 for
#: every m row) so that a real number stands out at a glance — but that also
#: means the template has no shape at all, and the minimum spans every row
#: from 0 to 109, so the *last* minimum (the trough, by this check's own
#: definition) lands at 109, outside the 1..15 bound. The five other checks
#: are satisfiable by a constant column, and these placeholders were chosen to
#: satisfy them — three of the five constrain values rather than layout: the
#: unit interval, the terminal row, and the two tables not being equal. This
#: check is the only one constraining the RELATIONSHIP BETWEEN rows, which no
#: constant column can express. That, not a layout-versus-values split, is what
#: qualifies a check for this set.
#: ``test_every_exempt_check_still_rejects_the_pristine_template`` below pins
#: this exemption live, so a stale exemption — one whose check would now pass
#: — fails loudly rather than sitting here unnoticed.
PRISTINE_TEMPLATE_EXEMPT: Final[frozenset[Callable[[str, ParamSet], None]]] = frozenset(
    {_check_the_table_falls_to_a_single_trough_then_rises}
)

#: ``MORTALITY_CHECKS`` minus the exemption above, for the pristine-template
#: test only. Every other test in this module still runs the full set.
_PRISTINE_TEMPLATE_CHECKS = tuple(
    check for check in MORTALITY_CHECKS if check not in PRISTINE_TEMPLATE_EXEMPT
)
_PRISTINE_TEMPLATE_CHECK_IDS = [check.__name__.lstrip("_") for check in _PRISTINE_TEMPLATE_CHECKS]


@pytest.mark.parametrize("check", _PRISTINE_TEMPLATE_CHECKS, ids=_PRISTINE_TEMPLATE_CHECK_IDS)
def test_the_pristine_template_satisfies_every_unexempted_check(
    pristine_mortality_set: ParamSet, check: Callable[[str, ParamSet], None]
) -> None:
    """The template starts from a layout the rest of the suite already accepts.

    This is what makes issue 5 safe. The human filling in real values starts
    from a layout that has already been proved acceptable, so a layout failure
    afterwards is one they introduced while filling it in rather than one they
    inherited. It says nothing about the edited copy: a deleted row, a mistyped
    age key, or a changed ``terminal_age_years`` are all still layout failures,
    and the live test above is what catches them once issue 5 lands.

    Runs every check except ``PRISTINE_TEMPLATE_EXEMPT``: the shape check
    constrains values, not layout, and the template's repdigit placeholders
    have no shape by design. See that constant.
    """
    check("mortality", pristine_mortality_set)


def test_every_exempt_check_still_rejects_the_pristine_template(
    pristine_mortality_set: ParamSet,
) -> None:
    """The exemption above is live, not stale.

    If the template's placeholder values were ever changed such that
    ``_check_the_table_falls_to_a_single_trough_then_rises`` started passing
    on them, that would mean the "obviously fake" repdigits had accidentally
    acquired a real shape — and the exemption in
    ``test_the_pristine_template_satisfies_every_unexempted_check`` would go
    stale, silently no longer testing anything. This fails loudly instead.
    """
    for check in PRISTINE_TEMPLATE_EXEMPT:
        with pytest.raises(AssertionError):
            check("mortality", pristine_mortality_set)


def test_a_correctly_shaped_synthetic_table_passes_the_shape_check(tmp_path: Path) -> None:
    """The base synthetic column the breakages above are built from is itself valid.

    Establishes that a breakage derived from ``_synthetic_shape_column`` fails
    because of what was done *to* it — the transposition, or the shifted
    trough — not because the base column was never shaped correctly to begin
    with.
    """
    unbroken = _mortality_set_with(
        tmp_path, lambda raw: _replace_q_x_with(raw, _synthetic_shape_column())
    )
    _check_the_table_falls_to_a_single_trough_then_rises("mortality", unbroken)


@pytest.mark.parametrize(
    ("check", "description", "mutate", "expected"),
    MORTALITY_BREAKAGES,
    ids=[description for _, description, _, _ in MORTALITY_BREAKAGES],
)
def test_a_broken_mortality_table_fails_the_check_that_covers_it(
    tmp_path: Path,
    check: Callable[[str, ParamSet], None],
    description: str,
    mutate: Callable[[dict], None],
    expected: str,
) -> None:
    """Each check actually fires on the damage it is there to catch.

    Only the named check is asserted. A mutation often trips others too, and
    pinning down which would make this table brittle for no gain.

    ``expected`` does pin which assertion *within* that check fires; see
    ``MORTALITY_BREAKAGES``.
    """
    broken = _mortality_set_with(tmp_path, mutate)
    with pytest.raises(AssertionError, match=re.escape(expected)):
        check("mortality", broken)


def test_every_mortality_check_has_a_breakage_exercising_it() -> None:
    """No check goes unexercised.

    A new entry in ``MORTALITY_CHECKS`` with nothing in
    ``MORTALITY_BREAKAGES`` pointed at it is a check that has never been seen
    to fail, which is indistinguishable from a check that cannot fail. This
    keeps the two tables in step.

    Note the granularity: this proves every *check* is exercised, not every
    assertion inside one. The same argument does apply one level down, and the
    breakage table covers each assertion individually today — but nothing here
    enforces that, so an assertion added inside an existing check arrives
    unexercised and this test stays green.
    """
    exercised = {check for check, _, _, _ in MORTALITY_BREAKAGES}
    assert exercised == set(MORTALITY_CHECKS), (
        f"Checks with no breakage: "
        f"{sorted(c.__name__ for c in set(MORTALITY_CHECKS) - exercised)}. "
        f"Breakages naming a check that is not registered: "
        f"{sorted(c.__name__ for c in exercised - set(MORTALITY_CHECKS))}."
    )


#: A Python identifier for one of the checks above, as a parameter file's
#: prose might name it.
_NAMES_A_CHECK = re.compile(r"_check_[a-z0-9_]+")


def _mortality_files() -> list[Path]:
    """The template and every live ``mortality.yaml``, which may name a check."""
    live = [
        directory / "mortality.yaml"
        for directory in sorted(DEFAULT_PARAMS_ROOT.iterdir())
        if directory.is_dir() and directory.name.isdigit()
    ]
    return [MORTALITY_TEMPLATE, *(path for path in live if path.exists())]


def test_every_check_a_mortality_file_names_by_identifier_exists() -> None:
    """No parameter file cites a check that has been renamed out from under it.

    The boxed warning in the template and in each live file names the check
    that may be assumed of the rows, by its Python identifier. CLAUDE.md
    forbids an agent from editing anything under ``params/``, so a rename of
    that check leaves a stale citation that the person who did the renaming
    cannot fix — a quieter version of the flat contradiction issue 27 was
    written to remove. This fails in the test module instead, where the rename
    happened and where an agent may act.
    """
    known = {check.__name__ for check in MORTALITY_CHECKS}
    cited: set[str] = set()
    for path in _mortality_files():
        named = set(_NAMES_A_CHECK.findall(path.read_text(encoding="utf-8")))
        cited |= named
        assert named <= known, (
            f"{path} names {sorted(named - known)}, which is not in "
            f"MORTALITY_CHECKS. Either the check was renamed and the comment "
            f"was left behind, or the comment is describing something that does "
            f"not exist. The fix belongs in this module: params/ is the human's."
        )
    assert cited, (
        "No mortality file names a check by identifier, so this test asserted "
        "nothing. The boxed warning is what it exists to keep honest — if that "
        "wording changed, this test needs to change with it or be deleted."
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

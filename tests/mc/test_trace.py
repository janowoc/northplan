# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``engine.mc.trace.trace_rows``, exercised against deterministic runs of the two
committed scenarios.

The example scenario's run and the couple's run (the latter with person 0's death month
forced to 27 via ``with_death_months``) are each built once, module-scoped: the deterministic
run itself is slow-ish, and every test in this module reads from one or the other without
mutating anything.
"""

from __future__ import annotations

import dataclasses
import enum
import re
from pathlib import Path

import numpy as np
import pytest

from engine.core.build import (
    build_deterministic_draws,
    build_draws,
    build_initial_state,
    draw_deaths,
    with_death_months,
)
from engine.core.build import build_market_inputs as _build_market_inputs
from engine.core.indexation import real_year
from engine.core.step import advance_month_traced
from engine.mc.simulate import SimulationResult, run
from engine.mc.trace import trace_rows
from engine.params.loader import load_year
from engine.policy.build import build_policy, expand_grid
from engine.scenario import load_scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "scenarios" / "example.yaml"
COUPLE = REPO_ROOT / "scenarios" / "late_life_couple.yaml"


class _Month(enum.IntEnum):
    JANUARY = 1


class _Province(enum.StrEnum):
    AB = "AB"


def _run(path: Path, death_months: tuple[int | None, ...] | None = None) -> SimulationResult:
    scenario = expand_grid(load_scenario(path))
    spec = scenario.policies[0]
    mortality = load_year(scenario.start_year)["mortality"]
    market = _build_market_inputs(scenario.assumptions)
    real_params = real_year(load_year(scenario.start_year), scenario.assumptions.inflation)
    draws = build_deterministic_draws(scenario, market, mortality)
    state = draw_deaths(
        build_initial_state(scenario, n_paths=draws.n_paths, policy=spec), draws, mortality
    )
    if death_months is not None:
        state = with_death_months(state, death_months)
    return run(
        state, build_policy(spec, scenario.household), draws, market, real_params, trace_path=0
    )


@pytest.fixture(scope="module")
def example_result() -> SimulationResult:
    return _run(EXAMPLE)


@pytest.fixture(scope="module")
def example_rows(example_result: SimulationResult) -> list[dict[str, object]]:
    return trace_rows(example_result.trace)


@pytest.fixture(scope="module")
def couple_result() -> SimulationResult:
    return _run(COUPLE, death_months=(27, None))


@pytest.fixture(scope="module")
def couple_rows(couple_result: SimulationResult) -> list[dict[str, object]]:
    return trace_rows(couple_result.trace)


# =============================================================================
# The example run
# =============================================================================


def test_row_count_matches_the_trace(example_rows, example_result) -> None:
    assert len(example_rows) == len(example_result.trace)


def test_january_2027_row_carries_the_hand_check_columns(example_rows) -> None:
    row = next(r for r in example_rows if r["context.year"] == 2027 and r["context.month"] == 1)
    for column in (
        "persons.0.rrsp.room",
        "persons.0.tfsa.room",
        "persons.0.rrif.annual_minimum",
        "persons.0.lif.annual_maximum",
        "persons.0.taxable.acb",
        "persons.0.balance_owing",
        "persons.0.income.employment",
    ):
        assert row[column] is not None, column


def test_december_rows_carry_assessments_matching_tax_assessed(
    example_rows, example_result
) -> None:
    tax_assessed_by_year = dict(
        zip(example_result.years.tolist(), example_result.tax_assessed[:, 0].tolist(), strict=True)
    )
    found_december = False
    for row in example_rows:
        if row["context.month"] == 12:
            found_december = True
            assert row["year_record.assessments.0.total"] is not None
            assert row["year_record.tax_assessed"] is not None
            assert row["year_record.assessments.0.total"] == row["year_record.tax_assessed"]
            assert row["year_record.tax_assessed"] == tax_assessed_by_year[row["year_record.year"]]
        else:
            for key, value in row.items():
                if key == "year_record" or key.startswith("year_record."):
                    assert value is None, key
    assert found_december


# =============================================================================
# The couple run, person 0 forced dead at month_index 27
# =============================================================================


def test_rolled_out_is_nonzero_only_at_the_forced_death_month(couple_rows) -> None:
    columns = [
        f"context.rolled_out.0.{field}"
        for field in ("rrsp", "rrif", "lira", "lif", "tfsa", "taxable")
    ]
    found_nonzero = False
    for row in couple_rows:
        values = [row[column] for column in columns]
        if row["context.month_index"] == 27:
            assert any(v != 0.0 for v in values)
            found_nonzero = True
        else:
            assert all(v == 0.0 for v in values)
    assert found_nonzero


def test_pension_split_transfers_mirror_each_other_every_december(couple_rows) -> None:
    found_nonzero = False
    for row in couple_rows:
        if row["context.month"] != 12:
            continue
        t_out0 = row["year_record.assessments.0.transfer_out"]
        t_in1 = row["year_record.assessments.1.transfer_in"]
        t_out1 = row["year_record.assessments.1.transfer_out"]
        t_in0 = row["year_record.assessments.0.transfer_in"]
        assert t_out0 == t_in1
        assert t_out1 == t_in0
        if t_out0 != 0.0 or t_out1 != 0.0:
            found_nonzero = True
    assert found_nonzero, "no December row carries a non-zero transfer; the check would be vacuous"


def test_terminal_assessments_sum_to_the_terminal_assessment_only_at_the_second_death(
    couple_rows,
) -> None:
    nonzero_rows = [row for row in couple_rows if row["context.terminal_assessment"] != 0.0]
    assert len(nonzero_rows) == 1
    target = nonzero_rows[0]

    total = 0.0
    for person_index in range(2):
        total += target[f"context.terminal_assessments.{person_index}.total"]
    assert total == target["context.terminal_assessment"]

    terminal_columns = [key for key in target if key.startswith("context.terminal_assessments.")]
    for row in couple_rows:
        if row is target:
            continue
        for column in terminal_columns:
            assert row[column] == 0.0, column


def test_couple_assessments_are_in_persons_order(couple_rows) -> None:
    """Person 0 is forced dead at month_index 27 (April 2028), so from 2029 on they have no
    income all year: their assessment must be zero and person 1's must carry the household
    tax. Swapping the two assessments would fail the second assertion.
    """
    count = 0
    for row in couple_rows:
        if row["context.month"] != 12:
            continue
        assert (
            0.0 + row["year_record.assessments.0.total"] + row["year_record.assessments.1.total"]
            == row["year_record.tax_assessed"]
        )
        if row["year_record.year"] >= 2029:
            assert (
                row["year_record.assessments.0.total"]
                == 0.0
                < row["year_record.assessments.1.total"]
            )
            count += 1
    assert count > 0


# =============================================================================
# The refusals
# =============================================================================


def test_trace_rows_refuses_an_unsliced_record() -> None:
    scenario = expand_grid(load_scenario(EXAMPLE))
    spec = scenario.policies[0]
    mortality = load_year(scenario.start_year)["mortality"]
    market = _build_market_inputs(scenario.assumptions)
    real_params = real_year(load_year(scenario.start_year), scenario.assumptions.inflation)

    draws = build_draws(scenario, market, n_paths=2, mortality=mortality)
    state = draw_deaths(build_initial_state(scenario, n_paths=2, policy=spec), draws, mortality)
    policy = build_policy(spec, scenario.household)

    _new_state, record = advance_month_traced(
        state, draws.real_returns[0], policy, market, real_params
    )

    with pytest.raises(ValueError, match=r"column 'context\.cash_opening'.*\(2,\).*at_path"):
        trace_rows([record])


def test_trace_rows_refuses_an_unknown_leaf_type(example_result) -> None:
    record = example_result.trace[0]
    poisoned_context = dataclasses.replace(record.context, month_index=np.int64(0))
    poisoned_record = dataclasses.replace(record, context=poisoned_context)

    with pytest.raises(TypeError, match=r"context\.month_index"):
        trace_rows([poisoned_record])


@pytest.mark.parametrize(
    "value, name",
    [(np.float64(0.0), "numpy.float64"), (np.str_("0"), "numpy.str_")],
)
def test_trace_rows_refuses_a_numpy_scalar_that_subclasses_a_python_scalar(
    example_result, value, name
) -> None:
    """``np.float64`` subclasses ``float`` and ``np.str_`` subclasses ``str``; neither is the
    exact type, so both fall through to the unsupported-leaf ``TypeError``.
    """
    record = example_result.trace[0]
    poisoned_context = dataclasses.replace(record.context, month_index=value)
    poisoned_record = dataclasses.replace(record, context=poisoned_context)

    with pytest.raises(TypeError, match=rf"context\.month_index.*{re.escape(name)}"):
        trace_rows([poisoned_record])


@pytest.mark.parametrize(
    "value, dtype_text",
    [
        (np.array([{"x": 1}], dtype=object), "object"),
        (np.array(["2026-01-01"], dtype="datetime64[D]"), "datetime64[D]"),
        (np.array([1 + 2j]), "complex128"),
        pytest.param(
            np.array([1.0], dtype=np.longdouble),
            str(np.dtype(np.longdouble)),
            marks=pytest.mark.skipif(
                np.dtype(np.longdouble).itemsize == 8,
                reason="longdouble is float64 on this platform, so .item() already returns float",
            ),
        ),
    ],
)
def test_trace_rows_refuses_an_unsupported_array_dtype(example_result, value, dtype_text) -> None:
    record = example_result.trace[0]
    poisoned_context = dataclasses.replace(record.context, month_index=value)
    poisoned_record = dataclasses.replace(record, context=poisoned_context)

    with pytest.raises(TypeError, match=rf"context\.month_index.*dtype {re.escape(dtype_text)}"):
        trace_rows([poisoned_record])


@pytest.mark.parametrize("value", [_Month.JANUARY, _Province.AB])
def test_trace_rows_refuses_a_python_scalar_subclass(example_result, value) -> None:
    """``IntEnum`` and ``StrEnum`` members are ``int``/``str`` subclasses, not the exact type,
    so both fall through to the unsupported-leaf ``TypeError``.
    """
    record = example_result.trace[0]
    poisoned_context = dataclasses.replace(record.context, month_index=value)
    poisoned_record = dataclasses.replace(record, context=poisoned_context)

    with pytest.raises(
        TypeError, match=rf"context\.month_index.*{re.escape(type(value).__qualname__)}"
    ):
        trace_rows([poisoned_record])


def test_trace_rows_refuses_an_ndarray_subclass(example_result) -> None:
    """``np.ma.MaskedArray`` subclasses ``np.ndarray``; it must fall through to the
    unsupported-leaf ``TypeError``, not be flattened to its underlying data, which would
    lose the mask.
    """
    record = example_result.trace[0]
    value = np.ma.array([1.0], mask=[True])
    poisoned_context = dataclasses.replace(record.context, month_index=value)
    poisoned_record = dataclasses.replace(record, context=poisoned_context)

    with pytest.raises(TypeError, match=r"context\.month_index.*MaskedArray"):
        trace_rows([poisoned_record])


# =============================================================================
# The flattening rules themselves
# =============================================================================


def test_columns_union_in_first_seen_order_and_a_missing_column_is_none(example_result) -> None:
    trace = example_result.trace
    first = trace[0]
    assert first.context.month != 12, "the trace opens in January, not December"
    december = next(r for r in trace if r.context.month == 12)

    rows = trace_rows([first, december])

    assert list(rows[0].keys()) == list(rows[1].keys())
    assert rows[0]["year_record.year"] is None

    first_keys = list(trace_rows([first])[0])
    december_keys = list(trace_rows([december])[0])
    december_only = [k for k in december_keys if k not in first_keys]
    first_only = [k for k in first_keys if k not in december_keys]
    assert december_only  # year_record.* columns
    assert first_only == ["year_record"]
    assert list(rows[0]) == first_keys + december_only
    assert list(trace_rows([december, first])[0]) == december_keys + first_only


def test_shape_one_leaves_become_python_scalars(example_rows) -> None:
    row = example_rows[0]
    assert type(row["context.cash_opening"]) is float
    assert type(row["persons.0.alive"]) is bool
    assert type(row["persons.0.death_month_index"]) is int


def test_empty_trace_gives_an_empty_list() -> None:
    assert trace_rows([]) == []

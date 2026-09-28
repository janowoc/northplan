# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``engine.mc.prepare``: ``prepare_run`` and ``evaluate`` against the hand-assembled pipeline.

Every statutory number is read from ``load_year(2026)`` (or the scenario's own
``start_year``), never typed in. ``EXAMPLE`` and ``COUPLE`` are the two
committed scenarios; each mutation below starts from a fresh
``yaml.safe_load`` of one of them, so no test can see another test's edit.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from engine.core.build import (
    build_deterministic_draws,
    build_draws,
    build_initial_state,
    build_market_inputs,
    draw_deaths,
    with_death_months,
)
from engine.core.indexation import real_year
from engine.core.timeline import MONTHS_PER_YEAR
from engine.mc.prepare import PreparedRun, evaluate, prepare_run
from engine.mc.simulate import SimulationResult, run
from engine.params.loader import DEFAULT_PARAMS_ROOT, load_year
from engine.policy.build import build_policy, expand_grid
from engine.scenario import (
    LifespanNotRepresentableError,
    PolicySpec,
    Scenario,
    StartAgeNotAllowedError,
    load_scenario,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "scenarios" / "example.yaml"
COUPLE = REPO_ROOT / "scenarios" / "late_life_couple.yaml"


def _values(path: Path) -> dict[str, Any]:
    """A fresh, mutable parse of the scenario at ``path``."""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(values, dict)
    return values


def _policy(values: dict[str, Any], name: str = "taxable-first") -> dict[str, Any]:
    for policy in values["policies"]:
        if policy["name"] == name:
            return policy
    raise AssertionError(f"no policy named {name!r}")


def _birth_for_age_at_open(start_year: int, age_months: int) -> tuple[int, int]:
    """A ``(birth_year, birth_month)`` giving exactly ``age_months`` on 1 January of ``start_year``."""
    years_back = -(-age_months // MONTHS_PER_YEAR)  # ceiling division
    birth_month = 1 - age_months + MONTHS_PER_YEAR * years_back
    assert 1 <= birth_month <= MONTHS_PER_YEAR
    return start_year - years_back, birth_month


def _append_minimal_person(
    values: dict[str, Any], person_id: str, birth_year: int, birth_month: int
) -> None:
    """Append a second, minimal adult -- CPP and OAS already in pay, no accounts or db_pensions --
    so nothing else about them can refuse the scenario first.
    """
    values["household"]["persons"].append(
        {
            "id": person_id,
            "birth_year": birth_year,
            "birth_month": birth_month,
            "sex": "f",
            "cpp": {"in_pay_monthly": 800.0},
            "oas": {"in_pay_monthly": 500.0},
            "prior_year_net_income": 0,
            "net_income_two_years_prior": 0,
        }
    )


def _hand_run(
    scenario: Scenario,
    spec: PolicySpec,
    *,
    n_paths: int | None,
    deterministic: bool = False,
    trace_path: int | None = None,
    death_months: tuple[int | None, ...] | None = None,
) -> SimulationResult:
    """The oracle: the pipeline ``tests/mc/test_trace.py::_run`` assembles by hand, minus the trace.

    ``scenario`` must already be grid-expanded, matching what :func:`prepare_run` hands
    :func:`~engine.mc.simulate.run`.
    """
    params = load_year(scenario.start_year)
    mortality = params["mortality"]
    market = build_market_inputs(scenario.assumptions)
    real_params = real_year(params, scenario.assumptions.inflation)
    if deterministic:
        draws = build_deterministic_draws(scenario, market, mortality)
    else:
        assert n_paths is not None
        draws = build_draws(scenario, market, n_paths, mortality)
    state = build_initial_state(scenario, draws.n_paths, policy=spec)
    state = draw_deaths(state, draws, mortality)
    if death_months is not None:
        state = with_death_months(state, death_months)
    policy = build_policy(spec, scenario.household)
    return run(state, policy, draws, market, real_params, trace_path=trace_path)


def _assert_results_equal(a: SimulationResult, b: SimulationResult) -> None:
    for field in dataclasses.fields(SimulationResult):
        va = getattr(a, field.name)
        vb = getattr(b, field.name)
        if field.name == "seed":
            assert va == vb, field.name
        elif field.name == "trace":
            assert va == () and vb == (), field.name
        else:
            assert isinstance(va, np.ndarray) and isinstance(vb, np.ndarray), field.name
            assert va.dtype == vb.dtype, field.name
            assert va.shape == vb.shape, field.name
            assert np.array_equal(va, vb), field.name


_EXAMPLE_RAW = load_scenario(EXAMPLE)
_COUPLE_RAW = load_scenario(COUPLE)
_EXAMPLE_EXPANDED = expand_grid(_EXAMPLE_RAW)
_COUPLE_EXPANDED = expand_grid(_COUPLE_RAW)


# =============================================================================
# Differential against the hand-assembled pipeline
# =============================================================================

_DIFFERENTIAL_CASES = (
    [
        pytest.param(
            _EXAMPLE_RAW, _EXAMPLE_EXPANDED, spec, 50, False, id=f"example-n50-{spec.name}"
        )
        for spec in _EXAMPLE_EXPANDED.policies
    ]
    + [
        pytest.param(_COUPLE_RAW, _COUPLE_EXPANDED, spec, 50, False, id=f"couple-n50-{spec.name}")
        for spec in _COUPLE_EXPANDED.policies
    ]
    + [
        pytest.param(
            _EXAMPLE_RAW,
            _EXAMPLE_EXPANDED,
            _EXAMPLE_EXPANDED.policies[0],
            None,
            True,
            id="example-deterministic",
        ),
        pytest.param(
            _COUPLE_RAW,
            _COUPLE_EXPANDED,
            _COUPLE_EXPANDED.policies[0],
            None,
            True,
            id="couple-deterministic",
        ),
    ]
)


@pytest.mark.parametrize(
    "raw_scenario,expanded_scenario,spec,n_paths,deterministic", _DIFFERENTIAL_CASES
)
def test_evaluate_matches_the_hand_assembled_pipeline(
    raw_scenario, expanded_scenario, spec, n_paths, deterministic
) -> None:
    prepared = prepare_run(raw_scenario, n_paths=n_paths, deterministic=deterministic)
    result = evaluate(prepared, spec)
    oracle = _hand_run(expanded_scenario, spec, n_paths=n_paths, deterministic=deterministic)
    _assert_results_equal(result, oracle)


# =============================================================================
# prepare_run / evaluate mechanics
# =============================================================================


def test_trace_path_is_forwarded() -> None:
    prepared = prepare_run(_EXAMPLE_RAW, deterministic=True)
    spec = prepared.scenario.policies[0]

    result = evaluate(prepared, spec, trace_path=0)
    oracle = _hand_run(_EXAMPLE_EXPANDED, spec, n_paths=None, deterministic=True, trace_path=0)

    assert len(result.trace) > 0
    assert len(result.trace) == len(oracle.trace)


def test_n_paths_none_uses_the_scenarios_n_paths() -> None:
    prepared = prepare_run(_EXAMPLE_RAW)
    assert prepared.draws.n_paths == _EXAMPLE_RAW.n_paths


def test_deterministic_gives_one_path() -> None:
    prepared = prepare_run(_EXAMPLE_RAW, deterministic=True)
    assert prepared.draws.n_paths == 1


def test_deterministic_with_n_paths_raises() -> None:
    with pytest.raises(ValueError, match="deterministic"):
        prepare_run(_EXAMPLE_RAW, deterministic=True, n_paths=5)


def test_deterministic_with_n_paths_raises_before_the_start_age_guard() -> None:
    """A scenario that would separately fail ``check_start_ages`` still raises the
    deterministic/``n_paths`` message, since that check runs first, before anything else.
    """
    values = _values(EXAMPLE)
    values["grid"] = {"elections.cpp_start_age_years.a": [50, 65]}
    scenario = Scenario.model_validate(values)

    with pytest.raises(StartAgeNotAllowedError):
        prepare_run(scenario, n_paths=2)

    with pytest.raises(
        ValueError, match=re.escape("deterministic=True and n_paths=5 were both given")
    ):
        prepare_run(scenario, deterministic=True, n_paths=5)


# =============================================================================
# The load-time guards, their order, and a committed scenario that passes them
# =============================================================================


def test_grid_cpp_start_age_50_is_refused() -> None:
    params = load_year(2026)
    assert params.cpp.number("start_age.earliest_months") > 50 * MONTHS_PER_YEAR

    values = _values(EXAMPLE)
    values["grid"] = {"elections.cpp_start_age_years.a": [50, 65]}
    scenario = Scenario.model_validate(values)

    with pytest.raises(StartAgeNotAllowedError) as excinfo:
        prepare_run(scenario)

    assert "elections.cpp_start_age_years.a=50" in str(excinfo.value)


def test_person_past_the_life_table_is_refused() -> None:
    params = load_year(2026)
    terminal_age_years = int(params["mortality"].number("terminal_age_years"))

    values = _values(EXAMPLE)
    age_months = (terminal_age_years + 1) * MONTHS_PER_YEAR
    birth_year, birth_month = _birth_for_age_at_open(values["start_year"], age_months)
    _append_minimal_person(values, "b", birth_year, birth_month)
    scenario = Scenario.model_validate(values)

    with pytest.raises(LifespanNotRepresentableError) as excinfo:
        prepare_run(scenario)

    assert "'b'" in str(excinfo.value)


def test_start_ages_runs_before_lifespan() -> None:
    """The scenario carries both a too-old person and an out-of-window CPP grid; the
    CPP refusal wins, since check_start_ages runs before check_lifespan.
    """
    params = load_year(2026)
    terminal_age_years = int(params["mortality"].number("terminal_age_years"))

    values = _values(EXAMPLE)
    age_months = (terminal_age_years + 1) * MONTHS_PER_YEAR
    birth_year, birth_month = _birth_for_age_at_open(values["start_year"], age_months)
    _append_minimal_person(values, "b", birth_year, birth_month)
    values["grid"] = {"elections.cpp_start_age_years.a": [50, 65]}
    scenario = Scenario.model_validate(values)

    with pytest.raises(StartAgeNotAllowedError, match=re.escape("cpp_start_age_years.a=50")):
        prepare_run(scenario)


def test_rrif_conversion_above_limit_is_refused() -> None:
    params = load_year(2026)
    limit = int(params.rrif.number("conversion_age_years"))

    values = _values(EXAMPLE)
    _policy(values)["elections"]["rrif_conversion"]["age_years"] = limit + 1
    scenario = Scenario.model_validate(values)

    with pytest.raises(StartAgeNotAllowedError, match=re.escape("rrif_conversion.age_years")):
        prepare_run(scenario)


def test_grid_rrif_conversion_above_limit_is_refused() -> None:
    """A grid value for the RRIF conversion election is checked like a written one, because
    check_start_ages runs on the expanded scenario: one combination, at the limit, would pass
    alone, but the other, one above it, refuses the whole expansion.
    """
    params = load_year(2026)
    limit = int(params.rrif.number("conversion_age_years"))

    values = _values(EXAMPLE)
    values["grid"] = {"elections.rrif_conversion.age_years": [limit, limit + 1]}
    scenario = Scenario.model_validate(values)

    with pytest.raises(StartAgeNotAllowedError) as excinfo:
        prepare_run(scenario)

    assert f"elections.rrif_conversion.age_years={limit + 1}" in str(excinfo.value)


def test_late_life_couple_prepares() -> None:
    prepared = prepare_run(_COUPLE_RAW, n_paths=2)
    assert isinstance(prepared, PreparedRun)


# =============================================================================
# The RESP offset check, and params_root
# =============================================================================


def test_resp_offset_check_runs_at_the_opening(tmp_path: Path) -> None:
    shutil.copytree(DEFAULT_PARAMS_ROOT / "2026", tmp_path / "2026")
    resp_file = tmp_path / "2026" / "resp.yaml"
    original = resp_file.read_text(encoding="utf-8")
    modified = original.replace("income_year_offset: -2", "income_year_offset: -1")
    assert modified != original
    resp_file.write_text(modified, encoding="utf-8")

    with pytest.raises(ValueError, match="income_year_offset"):
        prepare_run(_EXAMPLE_RAW, n_paths=2, params_root=tmp_path)


def test_params_root_is_honoured(tmp_path: Path) -> None:
    shutil.copytree(DEFAULT_PARAMS_ROOT / "2026", tmp_path / "2026")
    prepared = prepare_run(_EXAMPLE_RAW, n_paths=2, params_root=tmp_path)
    assert str(prepared.params.cpp.source).startswith(str(tmp_path))


# =============================================================================
# evaluate's membership-by-value check
# =============================================================================


def test_evaluate_refuses_the_unexpanded_policy() -> None:
    prepared = prepare_run(_EXAMPLE_RAW, n_paths=2)
    unexpanded_spec = _EXAMPLE_RAW.policies[0]

    with pytest.raises(ValueError, match=re.escape(f"spec {unexpanded_spec.name!r} is not one of")):
        evaluate(prepared, unexpanded_spec)


def test_evaluate_accepts_an_equal_copy_of_an_expanded_spec() -> None:
    prepared = prepare_run(_EXAMPLE_RAW, n_paths=2)
    spec = prepared.scenario.policies[0]
    copy = spec.model_copy()
    assert copy is not spec
    assert copy == spec

    result = evaluate(prepared, copy)
    assert isinstance(result, SimulationResult)


# =============================================================================
# death_months is forwarded
# =============================================================================


def test_death_months_is_forwarded() -> None:
    prepared = prepare_run(_COUPLE_RAW, n_paths=50)
    spec = prepared.scenario.policies[0]
    start_year = prepared.scenario.start_year
    forced_month = 27
    expected_year = start_year + forced_month // MONTHS_PER_YEAR

    forced_result = evaluate(prepared, spec, death_months=(forced_month, None))
    assert np.all(forced_result.death_year[0] == expected_year)

    unforced_result = evaluate(prepared, spec)
    assert not np.all(unforced_result.death_year[0] == expected_year)

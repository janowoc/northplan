# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A LIF that reaches someone below the unlocking age no longer stops a run (#124).

Two ends of one defect: a LIF stated for someone too young is refused at load, and a
LIF paid at a first death to a survivor too young for one runs to the end, because it
is paid into the survivor's RRIF. Scenarios are written to ``tmp_path`` from
``scenarios/minimal.yaml``. The unlocking age is read from ``params/2026/ab.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from engine.core.timeline import age_at_start_of_year
from engine.mc.prepare import evaluate, prepare_run
from engine.params.loader import load_year
from engine.scenario import LockedInAgeError, load_scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
MINIMAL = REPO_ROOT / "scenarios" / "minimal.yaml"
YOUNG_PATHS = 2000


def _unlocking_age() -> int:
    return load_year(2026).jurisdiction("ab").integer("lif.unlocking_age_years")


def _written(tmp_path: Path, values: dict[str, Any]) -> Path:
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def _minimal() -> dict[str, Any]:
    values = yaml.safe_load(MINIMAL.read_text(encoding="utf-8"))
    assert isinstance(values, dict)
    return values


def test_a_lif_stated_below_the_unlocking_age_is_refused_at_load(tmp_path: Path) -> None:
    values = _minimal()
    person = values["household"]["persons"][0]
    person["birth_year"] = 1981
    person["accounts"] = {"lif": {"balance": 10_000, "jurisdiction": "ab"}}
    scenario = load_scenario(_written(tmp_path, values))
    assert age_at_start_of_year(1981, 1, scenario.start_year) < _unlocking_age()  # guard

    with pytest.raises(LockedInAgeError) as excinfo:
        prepare_run(scenario)

    assert "'a'" in str(excinfo.value)
    assert str(age_at_start_of_year(1981, 1, scenario.start_year)) in str(excinfo.value)


def test_a_lif_holder_dying_before_a_young_survivor_runs_to_the_end(tmp_path: Path) -> None:
    values = _minimal()
    values["seed"] = 7
    holder = values["household"]["persons"][0]
    holder["birth_year"] = 1964
    holder["accounts"] = {"lif": {"balance": 100_000, "jurisdiction": "ab"}}
    survivor = {
        "id": "b",
        "birth_year": 1980,
        "birth_month": 1,
        "sex": "f",
        "cpp": {"contributory_history": 0.0},
        "prior_year_net_income": 0,
        "net_income_two_years_prior": 0,
    }
    values["household"]["persons"].append(survivor)
    elections = values["policies"][0]["elections"]
    elections["cpp_start_age_years"] = {"a": 65, "b": 65}
    elections["oas_start_age_years"] = {"a": 65, "b": 65}
    scenario = load_scenario(_written(tmp_path, values))

    prepared = prepare_run(scenario, n_paths=YOUNG_PATHS)
    result = evaluate(prepared, prepared.scenario.policies[0])

    death_a, death_b = result.death_year
    unlocking_age = _unlocking_age()
    too_young = np.array(
        [age_at_start_of_year(1980, 1, int(y) + 1) < unlocking_age for y in death_a]
    )
    reached = (death_a < death_b) & too_young
    assert np.any(reached)  # guard: some path pays a LIF to a survivor too young to hold one
    assert np.all(np.isfinite(result.estate_after_tax))

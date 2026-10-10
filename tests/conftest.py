# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Every scenario a test builds is one ``prepare_run`` would accept at load.

``engine.core.build.build_initial_state`` does not run the load checks; only
:func:`engine.mc.prepare.prepare_run` does. ``build_initial_state`` calls
``engine.core.build._select_policy`` by name on every build, to choose the policy
whose elections it takes. The session-scoped autouse fixture
:func:`_load_checks_on_every_build` replaces that name with
:func:`_checked_select` before any other fixture is set up, so every build in a
test or fixture is checked, whatever name or object it was called through. A
build during collection runs before the fixture and is not checked. A test that
reloads ``engine.core.build``, or assigns ``_select_policy`` without undoing it,
turns the guard off for every test after it; one that patches it through the
``monkeypatch`` fixture turns the guard off for that test alone.

:func:`_checked_select` runs what ``prepare_run`` runs on the scenario:
:func:`~engine.policy.build.expand_grid`, the parameter year for ``start_year``,
:func:`~engine.scenario.start_ages.check_start_ages`, then
:func:`~engine.scenario.lifespan.check_lifespan`. It then validates the scenario,
with only the chosen policy and no grid, against the schema, and checks that
policy's own elections, because a build can take a policy ``prepare_run`` never
sees: one passed as ``policy=``, or one written under a grid that replaces its
values. A refusal, or a start year with no parameter directory, fails the test
through ``pytest.fail``, which ``pytest.raises(ValueError)`` and
``except Exception`` do not catch. The checks use the repository's ``params/``,
whatever root a test passes to ``prepare_run``; a test whose synthetic root
disagrees with it opts out and says so.

A test that must build a scenario or policy the checks refuse opts out with
``@pytest.mark.unchecked_scenario(reason="...")``; the reason is required, as a
keyword, and must not be blank. A build is checked or not by the marker of the
test whose protocol is running, so a fixture wider than function scope is
judged by the first test that requests it.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from types import MappingProxyType

import pydantic
import pytest

import engine.core.build
from engine.params.loader import ParamYear, ParamYearMissingError, load_year
from engine.policy.build import expand_grid
from engine.scenario.lifespan import check_lifespan
from engine.scenario.load import ScenarioError
from engine.scenario.schema import PolicySpec, Scenario
from engine.scenario.start_ages import check_start_ages

_UNCHECKED_SELECT = engine.core.build._select_policy

_MARKER = "unchecked_scenario"
_OPT_OUT = f"@pytest.mark.{_MARKER}(reason=...)"

#: Each parameter year already loaded. A missing year is not cached.
_PARAM_YEARS: dict[int, ParamYear] = {}

#: The test whose protocol is running; ``None`` between tests.
_current_item: pytest.Item | None = None


def _param_year(year: int) -> ParamYear:
    if year not in _PARAM_YEARS:
        _PARAM_YEARS[year] = load_year(year)
    return _PARAM_YEARS[year]


@functools.wraps(_UNCHECKED_SELECT)
def _checked_select(scenario: Scenario, policy: PolicySpec | None) -> PolicySpec:
    """Choose the policy as the build would, then run the load checks before returning it.

    Skips the checks when the running test carries the opt-out marker.
    """
    chosen = _UNCHECKED_SELECT(scenario, policy)
    if _current_item is not None and _current_item.get_closest_marker(_MARKER) is not None:
        return chosen
    try:
        expanded = expand_grid(scenario)
        params = _param_year(expanded.start_year)
        check_start_ages(expanded, params)
        check_lifespan(expanded, params)
        only_chosen = scenario.model_copy(
            update={"policies": (chosen,), "grid": MappingProxyType({})}
        )
        Scenario.model_validate(only_chosen.model_dump(warnings=False))
        check_start_ages(only_chosen, params)
    except (ScenarioError, ParamYearMissingError, pydantic.ValidationError) as error:
        pytest.fail(
            "build_initial_state was given a scenario, or a policy for it, that the load "
            f"checks refuse: {type(error).__name__}: {error}\n"
            f"Fix it, or mark the test {_OPT_OUT} to say why it must build it."
        )
    return chosen


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item) -> Iterator[None]:
    global _current_item
    _current_item = item
    yield
    _current_item = None


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Fail, before any fixture is set up, a test whose opt-out has no usable reason."""
    marker = item.get_closest_marker(_MARKER)
    if marker is None:
        return
    reason = marker.kwargs.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        pytest.fail(
            f"{_OPT_OUT} needs a non-blank reason, given as a keyword; got "
            f"args={marker.args!r}, kwargs={marker.kwargs!r}."
        )


@pytest.fixture(scope="session", autouse=True)
def _load_checks_on_every_build() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(engine.core.build, "_select_policy", _checked_select)
        yield


@pytest.fixture
def lock() -> Iterator[Callable[[Path, int], None]]:
    """``lock(path, mode)`` chmods ``path`` for this test; modes are restored after it.

    Skips the test for the superuser, whom directory permissions do not stop. Lock an
    inner path before the directory holding it.
    """
    if os.geteuid() == 0:
        pytest.skip("the superuser is not stopped by directory permissions")
    saved: list[tuple[Path, int]] = []

    def _lock(path: Path, mode: int) -> None:
        saved.append((path, path.stat().st_mode & 0o7777))
        path.chmod(mode)

    try:
        yield _lock
    finally:
        for path, mode in reversed(saved):
            path.chmod(mode)

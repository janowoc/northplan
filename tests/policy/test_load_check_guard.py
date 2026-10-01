# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The load-check guard in ``tests/conftest.py``.

The refusal tests build through this module's own by-name import of
``build_initial_state``, the way most tests do; one test also builds through a
module attribute, a ``functools.partial``, a class attribute and a default
argument. The ``pytester`` tests run a copy of the guard in
an inner session, in process, under the repository's ``pyproject.toml``. They
opt out of the outer guard, so that the copy alone decides what the inner
session checks.
"""

from __future__ import annotations

import contextlib
import functools
import shutil
from pathlib import Path
from types import MappingProxyType

import pytest

import engine.core.build
from engine.core.build import build_initial_state
from engine.params.loader import DEFAULT_PARAMS_ROOT, load_year
from engine.policy.build import expand_grid
from engine.scenario.schema import CppEntitlement, OasEntitlement

pytest_plugins = ["pytester"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GUARD_SOURCE = _REPO_ROOT / "tests" / "conftest.py"

#: The opt-out reason of every pytester test here.
_INNER_REASON = (
    "its inner session runs a copy of the guard, which alone must decide what is checked"
)

#: A start year with no parameter directory; the test using it checks that premise.
_YEAR_WITHOUT_PARAMS = 2099


def _with_person_a(scenario, **update):
    person_a = scenario.household.persons[0].model_copy(update=update)
    new_household = scenario.household.model_copy(update={"persons": (person_a,)})
    return scenario.model_copy(update={"household": new_household})


def _with_cpp_in_pay(scenario):
    """``scenario`` with person ``a``, 718 months old at the opening, stating CPP in pay."""
    return _with_person_a(scenario, cpp=CppEntitlement(in_pay_monthly=1200.0))


def _aged_66(scenario, written_cpp_years):
    """Person ``a`` born 1959-03, 802 months (66 years 10 months) at the opening, OAS in pay.

    The policy's written CPP election is ``written_cpp_years``; the grid tries 66 and 70.
    """
    aged = _with_person_a(scenario, birth_year=1959, oas=OasEntitlement(in_pay_monthly=700.0))
    written = aged.policies[0]
    elections = written.elections.model_copy(
        update={"cpp_start_age_years": MappingProxyType({"a": written_cpp_years})}
    )
    return aged.model_copy(
        update={
            "policies": (written.model_copy(update={"elections": elections}),),
            "grid": MappingProxyType({"elections.cpp_start_age_years.a": (66, 70)}),
        }
    )


class _BuildHolder:
    build = staticmethod(build_initial_state)


def _build_by_default(scenario, build=build_initial_state):
    return build(scenario, n_paths=1)


_ROUTES = {
    "by_name": lambda scenario: build_initial_state(scenario, n_paths=1),
    "module_attribute": lambda scenario: engine.core.build.build_initial_state(scenario, n_paths=1),
    "partial": functools.partial(build_initial_state, n_paths=1),
    "class_attribute": lambda scenario: _BuildHolder.build(scenario, n_paths=1),
    "default_argument": _build_by_default,
}


def test_the_policy_choice_is_the_guards(request: pytest.FixtureRequest) -> None:
    guard = request.config.pluginmanager.get_plugin(str(_GUARD_SOURCE))
    assert guard is not None
    assert engine.core.build._select_policy is guard._checked_select
    assert build_initial_state is engine.core.build.build_initial_state


@pytest.mark.parametrize("route", list(_ROUTES.values()), ids=list(_ROUTES))
def test_every_route_into_a_build_is_checked(scenario, route) -> None:
    with pytest.raises(pytest.fail.Exception, match="CPP in pay"):
        route(_with_cpp_in_pay(scenario))


def test_a_start_age_refusal_fails_the_test(scenario) -> None:
    with pytest.raises(pytest.fail.Exception, match="unchecked_scenario") as failure:
        build_initial_state(_with_cpp_in_pay(scenario), n_paths=1)
    assert "StartAgeNotAllowedError" in str(failure.value)
    assert "CPP in pay" in str(failure.value)


def test_a_grid_value_is_checked(scenario) -> None:
    # 742 months (61 years 10 months) at the opening: the written CPP election at 65 is not
    # past, but the grid's 60 is.
    born_1964 = _with_person_a(scenario, birth_year=1964)
    assert 60 in born_1964.grid["elections.cpp_start_age_years.a"]
    with pytest.raises(pytest.fail.Exception, match="StartAgeNotAllowedError") as failure:
        build_initial_state(born_1964, n_paths=1)
    assert "cpp_start_age_years.a=60" in str(failure.value)


def test_a_lifespan_refusal_fails_the_test(scenario) -> None:
    terminal_age = int(load_year(scenario.start_year)["mortality"].number("terminal_age_years"))
    too_old = _with_person_a(
        scenario,
        birth_year=scenario.start_year - terminal_age - 2,
        cpp=CppEntitlement(in_pay_monthly=1200.0),
        oas=OasEntitlement(in_pay_monthly=700.0),
    )
    with pytest.raises(pytest.fail.Exception, match="LifespanNotRepresentableError"):
        build_initial_state(too_old, n_paths=1)


def test_a_scenario_the_schema_refuses_on_expansion_fails_the_test(scenario) -> None:
    # model_copy skips validation; expanding the grid revalidates, and the
    # example's first spending band begins after this start year.
    assert scenario.grid
    assert scenario.spending.schedule[0].from_year > 2025
    invalid = scenario.model_copy(update={"start_year": 2025})
    with pytest.raises(pytest.fail.Exception, match="ValidationError"):
        build_initial_state(invalid, n_paths=1)


def test_a_start_year_without_parameters_fails_the_test(scenario) -> None:
    assert not (DEFAULT_PARAMS_ROOT / str(_YEAR_WITHOUT_PARAMS)).exists()
    no_params = scenario.model_copy(
        update={"start_year": _YEAR_WITHOUT_PARAMS, "grid": MappingProxyType({})}
    )
    with pytest.raises(pytest.fail.Exception, match="ParamYearMissingError") as failure:
        build_initial_state(no_params, n_paths=1)
    assert "unchecked_scenario" in str(failure.value)


def test_a_policy_passed_outside_the_scenario_is_checked(scenario) -> None:
    aged = _aged_66(scenario, written_cpp_years=66)
    written = aged.policies[0]
    elections = written.elections.model_copy(
        update={"cpp_start_age_years": MappingProxyType({"a": 65})}
    )
    outside = written.model_copy(update={"name": "outside", "elections": elections})
    with pytest.raises(pytest.fail.Exception, match="StartAgeNotAllowedError") as failure:
        build_initial_state(aged, n_paths=1, policy=outside)
    assert "policies['outside'].elections.cpp_start_age_years['a']" in str(failure.value)
    assert "is below the age of 'a'" in str(failure.value)


def test_a_policy_from_the_expansion_builds(scenario) -> None:
    aged = _aged_66(scenario, written_cpp_years=66)
    for policy in (None, *expand_grid(aged).policies):
        build_initial_state(aged, n_paths=1, policy=policy)


def test_a_written_election_the_grid_replaces_is_checked(scenario) -> None:
    aged = _aged_66(scenario, written_cpp_years=65)
    with pytest.raises(pytest.fail.Exception, match="StartAgeNotAllowedError") as failure:
        build_initial_state(aged, n_paths=1)
    assert "policies['taxable-first'].elections.cpp_start_age_years['a']" in str(failure.value)
    assert "is below the age of 'a'" in str(failure.value)


def test_a_policy_the_schema_refuses_is_checked(scenario) -> None:
    written = scenario.policies[0]
    rrif_conversion = written.elections.rrif_conversion.model_copy(update={"fraction": 1.5})
    elections = written.elections.model_copy(update={"rrif_conversion": rrif_conversion})
    unvalidated = written.model_copy(update={"elections": elections})
    with pytest.raises(pytest.fail.Exception, match="ValidationError") as failure:
        build_initial_state(scenario, n_paths=1, policy=unvalidated)
    assert "rrif_conversion.fraction" in str(failure.value)
    assert "less than or equal to 1" in str(failure.value)


def test_a_refusal_is_not_swallowed_by_a_broad_except(scenario) -> None:
    with pytest.raises(pytest.fail.Exception), contextlib.suppress(Exception):
        build_initial_state(_with_cpp_in_pay(scenario), n_paths=1)


def test_an_accepted_scenario_builds(scenario) -> None:
    state = build_initial_state(scenario, n_paths=1)
    assert state.persons[0].cpp.in_pay_monthly is None


@pytest.mark.unchecked_scenario(reason="proves the opt-out builds a refused scenario")
def test_an_opted_out_test_builds_a_refused_scenario(scenario) -> None:
    state = build_initial_state(_with_cpp_in_pay(scenario), n_paths=1)
    assert (state.persons[0].cpp.in_pay_monthly == 1200.0).all()


def _inner_session(
    request: pytest.FixtureRequest, pytester: pytest.Pytester, files: dict[str, str]
) -> pytest.HookRecorder:
    """Run ``files`` (paths under ``tests/``) beside a copy of the guard, in process.

    The calling test must opt out of the outer guard; the outer guard's patch must be
    the same object after the inner session as before it.
    """
    assert request.node.get_closest_marker("unchecked_scenario") is not None
    outer_select = engine.core.build._select_policy
    shutil.copyfile(_REPO_ROOT / "pyproject.toml", pytester.path / "pyproject.toml")
    tests = pytester.mkdir("tests")
    (tests / "conftest.py").write_text(_GUARD_SOURCE.read_text())
    for relative, text in files.items():
        path = tests / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    recorder = pytester.inline_run()
    assert engine.core.build._select_policy is outer_select
    return recorder


_OPT_OUT_REASONS = """\
import pytest


@pytest.mark.unchecked_scenario(reason="a reason")
def test_with_reason():
    pass


@pytest.mark.unchecked_scenario
def test_without_reason():
    pass


@pytest.mark.unchecked_scenario(reason="  ")
def test_blank_reason():
    pass


@pytest.mark.unchecked_scenario("positional")
def test_positional_reason():
    pass
"""


@pytest.mark.unchecked_scenario(reason=_INNER_REASON)
def test_an_opt_out_needs_a_non_blank_keyword_reason(
    request: pytest.FixtureRequest, pytester: pytest.Pytester
) -> None:
    recorder = _inner_session(request, pytester, {"test_inner.py": _OPT_OUT_REASONS})

    passed, _skipped, failed = recorder.listoutcomes()
    assert [report.nodeid.split("::")[-1] for report in passed] == ["test_with_reason"]
    assert sorted(report.nodeid.split("::")[-1] for report in failed) == [
        "test_blank_reason",
        "test_positional_reason",
        "test_without_reason",
    ]
    for report in failed:
        assert report.when == "setup"
        assert "needs a non-blank reason" in str(report.longrepr)


_HOLDER_CONFTEST = f"""\
from pathlib import Path

import pytest

from engine.core.build import build_initial_state
from engine.scenario import load_scenario
from engine.scenario.schema import CppEntitlement


@pytest.fixture
def refused_build():
    scenario = load_scenario(Path({str(_REPO_ROOT / "scenarios" / "example.yaml")!r}))
    person_a = scenario.household.persons[0].model_copy(
        update={{"cpp": CppEntitlement(in_pay_monthly=1200.0)}}
    )
    household = scenario.household.model_copy(update={{"persons": (person_a,)}})
    return build_initial_state(scenario.model_copy(update={{"household": household}}), n_paths=1)
"""

_HOLDER_TESTS = """\
import sys
from pathlib import Path

HOLDER = str(Path(__file__).with_name("conftest.py"))


def test_premise_the_holder_is_a_plugin_but_not_in_sys_modules(request):
    plugins = [
        plugin
        for plugin in request.config.pluginmanager.get_plugins()
        if getattr(plugin, "__file__", None) == HOLDER
    ]
    assert len(plugins) == 1
    assert all(module is not plugins[0] for module in sys.modules.values())


def test_a_build_through_the_holder(refused_build):
    pass
"""


@pytest.mark.unchecked_scenario(reason=_INNER_REASON)
def test_a_conftest_held_only_as_a_plugin_is_checked(
    request: pytest.FixtureRequest, pytester: pytest.Pytester
) -> None:
    # holder/conftest.py is imported as "conftest"; collecting later/ imports
    # another "conftest", which pytest first removes from sys.modules.
    recorder = _inner_session(
        request,
        pytester,
        {
            "holder/conftest.py": _HOLDER_CONFTEST,
            "holder/test_holder.py": _HOLDER_TESTS,
            "later/conftest.py": "",
            "later/test_later.py": "def test_nothing():\n    pass\n",
        },
    )

    passed, _skipped, failed = recorder.listoutcomes()
    assert sorted(report.nodeid.split("::")[-1] for report in passed) == [
        "test_nothing",
        "test_premise_the_holder_is_a_plugin_but_not_in_sys_modules",
    ]
    assert [report.nodeid.split("::")[-1] for report in failed] == [
        "test_a_build_through_the_holder"
    ]
    assert failed[0].when == "setup"
    assert "StartAgeNotAllowedError" in str(failed[0].longrepr)


_WIDER_SCOPES = (
    f"EXAMPLE = {str(_REPO_ROOT / 'scenarios' / 'example.yaml')!r}\n"
    + """\
from pathlib import Path

import pytest

from engine.core.build import build_initial_state
from engine.scenario import load_scenario
from engine.scenario.schema import CppEntitlement


def _refused():
    scenario = load_scenario(Path(EXAMPLE))
    person_a = scenario.household.persons[0].model_copy(
        update={"cpp": CppEntitlement(in_pay_monthly=1200.0)}
    )
    household = scenario.household.model_copy(update={"persons": (person_a,)})
    return scenario.model_copy(update={"household": household})


@pytest.fixture(scope="module")
def built_per_module():
    return build_initial_state(_refused(), n_paths=1)


@pytest.fixture(scope="session")
def built_per_session():
    return build_initial_state(_refused(), n_paths=1)


def test_module_scoped(built_per_module):
    pass


def test_session_scoped(built_per_session):
    pass
"""
)


@pytest.mark.unchecked_scenario(reason=_INNER_REASON)
def test_a_build_in_a_wider_scoped_fixture_is_checked(
    request: pytest.FixtureRequest, pytester: pytest.Pytester
) -> None:
    recorder = _inner_session(request, pytester, {"test_wider.py": _WIDER_SCOPES})

    passed, _skipped, failed = recorder.listoutcomes()
    assert passed == []
    assert sorted(report.nodeid.split("::")[-1] for report in failed) == [
        "test_module_scoped",
        "test_session_scoped",
    ]
    for report in failed:
        assert report.when == "setup"
        assert "StartAgeNotAllowedError" in str(report.longrepr)

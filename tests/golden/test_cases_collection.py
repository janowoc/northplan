# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Collect ``test_cases.py`` itself against a ``cases/`` built in a temporary directory.

Each test copies the text of the real ``test_cases.py``, or that text with its
``if CASES:`` guard removed, into a package inside ``pytester``'s temporary
directory, beside a ``cases/`` directory it fills (or leaves empty, or leaves
out), and runs pytest's collection on that package in process. The package's
``conftest.py`` re-exports the real harness from ``golden.conftest`` and
rebinds only ``CASES_DIR`` to the package's own ``cases/``. The repository's
``pyproject.toml`` is copied beside the package, so the inner run collects
under the repository's pytest settings, among them
``empty_parameter_set_mark = "fail_at_collect"`` and the registered ``golden``
marker.

The package is not named ``golden``: the outer session has already imported
that name, and an in-process run would resolve the copy to the real modules.

Every case written here is obviously synthetic and lives in the temporary
directory, never under ``tests/golden/cases/``.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

_GOLDEN_DIR = Path(__file__).parent
_REPO_ROOT = _GOLDEN_DIR.parents[1]

#: The name of the copied package; anything but ``golden``.
_COPY_PACKAGE = "copied_golden"

#: The copied package's conftest: the real harness, with ``CASES_DIR`` rebound.
_CONFTEST = f"""\
from pathlib import Path

from {__package__}.conftest import *

CASES_DIR = Path(__file__).parent / "cases"
"""

#: One obviously synthetic case, doubling seven.
_SYNTHETIC_CASE = f"""\
target: {__package__}.test_harness.double
cases:
  - name: seven doubled
    source: "synthetic"
    checked: 2026-01-01
    inputs: {{x: 7.0}}
    expected: {{value: 14.0}}
"""

#: The node id of the copied ``test_cases.py`` in the inner run.
_MODULE_NODEID = f"{_COPY_PACKAGE}/test_cases.py"

_GUARD = "\nif CASES:\n"


def _real_source() -> str:
    return (_GOLDEN_DIR / "test_cases.py").read_text()


def _unguarded(source: str) -> str:
    """``source`` with the ``if CASES:`` guard removed and its body dedented."""
    assert source.count(_GUARD) == 1, "test_cases.py no longer has exactly one `if CASES:` guard"
    head, _, guarded = source.partition(_GUARD)
    return head + "\n" + textwrap.dedent(guarded)


def _collect(
    pytester: pytest.Pytester, source: str, case_files: dict[str, str] | None
) -> tuple[list[pytest.Item], pytest.HookRecorder]:
    """Collect ``source`` as ``test_cases.py`` beside a ``cases/`` holding ``case_files``.

    ``case_files`` maps a file name to its text; ``None`` leaves ``cases/``
    out. Returns the collected items and the inner run's hook recorder.
    """
    shutil.copyfile(_REPO_ROOT / "pyproject.toml", pytester.path / "pyproject.toml")
    package = pytester.path / _COPY_PACKAGE
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "conftest.py").write_text(_CONFTEST)
    (package / "test_cases.py").write_text(source)
    if case_files is not None:
        (package / "cases").mkdir()
        for name, text in case_files.items():
            (package / "cases" / name).write_text(text)
    return pytester.inline_genitems(package)


def _golden(items: list[pytest.Item]) -> list[pytest.Item]:
    return [item for item in items if item.get_closest_marker("golden") is not None]


def _module_report(recorder: pytest.HookRecorder) -> pytest.CollectReport:
    """The inner run's one collection report for the copied ``test_cases.py``."""
    all_reports = recorder.getreports("pytest_collectreport")
    reports = [report for report in all_reports if report.nodeid == _MODULE_NODEID]
    assert len(reports) == 1, (
        f"expected one collection report for {_MODULE_NODEID}, "
        f"saw {[report.nodeid for report in all_reports]}"
    )
    return reports[0]


@pytest.mark.parametrize("case_files", [{}, {".gitkeep": ""}], ids=["empty", "gitkeep_only"])
def test_an_empty_cases_directory_collects_no_golden_item_and_no_error(
    pytester: pytest.Pytester, case_files: dict[str, str]
) -> None:
    items, recorder = _collect(pytester, _real_source(), case_files)

    assert recorder.ret == pytest.ExitCode.NO_TESTS_COLLECTED
    assert recorder.getfailedcollections() == []
    assert _module_report(recorder).passed
    assert _golden(items) == []
    assert items == []


def test_one_synthetic_case_collects_exactly_one_golden_item(pytester: pytest.Pytester) -> None:
    items, recorder = _collect(pytester, _real_source(), {"synthetic.yaml": _SYNTHETIC_CASE})

    assert recorder.ret == pytest.ExitCode.OK
    assert recorder.getfailedcollections() == []
    assert len(_golden(items)) == 1


def test_a_missing_cases_directory_is_a_collection_error(pytester: pytest.Pytester) -> None:
    items, recorder = _collect(pytester, _real_source(), None)

    assert recorder.ret == pytest.ExitCode.INTERRUPTED
    assert items == []
    failed = recorder.getfailedcollections()
    assert len(failed) == 1
    assert failed[0].nodeid == _MODULE_NODEID
    assert "cases directory does not exist" in str(failed[0].longrepr)


def test_without_the_guard_an_empty_cases_directory_fails_at_collection(
    pytester: pytest.Pytester,
) -> None:
    items, recorder = _collect(pytester, _unguarded(_real_source()), {})

    assert recorder.ret == pytest.ExitCode.INTERRUPTED
    assert items == []
    failed = recorder.getfailedcollections()
    assert len(failed) == 1
    assert failed[0].nodeid == _MODULE_NODEID
    assert "Empty parameter set in 'test_golden_case'" in str(failed[0].longrepr)


def test_without_the_guard_one_synthetic_case_still_collects_one_golden_item(
    pytester: pytest.Pytester,
) -> None:
    """The unguarded copy is sound, so the failure above comes from the empty directory."""
    items, recorder = _collect(
        pytester, _unguarded(_real_source()), {"synthetic.yaml": _SYNTHETIC_CASE}
    )

    assert recorder.ret == pytest.ExitCode.OK
    assert recorder.getfailedcollections() == []
    assert len(_golden(items)) == 1

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``tests/golden/targets.py``.

Never reads the real ``params/`` tree: every parameter used below is a
synthetic fixture, written to ``tmp_path``, copied from
``tests/tax/test_combined.py``'s ``SYNTHETIC_FEDERAL_FLAT_RATE_TIE``,
``SYNTHETIC_AB_FLAT_RATE_TIE``, and ``SYNTHETIC_OAS_FLAT_RATE_TIE`` (they
cannot be imported across test directories). Obviously synthetic, never a
real tax parameter — see the ``SYNTHETIC TEST FIXTURE`` header on each.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import numpy as np
import pytest

from engine.core.indexation import RealParamYear, real_year
from engine.core.state import IncomeLedger
from engine.params.loader import load_year
from engine.tax import combined
from engine.tax.combined import Assessment

from . import targets
from .conftest import discover_cases, resolve_target, run_case

_THIS_MODULE = targets.__name__

#: The synthetic tax year these tests load, kept out of the way of any year a
#: real case under ``params/`` names.
_SYNTHETIC_YEAR = 2030

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
#: Copied verbatim from ``tests/tax/test_combined.py`` (cannot be imported
#: across test directories).
_SYNTHETIC_FEDERAL = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  brackets_and_credits:
    adjustment_months: [1]
    applies_to:
      - brackets.edges_annual
      - credits.basic_personal_amount_annual
      - credits.age_amount.amount_annual
      - credits.age_amount.reduction_threshold_annual
      - contribution_credit.cpp_maximum_annual
      - contribution_credit.ei_maximum_annual
  unindexed:
    adjustment_months: []
    applies_to:
      - credits.pension_income_amount_annual

brackets:
  edges_annual: []
  rates: [0.2]

credits:
  valuation_rate: 0.1
  basic_personal_amount_annual: 5000
  pension_income_amount_annual: 2000
  age_amount:
    eligibility_age_years: 999
    amount_annual: 1000
    reduction_threshold_annual: 100000
    reduction_rate: 0.1

contribution_credit:
  cpp_maximum_annual: 100
  ei_maximum_annual: 50

investment_income:
  capital_gains_inclusion_rate: 0.5
  eligible_dividend_gross_up_rate: 0.38
  eligible_dividend_credit_rate_of_gross_up: 0.3

pension_splitting:
  maximum_transfer_share: 0.5

eligible_pension_income:
  rrif_minimum_age_years: 65
"""

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
_SYNTHETIC_AB = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  brackets_and_credits:
    adjustment_months: [1]
    applies_to:
      - brackets.edges_annual
      - credits.basic_personal_amount_annual
      - credits.pension_income_amount_annual
      - credits.age_amount.amount_annual
      - credits.age_amount.reduction_threshold_annual

brackets:
  edges_annual: []
  rates: [0.1]

credits:
  valuation_rate: 0.08
  basic_personal_amount_annual: 4000
  pension_income_amount_annual: 1500
  age_amount:
    eligibility_age_years: 999
    amount_annual: 800
    reduction_threshold_annual: 100000
    reduction_rate: 0.1

investment_income:
  eligible_dividend_gross_up_rate: 0.38
  eligible_dividend_credit_rate_of_gross_up: 0.2
"""

#: SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
_SYNTHETIC_OAS = """
# SYNTHETIC TEST FIXTURE — these are not tax parameters and never were.
indexation:
  recovery_tax_threshold:
    adjustment_months: [1]
    applies_to:
      - recovery_tax.threshold_annual

recovery_tax:
  threshold_annual: 1000000
  rate: 0.15
"""


def _write_synthetic_params(root: Path) -> None:
    year_dir = root / str(_SYNTHETIC_YEAR)
    year_dir.mkdir(parents=True)
    (year_dir / "federal.yaml").write_text(_SYNTHETIC_FEDERAL, encoding="utf-8")
    (year_dir / "ab.yaml").write_text(_SYNTHETIC_AB, encoding="utf-8")
    (year_dir / "oas.yaml").write_text(_SYNTHETIC_OAS, encoding="utf-8")


@pytest.fixture
def params_root(tmp_path: Path) -> Path:
    """A ``tmp_path/"params"`` root holding the three synthetic files, at ``_SYNTHETIC_YEAR``."""
    root = tmp_path / "params"
    _write_synthetic_params(root)
    return root


@pytest.fixture
def params(params_root: Path) -> RealParamYear:
    """The synthetic year, in real dollars at zero inflation."""
    return real_year(load_year(_SYNTHETIC_YEAR, params_root), 0.0)


@pytest.fixture
def synthetic_params_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Same synthetic files, reachable through the harness's own resolution.

    Patches ``DEFAULT_PARAMS_ROOT`` on this package's ``conftest`` module,
    imported relatively — the name ``_load_year_cached`` actually reads at
    call time — rather than adding a root parameter to any harness function.
    The cache is cleared both before and after, exactly like
    ``synthetic_params_root`` in ``test_harness.py``. ``cases/`` is kept as a
    separate subdirectory of ``tmp_path``: ``discover_cases`` walks its own
    directory recursively and would reject a params YAML sitting inside it.
    """
    from . import conftest as conftest_module

    root = tmp_path / "params"
    _write_synthetic_params(root)

    conftest_module._load_year_cached.cache_clear()
    monkeypatch.setattr(conftest_module, "DEFAULT_PARAMS_ROOT", root)
    yield root
    conftest_module._load_year_cached.cache_clear()


# --- signature ----------------------------------------------------------------

_LEDGER_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(IncomeLedger))
_NON_LEDGER_NAMES = frozenset(
    {"age_at_end_of_year", "province", "params", "january_month_index", "transfer_in", "transfer_out"}
)
_REQUIRED_NAMES = frozenset({"age_at_end_of_year", "province", "params", "january_month_index"})


def test_signature_parameter_names_are_exactly_ledger_fields_plus_six() -> None:
    sig = inspect.signature(targets.person_assessment)
    assert set(sig.parameters) == _LEDGER_FIELD_NAMES | _NON_LEDGER_NAMES


def test_signature_is_entirely_keyword_only() -> None:
    sig = inspect.signature(targets.person_assessment)
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in sig.parameters.values())


def test_signature_required_parameters_are_exactly_the_four() -> None:
    sig = inspect.signature(targets.person_assessment)
    no_default = {
        name for name, p in sig.parameters.items() if p.default is inspect.Parameter.empty
    }
    assert no_default == _REQUIRED_NAMES


def test_signature_parameter_order_is_required_then_transfers_then_ledger_fields() -> None:
    assert list(inspect.signature(targets.person_assessment).parameters) == [
        "age_at_end_of_year",
        "province",
        "params",
        "january_month_index",
        "transfer_in",
        "transfer_out",
        *(f.name for f in dataclasses.fields(IncomeLedger)),
    ]


# --- pure construction: all-zero, shape, unknown keyword -----------------------


def test_only_required_arguments_gives_an_all_zero_assessment(params: RealParamYear) -> None:
    result = targets.person_assessment(
        age_at_end_of_year=70.0,
        province="ab",
        params=params,
        january_month_index=0,
    )
    for field in dataclasses.fields(Assessment):
        np.testing.assert_array_equal(getattr(result, field.name), np.zeros(1))


def test_every_assessment_field_has_shape_one_path(params: RealParamYear) -> None:
    result = targets.person_assessment(
        age_at_end_of_year=70.0,
        province="ab",
        params=params,
        january_month_index=0,
    )
    for field in dataclasses.fields(Assessment):
        assert getattr(result, field.name).shape == (1,)


def test_unknown_keyword_raises_type_error_naming_it(params: RealParamYear) -> None:
    with pytest.raises(TypeError, match="not_a_real_field"):
        targets.person_assessment(
            age_at_end_of_year=70.0,
            province="ab",
            params=params,
            january_month_index=0,
            not_a_real_field=1.0,
        )


# --- wiring: every argument reaches engine.tax.combined.person_assessment -----


def test_shim_wires_every_argument_through_to_the_engine_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every numeric argument becomes a distinct ``(1,)`` float64 array on the captured call.

    The stand-in records ``engine.tax.combined.person_assessment``'s keyword
    arguments and returns a sentinel unchanged, so that a wrong wiring (a
    field left off, a value dropped, a value routed to the wrong field) is
    caught here even though ``targets.person_assessment``'s own return value
    would otherwise hide it.
    """
    captured: dict[str, object] = {}
    sentinel = object()

    def _recording_stand_in(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(combined, "person_assessment", _recording_stand_in)

    ledger_names = tuple(f.name for f in dataclasses.fields(IncomeLedger))
    numeric_names = (*ledger_names, "age_at_end_of_year", "transfer_in", "transfer_out")
    values = {name: float(index + 1) for index, name in enumerate(numeric_names)}

    province = "ab"
    params_object = object()
    january_month_index = 3

    result = targets.person_assessment(
        province=province,
        params=params_object,
        january_month_index=january_month_index,
        **values,
    )

    assert result is sentinel
    assert captured["province"] is province
    assert captured["params"] is params_object
    assert captured["january_month_index"] is january_month_index

    for name in ("age_at_end_of_year", "transfer_in", "transfer_out"):
        array = captured[name]
        assert isinstance(array, np.ndarray)
        assert array.shape == (1,)
        assert array.dtype == np.float64
        np.testing.assert_array_equal(array, np.array([values[name]]))

    ledger = captured["ledger"]
    assert isinstance(ledger, IncomeLedger)
    for name in ledger_names:
        array = getattr(ledger, name)
        assert array.shape == (1,)
        assert array.dtype == np.float64
        np.testing.assert_array_equal(array, np.array([values[name]]))


# --- end to end, through the harness --------------------------------------------


def test_resolve_target_of_the_literal_dotted_path_is_this_module_object() -> None:
    """The dotted path a case author writes resolves to this same module object, not a re-import."""
    assert resolve_target("golden.targets.person_assessment") is targets.person_assessment


def test_golden_case_names_the_target_and_matches_direct_person_assessment(
    tmp_path: Path, synthetic_params_root: Path
) -> None:
    """A case naming ``golden.targets.person_assessment`` discovers, runs, and matches.

    The expected values are not hand-computed: they come from calling
    ``engine.tax.combined.person_assessment`` directly, on an equivalent
    hand-built ``IncomeLedger``, with the same synthetic parameters — so this
    checks that the harness reaches the shim and the shim reaches the engine
    function, not that either computes a particular number.
    """
    n = 1
    ledger = IncomeLedger(
        employment=np.array([60_000.0]),
        cpp=np.zeros(n),
        oas=np.array([8_000.0]),
        db_pension=np.array([20_000.0]),
        rrsp_withdrawals=np.zeros(n),
        rrif_lif_withdrawals=np.array([5_000.0]),
        interest=np.zeros(n),
        eligible_dividends=np.zeros(n),
        capital_gains=np.zeros(n),
        resp_accumulated_income=np.zeros(n),
        rrsp_deductions=np.zeros(n),
        cpp_base_contributions=np.array([2_000.0]),
        cpp_enhanced_contributions=np.zeros(n),
        ei_premiums=np.zeros(n),
        remitted=np.zeros(n),
    )
    age_at_end_of_year = 70.0
    transfer_in = np.array([1_000.0])
    transfer_out = np.array([1_500.0])
    province = "ab"
    january_month_index = 0

    direct_params = real_year(load_year(_SYNTHETIC_YEAR, synthetic_params_root), 0.0)
    expected = combined.person_assessment(
        ledger,
        age_at_end_of_year,
        transfer_in,
        transfer_out,
        province,
        direct_params,
        january_month_index,
    )
    expected_total = float(expected.total[0])
    expected_net_income = float(expected.net_income[0])
    assert expected_total != 0.0, "synthetic inputs must give a non-vacuous check"

    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    case_text = f"""
target: {_THIS_MODULE}.person_assessment
cases:
  - name: person_assessment shim end to end
    source: "synthetic"
    checked: 2026-01-01
    inputs:
      age_at_end_of_year: {age_at_end_of_year!r}
      province: {province!r}
      real_params_year: {{year: {_SYNTHETIC_YEAR}, inflation: 0.0}}
      january_month_index: {january_month_index}
      transfer_in: {float(transfer_in[0])!r}
      transfer_out: {float(transfer_out[0])!r}
      employment: {float(ledger.employment[0])!r}
      cpp: {float(ledger.cpp[0])!r}
      oas: {float(ledger.oas[0])!r}
      db_pension: {float(ledger.db_pension[0])!r}
      rrsp_withdrawals: {float(ledger.rrsp_withdrawals[0])!r}
      rrif_lif_withdrawals: {float(ledger.rrif_lif_withdrawals[0])!r}
      interest: {float(ledger.interest[0])!r}
      eligible_dividends: {float(ledger.eligible_dividends[0])!r}
      capital_gains: {float(ledger.capital_gains[0])!r}
      resp_accumulated_income: {float(ledger.resp_accumulated_income[0])!r}
      rrsp_deductions: {float(ledger.rrsp_deductions[0])!r}
      cpp_base_contributions: {float(ledger.cpp_base_contributions[0])!r}
      cpp_enhanced_contributions: {float(ledger.cpp_enhanced_contributions[0])!r}
      ei_premiums: {float(ledger.ei_premiums[0])!r}
      remitted: {float(ledger.remitted[0])!r}
    expected:
      total: {expected_total!r}
      net_income: {expected_net_income!r}
"""
    (cases_dir / "person_assessment.yaml").write_text(case_text, encoding="utf-8")

    cases = discover_cases(cases_dir)

    assert len(cases) == 1
    run_case(cases[0])  # must not raise

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The state classes: immutability, shape, dtype, and the invariants that are not typing.

Fixtures here are small, synthetic arrays built by hand rather than a real
scenario, so a failure points at the class under test rather than at
whatever ``scenarios/example.yaml`` happens to contain. The tree walker
itself lives in ``conftest.py`` and is imported from there, not defined
here, so ``test_build.py`` can reuse the identical check against a state
built from the example scenario.
"""

from __future__ import annotations

import dataclasses
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from engine.core.state import (
    DEATH_NOT_DRAWN,
    BeneficiaryState,
    BenefitState,
    CashState,
    Elections,
    EmploymentBand,
    HouseholdState,
    IncomeLedger,
    LifState,
    LiraState,
    PensionState,
    PersonState,
    RespState,
    RrifState,
    RrspState,
    SpendingLevel,
    TaxableState,
    TfsaState,
    YearRecord,
    freeze,
    updated,
)

from .conftest import walk

N_PATHS = 4

#: A generous floor, not an exact count: high enough that an accidentally
#: empty tree (a dropped tuple, a field quietly turned into None) cannot
#: pass, without being so exact that an unrelated field addition breaks this
#: test. The fixture below currently carries 53 arrays.
MINIMUM_ARRAYS = 40


def _zeros() -> np.ndarray:
    return np.zeros(N_PATHS, dtype=np.float64)


def _bools(value: bool) -> np.ndarray:
    return np.full(N_PATHS, value, dtype=np.bool_)


def _cash() -> CashState:
    return CashState(balance=np.full(N_PATHS, 100.0))


def _rrsp() -> RrspState:
    return RrspState(
        balance=_zeros(), room=_zeros(), contributed_ytd=_zeros(), converted_fraction_applied=False
    )


def _rrif(*, balance: float = 0.0, opened_year: int | None = None) -> RrifState:
    return RrifState(
        balance=np.full(N_PATHS, balance),
        annual_minimum=_zeros(),
        withdrawn_ytd=_zeros(),
        opened_year=opened_year,
    )


def _lira(*, balance: float = 0.0, jurisdiction: str = "") -> LiraState:
    return LiraState(balance=np.full(N_PATHS, balance), jurisdiction=jurisdiction)


def _lif(
    *, balance: float = 0.0, jurisdiction: str = "", opened_year: int | None = None
) -> LifState:
    return LifState(
        balance=np.full(N_PATHS, balance),
        jurisdiction=jurisdiction,
        annual_minimum=_zeros(),
        annual_maximum=_zeros(),
        withdrawn_ytd=_zeros(),
        opened_year=opened_year,
    )


def _tfsa() -> TfsaState:
    return TfsaState(balance=_zeros(), room=_zeros(), withdrawn_this_year=_zeros())


def _taxable() -> TaxableState:
    return TaxableState(balance=_zeros(), acb=_zeros())


def _benefit(
    *,
    start_age_months: int | None = 780,
    in_pay_monthly: np.ndarray | None = None,
    contributory_history: float | None = None,
) -> BenefitState:
    return BenefitState(
        start_age_months=start_age_months,
        in_pay_monthly=in_pay_monthly,
        contributory_history=contributory_history,
        monthly_amount=_zeros(),
    )


def _employment_band(*, from_month_index: int = -12, to_month_index: int = 60) -> EmploymentBand:
    return EmploymentBand(
        from_month_index=from_month_index,
        to_month_index=to_month_index,
        monthly_amount=np.full(N_PATHS, 5000.0),
    )


def _pension() -> PensionState:
    return PensionState(
        name="employer",
        monthly_amount=_zeros(),
        start_month_index=12,
        indexed=False,
        bridge_monthly=_zeros(),
        bridge_end_month_index=None,
        survivor_share=0.6,
    )


def _income_ledger() -> IncomeLedger:
    return IncomeLedger(*(_zeros() for _ in range(15)))


def _person(
    person_id: str = "a",
    *,
    lira: LiraState | None = None,
    lif: LifState | None = None,
    cpp: BenefitState | None = None,
) -> PersonState:
    return PersonState(
        person_id=person_id,
        sex="f",
        birth_year=1966,
        birth_month=3,
        alive=_bools(True),
        death_month_index=np.full(N_PATHS, DEATH_NOT_DRAWN, dtype=np.int64),
        rrsp=_rrsp(),
        rrif=_rrif(),
        lira=lira if lira is not None else _lira(),
        lif=lif if lif is not None else _lif(),
        tfsa=_tfsa(),
        taxable=_taxable(),
        cpp=cpp if cpp is not None else _benefit(contributory_history=0.85),
        oas=_benefit(contributory_history=None),
        employment=(_employment_band(),),
        pensions=(_pension(),),
        income=_income_ledger(),
        balance_owing=_zeros(),
        prior_year_net_income=_zeros(),
    )


def _resp() -> RespState:
    return RespState(
        contributions=_zeros(),
        grants=_zeros(),
        income=_zeros(),
        contributions_lifetime=_zeros(),
        grants_lifetime=_zeros(),
        grant_room=_zeros(),
        grant_received_ytd=_zeros(),
        contributed_ytd=_zeros(),
        subscriber_index=0,
        education_start_month_index=84,
        education_months=48,
        education_monthly_cost=1666.0,
        wound_up=_bools(False),
    )


def _beneficiary() -> BeneficiaryState:
    return BeneficiaryState(beneficiary_id="child1", birth_year=2015, birth_month=9, resp=_resp())


def _elections(*, cpp_start_age_months: tuple[int | None, ...] = (780,)) -> Elections:
    return Elections(
        cpp_start_age_months=cpp_start_age_months,
        oas_start_age_months=(780,),
        rrif_conversion_age_years=65,
        rrif_conversion_fraction=0.05,
        fill_pension_credit=True,
    )


def build_household(**overrides: object) -> HouseholdState:
    """A minimal, valid :class:`HouseholdState` for one person and one beneficiary."""
    fields_ = {
        "year": 2026,
        "month": 1,
        "month_index": 0,
        "n_paths": N_PATHS,
        "province": "ab",
        "persons": (_person(),),
        "beneficiaries": (_beneficiary(),),
        "cash": _cash(),
        "elections": _elections(),
        "spending_schedule": (SpendingLevel(from_year=2026, monthly_level=5000.0),),
        "spending_monthly": 5000.0,
        "spending_survivor_share": 0.75,
        "spending_achieved_ytd": _zeros(),
        "depleted": _bools(False),
        "estate_after_tax": np.full(N_PATHS, np.nan),
        "history": (),
    }
    fields_.update(overrides)
    return HouseholdState(**fields_)


def _year_record() -> YearRecord:
    return YearRecord(
        year=2026,
        net_worth=_zeros(),
        spending=_zeros(),
        tax_assessed=_zeros(),
        depleted=_bools(False),
    )


class TestWalker:
    def test_walk_passes_on_a_built_household(self) -> None:
        count = walk(build_household(), N_PATHS)
        assert count >= MINIMUM_ARRAYS, (
            f"walked only {count} arrays; the walk is vacuous if it can pass "
            "with an empty tree, so this floor exists to catch that."
        )

    def test_walk_fails_on_an_empty_tree(self) -> None:
        """The floor above is not vacuous: a plain scalar walks to zero."""
        assert walk(42, N_PATHS) == 0
        assert walk(None, N_PATHS) == 0
        assert walk((), N_PATHS) == 0

    def test_walk_descends_into_history(self) -> None:
        household = build_household(history=(_year_record(),))
        without_history = walk(build_household(), N_PATHS)
        with_history = walk(household, N_PATHS)
        assert with_history == without_history + 4  # net_worth, spending, tax_assessed, depleted

    def test_walk_descends_into_beneficiaries(self) -> None:
        """A delta proves descent; a floor only proves the tree is not empty."""
        without = walk(build_household(beneficiaries=()), N_PATHS)
        with_one = walk(build_household(), N_PATHS)
        assert with_one == without + 9  # RespState's nine array fields

    def test_walk_descends_into_a_second_employment_band(self) -> None:
        one_band = _person()
        two_bands = updated(one_band, employment=(*one_band.employment, _employment_band()))
        without = walk(build_household(persons=(one_band,)), N_PATHS)
        with_two = walk(build_household(persons=(two_bands,)), N_PATHS)
        assert with_two == without + 1  # EmploymentBand's one array field

    def test_walk_rejects_a_list(self) -> None:
        with pytest.raises(pytest.fail.Exception):
            walk([np.zeros(N_PATHS)], N_PATHS)


class TestWalkerFloorActuallyBites:
    """The floor in ``test_build.py`` is only meaningful if it can fail.

    ``test_walk_fails_on_an_empty_tree`` above shows the counter returns 0
    for a scalar; it never walks a *truncated* household, so nothing so far
    shows the floor assertion itself firing. This does, against a household
    with its RESP subtree removed — the shape ``test_build.py`` would take
    if the builder silently dropped it.
    """

    def test_a_household_missing_its_beneficiaries_falls_below_the_full_count(self) -> None:
        full = walk(build_household(), N_PATHS)
        truncated = walk(build_household(beneficiaries=()), N_PATHS)
        floor = full - 5  # well above "beneficiaries gone", below "full"
        assert truncated < floor, (
            "a household missing its whole RESP subtree should trip a floor "
            "set anywhere near the full count"
        )


class TestFreezeAndUpdated:
    def test_freeze_marks_non_writeable(self) -> None:
        array = np.zeros(3)
        frozen = freeze(array)
        assert frozen is array
        assert not frozen.flags.writeable

    def test_freeze_is_idempotent(self) -> None:
        array = np.zeros(3)
        freeze(array)
        freeze(array)  # must not raise
        assert not array.flags.writeable

    def test_freeze_rejects_2d_array(self) -> None:
        with pytest.raises(AssertionError):
            freeze(np.zeros((2, 3)))

    def test_writing_into_a_frozen_array_raises_value_error(self) -> None:
        household = build_household()
        with pytest.raises(ValueError, match="read-only"):
            household.cash.balance[0] = 1.0

    def test_assigning_a_field_on_a_frozen_dataclass_raises(self) -> None:
        household = build_household()
        with pytest.raises(FrozenInstanceError):
            household.month = 2

    def test_updated_returns_a_frozen_copy_and_leaves_the_original_alone(self) -> None:
        household = build_household()
        copy = updated(household, month=2)

        assert copy is not household
        assert copy.month == 2
        assert household.month == 1

        with pytest.raises(ValueError, match="read-only"):
            copy.spending_achieved_ytd[0] = 1.0

        assert walk(copy, N_PATHS) > 0
        assert walk(household, N_PATHS) > 0

    def test_updated_on_a_nested_class_reflects_in_a_new_household(self) -> None:
        household = build_household()
        new_rrsp = updated(household.persons[0].rrsp, balance=np.full(N_PATHS, 999.0))
        new_person = updated(household.persons[0], rrsp=new_rrsp)
        new_household = updated(household, persons=(new_person,))

        assert new_household.persons[0].rrsp.balance[0] == 999.0
        assert household.persons[0].rrsp.balance[0] == 0.0
        assert walk(new_household, N_PATHS) > 0


class TestFreezeCannotBeFooledByAView:
    """``array.flags.writeable = False`` only protects one array object.

    A slice shares memory with its base, so freezing the slice and leaving
    the base writeable is not read-only at all — it just delays the mutation
    by one indirection. ``freeze`` must refuse these outright.
    """

    def test_rejects_a_column_slice_of_a_bigger_buffer(self) -> None:
        # engine.mc.simulate produces real_returns shaped (n_assets, n_paths)
        # and hands slices of it downstream: this is that shape, not a
        # contrived one.
        real_returns = np.zeros((5, N_PATHS))
        column = real_returns[0, :]
        with pytest.raises(ValueError, match="view"):
            CashState(balance=column)

    def test_rejects_a_chained_view(self) -> None:
        base = np.zeros(10)
        chained = base[0:6][0:3]
        with pytest.raises(ValueError, match="view"):
            CashState(balance=chained)

    def test_accepts_a_copy_of_a_view(self) -> None:
        real_returns = np.zeros((5, N_PATHS))
        column = real_returns[0, :].copy()
        state = CashState(balance=column)
        assert not state.balance.flags.writeable

    def test_mutating_the_buffer_behind_a_rejected_view_would_have_corrupted_state(self) -> None:
        """Demonstrates the bug this prevents, without going through freeze."""
        real_returns = np.zeros((5, N_PATHS))
        column = real_returns[0, :]
        column.flags.writeable = False  # what the old, unfixed freeze() did
        real_returns[0, 0] = 1e9  # a later caller mutates the buffer it owns
        assert column[0] == 1e9, "the 'frozen' view moved anyway: freezing a view protects nothing"


class TestFreezeGuardIsOneDirectional:
    """``freeze`` closes one hole and cannot close the other; pin both.

    It refuses an array *handed to state* whose ``.base`` is still
    writeable. It cannot refuse a view *taken from* an array that was, at
    the moment it was handed to state, an ordinary array with no base of
    its own — nothing distinguishes that array from one nobody else still
    holds a reference to, and a reference taken before the freeze stays
    exactly as writeable as it always was.
    """

    def test_a_view_taken_before_the_array_is_frozen_stays_writeable(self) -> None:
        big = np.zeros(10)
        col = big[0:4]  # taken while big is still ordinary
        state = CashState(balance=big)  # big.base is None, so freeze() accepts it whole
        col[0] = 999.0  # col was never itself frozen
        assert state.balance[0] == 999.0, (
            "freeze() protects against handing state a view of a live buffer, "
            "not against handing state a buffer you kept a view of"
        )

    def test_a_view_taken_after_the_array_is_frozen_is_read_only_too(self) -> None:
        big = np.zeros(10)
        CashState(balance=big)  # freezes big itself
        col = big[0:4]  # taken only after big is already read-only
        assert not col.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            col[0] = 1.0


class TestLiraJurisdictionInvariant:
    def test_empty_jurisdiction_and_zero_balance_is_fine(self) -> None:
        _lira(balance=0.0, jurisdiction="")

    def test_empty_jurisdiction_with_nonzero_balance_fails(self) -> None:
        with pytest.raises(ValueError, match="jurisdiction"):
            _lira(balance=1.0, jurisdiction="")

    def test_named_jurisdiction_with_nonzero_balance_is_fine(self) -> None:
        _lira(balance=1.0, jurisdiction="ab")


class TestLifJurisdictionInvariant:
    def test_empty_jurisdiction_and_zero_balance_is_fine(self) -> None:
        _lif(balance=0.0, jurisdiction="")

    def test_empty_jurisdiction_with_nonzero_balance_fails(self) -> None:
        with pytest.raises(ValueError, match="jurisdiction"):
            # opened_year is set so only the jurisdiction rule is under test.
            _lif(balance=1.0, jurisdiction="", opened_year=2025)

    def test_named_jurisdiction_with_nonzero_balance_is_fine(self) -> None:
        _lif(balance=1.0, jurisdiction="ab", opened_year=2025)


class TestRrifOpenedYearInvariant:
    def test_opened_year_none_and_zero_balance_is_fine(self) -> None:
        _rrif(balance=0.0, opened_year=None)

    def test_opened_year_set_and_nonzero_balance_is_fine(self) -> None:
        _rrif(balance=1.0, opened_year=2025)

    def test_opened_year_none_with_nonzero_balance_fails(self) -> None:
        with pytest.raises(ValueError, match="opened_year"):
            _rrif(balance=1.0, opened_year=None)


class TestLifOpenedYearInvariant:
    def test_opened_year_none_and_zero_balance_is_fine(self) -> None:
        _lif(balance=0.0, opened_year=None)

    def test_opened_year_set_and_nonzero_balance_is_fine(self) -> None:
        _lif(balance=1.0, jurisdiction="ab", opened_year=2025)

    def test_opened_year_none_with_nonzero_balance_fails(self) -> None:
        with pytest.raises(ValueError, match="opened_year"):
            _lif(balance=1.0, jurisdiction="ab", opened_year=None)


class TestDeathConsistency:
    """Only the half of the invariant that holds regardless of month_index.

    ``alive == (death_month_index > month_index)`` is NOT checked here — see
    the comment in ``PersonState.__post_init__`` for why that half needs a
    ``month_index`` this class does not carry, and a decision that belongs
    to issue 19.
    """

    def test_not_alive_with_the_sentinel_still_set_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="death_month_index"):
            updated(_person(), alive=np.array([False, True, True, True]))

    def test_not_alive_with_a_real_death_month_is_fine(self) -> None:
        updated(
            _person(),
            alive=np.array([False, True, True, True]),
            death_month_index=np.array(
                [5, DEATH_NOT_DRAWN, DEATH_NOT_DRAWN, DEATH_NOT_DRAWN], dtype=np.int64
            ),
        )

    def test_all_alive_with_the_sentinel_is_fine(self) -> None:
        _person()  # the default fixture: alive everywhere, sentinel everywhere


class TestBenefitStateInPayMonthly:
    def test_in_pay_monthly_as_an_array_is_frozen_and_shaped(self) -> None:
        benefit = _benefit(
            start_age_months=None,
            in_pay_monthly=np.full(N_PATHS, 1200.0),
            contributory_history=None,
        )
        assert benefit.start_age_months is None
        assert benefit.in_pay_monthly.shape == (N_PATHS,)
        assert benefit.in_pay_monthly.dtype == np.float64
        assert not benefit.in_pay_monthly.flags.writeable

    def test_contributory_history_and_in_pay_monthly_together_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at most one"):
            _benefit(in_pay_monthly=np.full(N_PATHS, 1200.0), contributory_history=0.85)


class TestShapeValidation:
    def test_leaf_rejects_a_2d_array(self) -> None:
        with pytest.raises(AssertionError):
            CashState(balance=np.zeros((2, 2)))

    def test_household_rejects_arrays_disagreeing_with_n_paths(self) -> None:
        with pytest.raises(AssertionError):
            build_household(spending_achieved_ytd=np.zeros(N_PATHS + 1))

    def test_household_rejects_a_person_array_disagreeing_with_n_paths(self) -> None:
        mismatched_person = _person()
        object.__setattr__(  # bypass PersonState's own freezing to build a bad fixture
            mismatched_person, "balance_owing", np.zeros(N_PATHS + 1)
        )
        with pytest.raises(AssertionError):
            build_household(persons=(mismatched_person,))


class TestPriorYearNetIncome:
    def test_present_zero_and_read_only_by_default(self) -> None:
        person = _person()
        assert person.prior_year_net_income.shape == (N_PATHS,)
        assert (person.prior_year_net_income == 0.0).all()
        assert not person.prior_year_net_income.flags.writeable


def test_person_state_has_no_cash_field() -> None:
    """Cash is household-level; a person carries no share of it at all."""
    assert "cash" not in {f.name for f in dataclasses.fields(PersonState)}


class TestSpendingMonthlyInvariant:
    def test_empty_spending_schedule_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="spending_schedule"):
            build_household(spending_schedule=())

    def test_spending_monthly_out_of_step_with_the_schedule_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="spending_monthly"):
            build_household(spending_monthly=9_999.0)

    def test_spending_monthly_matching_the_schedule_is_fine(self) -> None:
        build_household(spending_monthly=5_000.0)  # the fixture's own consistent value

    def test_a_later_year_selects_a_later_level(self) -> None:
        schedule = (
            SpendingLevel(from_year=2026, monthly_level=5_000.0),
            SpendingLevel(from_year=2030, monthly_level=4_000.0),
        )
        build_household(year=2030, spending_schedule=schedule, spending_monthly=4_000.0)
        with pytest.raises(ValueError, match="spending_monthly"):
            build_household(year=2030, spending_schedule=schedule, spending_monthly=5_000.0)

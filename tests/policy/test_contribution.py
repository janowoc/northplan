# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``SplitContribution.transfers``: weighted placement, spill, and RESP eligibility.

``example.yaml``'s beneficiary ``child1`` (born 2015-09) and ``two_person_scenario``'s clone
``child2`` (identical) supply the RESP fixtures; ``params/2026/resp.yaml``'s
``grant.cessation_age_years`` and ``contributions.maximum_lifetime`` are read through
``real_params``, never inlined.
"""

from __future__ import annotations

import numpy as np

from engine.core.build import build_initial_state
from engine.core.state import DEATH_NOT_DRAWN, updated
from engine.core.timeline import age_at_end_of_year
from engine.policy.contribution import SplitContribution


def test_spill_when_the_first_account_has_no_room(scenario, real_params, set_person, make_context):
    state = build_initial_state(scenario, n_paths=1)
    state = set_person(state, 0, rrsp=updated(state.persons[0].rrsp, room=np.zeros(1)))
    contribution = SplitContribution(weights={"rrsp": 1.0}, spill_order=("rrsp", "tfsa", "taxable"))
    context = make_context(
        n_paths=1,
        n_persons=1,
        n_beneficiaries=1,
        year=2026,
        month=1,
        month_index=0,
        cash_after_flows=np.zeros(1),
    )
    budget = np.array([5000.0])

    transfers = contribution.transfers(state, context, real_params, budget)

    assert len(transfers) == 1
    assert transfers[0].to_kind == "tfsa"
    np.testing.assert_allclose(transfers[0].amount, budget)


def test_a_kind_absent_from_spill_order_gets_nothing(
    scenario, real_params, set_person, make_context
):
    state = build_initial_state(scenario, n_paths=1)
    state = set_person(state, 0, rrsp=updated(state.persons[0].rrsp, room=np.zeros(1)))
    contribution = SplitContribution(weights={"rrsp": 1.0}, spill_order=("taxable",))
    context = make_context(
        n_paths=1,
        n_persons=1,
        n_beneficiaries=1,
        year=2026,
        month=1,
        month_index=0,
        cash_after_flows=np.zeros(1),
    )
    budget = np.array([5000.0])

    transfers = contribution.transfers(state, context, real_params, budget)

    assert all(t.to_kind != "tfsa" for t in transfers)
    assert len(transfers) == 1
    assert transfers[0].to_kind == "taxable"
    np.testing.assert_allclose(transfers[0].amount, budget)


def test_remainder_stays_in_cash_when_taxable_is_absent(
    scenario, real_params, set_person, make_context
):
    state = build_initial_state(scenario, n_paths=1)
    state = set_person(state, 0, rrsp=updated(state.persons[0].rrsp, room=np.zeros(1)))
    contribution = SplitContribution(weights={"rrsp": 1.0}, spill_order=())
    context = make_context(
        n_paths=1,
        n_persons=1,
        n_beneficiaries=1,
        year=2026,
        month=1,
        month_index=0,
        cash_after_flows=np.zeros(1),
    )
    budget = np.array([5000.0])

    transfers = contribution.transfers(state, context, real_params, budget)

    assert transfers == ()


def test_rrsp_with_room_zero_spills(scenario, real_params, set_person, make_context):
    state = build_initial_state(scenario, n_paths=1)
    state = set_person(state, 0, rrsp=updated(state.persons[0].rrsp, room=np.zeros(1)))
    contribution = SplitContribution(weights={"rrsp": 1.0}, spill_order=("taxable",))
    context = make_context(
        n_paths=1,
        n_persons=1,
        n_beneficiaries=1,
        year=2026,
        month=1,
        month_index=0,
        cash_after_flows=np.zeros(1),
    )
    budget = np.array([3000.0])

    transfers = contribution.transfers(state, context, real_params, budget)

    assert len(transfers) == 1
    assert transfers[0].to_kind == "taxable"
    np.testing.assert_allclose(transfers[0].amount, budget)


def test_couple_equal_split_and_one_dead_places_everything_with_the_living(
    two_person_scenario, real_params, set_person, make_context
):
    state = build_initial_state(two_person_scenario, n_paths=2)
    # Path 0: both alive. Path 1: person 1 (index 1) is dead -- death_month_index must be a
    # real month (here, one before the opening) or PersonState's own invariant rejects it.
    dead_on_path1 = np.array([True, False])
    death_month_index = np.array([DEATH_NOT_DRAWN, -1], dtype=np.int64)
    state = set_person(state, 1, alive=dead_on_path1, death_month_index=death_month_index)
    contribution = SplitContribution(weights={"taxable": 1.0}, spill_order=("taxable",))
    context = make_context(
        n_paths=2,
        n_persons=2,
        n_beneficiaries=2,
        year=2026,
        month=1,
        month_index=0,
        cash_after_flows=np.zeros(2),
    )
    budget = np.array([10000.0, 10000.0])

    transfers = contribution.transfers(state, context, real_params, budget)

    to_person0 = next(t for t in transfers if t.person_index == 0 and t.to_kind == "taxable")
    to_person1 = next(t for t in transfers if t.person_index == 1 and t.to_kind == "taxable")
    # Path 0: split equally.
    np.testing.assert_allclose(to_person0.amount[0], 5000.0)
    np.testing.assert_allclose(to_person1.amount[0], 5000.0)
    # Path 1: person 1 is dead; person 0 takes the whole budget.
    np.testing.assert_allclose(to_person0.amount[1], 10000.0)
    np.testing.assert_allclose(to_person1.amount[1], 0.0)


class TestResp:
    def _state_and_beneficiary(self, scenario, n_paths=1):
        state = build_initial_state(scenario, n_paths=n_paths)
        return state, state.beneficiaries[0]

    def test_age_end_equal_to_cessation_receives_a_share(self, scenario, real_params, make_context):
        state, beneficiary = self._state_and_beneficiary(scenario)
        cessation_age = real_params.resp.number("grant.cessation_age_years")
        year = beneficiary.birth_year + int(cessation_age)
        assert age_at_end_of_year(beneficiary.birth_year, beneficiary.birth_month, year) == int(
            cessation_age
        )
        month_index = 12 * (year - scenario.start_year)

        contribution = SplitContribution(weights={"resp": 1.0}, spill_order=())
        context = make_context(
            n_paths=1,
            n_persons=1,
            n_beneficiaries=1,
            year=year,
            month=1,
            month_index=month_index,
            cash_after_flows=np.zeros(1),
        )
        transfers = contribution.transfers(state, context, real_params, np.array([1000.0]))

        assert len(transfers) == 1
        assert transfers[0].to_kind == "resp"
        np.testing.assert_allclose(transfers[0].amount, [1000.0])

    def test_age_end_one_past_cessation_gets_no_transfer_at_all(
        self, scenario, real_params, make_context
    ):
        state, beneficiary = self._state_and_beneficiary(scenario)
        cessation_age = real_params.resp.number("grant.cessation_age_years")
        year = beneficiary.birth_year + int(cessation_age) + 1
        assert (
            age_at_end_of_year(beneficiary.birth_year, beneficiary.birth_month, year)
            == int(cessation_age) + 1
        )
        month_index = 12 * (year - scenario.start_year)

        contribution = SplitContribution(weights={"resp": 1.0}, spill_order=())
        context = make_context(
            n_paths=1,
            n_persons=1,
            n_beneficiaries=1,
            year=year,
            month=1,
            month_index=month_index,
            cash_after_flows=np.zeros(1),
        )
        transfers = contribution.transfers(state, context, real_params, np.array([1000.0]))

        assert not any(t.to_kind == "resp" for t in transfers)

    def test_no_transfer_at_the_example_beneficiarys_window_end(
        self, scenario, real_params, make_context
    ):
        """At the month the example beneficiary's education window ends, the contribution
        component emits no RESP transfer.
        """
        state, beneficiary = self._state_and_beneficiary(scenario)
        resp_state = beneficiary.resp
        threshold = resp_state.education_start_month_index + resp_state.education_months
        year, month = scenario.start_year + threshold // 12, threshold % 12 + 1

        contribution = SplitContribution(weights={"resp": 1.0}, spill_order=())
        context = make_context(
            n_paths=1,
            n_persons=1,
            n_beneficiaries=1,
            year=year,
            month=month,
            month_index=threshold,
            cash_after_flows=np.zeros(1),
        )
        transfers = contribution.transfers(state, context, real_params, np.array([1000.0]))

        assert transfers == ()

    def test_transfer_the_month_before_the_window_ends_none_at_it(
        self, example_values, build_from_values, real_params, make_context
    ):
        """A synthetic window that ends while the beneficiary is still well under the
        cessation age, so the age condition alone cannot explain the rejection at
        ``threshold`` -- only the window condition can. ``example.yaml``'s own beneficiary
        (``test_no_transfer_at_the_example_beneficiarys_window_end``, above) is
        already past cessation by the time its window ends, and would reject at ``threshold``
        for the age reason alone even if the window condition were dropped.
        """
        values = example_values()
        values["household"]["beneficiaries"][0]["education"] = {
            "start_year": 2026,
            "start_month": 1,
            "months": 1,
            "annual_cost": 20000,
        }
        _scenario, state = build_from_values(values, n_paths=1)
        beneficiary = state.beneficiaries[0]
        resp_state = beneficiary.resp
        threshold = resp_state.education_start_month_index + resp_state.education_months
        assert threshold == 1

        cessation_age = real_params.resp.number("grant.cessation_age_years")
        age_end = age_at_end_of_year(beneficiary.birth_year, beneficiary.birth_month, 2026)
        assert age_end <= cessation_age, "the age condition alone must not explain a rejection"

        contribution = SplitContribution(weights={"resp": 1.0}, spill_order=())

        context_before = make_context(
            n_paths=1,
            n_persons=1,
            n_beneficiaries=1,
            year=2026,
            month=1,
            month_index=threshold - 1,
            cash_after_flows=np.zeros(1),
        )
        before = contribution.transfers(state, context_before, real_params, np.array([1000.0]))
        assert len(before) == 1
        assert before[0].to_kind == "resp"
        np.testing.assert_allclose(before[0].amount, [1000.0])

        from engine.core.step import _validate_transfer

        _validate_transfer(
            before[0], 1, len(state.persons), list(state.beneficiaries), threshold - 1
        )

        context_at = make_context(
            n_paths=1,
            n_persons=1,
            n_beneficiaries=1,
            year=2026,
            month=2,
            month_index=threshold,
            cash_after_flows=np.zeros(1),
        )
        at_threshold = contribution.transfers(state, context_at, real_params, np.array([1000.0]))
        assert at_threshold == ()

    def test_wound_up_blocks_only_that_path(self, scenario, real_params, make_context):
        """``wound_up`` set True on one path only, both inside the age limit and the
        education window -- that path gets nothing, the other path still does.
        """
        state, beneficiary = self._state_and_beneficiary(scenario, n_paths=2)
        wound_up = np.array([True, False])
        beneficiaries = list(state.beneficiaries)
        beneficiaries[0] = updated(beneficiary, resp=updated(beneficiary.resp, wound_up=wound_up))
        state = updated(state, beneficiaries=tuple(beneficiaries))

        contribution = SplitContribution(weights={"resp": 1.0}, spill_order=())
        context = make_context(
            n_paths=2,
            n_persons=1,
            n_beneficiaries=1,
            year=2026,
            month=1,
            month_index=0,
            cash_after_flows=np.zeros(2),
        )
        transfers = contribution.transfers(state, context, real_params, np.array([1000.0, 1000.0]))

        resp_transfer = next(t for t in transfers if t.to_kind == "resp")
        np.testing.assert_allclose(resp_transfer.amount, [0.0, 1000.0])

        from engine.core.step import _validate_transfer

        _validate_transfer(resp_transfer, 2, len(state.persons), list(state.beneficiaries), 0)

    def test_equal_split_across_two_beneficiaries(
        self, two_person_scenario, real_params, make_context
    ):
        state = build_initial_state(two_person_scenario, n_paths=1)
        contribution = SplitContribution(weights={"resp": 1.0}, spill_order=())
        context = make_context(
            n_paths=1,
            n_persons=2,
            n_beneficiaries=2,
            year=2026,
            month=1,
            month_index=0,
            cash_after_flows=np.zeros(1),
        )
        budget = np.array([2000.0])

        transfers = contribution.transfers(state, context, real_params, budget)

        resp_transfers = [t for t in transfers if t.to_kind == "resp"]
        assert len(resp_transfers) == 2
        for transfer in resp_transfers:
            np.testing.assert_allclose(transfer.amount, [1000.0])

    def test_lifetime_cap_spills(self, scenario, real_params, make_context):
        state, beneficiary = self._state_and_beneficiary(scenario)
        resp_state = beneficiary.resp
        maximum_lifetime = real_params.resp.annual_amount("contributions.maximum_lifetime", 0)
        room_left = maximum_lifetime - float(resp_state.contributions_lifetime[0])
        assert room_left > 0

        contribution = SplitContribution(weights={"resp": 1.0}, spill_order=("taxable",))
        context = make_context(
            n_paths=1,
            n_persons=1,
            n_beneficiaries=1,
            year=2026,
            month=1,
            month_index=0,
            cash_after_flows=np.zeros(1),
        )
        budget = np.array([room_left + 5000.0])

        transfers = contribution.transfers(state, context, real_params, budget)

        resp_transfer = next(t for t in transfers if t.to_kind == "resp")
        taxable_transfer = next(t for t in transfers if t.to_kind == "taxable")
        np.testing.assert_allclose(resp_transfer.amount, [room_left])
        np.testing.assert_allclose(taxable_transfer.amount, [5000.0])

    def test_spill_takes_up_to_its_room_left_when_one_beneficiary_caps(
        self, two_person_scenario, real_params, make_context
    ):
        """The spill re-offers what a capped beneficiary refuses, rather than losing it.

        Beneficiary 0 is 100 short of the lifetime cap; beneficiary 1 has ample room. A
        single equal-split offer of the 2,000 budget would give each 1,000, cap beneficiary
        0 at 100, and strand the other 900 in the pool -- the bug this test would have
        caught. Re-offering it to beneficiary 1 places the whole 2,000.
        """
        state = build_initial_state(two_person_scenario, n_paths=1)
        maximum_lifetime = real_params.resp.annual_amount("contributions.maximum_lifetime", 0)
        beneficiary0 = state.beneficiaries[0]
        room_left0 = 100.0
        new_resp0 = updated(
            beneficiary0.resp,
            contributions_lifetime=np.full(1, maximum_lifetime - room_left0),
        )
        beneficiaries = list(state.beneficiaries)
        beneficiaries[0] = updated(beneficiary0, resp=new_resp0)
        state = updated(state, beneficiaries=tuple(beneficiaries))

        context = make_context(
            n_paths=1,
            n_persons=2,
            n_beneficiaries=2,
            year=2026,
            month=1,
            month_index=0,
            cash_after_flows=np.zeros(1),
        )
        budget = np.array([2000.0])

        contribution_resp_only = SplitContribution(weights={"resp": 1.0}, spill_order=("resp",))
        transfers = contribution_resp_only.transfers(state, context, real_params, budget)
        resp0 = next(t for t in transfers if t.person_index == 0 and t.to_kind == "resp")
        resp1 = next(t for t in transfers if t.person_index == 1 and t.to_kind == "resp")
        np.testing.assert_allclose(resp0.amount, [room_left0])
        np.testing.assert_allclose(resp1.amount, [1900.0])
        assert not any(t.to_kind == "taxable" for t in transfers)

        contribution_with_taxable = SplitContribution(
            weights={"resp": 1.0}, spill_order=("resp", "taxable")
        )
        transfers_with_taxable = contribution_with_taxable.transfers(
            state, context, real_params, budget
        )
        assert not any(t.to_kind == "taxable" for t in transfers_with_taxable)

    def test_spill_over_two_rounds(self, scenario, real_params, set_person, make_context):
        """Room left under the lifetime cap differs widely across three eligible
        beneficiaries (100, 500, 40,000), so a single equal-split offer caps two of them and
        a second round is needed to place what they refused -- the round-based spill, now
        exercised with more than two beneficiaries.
        """
        state, beneficiary0 = self._state_and_beneficiary(scenario)
        state = set_person(state, 0, rrsp=updated(state.persons[0].rrsp, room=np.zeros(1)))
        maximum_lifetime = real_params.resp.annual_amount("contributions.maximum_lifetime", 0)
        room_lefts = (100.0, 500.0, 40_000.0)
        beneficiaries = tuple(
            updated(
                beneficiary0,
                beneficiary_id=f"synthetic{index}",
                resp=updated(
                    beneficiary0.resp,
                    contributions_lifetime=np.full(1, maximum_lifetime - room_left),
                ),
            )
            for index, room_left in enumerate(room_lefts)
        )
        state = updated(state, beneficiaries=beneficiaries)

        context = make_context(
            n_paths=1,
            n_persons=1,
            n_beneficiaries=3,
            year=2026,
            month=1,
            month_index=0,
            cash_after_flows=np.zeros(1),
        )
        budget = np.array([3000.0])
        expected = (100.0, 500.0, 2400.0)

        resp_only = SplitContribution(weights={"rrsp": 1.0}, spill_order=("resp",))
        transfers = resp_only.transfers(state, context, real_params, budget)
        resp_transfers = {t.person_index: t.amount for t in transfers if t.to_kind == "resp"}
        for index, amount in enumerate(expected):
            np.testing.assert_allclose(resp_transfers[index], [amount])
        assert not any(t.to_kind == "taxable" for t in transfers)

        with_taxable = SplitContribution(weights={"rrsp": 1.0}, spill_order=("resp", "taxable"))
        transfers_with_taxable = with_taxable.transfers(state, context, real_params, budget)
        resp_transfers_2 = {
            t.person_index: t.amount for t in transfers_with_taxable if t.to_kind == "resp"
        }
        for index, amount in enumerate(expected):
            np.testing.assert_allclose(resp_transfers_2[index], [amount])
        assert not any(t.to_kind == "taxable" for t in transfers_with_taxable)

    def test_spill_over_three_rounds(self, scenario, real_params, set_person, make_context):
        """Room left under the lifetime cap forces a third round: round 1 offers 1,000 each
        and caps beneficiary 0 at its 100 left, leaving a pool of 900; round 2 offers 450 to
        each of the remaining two, capping beneficiary 1 at its 200 left and leaving
        beneficiary 2 with 450 placed and 250 still in the pool; round 3 gives beneficiary 2
        that last 250.
        """
        state, beneficiary0 = self._state_and_beneficiary(scenario)
        state = set_person(state, 0, rrsp=updated(state.persons[0].rrsp, room=np.zeros(1)))
        maximum_lifetime = real_params.resp.annual_amount("contributions.maximum_lifetime", 0)
        room_lefts = (100.0, 1_200.0, 40_000.0)
        beneficiaries = tuple(
            updated(
                beneficiary0,
                beneficiary_id=f"synthetic{index}",
                resp=updated(
                    beneficiary0.resp,
                    contributions_lifetime=np.full(1, maximum_lifetime - room_left),
                ),
            )
            for index, room_left in enumerate(room_lefts)
        )
        state = updated(state, beneficiaries=beneficiaries)

        context = make_context(
            n_paths=1,
            n_persons=1,
            n_beneficiaries=3,
            year=2026,
            month=1,
            month_index=0,
            cash_after_flows=np.zeros(1),
        )
        budget = np.array([3000.0])
        expected = (100.0, 1_200.0, 1_700.0)

        resp_only = SplitContribution(weights={"rrsp": 1.0}, spill_order=("resp",))
        transfers = resp_only.transfers(state, context, real_params, budget)
        resp_transfers = {t.person_index: t.amount for t in transfers if t.to_kind == "resp"}
        for index, amount in enumerate(expected):
            np.testing.assert_allclose(resp_transfers[index], [amount])

    def test_no_eligible_beneficiary_spills_the_whole_share(
        self, scenario, real_params, make_context
    ):
        state, beneficiary = self._state_and_beneficiary(scenario)
        cessation_age = real_params.resp.number("grant.cessation_age_years")
        year = beneficiary.birth_year + int(cessation_age) + 1  # ineligible by age
        month_index = 12 * (year - scenario.start_year)

        contribution = SplitContribution(weights={"resp": 1.0}, spill_order=("taxable",))
        context = make_context(
            n_paths=1,
            n_persons=1,
            n_beneficiaries=1,
            year=year,
            month=1,
            month_index=month_index,
            cash_after_flows=np.zeros(1),
        )
        budget = np.array([1000.0])

        transfers = contribution.transfers(state, context, real_params, budget)

        assert not any(t.to_kind == "resp" for t in transfers)
        taxable_transfer = next(t for t in transfers if t.to_kind == "taxable")
        np.testing.assert_allclose(taxable_transfer.amount, budget)

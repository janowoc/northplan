# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""End to end: ``SplitContribution`` sees the RRSP room ``open_year`` zeroed for a person
already past the statutory conversion age, through ``build_policy`` and
``advance_month_traced`` -- not a scripted stand-in for the policy.
"""

from __future__ import annotations

import numpy as np

from engine.core.build import build_initial_state
from engine.core.step import advance_month_traced
from engine.policy.build import build_policy
from engine.scenario.schema import CppEntitlement, OasEntitlement


def test_the_budget_reaches_taxable_in_the_first_month_when_room_is_zeroed(
    scenario, real_params, market
):
    """With the person already past the conversion age at the run's opening, room is zero
    from the very first ``open_year``. With an RRSP weight of 1 and
    ``spill_order=("taxable",)``, ``SplitContribution`` must see that zero and spill the
    whole month's budget to taxable, in the first month.
    """
    rrif_age = int(real_params.rrif.number("conversion_age_years"))
    person_a = scenario.household.persons[0]
    shifted_birth_year = scenario.start_year - (rrif_age + 1)
    # At this age the example's elections at 65, and its grid's CPP elections at 60
    # and 65, are already past, so CPP and OAS are stated in pay, at zero.
    new_person = person_a.model_copy(
        update={
            "birth_year": shifted_birth_year,
            "cpp": CppEntitlement(in_pay_monthly=0.0),
            "oas": OasEntitlement(in_pay_monthly=0.0),
        }
    )
    new_household = scenario.household.model_copy(update={"persons": (new_person,)})
    new_contribution = scenario.policies[0].contribution.model_copy(
        update={
            "weights": {"rrsp": 1.0, "tfsa": 0.0, "taxable": 0.0, "resp": 0.0},
            "spill_order": ("taxable",),
        }
    )
    new_policy = scenario.policies[0].model_copy(update={"contribution": new_contribution})
    new_scenario = scenario.model_copy(
        update={"household": new_household, "policies": (new_policy,)}
    )

    state = build_initial_state(new_scenario, n_paths=1)
    assert np.all(state.persons[0].rrsp.room > 0.0), "room must start positive to test this"
    policy = build_policy(new_scenario.policies[0], new_scenario.household)

    n_assets = len(market.asset_class_names)
    month_returns = np.zeros((n_assets, 1))
    _new_state, record = advance_month_traced(state, month_returns, policy, market, real_params)

    np.testing.assert_allclose(record.contributions[0].rrsp, 0.0)
    assert np.all(record.contributions[0].taxable > 0.0), "the budget must reach taxable"

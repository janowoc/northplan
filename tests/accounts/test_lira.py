# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Locked-In Retirement Account: unlocking age, must_convert, and the LIRA-to-LIF merge guard.

Runs entirely against the real Alberta jurisdiction file — this module reads
only three small parameters, none of them a table worth faking.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.accounts import lira
from engine.core.indexation import real_year
from engine.core.state import LifState, LiraState
from engine.params.loader import load_year


@pytest.fixture
def ab():
    return real_year(load_year(2026), 0.0).jurisdiction("ab")


def _lira(balance=0.0, jurisdiction="ab") -> LiraState:
    return LiraState(balance=np.array([balance], dtype=np.float64), jurisdiction=jurisdiction)


def _lif(balance=0.0, jurisdiction="", opened_year=None) -> LifState:
    return LifState(
        balance=np.array([balance], dtype=np.float64),
        jurisdiction=jurisdiction,
        annual_minimum=np.array([0.0]),
        annual_maximum=np.array([0.0]),
        withdrawn_ytd=np.array([0.0]),
        opened_year=opened_year,
    )


def test_withdrawals_permitted_at_and_after_unlocking_age(ab) -> None:
    unlocking_age = int(ab.number("lif.unlocking_age_years"))
    assert lira.withdrawals_permitted(unlocking_age - 1, ab) is False
    assert lira.withdrawals_permitted(unlocking_age, ab) is True


def test_must_convert_at_and_after_the_deadline(ab) -> None:
    deadline = int(ab.number("lif.conversion_deadline_age_years"))
    assert lira.must_convert(deadline - 1, ab) is False
    assert lira.must_convert(deadline, ab) is True


def test_convert_to_lif_moves_the_balance_and_zeroes_the_lira() -> None:
    lira_state = _lira(balance=1000.0, jurisdiction="ab")
    lif_state = _lif(balance=500.0, jurisdiction="ab", opened_year=2020)
    new_lira, new_lif = lira.convert_to_lif(lira_state, lif_state, year=2030)
    np.testing.assert_allclose(new_lira.balance, [0.0])
    np.testing.assert_allclose(new_lif.balance, [1500.0])
    assert new_lif.opened_year == 2020  # already open: left alone


def test_convert_to_lif_opens_the_lif_when_it_was_empty() -> None:
    lira_state = _lira(balance=1000.0, jurisdiction="ab")
    lif_state = _lif(balance=0.0, jurisdiction="", opened_year=None)
    _, new_lif = lira.convert_to_lif(lira_state, lif_state, year=2030)
    assert new_lif.opened_year == 2030


def test_convert_to_lif_carries_the_liras_jurisdiction_when_the_lifs_is_empty() -> None:
    lira_state = _lira(balance=1000.0, jurisdiction="ab")
    lif_state = _lif(balance=0.0, jurisdiction="", opened_year=None)
    _, new_lif = lira.convert_to_lif(lira_state, lif_state, year=2030)
    assert new_lif.jurisdiction == "ab"


def test_convert_to_lif_leaves_the_jurisdiction_unchanged_when_both_already_agree() -> None:
    # Both states already name "ab", so this cannot distinguish "the LIF's
    # jurisdiction wins" from "the LIRA's does" — when the two differ,
    # convert_to_lif raises instead of picking one (see the raise test
    # below), so that distinction is not observable at all. This only
    # confirms the ordinary case still resolves to the shared jurisdiction.
    lira_state = _lira(balance=1000.0, jurisdiction="ab")
    lif_state = _lif(balance=500.0, jurisdiction="ab", opened_year=2020)
    _, new_lif = lira.convert_to_lif(lira_state, lif_state, year=2030)
    assert new_lif.jurisdiction == "ab"


def test_convert_to_lif_raises_when_jurisdictions_differ() -> None:
    lira_state = _lira(balance=1000.0, jurisdiction="ab")
    lif_state = _lif(balance=500.0, jurisdiction="on", opened_year=2020)
    with pytest.raises(ValueError, match=r"ab.*on|on.*ab"):
        lira.convert_to_lif(lira_state, lif_state, year=2030)

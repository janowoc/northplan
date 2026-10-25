# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The form helpers' waits on the state line wait for the line to be shown."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from form_helpers import state, wait_clean, wait_settled
from offline import Offline
from playwright.sync_api import Page, expect


@pytest.mark.parametrize(
    "wait", [pytest.param(wait_settled, id="settled"), pytest.param(wait_clean, id="clean")]
)
def test_a_wait_does_not_pass_on_the_hidden_state_line_of_a_closed_form(
    api_form_page: tuple[Page, Offline], wait: Callable[[Page], None]
) -> None:
    page, _ = api_form_page
    expect(state(page)).to_be_hidden()
    expect(state(page)).to_have_text("No errors.")
    with pytest.raises(AssertionError):
        wait(page)

# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The form helpers' waits on the state line wait for the line to be shown, do not read a line
Alpine is about to hide as shown, and fail when no animation frame comes."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest
from form_helpers import open_example, state, wait_clean, wait_settled
from offline import Offline
from playwright.sync_api import Error, Page, expect

LATE_FRAMES = (
    "() => { const frame = window.requestAnimationFrame.bind(window); "
    "window.requestAnimationFrame = (callback) => setTimeout(() => frame(callback), 1000); }"
)
NO_FRAMES = "() => { window.requestAnimationFrame = () => 0; }"


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


@pytest.mark.parametrize(
    "wait", [pytest.param(wait_settled, id="settled"), pytest.param(wait_clean, id="clean")]
)
def test_a_wait_does_not_pass_on_a_state_line_alpine_is_about_to_hide(
    api_form_page: tuple[Page, Offline], wait: Callable[[Page], None]
) -> None:
    page, _ = api_form_page
    open_example(page)
    wait_clean(page)
    page.evaluate(LATE_FRAMES)
    page.evaluate("() => { Alpine.$data(document.body).formOpen = false; }")
    expect(state(page)).to_be_visible()
    expect(state(page)).to_have_text("No errors.")
    with pytest.raises(AssertionError):
        wait(page)
    expect(state(page)).to_be_hidden()


def test_a_wait_fails_when_no_animation_frame_comes(api_form_page: tuple[Page, Offline]) -> None:
    page, _ = api_form_page
    open_example(page)
    wait_clean(page)
    page.evaluate(NO_FRAMES)
    start = time.monotonic()
    with pytest.raises(Error, match="no animation frame in 5 s"):
        wait_clean(page)
    elapsed = time.monotonic() - start
    assert 4.5 < elapsed < 15

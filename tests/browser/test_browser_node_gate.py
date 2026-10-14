# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The Node gate has three outcomes, and none of them needs Node to be tested."""

from __future__ import annotations

from pathlib import Path

import pytest
from node_gate import REQUIRE_VARIABLE, require_node


def test_node_and_the_package_present_returns_the_node_path(tmp_path: Path) -> None:
    assert (
        require_node(environ={}, which=lambda _name: "/usr/bin/node", package=tmp_path)
        == "/usr/bin/node"
    )


def test_missing_and_unset_skips_with_the_way_to_install(tmp_path: Path) -> None:
    with pytest.raises(pytest.skip.Exception, match=r"npm ci --ignore-scripts"):
        require_node(environ={}, which=lambda _name: None, package=tmp_path / "absent")


def test_missing_and_not_one_still_skips(tmp_path: Path) -> None:
    with pytest.raises(pytest.skip.Exception):
        require_node(
            environ={REQUIRE_VARIABLE: "0"}, which=lambda _name: None, package=tmp_path / "absent"
        )


def test_missing_and_set_fails_with_the_same_text(tmp_path: Path) -> None:
    with pytest.raises(pytest.fail.Exception, match=r"npm ci --ignore-scripts"):
        require_node(
            environ={REQUIRE_VARIABLE: "1"}, which=lambda _name: None, package=tmp_path / "absent"
        )


def test_the_package_missing_alone_is_enough(tmp_path: Path) -> None:
    with pytest.raises(pytest.fail.Exception, match="is absent"):
        require_node(
            environ={REQUIRE_VARIABLE: "1"},
            which=lambda _name: "/usr/bin/node",
            package=tmp_path / "absent",
        )


def test_node_missing_alone_is_enough(tmp_path: Path) -> None:
    with pytest.raises(pytest.fail.Exception, match="node is not on PATH"):
        require_node(environ={REQUIRE_VARIABLE: "1"}, which=lambda _name: None, package=tmp_path)

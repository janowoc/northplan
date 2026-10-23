# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Refusing a bracket fill aimed at a federal bracket edge the parameter year does not have.

One refusal, raised as :class:`CeilingBracketNotAllowedError`: a policy whose
``withdrawal.taxable_ceiling_bracket`` is not ``None`` and is not below the number of
``brackets.edges_annual`` in the parameter year's ``federal`` file. The index counts the
edges from zero, as the bracket fill reads them. ``None`` turns the fill off and is not
checked.

A parameter year that does not match ``scenario.start_year`` is a caller
mistake rather than a bad scenario, and raises ``ValueError`` instead, like
:func:`engine.scenario.start_ages.check_start_ages`.

:func:`engine.mc.prepare.check_scenario` calls :func:`check_ceiling_bracket` once, on
the grid-expanded scenario, right after :func:`~engine.scenario.lifespan.check_lifespan`.
"""

from __future__ import annotations

from engine.params.loader import ParamYear
from engine.scenario.load import ScenarioError
from engine.scenario.schema import Scenario

__all__ = ["CeilingBracketNotAllowedError", "check_ceiling_bracket"]


class CeilingBracketNotAllowedError(ScenarioError):
    """A ``taxable_ceiling_bracket`` past the last federal bracket edge of the parameter year.

    Raised by :func:`check_ceiling_bracket`. The message names the policy, the field, the
    federal parameter file's source, and its number of edges.
    """


def check_ceiling_bracket(scenario: Scenario, params: ParamYear) -> None:
    """Refuse a ``taxable_ceiling_bracket`` past the last federal edge in ``params``.

    Policies are checked in file order, and this raises on the first refusal. The edge
    table is read before any policy is looked at, so it must be present even when no
    policy has a ceiling.

    Args:
        scenario: The validated scenario.
        params: The parameter year matching ``scenario.start_year``.

    Raises:
        ValueError: If ``params.year != scenario.start_year``.
        MissingParameterError: If ``brackets.edges_annual`` is absent.
        MalformedParamFileError: If it is not a list.
        CeilingBracketNotAllowedError: On the first policy whose index is not below the
            number of edges.
    """
    if params.year != scenario.start_year:
        raise ValueError(
            f"check_ceiling_bracket: params is for tax year {params.year} but "
            f"scenario.start_year is {scenario.start_year}. Load "
            f"params/{scenario.start_year}/ for this scenario."
        )

    count = len(params.federal.sequence("brackets.edges_annual"))
    for policy in scenario.policies:
        index = policy.withdrawal.taxable_ceiling_bracket
        if index is not None and index >= count:
            raise CeilingBracketNotAllowedError(
                f"policies[{policy.name!r}].withdrawal.taxable_ceiling_bracket: no federal "
                f"bracket edge has this index. brackets.edges_annual in {params.federal.source} "
                f"has {count} edge(s), counted from 0, so the index must be below {count}."
            )

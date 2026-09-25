# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Minimal :class:`~engine.policy.base.Policy` implementations for ``tests/core``.

Test code only: none of these belongs under ``engine/``. Each is deliberately dumb —
no financial reasoning at all — so that a test exercising ``engine.core.step`` or
``engine.mc.simulate`` is exercising the step, not a policy's own logic.
"""

from __future__ import annotations

from engine.core.context import MonthContext
from engine.core.indexation import RealParamYear
from engine.core.state import Elections, HouseholdState
from engine.policy.base import Decision, Policy, Transfer


class DoNothingPolicy:
    """Never transfers anything. The baseline every identity test runs against."""

    def __init__(self, elections: Elections, order: tuple[str, ...]) -> None:
        self._elections = elections
        self._order = order

    def decide(
        self, _state: HouseholdState, _context: MonthContext, _real_params: RealParamYear
    ) -> Decision:
        return Decision(transfers=())

    def elections(self) -> Elections:
        return self._elections

    def withdrawal_order(self) -> tuple[str, ...]:
        return self._order

    def free_parameters(self) -> dict[str, float]:
        return {}


class RecordingPolicy:
    """Wraps another policy, recording ``(state, context)`` for every call to ``decide``.

    Attributes:
        calls: Every ``(state, context)`` pair seen, in call order. A plain list on the
            instance is fine here — this is test code, not a frozen dataclass under
            ``engine/``.
    """

    def __init__(self, wrapped: Policy) -> None:
        self.wrapped = wrapped
        self.calls: list[tuple[HouseholdState, MonthContext]] = []

    def decide(
        self, state: HouseholdState, context: MonthContext, real_params: RealParamYear
    ) -> Decision:
        self.calls.append((state, context))
        return self.wrapped.decide(state, context, real_params)

    def elections(self) -> Elections:
        return self.wrapped.elections()

    def withdrawal_order(self) -> tuple[str, ...]:
        return self.wrapped.withdrawal_order()

    def free_parameters(self) -> dict[str, float]:
        return self.wrapped.free_parameters()


class ScriptedPolicy:
    """Returns a fixed, pre-written set of transfers per month index.

    Attributes:
        script: Month index to the transfers to return that month. A month index absent
            from ``script`` gets no transfers.
    """

    def __init__(
        self,
        elections: Elections,
        order: tuple[str, ...],
        script: dict[int, tuple[Transfer, ...]],
    ) -> None:
        self._elections = elections
        self._order = order
        self._script = script

    def decide(
        self, _state: HouseholdState, context: MonthContext, _real_params: RealParamYear
    ) -> Decision:
        return Decision(transfers=self._script.get(context.month_index, ()))

    def elections(self) -> Elections:
        return self._elections

    def withdrawal_order(self) -> tuple[str, ...]:
        return self._order

    def free_parameters(self) -> dict[str, float]:
        return {}

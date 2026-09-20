# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Classifies every dollar-typed field in :mod:`engine.core.state` as nominal or real.

This adds no production code: nothing in ``engine/`` imports this module, and it does not apply
:func:`engine.core.indexation.nominal_carry_factor` to anything. Its only job is to make the
classification explicit, in one table, and to fail the moment a new field arrives unclassified.

Scope of the table
-------------------

Every dataclass defined in :mod:`engine.core.state` — discovered from the module, not listed by
hand — and, within each, every field whose resolved annotation is ``NDArray[np.float64]`` or
``NDArray[np.float64] | None``. That is every dollar amount the module carries today, and nothing
else: the integer and boolean arrays (``alive``, ``death_month_index``, ``wound_up``,
``depleted``) are out of scope, and so are the plain ``float`` fields (``survivor_share``,
``contributory_history``, ``rrif_conversion_fraction``, ``education_monthly_cost``,
``spending_monthly``, ``spending_survivor_share``, ``monthly_level``) — fractions in some cases,
scenario inputs rather than carried balances in the rest. The match is exact equality against the
two spellings above, so a per-path dollar field typed any other way — the longhand
``np.ndarray[tuple[int, ...], np.dtype[np.float64]]``, a ``type`` alias, a three-way union, or
anything else that is not one of those two exact objects — would slip past this rule; nothing here
can catch that.

The two classifications
------------------------

``Basis.REAL`` — the field is in January dollars of the scenario's start year at every month of
the run. Nothing decays it. This is the engine's default (``docs/limitations.md`` L5), and it
covers both a balance earning a real return (L6: returns are real) and a within-year accumulator
that is zeroed each January before any decay could apply to it.

``Basis.NOMINAL`` — the quantity the field stands for is fixed in nominal terms, so its real value
falls by :func:`~engine.core.indexation.nominal_carry_factor` every January it survives. The field
is not necessarily the place that applies the factor: whether ``open_year`` restates the stored
figure or a consumer deflates at the point of use is a later issue's decision, and
``PensionState.monthly_amount`` already takes the second route. ``NOMINAL`` here means "a factor
is owed", not "``open_year`` owns it".
"""

from __future__ import annotations

import dataclasses
import inspect
import typing
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Final

import numpy as np
from numpy.typing import NDArray

from engine.core import state

#: The two per-path dollar array shapes this rule recognizes.
_DOLLAR_ARRAY_TYPES: Final[tuple[object, ...]] = (
    NDArray[np.float64],
    NDArray[np.float64] | None,
)


class Basis(Enum):
    """Whether a state field's real value is held constant, or decays as prices rise."""

    NOMINAL = "nominal"
    REAL = "real"


#: One entry per in-scope field, keyed by (class name, field name). See the module docstring for
#: what each classification means. ``docs/limitations.md`` line numbers are cited as ``L<n>``.
TABLE: Final[dict[tuple[str, str], tuple[Basis, str]]] = {
    ("CashState", "balance"): (
        Basis.REAL,
        "Cash pays zero real return (L38), so it holds its real value by assumption.",
    ),
    ("RrspState", "balance"): (Basis.REAL, "Returns are real (L6)."),
    ("RrspState", "room"): (
        Basis.NOMINAL,
        "Unused contribution room is a dollar figure fixed in nominal terms; held constant it "
        "overstates the room.",
    ),
    ("RrspState", "contributed_ytd"): (
        Basis.REAL,
        "A within-year accumulator, zeroed each January, compared against room already stated "
        "in real dollars.",
    ),
    ("RrifState", "balance"): (Basis.REAL, "Returns are real (L6)."),
    ("RrifState", "annual_minimum"): (
        Basis.REAL,
        "Fixed each January from that January's real balance and consumed within the year.",
    ),
    ("RrifState", "withdrawn_ytd"): (
        Basis.REAL,
        "A within-year accumulator, zeroed each January.",
    ),
    ("LiraState", "balance"): (Basis.REAL, "Returns are real (L6)."),
    ("LifState", "balance"): (Basis.REAL, "Returns are real (L6)."),
    ("LifState", "annual_minimum"): (
        Basis.REAL,
        "Fixed each January from that January's real balance.",
    ),
    ("LifState", "annual_maximum"): (
        Basis.REAL,
        "Fixed each January from that January's real balance.",
    ),
    ("LifState", "withdrawn_ytd"): (
        Basis.REAL,
        "A within-year accumulator, zeroed each January.",
    ),
    ("TfsaState", "balance"): (Basis.REAL, "Returns are real (L6)."),
    ("TfsaState", "room"): (
        Basis.NOMINAL,
        "Unused room is a nominal dollar figure carried forward indefinitely.",
    ),
    ("TfsaState", "withdrawn_this_year"): (
        Basis.NOMINAL,
        "Next January restores to room the nominal amount withdrawn, not its real value at "
        "the time.",
    ),
    ("TaxableState", "balance"): (Basis.REAL, "Market value; returns are real (L6)."),
    ("TaxableState", "acb"): (
        Basis.NOMINAL,
        "A capital gain is taxed on the nominal gain; a real ACB would exempt the inflation "
        "part of every gain.",
    ),
    ("EmploymentBand", "monthly_amount"): (Basis.REAL, "A real step schedule per person (L39)."),
    ("BenefitState", "in_pay_monthly"): (
        Basis.REAL,
        "An amount already in pay is held in start-year January dollars and not eroded (L52).",
    ),
    ("BenefitState", "monthly_amount"): (
        Basis.REAL,
        "The amount in pay this month, on the same basis as in_pay_monthly (L52).",
    ),
    ("PensionState", "monthly_amount"): (
        Basis.NOMINAL,
        "Fixed in nominal terms when indexed is false; engine/benefits/pension.py's "
        "db_pension_monthly applies unindexed_factor at the point of payment rather than "
        "restating the field, which is why open_year must leave it alone.",
    ),
    ("PensionState", "bridge_monthly"): (
        Basis.NOMINAL,
        "Same route and same reason as monthly_amount.",
    ),
    ("IncomeLedger", "employment"): (
        Basis.REAL,
        "Current-year flows, zeroed each January; assessed against a real-dollar tax table.",
    ),
    ("IncomeLedger", "cpp"): (
        Basis.REAL,
        "Current-year flows, zeroed each January; assessed against a real-dollar tax table.",
    ),
    ("IncomeLedger", "oas"): (
        Basis.REAL,
        "Current-year flows, zeroed each January; assessed against a real-dollar tax table.",
    ),
    ("IncomeLedger", "db_pension"): (
        Basis.REAL,
        "Current-year flows, zeroed each January; assessed against a real-dollar tax table.",
    ),
    ("IncomeLedger", "rrsp_withdrawals"): (
        Basis.REAL,
        "Current-year flows, zeroed each January; assessed against a real-dollar tax table.",
    ),
    ("IncomeLedger", "rrif_lif_withdrawals"): (
        Basis.REAL,
        "Current-year flows, zeroed each January; assessed against a real-dollar tax table.",
    ),
    ("IncomeLedger", "interest"): (
        Basis.REAL,
        "Current-year flows, zeroed each January; assessed against a real-dollar tax table.",
    ),
    ("IncomeLedger", "eligible_dividends"): (
        Basis.REAL,
        "Current-year flows, zeroed each January; assessed against a real-dollar tax table.",
    ),
    ("IncomeLedger", "capital_gains"): (
        Basis.REAL,
        "Current-year flows, zeroed each January; assessed against a real-dollar tax table.",
    ),
    ("IncomeLedger", "resp_accumulated_income"): (
        Basis.REAL,
        "Current-year flows, zeroed each January; assessed against a real-dollar tax table.",
    ),
    ("IncomeLedger", "rrsp_deductions"): (
        Basis.REAL,
        "Current-year flows, zeroed each January; assessed against a real-dollar tax table.",
    ),
    ("IncomeLedger", "cpp_base_contributions"): (
        Basis.REAL,
        "Current-year flows, zeroed each January; assessed against a real-dollar tax table.",
    ),
    ("IncomeLedger", "cpp_enhanced_contributions"): (
        Basis.REAL,
        "Current-year flows, zeroed each January; assessed against a real-dollar tax table.",
    ),
    ("IncomeLedger", "ei_premiums"): (
        Basis.REAL,
        "Current-year flows, zeroed each January; assessed against a real-dollar tax table.",
    ),
    ("IncomeLedger", "remitted"): (
        Basis.REAL,
        "Current-year flows, zeroed each January; assessed against a real-dollar tax table.",
    ),
    ("PersonState", "balance_owing"): (
        Basis.REAL,
        "Assessed at the December close and settled a few months later, within the same "
        "real-dollar year.",
    ),
    ("PersonState", "prior_year_net_income"): (
        Basis.REAL,
        "Written at each December close from that year's real income. The opening figure a "
        "scenario states is the documented exception, taken as filed (engine/__init__.py, "
        "and L5's second sentence); it is one year stale on purpose and is not decayed.",
    ),
    ("RespState", "contributions"): (
        Basis.NOMINAL,
        "A principal bucket, returned at its nominal amount; growth accrues to income.",
    ),
    ("RespState", "grants"): (
        Basis.NOMINAL,
        "A principal bucket, repaid at its nominal amount; growth accrues to income.",
    ),
    ("RespState", "income"): (
        Basis.REAL,
        "The bucket that earns the plan's return, and returns are real (L6).",
    ),
    ("RespState", "contributions_lifetime"): (
        Basis.NOMINAL,
        "Measured against a lifetime cap that the parameter view decays, so the running total "
        "must decay with it.",
    ),
    ("RespState", "grants_lifetime"): (
        Basis.NOMINAL,
        "Same as contributions_lifetime.",
    ),
    ("RespState", "grant_room"): (
        Basis.NOMINAL,
        "Grant room accrues in nominal grant dollars and carries forward.",
    ),
    ("RespState", "grant_received_ytd"): (
        Basis.REAL,
        "A within-year accumulator, zeroed each January, compared against an annual maximum "
        "already stated in real dollars.",
    ),
    ("RespState", "contributed_ytd"): (
        Basis.REAL,
        "A within-year accumulator, zeroed each January.",
    ),
    ("YearRecord", "net_worth"): (Basis.REAL, "An output snapshot of real balances."),
    ("YearRecord", "spending"): (
        Basis.REAL,
        "An output snapshot; spending is a real schedule.",
    ),
    ("YearRecord", "tax_assessed"): (
        Basis.REAL,
        "An output snapshot, assessed on real-dollar income.",
    ),
    ("HouseholdState", "spending_achieved_ytd"): (
        Basis.REAL,
        "A within-year accumulator, zeroed each January.",
    ),
    ("HouseholdState", "estate_after_tax"): (
        Basis.REAL,
        "An output built from real balances.",
    ),
}

#: A generous floor, not an exact count: high enough that a collector run over every state
#: dataclass and silently returning too little could not pass. It classifies 52 fields today.
MINIMUM_CLASSIFIED = 45


def _state_dataclasses() -> tuple[type, ...]:
    """Every dataclass defined in :mod:`engine.core.state`, in module order."""
    return tuple(
        obj
        for _, obj in inspect.getmembers(state, inspect.isclass)
        if dataclasses.is_dataclass(obj) and obj.__module__ == state.__name__
    )


def in_scope_fields(classes: Iterable[type]) -> dict[tuple[str, str], object]:
    """Every ``(class name, field name)`` among ``classes`` typed as a per-path dollar array.

    Args:
        classes: Dataclasses to inspect. Not necessarily all of them, or all real ones: the
            "guard bites" test below passes a synthetic class here too.

    Returns:
        Resolved annotation keyed by ``(class.__name__, field.name)``, for every field whose
        annotation, resolved via :func:`typing.get_type_hints`, is ``NDArray[np.float64]`` or
        ``NDArray[np.float64] | None``.
    """
    found: dict[tuple[str, str], object] = {}
    for cls in classes:
        hints = typing.get_type_hints(cls, include_extras=False)
        for f in dataclasses.fields(cls):
            annotation = hints[f.name]
            if annotation in _DOLLAR_ARRAY_TYPES:
                found[(cls.__name__, f.name)] = annotation
    return found


def unclassified(
    classes: Iterable[type], table: Mapping[tuple[str, str], object]
) -> set[tuple[str, str]]:
    """In-scope fields among ``classes`` that ``table`` does not key."""
    return set(in_scope_fields(classes)) - set(table)


class TestEveryInScopeFieldIsClassified:
    def test_no_field_is_missing_from_the_table(self) -> None:
        missing = unclassified(_state_dataclasses(), TABLE)
        assert not missing, (
            f"the following fields of engine.core.state are NDArray[np.float64] but not "
            f"classified in TABLE: {sorted(missing)}. For each: decide whether the quantity "
            f"is fixed in nominal terms, add it to TABLE with a one-line reason, and if it is "
            f"NOMINAL, note that a nominal_carry_factor call is owed somewhere."
        )

    def test_no_stale_entry_names_a_field_that_no_longer_exists(self) -> None:
        present = set(in_scope_fields(_state_dataclasses()))
        stale = set(TABLE) - present
        assert not stale, (
            f"TABLE classifies the following fields, but they are no longer in scope in "
            f"engine.core.state (renamed, removed, or no longer NDArray[np.float64]): "
            f"{sorted(stale)}. Update TABLE to match."
        )

    def test_the_table_is_not_vacuous(self) -> None:
        """A floor on the collector's own output, so a collector that silently stopped finding
        fields fails rather than passing test_no_field_is_missing_from_the_table vacuously."""
        assert len(in_scope_fields(_state_dataclasses())) >= MINIMUM_CLASSIFIED
        assert len(TABLE) >= MINIMUM_CLASSIFIED

    def test_the_floor_can_actually_bite(self) -> None:
        """Shows the assertion above can fail, not just that it currently passes.

        A collector restricted to one class finds far fewer fields than the floor, the way
        ``tests/core/test_build.py``'s ``MINIMUM_ARRAYS`` test shows its own floor can bite.
        """
        assert len(in_scope_fields([state.CashState])) < MINIMUM_CLASSIFIED


class TestTheGuardBites:
    def test_an_unclassified_field_on_a_new_class_is_reported(self) -> None:
        """The success criterion: adding a field fails this test until it is classified.

        A synthetic frozen dataclass with one ``NDArray[np.float64]`` field, run through the
        same collector the real test uses, against the real ``TABLE`` — which does not, and
        cannot, know about a class defined inside this test.
        """

        @dataclasses.dataclass(frozen=True, slots=True)
        class NotYetClassified:
            new_balance: NDArray[np.float64]

        missing = unclassified([NotYetClassified], TABLE)
        assert missing == {("NotYetClassified", "new_balance")}

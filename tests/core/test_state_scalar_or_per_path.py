# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Classifies every non-array field in :mod:`engine.core.state` as scalar or per-path.

This adds no production code: nothing in ``engine/`` imports this module. Its only job is to
make the classification explicit, in one table, and to fail the moment a new field arrives
unclassified.

The danger this guards against is a scalar that *should* have varied by path — one path's answer
silently applied to every path, raising nothing. ``walk`` in ``tests/core/conftest.py`` already
asserts that every array is ``(n_paths,)``; nothing asserts that a scalar is legitimately scalar.
This table is that assertion.

Scope of the table
-------------------

Every dataclass defined in :mod:`engine.core.state` — discovered from the module, not listed by
hand — and, within each, every field whose resolved annotation (after stripping a ``| None``) is
**none of**:

- an ndarray, detected via ``typing.get_origin(annotation) is NDArray`` (``NDArray[np.float64]``'s
  origin is ``NDArray`` itself, not ``np.ndarray``);
- a dataclass defined in :mod:`engine.core.state` (nested structure, collected in its own right);
- a ``tuple[...]`` whose element type is a dataclass defined in :mod:`engine.core.state` (same
  reason).

This deliberately **includes** ``Elections.cpp_start_age_months`` and
``Elections.oas_start_age_months`` — ``tuple[int | None, ...]``, a tuple of scalars, one per
person, and exactly the kind of field this guard is for. It deliberately **excludes** the
non-float arrays ``alive``, ``death_month_index``, ``wound_up``, ``depleted``: they are arrays,
and ``walk`` already covers them.

The four classifications
--------------------------

``Scalarity.IDENTITY`` — a name, id, code, jurisdiction, or index that labels something; never a
quantity.

``Scalarity.SCENARIO_INPUT`` — stated in the scenario or the policy spec and carried through
without a per-path decision, at most unit-converted. The scenario, or the spec, states one value
(or a fixed formula's worth of arithmetic on one), so every path carries that value.

``Scalarity.CALENDAR`` — a year, month, or month index derived from the calendar and the
scenario's dates. The calendar does not vary by path. Among fields holding a year, a month, or a
month index, and never set by a run-time trigger (those are ``DETERMINISTIC_TRIGGER``): the
field is ``CALENDAR`` iff the engine consulted the calendar to produce its value —
``EmploymentBand.from_month_index`` converts a stated year into a month index and
``HouseholdState.year`` advances every January, while ``SpendingLevel.from_year`` is copied
through untouched.

``Scalarity.DETERMINISTIC_TRIGGER`` — computed during the run, but from quantities that do not
vary by path, so the same on every path. The one category whose reason has to make an argument
rather than simply naming the field's nature.
"""

from __future__ import annotations

import dataclasses
import inspect
import types
import typing
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Final

from numpy.typing import NDArray

from engine.core import state


class Scalarity(Enum):
    """Why an in-scope field of :mod:`engine.core.state` is legitimately scalar (not per-path)."""

    IDENTITY = "identity"
    SCENARIO_INPUT = "scenario_input"
    CALENDAR = "calendar"
    DETERMINISTIC_TRIGGER = "deterministic_trigger"


#: One entry per in-scope field, keyed by (class name, field name). See the module docstring for
#: what each classification means.
TABLE: Final[dict[tuple[str, str], tuple[Scalarity, str]]] = {
    ("BeneficiaryState", "beneficiary_id"): (Scalarity.IDENTITY, "An identifier, not a quantity."),
    ("BeneficiaryState", "birth_year"): (
        Scalarity.SCENARIO_INPUT,
        "The scenario states one birth date for this beneficiary.",
    ),
    ("BeneficiaryState", "birth_month"): (
        Scalarity.SCENARIO_INPUT,
        "The scenario states one birth date for this beneficiary.",
    ),
    ("BenefitState", "start_age_months"): (
        Scalarity.SCENARIO_INPUT,
        "The policy's election, one age for every path.",
    ),
    ("BenefitState", "contributory_history"): (
        Scalarity.SCENARIO_INPUT,
        "The scenario states one fraction of the maximum CPP pension earned.",
    ),
    ("Elections", "cpp_start_age_months"): (
        Scalarity.SCENARIO_INPUT,
        "The policy's election, one age per person, the same on every path.",
    ),
    ("Elections", "oas_start_age_months"): (
        Scalarity.SCENARIO_INPUT,
        "The policy's election, one age per person, the same on every path.",
    ),
    ("Elections", "rrif_conversion_age_years"): (
        Scalarity.SCENARIO_INPUT,
        "The policy's election, household-wide, the same on every path.",
    ),
    ("Elections", "rrif_conversion_fraction"): (
        Scalarity.SCENARIO_INPUT,
        "The policy's election, household-wide, the same on every path.",
    ),
    ("Elections", "fill_pension_credit"): (
        Scalarity.SCENARIO_INPUT,
        "The policy's election, a flag, the same on every path.",
    ),
    ("EmploymentBand", "from_month_index"): (
        Scalarity.CALENDAR,
        "Derived from the scenario's stated calendar dates, not from anything a path draws.",
    ),
    ("EmploymentBand", "to_month_index"): (
        Scalarity.CALENDAR,
        "Derived from the scenario's stated calendar dates, not from anything a path draws.",
    ),
    ("HouseholdState", "year"): (Scalarity.CALENDAR, "The calendar does not vary by path."),
    ("HouseholdState", "month"): (Scalarity.CALENDAR, "The calendar does not vary by path."),
    ("HouseholdState", "month_index"): (
        Scalarity.CALENDAR,
        "The calendar does not vary by path.",
    ),
    ("HouseholdState", "n_paths"): (
        Scalarity.IDENTITY,
        "A path count, not a quantity carried by any one path.",
    ),
    ("HouseholdState", "province"): (
        Scalarity.IDENTITY,
        "A jurisdiction code, not a quantity.",
    ),
    ("HouseholdState", "spending_monthly"): (
        Scalarity.SCENARIO_INPUT,
        "The spending level in force for the current year, from the scenario's schedule; "
        "household-wide, not per path.",
    ),
    ("HouseholdState", "spending_survivor_share"): (
        Scalarity.SCENARIO_INPUT,
        "The scenario states one survivor share.",
    ),
    ("LifState", "jurisdiction"): (
        Scalarity.IDENTITY,
        "A pension-jurisdiction code, not a quantity.",
    ),
    ("LifState", "opened_year"): (
        Scalarity.DETERMINISTIC_TRIGGER,
        "LifState.__post_init__ refuses opened_year is None with a nonzero balance on any path; "
        "a LIF's minimum is the RRIF minimum (engine.accounts.lif's own module docstring), "
        "computed by engine.accounts.rrif.minimum_withdrawal from the per-path opening balance, "
        "so a shared open year costs nothing on a path where the plan holds nothing — including "
        "a path where the holder died before the conversion age.",
    ),
    ("LiraState", "jurisdiction"): (
        Scalarity.IDENTITY,
        "A pension-jurisdiction code, not a quantity.",
    ),
    ("PensionState", "name"): (Scalarity.IDENTITY, "A label, not a quantity."),
    ("PensionState", "start_month_index"): (
        Scalarity.CALENDAR,
        "Derived from the scenario's stated calendar dates, not from anything a path draws.",
    ),
    ("PensionState", "indexed"): (
        Scalarity.SCENARIO_INPUT,
        "The scenario states whether this pension is indexed, a flag the same on every path.",
    ),
    ("PensionState", "bridge_end_month_index"): (
        Scalarity.CALENDAR,
        "Derived from the scenario's stated calendar dates, not from anything a path draws.",
    ),
    ("PensionState", "survivor_share"): (
        Scalarity.SCENARIO_INPUT,
        "The scenario states one survivor share for this pension; it multiplies the per-path "
        "amount at the point of payment rather than being drawn per path itself.",
    ),
    ("PersonState", "person_id"): (Scalarity.IDENTITY, "An identifier, not a quantity."),
    ("PersonState", "sex"): (
        Scalarity.IDENTITY,
        "Selects the life table, a scenario-stated attribute of the person, not a per-path draw.",
    ),
    ("PersonState", "birth_year"): (
        Scalarity.SCENARIO_INPUT,
        "The scenario states one birth date for this person.",
    ),
    ("PersonState", "birth_month"): (
        Scalarity.SCENARIO_INPUT,
        "The scenario states one birth date for this person.",
    ),
    ("RespState", "subscriber_index"): (
        Scalarity.IDENTITY,
        "Resolved once in engine.core.build from the scenario's named subscriber; the engine "
        "models no transfer of the plan on the subscriber's death, so the index cannot change "
        "on one path and not another.",
    ),
    ("RespState", "education_start_month_index"): (
        Scalarity.CALENDAR,
        "Derived from the scenario's stated calendar dates, not from anything a path draws.",
    ),
    ("RespState", "education_months"): (
        Scalarity.SCENARIO_INPUT,
        "The scenario states one enrolment length in months.",
    ),
    ("RespState", "education_monthly_cost"): (
        Scalarity.SCENARIO_INPUT,
        "The scenario states one monthly cost; not drawn per path.",
    ),
    ("RrifState", "opened_year"): (
        Scalarity.DETERMINISTIC_TRIGGER,
        "RrifState.__post_init__ refuses opened_year is None with a nonzero balance on any path, "
        "and engine.accounts.rrif.minimum_withdrawal multiplies the per-path opening balance, so "
        "a shared open year costs nothing on a path where the plan holds nothing — including a "
        "path where the holder died before the conversion age.",
    ),
    ("RrspState", "converted_fraction_applied"): (
        Scalarity.DETERMINISTIC_TRIGGER,
        "engine.accounts.rrsp.must_convert takes age_at_end_of_year, which its own docstring "
        "says is 'not per-path: a person's age does not vary by path', and convert(state, "
        "fraction) takes a scalar fraction — nothing balance- or depletion-driven decides "
        "whether the flag fires. convert() still moves only state.balance * fraction "
        "elementwise, so on a path where the holder already died before that age the shared "
        "trigger never mixes one path's balance into another's.",
    ),
    ("SpendingLevel", "from_year"): (
        Scalarity.SCENARIO_INPUT,
        "Copied unchanged from the scenario's stated spending schedule (engine.core.build), not "
        "converted through the calendar the way a month index is.",
    ),
    ("SpendingLevel", "monthly_level"): (
        Scalarity.SCENARIO_INPUT,
        "Derived once from the scenario's spending schedule; household-wide, not per path.",
    ),
    ("YearRecord", "year"): (Scalarity.CALENDAR, "The calendar does not vary by path."),
}

#: A generous floor, not an exact count: high enough that a collector run over every state
#: dataclass and silently returning too little could not pass. It classifies 40 fields today.
MINIMUM_CLASSIFIED = 35


def _state_dataclasses() -> tuple[type, ...]:
    """Every dataclass defined in :mod:`engine.core.state`, in module order."""
    return tuple(
        obj
        for _, obj in inspect.getmembers(state, inspect.isclass)
        if dataclasses.is_dataclass(obj) and obj.__module__ == state.__name__
    )


def _strip_optional(annotation: object) -> object:
    """``X | None`` becomes ``X``; anything else is returned unchanged."""
    origin = typing.get_origin(annotation)
    if origin is types.UnionType or origin is typing.Union:
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _is_own_dataclass(annotation: object) -> bool:
    """Whether ``annotation`` is itself a dataclass defined in :mod:`engine.core.state`."""
    return (
        inspect.isclass(annotation)
        and dataclasses.is_dataclass(annotation)
        and annotation.__module__ == state.__name__
    )


def _is_tuple_of_own_dataclass(annotation: object) -> bool:
    """Whether ``annotation`` is ``tuple[X, ...]`` for some dataclass ``X`` in this module."""
    if typing.get_origin(annotation) is not tuple:
        return False
    args = typing.get_args(annotation)
    return bool(args) and _is_own_dataclass(args[0])


def in_scope_fields(classes: Iterable[type]) -> dict[tuple[str, str], object]:
    """Every ``(class name, field name)`` among ``classes`` that is not an array or nested state.

    Args:
        classes: Dataclasses to inspect. Not necessarily all of them, or all real ones: the
            "guard bites" test below passes a synthetic class here too.

    Returns:
        Resolved (``| None``-stripped) annotation keyed by ``(class.__name__, field.name)``, for
        every field in scope per the module docstring.
    """
    found: dict[tuple[str, str], object] = {}
    for cls in classes:
        hints = typing.get_type_hints(cls, include_extras=False)
        for f in dataclasses.fields(cls):
            annotation = _strip_optional(hints[f.name])
            if typing.get_origin(annotation) is NDArray:
                continue
            if _is_own_dataclass(annotation) or _is_tuple_of_own_dataclass(annotation):
                continue
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
            f"the following fields of engine.core.state are in scope but not classified in "
            f"TABLE: {sorted(missing)}. For each: decide whether it is an IDENTITY, a "
            f"SCENARIO_INPUT, a CALENDAR value, or a DETERMINISTIC_TRIGGER, and add it to TABLE "
            f"with a one-line reason it is legitimately the same on every path."
        )

    def test_no_stale_entry_names_a_field_that_no_longer_exists(self) -> None:
        present = set(in_scope_fields(_state_dataclasses()))
        stale = set(TABLE) - present
        assert not stale, (
            f"TABLE classifies the following fields, but they are no longer in scope in "
            f"engine.core.state (renamed, removed, or no longer in scope): {sorted(stale)}. "
            f"Update TABLE to match."
        )

    def test_the_table_is_not_vacuous(self) -> None:
        """A floor on the collector's own output, so a collector that silently stopped finding
        fields fails rather than passing test_no_field_is_missing_from_the_table vacuously."""
        assert len(in_scope_fields(_state_dataclasses())) >= MINIMUM_CLASSIFIED
        assert len(TABLE) >= MINIMUM_CLASSIFIED

    def test_the_floor_can_actually_bite(self) -> None:
        """Shows the assertion above can fail, not just that it currently passes.

        A collector restricted to one class finds far fewer fields than the floor, the way
        ``tests/core/test_state_nominal_or_real.py``'s ``MINIMUM_CLASSIFIED`` test shows its own
        floor can bite.
        """
        assert len(in_scope_fields([state.YearRecord])) < MINIMUM_CLASSIFIED


class TestTheGuardBites:
    def test_an_unclassified_field_on_a_new_class_is_reported(self) -> None:
        """The success criterion: adding a field fails this test until it is classified.

        A synthetic frozen dataclass with one in-scope scalar field, run through the same
        collector the real test uses, against the real ``TABLE`` — which does not, and cannot,
        know about a class defined inside this test.
        """

        @dataclasses.dataclass(frozen=True, slots=True)
        class NotYetClassified:
            new_label: str

        missing = unclassified([NotYetClassified], TABLE)
        assert missing == {("NotYetClassified", "new_label")}

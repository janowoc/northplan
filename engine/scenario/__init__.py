# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The scenario: everything a run is told, and nothing it works out.

A scenario is the household's side of the calculation — who they are, what they
hold, what they expect the market to do, and which policies to try. The other
side is ``params/``, which holds the rules, and the two never mix: no tax
figure appears in a scenario file and no household figure appears in a
parameter file.

``schema`` says what a well-formed scenario is; ``load`` turns a file into one.
``start_ages`` refuses a CPP or OAS start age, or a RRIF conversion age, the
scenario's parameter year does not allow. ``lifespan`` refuses a scenario the
parameter year's life table cannot represent, either for one person past the
terminal age or for a household whose run cannot reach its first December
close. Building engine state from a scenario lives in
``engine/core/build.py``, not here — this package hands out no NumPy arrays.

Names ending in ``Spec`` are the deliberate exceptions to plain naming: they
describe an input that the engine also has a *runtime* class for, and the two
have to be importable side by side in the builder without an alias.
"""

from engine.scenario.lifespan import (
    LifespanNotRepresentableError,
    check_lifespan,
)
from engine.scenario.load import (
    DuplicateKeyError,
    InvalidScenarioError,
    MalformedScenarioFileError,
    ScenarioError,
    ScenarioFileMissingError,
    load_scenario,
)
from engine.scenario.schema import (
    CONTRIBUTION_KINDS,
    DEFAULT_ALLOCATION,
    INVESTABLE_KINDS,
    WITHDRAWAL_KINDS,
    Accounts,
    AssetClass,
    Assumptions,
    Beneficiary,
    CashAccount,
    ContributionRule,
    CppEntitlement,
    DbPension,
    Education,
    ElectionsSpec,
    Employment,
    Household,
    LifAccount,
    LiraAccount,
    OasEntitlement,
    Person,
    PolicySpec,
    Resp,
    RrifAccount,
    RrifConversion,
    RrspAccount,
    Scenario,
    Spending,
    SpendingBand,
    TaxableAccount,
    TfsaAccount,
    WithdrawalRule,
    resolve_policy_path,
)
from engine.scenario.start_ages import (
    StartAgeNotAllowedError,
    check_start_ages,
)

__all__ = [
    "CONTRIBUTION_KINDS",
    "DEFAULT_ALLOCATION",
    "INVESTABLE_KINDS",
    "WITHDRAWAL_KINDS",
    "Accounts",
    "AssetClass",
    "Assumptions",
    "Beneficiary",
    "CashAccount",
    "ContributionRule",
    "CppEntitlement",
    "DbPension",
    "DuplicateKeyError",
    "Education",
    "ElectionsSpec",
    "Employment",
    "Household",
    "InvalidScenarioError",
    "LifAccount",
    "LifespanNotRepresentableError",
    "LiraAccount",
    "MalformedScenarioFileError",
    "OasEntitlement",
    "Person",
    "PolicySpec",
    "Resp",
    "RrifAccount",
    "RrifConversion",
    "RrspAccount",
    "Scenario",
    "ScenarioError",
    "ScenarioFileMissingError",
    "Spending",
    "SpendingBand",
    "StartAgeNotAllowedError",
    "TaxableAccount",
    "TfsaAccount",
    "WithdrawalRule",
    "check_lifespan",
    "check_start_ages",
    "load_scenario",
    "resolve_policy_path",
]

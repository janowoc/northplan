# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``MarketInputs``: the per-scenario market assumptions the run reads.

Owns :class:`MarketInputs`, the array-valued form of a scenario's capital
market assumptions that the draws and the monthly step read, always in
``asset_class_names`` order. Every array on it is a read-only copy: the
caller's own arrays are never frozen, and nothing this object holds can
change out from under a path in flight.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

#: The key of the allocation every account without its own entry holds.
#: Same key as ``engine.scenario.schema.DEFAULT_ALLOCATION``.
DEFAULT_KIND: Final[str] = "default"


@dataclass(frozen=True, slots=True)
class MarketInputs:
    """A scenario's capital market assumptions, as arrays.

    Attributes:
        asset_class_names: Class names, the order every array below is indexed by.
        annual_means: Real annual arithmetic means, bare fractions, ``(n_assets,)``.
        annual_covariance: Covariance of real annual simple returns,
            ``(n_assets, n_assets)``.
        interest_yields: Annual fraction of balance distributed as interest,
            ``(n_assets,)``.
        dividend_yields: Annual fraction distributed as dividends, ``(n_assets,)``.
        distributed_gains_yields: Annual fraction distributed as realized capital
            gains, ``(n_assets,)``.
        weights_by_kind: Portfolio weights by account kind, ``(n_assets,)`` each.
            Must contain :data:`DEFAULT_KIND`, held by any account kind absent
            from this mapping.
        investable_kinds: Account kinds that hold investments.

    Every array field, including each entry of ``weights_by_kind``, is copied and
    frozen on construction. Shapes only are validated here; attainability of the
    moments, weights summing to one, and ``weights_by_kind`` naming only
    investable kinds are the schema's job, already checked when the scenario
    loaded.
    """

    asset_class_names: tuple[str, ...]
    annual_means: NDArray[np.float64]
    annual_covariance: NDArray[np.float64]
    interest_yields: NDArray[np.float64]
    dividend_yields: NDArray[np.float64]
    distributed_gains_yields: NDArray[np.float64]
    weights_by_kind: Mapping[str, NDArray[np.float64]]
    investable_kinds: frozenset[str]

    def __post_init__(self) -> None:
        names = tuple(self.asset_class_names)
        n = len(names)
        if n < 1:
            raise ValueError("MarketInputs.asset_class_names: no asset classes given.")

        for field_name, expected in (
            ("annual_means", (n,)),
            ("annual_covariance", (n, n)),
            ("interest_yields", (n,)),
            ("dividend_yields", (n,)),
            ("distributed_gains_yields", (n,)),
        ):
            array = np.array(getattr(self, field_name), dtype=np.float64)
            if array.shape != expected:
                raise ValueError(
                    f"MarketInputs.{field_name}: expected shape {expected} for asset "
                    f"classes {', '.join(names)}, got {array.shape}."
                )
            array.flags.writeable = False
            object.__setattr__(self, field_name, array)

        if DEFAULT_KIND not in self.weights_by_kind:
            known = ", ".join(sorted(self.weights_by_kind)) or "none"
            raise ValueError(
                f"MarketInputs.weights_by_kind: a {DEFAULT_KIND!r} entry is required; got {known}."
            )

        frozen_weights: dict[str, NDArray[np.float64]] = {}
        for kind, weights in self.weights_by_kind.items():
            array = np.array(weights, dtype=np.float64)
            if array.shape != (n,):
                raise ValueError(
                    f"MarketInputs.weights_by_kind[{kind!r}]: expected shape ({n},) for "
                    f"asset classes {', '.join(names)}, got {array.shape}."
                )
            array.flags.writeable = False
            frozen_weights[kind] = array

        object.__setattr__(self, "asset_class_names", names)
        object.__setattr__(self, "weights_by_kind", MappingProxyType(frozen_weights))
        object.__setattr__(self, "investable_kinds", frozenset(self.investable_kinds))

    def weights(self, kind: str) -> NDArray[np.float64]:
        """The allocation weight vector for ``kind``, ``(n_assets,)``.

        Returns ``kind``'s own entry if present; otherwise the
        :data:`DEFAULT_KIND` entry, if ``kind`` is an investable account.

        Raises:
            ValueError: If ``kind`` is not an account kind that holds
                investments.
        """
        if kind in self.weights_by_kind:
            return self.weights_by_kind[kind]
        if kind in self.investable_kinds:
            return self.weights_by_kind[DEFAULT_KIND]
        known = ", ".join(sorted(self.investable_kinds))
        raise ValueError(
            f"MarketInputs.weights: {kind!r} is not an account kind that holds "
            f"investments. Known: {known}."
        )

    def weighted_yields(self, kind: str) -> tuple[float, float, float]:
        """The three yields for ``kind``'s allocation, weighted by its portfolio.

        Returns:
            ``(interest, dividends, distributed_gains)``, each an annual
            fraction of balance.
        """
        w = self.weights(kind)
        return (
            float(w @ self.interest_yields),
            float(w @ self.dividend_yields),
            float(w @ self.distributed_gains_yields),
        )

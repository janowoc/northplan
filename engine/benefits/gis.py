"""Guaranteed Income Supplement.

Income-tested, non-taxable, and reduced against a different income base than
the OAS recovery tax uses — notably OAS itself is excluded from that base.
Modelled only for households whose projected income makes it reachable.

Parameters from ``params/{year}/oas.yaml``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engine.params.loader import ParamSet


def countable_income(
    net_income: ArrayLike,
    oas_received: ArrayLike,
    employment_income: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Income base against which GIS is reduced.

    Not the same as net income: OAS is excluded, and part of employment income
    is exempt.

    Args:
        net_income: Net income for the relevant year, ``(n_paths,)``.
        oas_received: OAS included in that net income, to be excluded here.
        employment_income: Employment and self-employment income, for the
            earnings exemption.
        params: The ``oas`` parameter set.

    Returns:
        Countable income, non-negative.
    """
    raise NotImplementedError


def supplement_annual(
    countable: ArrayLike,
    has_spouse: ArrayLike,
    spouse_receives_oas: ArrayLike,
    params: ParamSet,
) -> NDArray[np.float64]:
    """Annual GIS payable.

    The maximum and the reduction rate both depend on marital status and on
    whether a spouse receives OAS, which is why those flags are arguments
    rather than being folded into a single rate.

    Args:
        countable: Output of :func:`countable_income`.
        has_spouse: Whether the recipient has a spouse or common-law partner.
        spouse_receives_oas: Whether that spouse receives OAS.
        params: The ``oas`` parameter set.

    Returns:
        Annual GIS in real dollars, non-negative and non-taxable.
    """
    raise NotImplementedError

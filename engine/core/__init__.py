"""The deterministic annual step: ``(state, year, policy) -> state``.

There is exactly one simulation loop in this repository and it lives here.
Monte Carlo (``engine/mc/``) drives this step across years for all paths at
once; the optimizer (``engine/optimize/``) drives Monte Carlo across policies.
Neither reimplements the year.

All amounts are real dollars. All arrays are shape ``(n_paths,)``.
"""

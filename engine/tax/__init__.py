"""Array-valued pure functions for Canadian income tax.

Every function here takes and returns NumPy arrays of shape ``(n_paths,)`` and
is pure: same inputs, same outputs, no state. Bracket logic uses ``np.clip``
and ``np.where`` rather than Python branching, so that a scalar input works by
broadcasting and a golden-number test exercises the identical code path as a
100,000-path Monte Carlo run.

All dollar amounts in and out are real dollars (see ``engine/__init__.py``).
All parameters come from ``params/`` via ``engine.params.loader``; there is
never a numeric tax constant in this package.
"""

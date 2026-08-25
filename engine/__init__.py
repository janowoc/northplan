"""northplan simulation engine.

Pure Python. This package must never import ``api``, ``fastapi``, or ``cli``;
``tests/test_layering.py`` enforces that by walking the AST of every module
here.

Conventions that hold everywhere below this package
---------------------------------------------------

**Real dollars, internally, always.** Every dollar amount that crosses a
function boundary inside ``engine/`` is in real (constant purchasing power)
dollars, expressed in the base year of the scenario. Consequences:

- CPI-indexed tax brackets, credits, and benefit thresholds are *constant* in
  real terms across simulated years. They are loaded once and reused. Do not
  index them forward; that would double-count inflation.
- Non-indexed amounts are the exception and must decay explicitly. A DB pension
  with no indexation loses real value every year, and the code that models it
  must apply that decay by hand and say so in a comment. Silence here is a bug.
- Return assumptions are real returns. Contribution and withdrawal amounts are
  real amounts.
- Conversion to nominal dollars happens exactly once, at display, in ``api/``
  or ``cli/``. No function in ``engine/`` returns a nominal figure.

**Shapes.** The simulation is vectorized across paths, not years. Years are
sequentially dependent so the outer loop steps through them one at a time; the
paths within a year are independent, so the inner computation handles all of
them at once. Every function in ``tax/`` and ``benefits/`` takes and returns
NumPy arrays of shape ``(n_paths,)``, using ``np.clip`` and ``np.where`` for
bracket logic rather than Python branching. Because NumPy broadcasts, a scalar
is a valid input: a golden-number test calling a function with a single float
exercises the identical code path as production. One implementation, two use
sites.

**Common random numbers.** The return matrix is generated once, from a fixed
seed, and reused across every policy the optimizer evaluates. Redrawing per
policy would make the optimizer chase Monte Carlo noise instead of signal.

**No clairvoyance.** A policy function may read only what is knowable at that
simulated point in time — current balances, current age, current-year brackets,
realized history. It may never read a future return, a future balance, or a
terminal value. A clairvoyant policy produces an answer that cannot be acted on.

**Household is a list of persons**, from day one, even when only one person is
modelled. Pension splitting, survivor benefits, the RRIF spousal rollover, OAS
ceasing at first death, and two mortality timelines all need two people.

**No invented parameters.** No tax, benefit, or account constant is ever a
literal in a ``.py`` file under this package. Every one is loaded from
``params/`` through ``engine.params.loader``, which raises
``MissingParameterError`` rather than substituting a default.
"""

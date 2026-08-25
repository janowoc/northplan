"""Search over policy parameters.

The outermost of the three layers: the optimizer proposes a policy, Monte Carlo
evaluates it across paths, and the annual step runs each year. The optimizer
searches *policy parameters*, never paths and never per-path decisions.

Two rules hold everywhere here:

- **Common random numbers.** Every candidate policy is evaluated against the
  same :class:`~engine.mc.returns.RandomDraws`. Regenerating draws per
  candidate turns the objective into noise and the optimizer into a noise
  maximiser.
- **Brute force first.** A grid search over a vectorized simulator is fast
  enough and is obviously correct. Anything cleverer needs the grid search
  as its reference before it is trusted.
"""

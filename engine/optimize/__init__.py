"""Search over policy parameters.

The outermost of the three layers: the optimizer proposes a policy, Monte Carlo
evaluates it across paths, and the monthly step runs each month. The optimizer
searches *policy parameters*, never paths and never per-path decisions.

The monthly timestep costs twelve times the steps per evaluation, and the
optimizer is what multiplies that cost by the size of the grid. Vectorization
across paths is what makes it affordable; a Python loop over paths inside the
month would not be.

Two rules hold everywhere here:

- **Common random numbers.** Every candidate policy is evaluated against the
  same :class:`~engine.mc.returns.RandomDraws`. Regenerating draws per
  candidate turns the objective into noise and the optimizer into a noise
  maximiser.
- **Brute force first.** A grid search over a vectorized simulator is fast
  enough and is obviously correct. Anything cleverer needs the grid search
  as its reference before it is trusted.
"""

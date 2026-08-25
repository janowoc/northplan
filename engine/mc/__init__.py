"""Monte Carlo: return generation and the loop over years.

**Common random numbers.** The return matrix and the mortality draws are
generated once, from a fixed seed, and reused unchanged across every policy the
optimizer evaluates. Two policies must be compared on the same futures. Drawing
fresh numbers per policy makes the difference between two policies mostly
sampling noise, and the optimizer will happily maximise that noise.

Vectorization is across paths. The loop here is over years, because years are
sequentially dependent; within a year, all paths are computed at once.
"""

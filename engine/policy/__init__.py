"""Parameterized decision rules — what the optimizer searches over.

A policy is a small set of numbers plus the rules that turn state into
decisions. The optimizer searches the numbers; the rules are fixed code.

**The constraint that makes any of this meaningful:** a policy function may
read only what is knowable at that simulated moment — current balances, current
ages, realized history, and the current year's parameters. It may never read a
future return, a future balance, a terminal value, or any array sliced past the
current year index.

A policy that peeks produces a plan nobody can follow, and it will look
excellent. That failure mode is silent, which is why the verifier checks for it
specifically.
"""

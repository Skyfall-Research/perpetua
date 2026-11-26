# pp.drift

Drift processes change parameters inside the lifetime: `Stationary`, `Cyclic`,
`RandomWalk`, `Procedural`, the agent-driven `Accumulate`, `Schedule` (a
schedule held as data, so each vmapped lifetime can have its own), and the
`Sequence` and `Sum` combinators. A phase change is a declared switch.

```python
schedule = pp.drift.Sequence(
    [(pp.drift.Stationary(), 1_000), (pp.drift.Cyclic([(7, 7), (0, 0)], 256), 4_096)]
)
```

See [lifetimes and drift](../native-api.md) for transition order and timing.

```{eval-rst}
.. automodule:: perpetua.drift
   :members:
```

# pp.runtime (runners)

`init_lifetime` builds the carry (state, public observation and info, separate
environment and agent keys, metrics). `Runner` compiles a chunk of ticks once;
`HostRunner` steps host simulators tick by tick in a Python loop. `transition`, `interact` and `tick` are the
per-tick body every runner shares. `assign_schedule` gives a lifetime its own
drift schedule. `perpetua.compute` adds compute-aware time.

`ChunkRunner` is the contract every runner follows, including the learner
runners and `ProbedRunner`: `runner(carry)` returns the next carry, `metadata`
is the configuration checked on restore, and `donates` tells whether the call
consumes its input carry. Code that accepts "any runner" should rely on these
three members only.

```python
env = pp.make("ContinualGrid-v0")
carry = pp.runtime.init_lifetime(env, seed=0, window_size=128)
runner = pp.runtime.Runner(env, pp.policies.random, ticks=256, donate=True)
carry = runner(carry)
```

```{eval-rst}
.. autoclass:: perpetua.runtime.ChunkRunner
   :members:
   :special-members: __call__

.. automodule:: perpetua.runtime
   :members:
   :exclude-members: ChunkRunner

.. automodule:: perpetua.compute
   :members:

.. automodule:: perpetua.policies
   :members:
```

# pp.make and registration

`pp.make(id, **kwargs)` builds a registered environment (your own
registrations are checked first); `drift=` attaches a drift process (or a
mapping of parameter names to processes), and without it nothing drifts. `pp.register` adds
your own factories, `pp.compose(env, drift)` rebinds drift on an environment
that supports it, and `pp.presets` builds benchmark sequences.
`pp.environments()` lists every id `make` accepts, `pp.environments(builtin=True)`
only the built-in ones, and `pp.registered()` only the ones you registered.
`pp.spec(id)` returns what the registry records about a built-in id without
constructing it: the extra it needs, whether it steps a simulator outside JAX,
the platform it is limited to and the script that builds its runtime.

```python
import perpetua as pp

env = pp.make("ContinualGrid-v0", size=8, drift=pp.drift.RandomWalk(low=0, high=7))
pp.register("my/Grid-v0", lambda **kw: pp.make("ContinualGrid-v0", **kw))
sequence = pp.presets.switching_minatar(period=100_000)
```

```{eval-rst}
.. automodule:: perpetua.registration
   :members:

.. automodule:: perpetua.compose
   :members:

.. automodule:: perpetua.presets
   :members:
```

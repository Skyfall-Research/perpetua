# perpetua.envs (native worlds)

The JAX parameter environments behind the native ids. Construct them through
`pp.make`; the classes document their dynamics. `pp.make` attaches no drift
unless you pass `drift=`, so `ForagerGrid-v0` keeps a fixed mass until given a
`pp.drift.Accumulate` process.

```{eval-rst}
.. automodule:: perpetua.envs.continual_grid
   :members:

.. automodule:: perpetua.envs.door_grid
   :members:

.. automodule:: perpetua.envs.forager_grid
   :members:

.. automodule:: perpetua.envs.bean_world
   :members:

.. automodule:: perpetua.envs.cooperative_grid
   :members:
   :exclude-members: check_slots

.. automodule:: perpetua.envs.partner_grid
   :members:

.. automodule:: perpetua.envs.opponent_grid
   :members:
```

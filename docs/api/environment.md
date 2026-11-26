# Environment and pp.Oracle

An environment is `init(key) -> (state, obs, info)` and
`step(state, action, key) -> (state, obs, reward, info)`, with spaces,
metadata and Oracle fields. `DriftEnv` wraps a parameter environment and owns
drift state, its random stream, declared switches, the clock and the Oracle
fields; `Oracle` exposes chosen privileged fields under `info["oracle"]` and
records the choice.

```python
env = pp.Oracle(pp.make("ContinualGrid-v0"), expose=["task_id", "boundary"])
env.metadata["oracle_exposure"]  # ('task_id', 'boundary')
```

The contract in full, with a worked example, is in the
[native environment guide](../native-api.md); `Info`, `EnvInfo` and `Metadata`
below list every key with its shape and meaning.

```{eval-rst}
.. automodule:: perpetua.core
   :members:
   :exclude-members: as_int, requires_extra

.. automodule:: perpetua.params
   :members:
```

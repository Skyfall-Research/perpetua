# pp.checkpoint

`save` writes a whole carry (lifetime, learner, optimizer, replay, RNG) and its
configuration atomically; `restore` checks both against a template; `replay`
repeats a lifetime segment from a checkpoint exactly on the same software and
backend. `save` also records what wrote the checkpoint (Perpetua version and
source revision, JAX, Orbax, backend, Python, platform); `provenance` reads
that record back, and `restore` warns when the running software differs or
when a revision does not identify the code (unknown on one side, or
uncommitted changes on either). `restore` also compares the shapes and dtypes
of the stored arrays with the template before loading them.

The three functions take a complete carry. `perpetua.torch_bridge.TorchRunner`
is the exception among the library's runners: its agent's state lives in the
PyTorch agent, not in the carry, so `save`, `restore` and `replay` refuse a
`TorchRunner` and its `TorchTrainingState` with a `TypeError`. Use
`TorchRunner.save` and `TorchRunner.restore` for it.

```python
env = pp.make("ContinualGrid-v0")
runner = pp.runtime.Runner(env, pp.policies.random, ticks=256)
template = pp.runtime.init_lifetime(env, seed=0)
state = runner(template)
pp.checkpoint.save("checkpoints/001", state, metadata=runner.metadata)
state = pp.checkpoint.restore("checkpoints/001", template, expected_metadata=runner.metadata)
segment = pp.checkpoint.replay("checkpoints/001", template, runner, chunks=4)
pp.checkpoint.provenance("checkpoints/001")["revision"]
```

```{eval-rst}
.. automodule:: perpetua.checkpoint
   :members:
```

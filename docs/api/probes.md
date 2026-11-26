# pp.probes

Dormant-unit ratio and effective rank, on demand or on the update clock with
`ProbedRunner`.

```python
from perpetua.agents import PPO, PPORunner

trainer = PPORunner(pp.make("ContinualGrid-v0"), PPO(action_size=5))
tracked = pp.probes.ProbedRunner(trainer, every=10)
carry = tracked(tracked.init(seed=0))
carry.trace.latest
```

```{eval-rst}
.. automodule:: perpetua.probes
   :members:
```

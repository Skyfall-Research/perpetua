# perpetua.agents (learners)

Continuing PPO and IPPO, DQN and IQL, and SAC, with traced hyperparameters and
plasticity options; `perpetua.torch_bridge` runs PyTorch agents. See
[training](../training.md).

```python
from perpetua.agents import PPO, PPORunner

env = pp.make("ContinualGrid-v0")
trainer = PPORunner(env, PPO(action_size=5), donate=True)
state = trainer(trainer.init(seed=0))
```

```{eval-rst}
.. automodule:: perpetua.agents.ppo
   :members:

.. automodule:: perpetua.agents.dqn
   :members:

.. automodule:: perpetua.agents.sac
   :members:

.. automodule:: perpetua.agents.plasticity
   :members:

.. automodule:: perpetua.agents.encoders
   :members:

.. automodule:: perpetua.agents.compute_ppo
   :members:

.. automodule:: perpetua.torch_bridge
   :members:
```

# Basic usage

## The lifetime

A Perpetua environment is a pure function of state and parameters:

```python
import jax
import perpetua as pp

env = pp.make("ContinualGrid-v0")
init_key, act_key, step_key = jax.random.split(jax.random.key(0), 3)
state, obs, info = env.init(init_key)
action = pp.policies.random(obs, info, act_key)
state, obs, reward, info = env.step(state, action, step_key)
```

There is no `reset()` and no `done`. `info["respawn"]` reports an event inside
the lifetime, not a boundary. Drift changes parameters, and you never write the drift
bookkeeping yourself: the framework keeps the drift state, its random stream,
the switch events and the clock.

## Construct and run

```python
import jax
import perpetua as pp

env = pp.make("ContinualGrid-v0", size=8, drift=pp.drift.Cyclic([(7, 7), (0, 0)], 127))
# greedy is a scripted policy that walks to the visible goal (grid worlds only)
runner = pp.runtime.Runner(env, pp.policies.greedy, ticks=256)
carry = runner(pp.runtime.init_lifetime(env, seed=0, window_size=128))
print(carry.metrics.mean)  # mean reward per tick, per agent
```

This prints `[0.13671875]`: one mean per agent slot (this world has one).

## Learners

```python
from perpetua.agents import PPO, PPORunner

# `env` is the ContinualGrid from the block above.
# ContinualGrid has 5 actions (stay, up, down, left, right): env.action_space is
# MultiDiscrete([5]), one entry per agent.
trainer = PPORunner(env, PPO(action_size=5))
train_state = trainer(trainer.init(seed=0))
```

PPO/IPPO and DQN/IQL handle discrete actions, SAC handles continuous `Box`
actions. Every learner runner records the same metrics (see
[evaluation](../evaluation.md)).

## Optional features

- **Learn average reward instead of discounted return:** `reward_centering=True`
  on a learner config subtracts a running estimate of the average reward (off by
  default). See [average reward with PPO](../tutorials/reward-centering.md).
- **Learn from pixels:** `torso="cnn"` gives PPO and DQN a convolutional
  network. See [learner options](../training-options.md#pixel-observations-cnn-torso).
- **Switch between environments in one lifetime:** `pp.tasks.switch`, where each
  task either resumes where it stopped or restarts. See
  [task switching](../task-switching.md).
- **Tell the agent what changed:** `pp.Oracle` exposes chosen privileged fields,
  for example `expose=["task_id", "boundary"]`, and records that it did. See
  [concepts](../introduction/concepts.md).
- **Save memory on long runs:** `donate=True` on `Runner` and the compiled
  learner runners update the carry's buffers in place instead of copying them.
  The carry you pass in is consumed, so keep only the returned one. See
  [lifetimes and drift](../native-api.md).
- **Watch for loss of plasticity:** `pp.probes.ProbedRunner` records the share
  of inactive ("dormant") units and the effective rank of the network's
  features as training goes. See [evaluation](../evaluation.md).
- **Reproduce a segment exactly:** `pp.checkpoint.replay` re-runs part of a
  lifetime from a checkpoint. See [`pp.checkpoint`](../api/checkpoints.md).
- **Account for tuning:** `pp.tuning.tune` tries each hyperparameter candidate
  on a declared first fraction of the lifetime, keeps the best, and reports the
  ticks that tuning cost. See [tuning](../tuning.md#declared-tuning-protocol).

See the [API reference](../api/index.md) and the
[environment index](../environments/index.md).

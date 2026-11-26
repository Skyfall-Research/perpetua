# Why the API looks the way it does

Perpetua's API is the smallest one that keeps a lifetime explicit, compiled and
free of leaked privilege. A few choices differ from the obvious first sketch,
and this page records why.

## `init` and `step` return observations and info

```text
state, obs, info = env.init(key)
state, obs, reward, info = env.step(state, action, key)
```

A sketch with `state = env.init(key)` and `state, out = env.step(...)` would
keep the observation inside the state. Returning it separately keeps the agent's
view (`obs`, `info`) apart from the world's full state, which also holds hidden
drift parameters. A policy that only receives `obs` and `info` cannot read
privileged data by accident. The four-tuple also reads like Gymnasium's
`step` without `terminated` and `truncated`.

## What `info` holds, and what it does not

By default `info` holds `action_mask`, `active_mask`, `respawn`, `space_change`
and the lifetime clock `t`. It holds no task identity, no boundary flag and no
drift parameters. **A switch is not a public event:** a flag for it would let agents
reset at switches. The flag exists as the privileged `boundary`
field, opt-in through an Oracle and recorded in the metadata:

```python
import jax
import perpetua as pp

env = pp.Oracle(
    pp.make("ContinualGrid-v0", drift=pp.drift.Cyclic([(3, 3), (0, 0)], 4)),
    expose=["boundary", "task_id"],
)
state, obs, info = env.init(jax.random.key(0))
assert "boundary" not in info and set(info["oracle"]) == {"boundary", "task_id"}
assert env.metadata["oracle_exposure"] == ("boundary", "task_id")
```

`space_change` stays public because it describes the agent's own action set,
which the agent sees through `action_mask` anyway.

## Learners act per tick and update per chunk

```text
out = agent.act(agent_state, obs, info, key)   # PPO, SAC: out.action; DQN: the action array
agent_state, ... = agent.update(agent_state, rollout_or_replay, ..., key)
```

`act` is pure: it does not return a new agent state, because acting changes
nothing a learner keeps. Learning happens in `update`, which takes a rollout
(PPO) or a replay batch (DQN, SAC) rather than one tick's output, since every
learner here updates from batches. Runners such as `PPORunner` interleave the
two inside one compiled program, and the lifetime, learner, optimizer and
replay state form one pytree.

## Metrics update from the reward

`pp.metrics.update(metrics, reward, switch=...)` takes the reward and, for
recovery, the declared switch signal. The runners pass that signal from
inside the world, so recovery after a switch is measured without exposing
it to the agent.

## Versioned ids

Perpetua's own ids carry a version (`agarcl/grand-arena-v0`), as in Gymnasium,
so a change in dynamics gets a new id instead of silently changing old
results. Unversioned aliases are not provided. Wrapped gymnax environments
keep gymnax's own names, some of which have no version
(`gymnax/Asterix-MinAtar`).

## Version

This is a pre-release (`0.0.1a0`). The API may still change before `0.1`.

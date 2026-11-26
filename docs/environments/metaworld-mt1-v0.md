# Meta-World MT1

This environment is part of the [Continual benchmarks](families/benchmarks.md) family; its page lists the install command and the family's other environments.

| | |
|---|---|
| Creation | `pp.make("metaworld/MT1-v0")` |
| Install | `uv sync --locked --extra metaworld` |
| Actions | Continuous |
| Agents | `1` |
| Action Shape | `(1, 4)` |
| Action Values | `[-1, 1]` |
| Observation Shape | `(1, 39)` |
| Observation Values | `(-inf, inf)` |
| Execution | `host` |
| Checkpointable | no |
| Backend | `metaworld-3.1.1` |
| Default drift | `stationary` |
| Driftable parameters | `reward_multiplier` |
| Default drift field | `reward_multiplier` |
| Privileged (Oracle) | `task_id`, `boundary`, drift values, `success` ([all](#privileged-fields)) |
| Upstream | [Meta-World](https://metaworld.farama.org/benchmark/benchmark_descriptions/) |

One Farama Meta-World 3.1.1 MT1 task on MuJoCo 3.3.0, host-controlled. `seed` fixes the 50 goal and object configurations MT1 generates and `task_index` pins one of them, so lifetimes with different `seed` values face different goals for the same `task_index`. `pp.presets.cw10()`/`cw20()` give the Continual World order.

![A frame of Meta-World MT1](../figures/environments/metaworld-mt1-v0.png)

*A MuJoCo render of the default task, `hammer-v3`, after 12 random actions. It is rendered from the upstream environment; the adapter runs without a renderer.*

The observation, action and reward definitions are upstream's: see [Meta-World](https://metaworld.farama.org/benchmark/benchmark_descriptions/). This page lists only what changes in Perpetua.

## What changes in Perpetua

### Respawns

Backend episodes (200 steps by default) end in respawns.

### Drift

`reward_multiplier`; task switching is a preset.

## Arguments

| Argument | Default |
|---|---|
| `env_name` | `'hammer-v3'` |
| `seed` | `0` |
| `max_episode_steps` | `200` |
| `task_index` | `0` |
| `drift` | `None` |
| `compiled` | `False` |
| `randomize_goals` | `False` |

## Usage

```python
import jax
import jax.numpy as jnp
import perpetua as pp

env = pp.make("metaworld/MT1-v0")
state, obs, info = env.init(jax.random.key(0))
for tick in range(4):
    action = jnp.zeros(env.action_space.shape, dtype=jnp.float32)
    state, obs, reward, info = env.step(state, action, jax.random.key(1000 + tick))
print(obs.shape, reward, info["t"].to_int())

# Make it non-stationary: `reward_multiplier` alternates between two values every 1,000 ticks.
drifting = pp.make(
    "metaworld/MT1-v0",
    drift={"reward_multiplier": pp.drift.Cyclic([(1,), (0.5,)], period=1000)},
)
state, obs, info = drifting.init(jax.random.key(0))
drifting.close()
env.close()  # host simulators hold an external process or library
```

## Limits

Opaque simulator state: not checkpointable. `max_episode_steps` may not exceed Meta-World's own horizon of 500 steps. Upstream declares the goal entries of the observation (36 to 38) as a single point; the lifted space declares them `(-inf, inf)`.

## See also

The [Benchmark setup: MEAL, Continual World and COOM](../benchmarks.md) guide.

## Privileged fields

Hidden from the agent; expose any of them with `pp.Oracle(env, expose=["task_id", ...])`, which records the choice in `metadata["oracle_exposure"]`.

| Field | Meaning |
|---|---|
| `phase` | the drift phase: an integer that changes at every declared switch (for `Cyclic`, the index of the current value) |
| `drift_parameters` | the current value of the default drifting parameter |
| `drift_changed` | deprecated alias of `boundary` |
| `task_id` | the active task: the task index under a task switch, the regime under MORPHEUS, otherwise the phase of the default drifting parameter |
| `boundary` | whether a declared switch happened on this tick (the name task-sequence protocols use, for example SAC's `task_boundary_reset`) |
| `reward_multiplier_parameters`, `reward_multiplier_phase`, `reward_multiplier_changed` | the current value, phase and switch flag of `reward_multiplier` |
| `success` | Meta-World's task-success flag for the last step |

## Version History

- `v0`: initial release.

## API

```{eval-rst}
.. autoclass:: perpetua.adapters.host.MetaWorldLifetime
   :no-index:
```

# EnvPool CartPole

This environment is part of the [CPU bridge](families/cpu-bridge.md) family; its page lists the install command and the family's other environments.

| | |
|---|---|
| Creation | `pp.make("envpool/CartPole-v1")` |
| Install | `uv sync --locked --extra cpu-bridge  # Linux` |
| Actions | Discrete |
| Agents | `32` |
| Action Shape | `(32,)` |
| Action Values | `Discrete(2)` per agent |
| Observation Shape | `(32, 4)` |
| Observation Values | `(-inf, inf)` |
| Execution | `host` |
| Checkpointable | no |
| Backend | `envpool-1.2.7` |
| Default drift | `stationary` |
| Driftable parameters | `reward_multiplier` |
| Default drift field | `reward_multiplier` |
| Privileged (Oracle) | `task_id`, `boundary`, drift values ([all](#privileged-fields)) |
| Upstream | [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) |

EnvPool's batched C++ CartPole, one lifetime over `num_envs` slots.

The observation, action and reward definitions are upstream's: see [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/). This page lists only what changes in Perpetua.

## What changes in Perpetua

### Respawns

EnvPool's next-step autoreset becomes a respawn event.

### Drift

`reward_multiplier`.

## Arguments

| Argument | Default |
|---|---|
| `num_envs` | `32` |
| `num_threads` | `4` |
| `episode_steps` | `500` |
| `snapshot` | `False` |
| `drift` | `None` |
| `compiled` | `False` |
| `overlap` | `False` |

## Usage

```python
import jax
import perpetua as pp

env = pp.make("envpool/CartPole-v1")
state, obs, info = env.init(jax.random.key(0))
for tick in range(4):
    action = pp.policies.random(obs, info, jax.random.key(100 + tick))
    state, obs, reward, info = env.step(state, action, jax.random.key(1000 + tick))
print(obs.shape, reward, info["t"].to_int())

# Make it non-stationary: `reward_multiplier` alternates between two values every 1,000 ticks.
drifting = pp.make(
    "envpool/CartPole-v1",
    drift={"reward_multiplier": pp.drift.Cyclic([(1,), (0.5,)], period=1000)},
)
state, obs, info = drifting.init(jax.random.key(0))
drifting.close()
env.close()  # host simulators hold an external process or library
```

## Limits

Linux only. `snapshot=True` needs the optional native binding; `overlap=True` steps half batches (not bit-reproducible).

## See also

The [Environment adapters and the CPU bridge](../adapters.md) guide.

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

## Version History

- `v1`: initial release.

## API

```{eval-rst}
.. autoclass:: perpetua.adapters.host.EnvPoolCartPole
   :no-index:
```

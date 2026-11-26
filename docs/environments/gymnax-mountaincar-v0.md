# Mountain Car

This environment is part of the [Classic and MuJoCo control](families/control.md) family; its page lists the install command and the family's other environments.

| | |
|---|---|
| Creation | `pp.make("gymnax/MountainCar-v0")` |
| Install | `uv sync --locked --extra interop` |
| Actions | Discrete |
| Agents | `1` |
| Action Shape | `(1,)` |
| Action Values | `Discrete(3)` per agent |
| Observation Shape | `(1, 2)` |
| Observation Values | `[-1.2, 0.6]` |
| Execution | `jax` |
| Checkpointable | yes |
| Backend | `gymnax-0.0.9` |
| Default drift | `stationary` |
| Driftable parameters | `force`, `goal_position`, `goal_velocity`, `gravity`, `max_position`, `max_speed`, `min_position` |
| Default drift field | `min_position` |
| Privileged (Oracle) | `task_id`, `boundary`, drift values ([all](#privileged-fields)) |
| Upstream | [Gymnasium Mountain Car](https://gymnasium.farama.org/environments/classic_control/mountain_car/) |

gymnax's `MountainCar-v0` through the generic `GymnaxParameters` adapter.

![A frame of Mountain Car](../figures/environments/gymnax-mountaincar-v0.svg)

*A drawing of the observation after 12 random ticks (positions and angles; not a simulator render).*

The observation, action and reward definitions are upstream's: see [Gymnasium Mountain Car](https://gymnasium.farama.org/environments/classic_control/mountain_car/). This page lists only what changes in Perpetua.

## What changes in Perpetua

### Rewards

The backend's reward, unchanged.

### Respawns

The backend time limit is removed; terminations become respawns inside the lifetime.

### Drift

Any float parameter listed under Driftable parameters, by name.

## Arguments

| Argument | Default |
|---|---|
| `drift_fields` | `None` |
| `drift` | `None` |

## Usage

```python
import jax
import perpetua as pp

env = pp.make("gymnax/MountainCar-v0")
state, obs, info = env.init(jax.random.key(0))
for tick in range(4):
    action = pp.policies.random(obs, info, jax.random.key(100 + tick))
    state, obs, reward, info = env.step(state, action, jax.random.key(1000 + tick))
print(obs.shape, reward, info["t"].to_int())

# Make it non-stationary: `min_position` alternates between two values every 1,000 ticks.
drifting = pp.make(
    "gymnax/MountainCar-v0",
    drift={"min_position": pp.drift.Cyclic([-1.2, -0.6], period=1000)},
)
state, obs, info = drifting.init(jax.random.key(0))
```

## Limits

Actions are checked for shape and dtype (an integer array of shape `(1,)`), not for value: an index outside the action range reaches the backend as it is. Most backends look the index up in a table, so an index past the end acts as the last action and a negative one counts from the end; MountainCar scales the push with it. Clip in the policy.

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
| `force_parameters`, `force_phase`, `force_changed` | the current value, phase and switch flag of `force` |
| `goal_position_parameters`, `goal_position_phase`, `goal_position_changed` | the current value, phase and switch flag of `goal_position` |
| `goal_velocity_parameters`, `goal_velocity_phase`, `goal_velocity_changed` | the current value, phase and switch flag of `goal_velocity` |
| `gravity_parameters`, `gravity_phase`, `gravity_changed` | the current value, phase and switch flag of `gravity` |
| `max_position_parameters`, `max_position_phase`, `max_position_changed` | the current value, phase and switch flag of `max_position` |
| `max_speed_parameters`, `max_speed_phase`, `max_speed_changed` | the current value, phase and switch flag of `max_speed` |
| `min_position_parameters`, `min_position_phase`, `min_position_changed` | the current value, phase and switch flag of `min_position` |
| `max_steps_in_episode_parameters`, `max_steps_in_episode_phase`, `max_steps_in_episode_changed` | the backend's original time limit, as a constant; the adapter removes the limit, so it is not enforced |

## Version History

- `v0`: wraps gymnax 0.0.9; the version follows gymnax's id.

## API

```{eval-rst}
.. autoclass:: perpetua.adapters.gymnax_params.GymnaxParameters
   :no-index:
```

# FourRooms (misc)

This environment is part of the [MinAtar, bsuite and misc (gymnax)](families/gymnax.md) family; its page lists the install command and the family's other environments.

| | |
|---|---|
| Creation | `pp.make("gymnax/FourRooms-misc")` |
| Install | `uv sync --locked --extra interop` |
| Actions | Discrete |
| Agents | `1` |
| Action Shape | `(1,)` |
| Action Values | `Discrete(4)` per agent |
| Observation Shape | `(1, 4)` |
| Observation Values | `[1, 11]` |
| Execution | `jax` |
| Checkpointable | yes |
| Backend | `gymnax-0.0.9` |
| Default drift | `stationary` |
| Driftable parameters | `fail_prob` |
| Default drift field | `fail_prob` |
| Privileged (Oracle) | `task_id`, `boundary`, drift values ([all](#privileged-fields)) |
| Upstream | [gymnax](https://github.com/RobertTLange/gymnax) |

gymnax's `FourRooms-misc` through the generic `GymnaxParameters` adapter.

![A frame of FourRooms (misc)](../figures/environments/gymnax-fourrooms-misc.svg)

*The map after a few random ticks: walls in grey, the agent in blue and the goal in orange.*

## Definition

### Observation Space

float32 `(1, 4)`: the agent's row and column, then the goal's row and column, on a 13-by-13 map (values 1 to 11). The agent starts at (4, 1) and the goal is at (8, 9) unless the backend's `resample_*` parameters are changed.

### Action Space

`Discrete(4)`: 0 up, 1 right, 2 down, 3 left. With probability `fail_prob * 4/3` (4/9) a uniformly random action replaces the chosen one; a wall blocks the move.

### Rewards

float32 `(1,)`: 1 when the agent moves onto the goal, else 0.

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

env = pp.make("gymnax/FourRooms-misc")
state, obs, info = env.init(jax.random.key(0))
for tick in range(4):
    action = pp.policies.random(obs, info, jax.random.key(100 + tick))
    state, obs, reward, info = env.step(state, action, jax.random.key(1000 + tick))
print(obs.shape, reward, info["t"].to_int())

# Make it non-stationary: `fail_prob` alternates between two values every 1,000 ticks.
drifting = pp.make(
    "gymnax/FourRooms-misc",
    drift={"fail_prob": pp.drift.Cyclic([0.333333, 0.166667], period=1000)},
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
| `fail_prob_parameters`, `fail_prob_phase`, `fail_prob_changed` | the current value, phase and switch flag of `fail_prob` |
| `<name>_parameters`, `<name>_phase`, `<name>_changed` for `resample_goal_pos`, `resample_init_pos` | constants: the backend's value of each field, phase 0, never changed; `drift=` rejects these fields |
| `max_steps_in_episode_parameters`, `max_steps_in_episode_phase`, `max_steps_in_episode_changed` | the backend's original time limit, as a constant; the adapter removes the limit, so it is not enforced |

## Version History

- `v0`: wraps gymnax 0.0.9; the version follows gymnax's id.

## API

```{eval-rst}
.. autoclass:: perpetua.adapters.gymnax_params.GymnaxParameters
   :no-index:
```

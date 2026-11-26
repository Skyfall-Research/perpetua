# MetaMaze (misc)

This environment is part of the [MinAtar, bsuite and misc (gymnax)](families/gymnax.md) family; its page lists the install command and the family's other environments.

| | |
|---|---|
| Creation | `pp.make("gymnax/MetaMaze-misc")` |
| Install | `uv sync --locked --extra interop` |
| Actions | Discrete |
| Agents | `1` |
| Action Shape | `(1,)` |
| Action Values | `Discrete(4)` per agent |
| Observation Shape | `(1, 15)` |
| Observation Values | `(-inf, inf)` |
| Execution | `jax` |
| Checkpointable | yes |
| Backend | `gymnax-0.0.9` |
| Default drift | `stationary` |
| Driftable parameters | `punishment`, `reward` |
| Default drift field | `reward` |
| Privileged (Oracle) | `task_id`, `boundary`, drift values ([all](#privileged-fields)) |
| Upstream | [gymnax](https://github.com/RobertTLange/gymnax) |

gymnax's `MetaMaze-misc` through the generic `GymnaxParameters` adapter.

![A frame of MetaMaze (misc)](../figures/environments/gymnax-metamaze-misc.svg)

*The map after a few random ticks: walls in grey, the agent in blue and the goal in orange.*

## Definition

### Observation Space

float32 `(1, 15)`: the 3-by-3 window around the agent (1 for a wall), flattened row-major (9 values), a one-hot of the last action (4), the last reward and the unnormalised step count since the respawn.

### Action Space

`Discrete(4)`: 0 up, 1 right, 2 down, 3 left; a wall blocks the move.

### Rewards

float32 `(1,)`: `reward` (10) when the agent reaches the goal, else 0 (the backend's wall `punishment` is 0).

### Respawns

Rare. Reaching the goal pays the reward and moves the agent to a resampled start cell without ending the episode. With the 200-step limit removed, an episode ends only when that resampled start cell is the goal itself, and only then is `info['respawn']` true and a new goal drawn: 11 respawns and 678 goal reaches in 200,000 random ticks. The last observation entry is the backend's step count since the last respawn, so it grows without bound between respawns (1,000 at tick 1,000).

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

env = pp.make("gymnax/MetaMaze-misc")
state, obs, info = env.init(jax.random.key(0))
for tick in range(4):
    action = pp.policies.random(obs, info, jax.random.key(100 + tick))
    state, obs, reward, info = env.step(state, action, jax.random.key(1000 + tick))
print(obs.shape, reward, info["t"].to_int())

# Make it non-stationary: `reward` alternates between two values every 1,000 ticks.
drifting = pp.make(
    "gymnax/MetaMaze-misc",
    drift={"reward": pp.drift.Cyclic([10, 5], period=1000)},
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
| `punishment_parameters`, `punishment_phase`, `punishment_changed` | the current value, phase and switch flag of `punishment` |
| `reward_parameters`, `reward_phase`, `reward_changed` | the current value, phase and switch flag of `reward` |
| `<name>_parameters`, `<name>_phase`, `<name>_changed` for `normalize_time` | constants: the backend's value of each field, phase 0, never changed; `drift=` rejects these fields |
| `max_steps_in_episode_parameters`, `max_steps_in_episode_phase`, `max_steps_in_episode_changed` | the backend's original time limit, as a constant; the adapter removes the limit, so it is not enforced |

## Version History

- `v0`: wraps gymnax 0.0.9; the version follows gymnax's id.

## API

```{eval-rst}
.. autoclass:: perpetua.adapters.gymnax_params.GymnaxParameters
   :no-index:
```

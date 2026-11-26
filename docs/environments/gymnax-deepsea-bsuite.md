# DeepSea (bsuite)

This environment is part of the [MinAtar, bsuite and misc (gymnax)](families/gymnax.md) family; its page lists the install command and the family's other environments.

| | |
|---|---|
| Creation | `pp.make("gymnax/DeepSea-bsuite")` |
| Install | `uv sync --locked --extra interop` |
| Actions | Discrete |
| Agents | `1` |
| Action Shape | `(1,)` |
| Action Values | `Discrete(2)` per agent |
| Observation Shape | `(1, 8, 8)` |
| Observation Values | `[0, 1]` |
| Execution | `jax` |
| Checkpointable | yes |
| Backend | `gymnax-0.0.9` |
| Default drift | `stationary` |
| Driftable parameters | `unscaled_move_cost` |
| Default drift field | `unscaled_move_cost` |
| Privileged (Oracle) | `task_id`, `boundary`, drift values ([all](#privileged-fields)) |
| Upstream | [bsuite](https://github.com/google-deepmind/bsuite) |

gymnax's `DeepSea-bsuite` through the generic `GymnaxParameters` adapter.

![A frame of DeepSea (bsuite)](../figures/environments/gymnax-deepsea-bsuite.svg)

*The observation after 40 random ticks.*

The observation, action and reward definitions are upstream's: see [bsuite](https://github.com/google-deepmind/bsuite). This page lists only what changes in Perpetua.

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

env = pp.make("gymnax/DeepSea-bsuite")
state, obs, info = env.init(jax.random.key(0))
for tick in range(4):
    action = pp.policies.random(obs, info, jax.random.key(100 + tick))
    state, obs, reward, info = env.step(state, action, jax.random.key(1000 + tick))
print(obs.shape, reward, info["t"].to_int())

# Make it non-stationary: `unscaled_move_cost` alternates between two values every 1,000 ticks.
drifting = pp.make(
    "gymnax/DeepSea-bsuite",
    drift={"unscaled_move_cost": pp.drift.Cyclic([0.01, 0.005], period=1000)},
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
| `unscaled_move_cost_parameters`, `unscaled_move_cost_phase`, `unscaled_move_cost_changed` | the current value, phase and switch flag of `unscaled_move_cost` |
| `<name>_parameters`, `<name>_phase`, `<name>_changed` for `deterministic`, `randomize_actions`, `sample_action_map` | constants: the backend's value of each field, phase 0, never changed; `drift=` rejects these fields |
| `max_steps_in_episode_parameters`, `max_steps_in_episode_phase`, `max_steps_in_episode_changed` | the backend's original time limit, as a constant; the adapter removes the limit, so it is not enforced |

## Version History

- `v0`: wraps gymnax 0.0.9; the version follows gymnax's id.

## API

```{eval-rst}
.. autoclass:: perpetua.adapters.gymnax_params.GymnaxParameters
   :no-index:
```

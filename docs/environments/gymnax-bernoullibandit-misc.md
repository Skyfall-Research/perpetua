# BernoulliBandit (misc)

This environment is part of the [MinAtar, bsuite and misc (gymnax)](families/gymnax.md) family; its page lists the install command and the family's other environments.

| | |
|---|---|
| Creation | `pp.make("gymnax/BernoulliBandit-misc")` |
| Install | `uv sync --locked --extra interop` |
| Actions | Discrete |
| Agents | `1` |
| Action Shape | `(1,)` |
| Action Values | `Discrete(2)` per agent |
| Observation Shape | `(1, 4)` |
| Observation Values | `[-1, 2]` |
| Execution | `jax` |
| Checkpointable | yes |
| Backend | `gymnax-0.0.9` |
| Default drift | `stationary` |
| Driftable parameters | `max_lim`, `min_lim`, `reward_prob` |
| Default drift field | `reward_prob` |
| Privileged (Oracle) | `task_id`, `boundary`, drift values ([all](#privileged-fields)) |
| Upstream | [gymnax](https://github.com/RobertTLange/gymnax) |

gymnax's `BernoulliBandit-misc` through the generic `GymnaxParameters` adapter.

## Definition

### Observation Space

float32 `(1, 4)`: the last reward (0 or 1), a one-hot of the last action (2 values) and the step count since the last respawn, scaled to [-1, 1] over the 100-step limit.

### Action Space

`Discrete(2)`: the arm to pull.

### Rewards

float32 `(1,)`: 1 with the pulled arm's probability, else 0. The arms pay with probabilities `reward_prob` (0.1) and `1 - reward_prob`, in an order drawn at each respawn.

### Respawns

The environment has no termination other than its time limit, so the limit is kept and reported as a respawn: `max_steps_in_episode` is 100, so `info['respawn']` is true on ticks 100, 200, 300 and so on. Parameters it reads only at reset take effect at the next respawn.

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

env = pp.make("gymnax/BernoulliBandit-misc")
state, obs, info = env.init(jax.random.key(0))
for tick in range(4):
    action = pp.policies.random(obs, info, jax.random.key(100 + tick))
    state, obs, reward, info = env.step(state, action, jax.random.key(1000 + tick))
print(obs.shape, reward, info["t"].to_int())

# Make it non-stationary: `reward_prob` alternates between two values every 1,000 ticks.
drifting = pp.make(
    "gymnax/BernoulliBandit-misc",
    drift={"reward_prob": pp.drift.Cyclic([0.1, 0.05], period=1000)},
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
| `max_lim_parameters`, `max_lim_phase`, `max_lim_changed` | the current value, phase and switch flag of `max_lim` |
| `min_lim_parameters`, `min_lim_phase`, `min_lim_changed` | the current value, phase and switch flag of `min_lim` |
| `reward_prob_parameters`, `reward_prob_phase`, `reward_prob_changed` | the current value, phase and switch flag of `reward_prob` |
| `<name>_parameters`, `<name>_phase`, `<name>_changed` for `normalize_time`, `t_max` | constants: the backend's value of each field, phase 0, never changed; `drift=` rejects these fields |
| `max_steps_in_episode_parameters`, `max_steps_in_episode_phase`, `max_steps_in_episode_changed` | the backend's time limit, which this environment keeps (it ends the episode, as a respawn) |

## Version History

- `v0`: wraps gymnax 0.0.9; the version follows gymnax's id.

## API

```{eval-rst}
.. autoclass:: perpetua.adapters.gymnax_params.GymnaxParameters
   :no-index:
```

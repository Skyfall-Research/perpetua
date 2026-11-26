# GaussianBandit (misc)

This environment is part of the [MinAtar, bsuite and misc (gymnax)](families/gymnax.md) family; its page lists the install command and the family's other environments.

| | |
|---|---|
| Creation | `pp.make("gymnax/GaussianBandit-misc")` |
| Install | `uv sync --locked --extra interop` |
| Actions | Discrete |
| Agents | `1` |
| Action Shape | `(1,)` |
| Action Values | `Discrete(2)` per agent |
| Observation Shape | `(1, 4)` |
| Observation Values | `(-inf, inf)` |
| Execution | `jax` |
| Checkpointable | yes |
| Backend | `gymnax-0.0.9` |
| Default drift | `stationary` |
| Driftable parameters | `mean_mu`, `sigma_l`, `sigma_p` |
| Default drift field | `mean_mu` |
| Privileged (Oracle) | `task_id`, `boundary`, drift values ([all](#privileged-fields)) |
| Upstream | [gymnax](https://github.com/RobertTLange/gymnax) |

gymnax's `GaussianBandit-misc` through the generic `GymnaxParameters` adapter.

## Definition

### Observation Space

float32 `(1, 4)`: the last reward, a one-hot of the last action (2 values) and the step count since the last respawn, scaled to [-1, 1] over the 100-step limit.

### Action Space

`Discrete(2)`: the arm to pull.

### Rewards

float32 `(1,)`: arm 0 pays 0. Arm 1 pays `mu` plus Gaussian noise with standard deviation `sigma_l` (0.1); `mu` is drawn at each respawn from a Gaussian with mean `mean_mu` (-1) and standard deviation `sigma_p` (1).

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

env = pp.make("gymnax/GaussianBandit-misc")
state, obs, info = env.init(jax.random.key(0))
for tick in range(4):
    action = pp.policies.random(obs, info, jax.random.key(100 + tick))
    state, obs, reward, info = env.step(state, action, jax.random.key(1000 + tick))
print(obs.shape, reward, info["t"].to_int())

# Make it non-stationary: `mean_mu` alternates between two values every 1,000 ticks.
drifting = pp.make(
    "gymnax/GaussianBandit-misc",
    drift={"mean_mu": pp.drift.Cyclic([-1, -0.5], period=1000)},
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
| `mean_mu_parameters`, `mean_mu_phase`, `mean_mu_changed` | the current value, phase and switch flag of `mean_mu` |
| `sigma_l_parameters`, `sigma_l_phase`, `sigma_l_changed` | the current value, phase and switch flag of `sigma_l` |
| `sigma_p_parameters`, `sigma_p_phase`, `sigma_p_changed` | the current value, phase and switch flag of `sigma_p` |
| `<name>_parameters`, `<name>_phase`, `<name>_changed` for `normalize_time` | constants: the backend's value of each field, phase 0, never changed; `drift=` rejects these fields |
| `max_steps_in_episode_parameters`, `max_steps_in_episode_phase`, `max_steps_in_episode_changed` | the backend's time limit, which this environment keeps (it ends the episode, as a respawn) |

## Version History

- `v0`: wraps gymnax 0.0.9; the version follows gymnax's id.

## API

```{eval-rst}
.. autoclass:: perpetua.adapters.gymnax_params.GymnaxParameters
   :no-index:
```

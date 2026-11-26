# Pendulum

This environment is part of the [Classic and MuJoCo control](families/control.md) family; its page lists the install command and the family's other environments.

| | |
|---|---|
| Creation | `pp.make("gymnax/Pendulum-v1")` |
| Install | `uv sync --locked --extra interop` |
| Actions | Continuous |
| Agents | `1` |
| Action Shape | `(1, 1)` |
| Action Values | `[-2, 2]` |
| Observation Shape | `(1, 3)` |
| Observation Values | `[-8, 8]` |
| Execution | `jax` |
| Checkpointable | yes |
| Backend | `gymnax-0.0.9` |
| Default drift | `stationary` |
| Driftable parameters | `g` |
| Default drift field | `g` |
| Privileged (Oracle) | `task_id`, `boundary`, drift values ([all](#privileged-fields)) |
| Upstream | [Gymnasium Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/) |

gymnax Pendulum with continuous torque and drifting gravity.

![A frame of Pendulum](../figures/environments/gymnax-pendulum-v1.svg)

*A drawing of the observation after 12 random ticks (positions and angles; not a simulator render).*

The observation, action and reward definitions are upstream's: see [Gymnasium Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/). This page lists only what changes in Perpetua.

## What changes in Perpetua

### Rewards

float32 `(1,)`: Gymnasium's pendulum cost, negated.

### Respawns

None: Pendulum has no failure states.

### Drift

`g` (gravity).

## Arguments

| Argument | Default |
|---|---|
| `drift` | `None` |

## Usage

```python
import jax
import jax.numpy as jnp
import perpetua as pp

env = pp.make("gymnax/Pendulum-v1")
state, obs, info = env.init(jax.random.key(0))
for tick in range(4):
    action = jnp.zeros(env.action_space.shape, dtype=jnp.float32)
    state, obs, reward, info = env.step(state, action, jax.random.key(1000 + tick))
print(obs.shape, reward, info["t"].to_int())

# Make it non-stationary: `g` alternates between two values every 1,000 ticks.
drifting = pp.make(
    "gymnax/Pendulum-v1",
    drift={"g": pp.drift.Cyclic([(10,), (5,)], period=1000)},
)
state, obs, info = drifting.init(jax.random.key(0))
```

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
| `g_parameters`, `g_phase`, `g_changed` | the current value, phase and switch flag of `g` |

## Version History

- `v1`: initial release.

## API

```{eval-rst}
.. autoclass:: perpetua.adapters.gymnax.GymnaxPendulum
   :no-index:
```

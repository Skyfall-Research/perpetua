# Door Grid

This environment is part of the [Native continual worlds](families/native.md) family; its page lists the install command and the family's other environments.

| | |
|---|---|
| Creation | `pp.make("DoorGrid-v0")` |
| Install | `uv sync --locked` |
| Actions | Discrete |
| Agents | `1` |
| Action Shape | `(1,)` |
| Action Values | `Discrete(5)` per agent |
| Observation Shape | `(1, 4)` |
| Observation Values | `[0, 1]` |
| Execution | `jax` |
| Checkpointable | yes |
| Backend | `native JAX` |
| Default drift | `stationary` |
| Driftable parameters | `doors` |
| Default drift field | `doors` |
| Privileged (Oracle) | `task_id`, `boundary`, drift values ([all](#privileged-fields)) |

Reach a fixed goal while the up/down/left/right doors open and close. The action space keeps its shape but its meaning evolves: a closed door masks and blocks the matching move.

![A frame of Door Grid](../figures/environments/doorgrid-v0.svg)

*One frame after a few random ticks: the agent in blue and the goal in orange. Arrows show the four moves: green is available, red is masked by a closed door or a wall.*

## Definition

### Observation Space

float32 `(agents, 4)`: normalized row, column, goal row, goal column.

### Action Space

Integer `(agents,)`: stay/up/down/left/right = 0/1/2/3/4. Masked or invalid actions become stay.

### Rewards

float32 `(agents,)`: +1 on reaching the goal.

### Initial State

Agents spawn outside the goal; all doors start open.

### Respawns

Collecting respawns the agent elsewhere on the same tick.

### Drift

`doors`, a `(4,)` vector (1 open, 0 closed). `info['space_change']` is true exactly on ticks where the effective action availability changes.

## Arguments

| Argument | Default |
|---|---|
| `size` | `5` |
| `num_agents` | `1` |
| `drift` | `None` |

## Usage

```python
import jax
import perpetua as pp

env = pp.make("DoorGrid-v0")
state, obs, info = env.init(jax.random.key(0))
for tick in range(4):
    action = pp.policies.random(obs, info, jax.random.key(100 + tick))
    state, obs, reward, info = env.step(state, action, jax.random.key(1000 + tick))
print(obs.shape, reward, info["t"].to_int())

# Make it non-stationary: `doors` alternates between two values every 1,000 ticks.
drifting = pp.make(
    "DoorGrid-v0",
    drift={"doors": pp.drift.Cyclic([(1, 1, 1, 1), (1, 0, 1, 0)], period=1000)},
)
state, obs, info = drifting.init(jax.random.key(0))
```

## See also

The [Evolving spaces: masks, events and the space manager](../evolving-spaces.md) guide.

## Privileged fields

Hidden from the agent; expose any of them with `pp.Oracle(env, expose=["task_id", ...])`, which records the choice in `metadata["oracle_exposure"]`.

| Field | Meaning |
|---|---|
| `phase` | the drift phase: an integer that changes at every declared switch (for `Cyclic`, the index of the current value) |
| `drift_parameters` | the current value of the default drifting parameter |
| `drift_changed` | deprecated alias of `boundary` |
| `task_id` | the active task: the task index under a task switch, the regime under MORPHEUS, otherwise the phase of the default drifting parameter |
| `boundary` | whether a declared switch happened on this tick (the name task-sequence protocols use, for example SAC's `task_boundary_reset`) |
| `doors_parameters`, `doors_phase`, `doors_changed` | the current value, phase and switch flag of `doors` |

## Version History

- `v0`: initial release.

## API

```{eval-rst}
.. autoclass:: perpetua.envs.door_grid.DoorGrid
   :no-index:
```

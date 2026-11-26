# AgarCL Grand Arena

This environment is part of the [Continual benchmarks](families/benchmarks.md) family; its page lists the install command and the family's other environments.

| | |
|---|---|
| Creation | `pp.make("agarcl/grand-arena-v0")` |
| Install | `uv run python -m perpetua.adapters.agarcl_setup` |
| Actions | Continuous |
| Agents | `1` |
| Action Shape | `(1, 3)` |
| Action Values | `[-1, 2]` |
| Observation Shape | `(1, 128, 128, 3)` |
| Observation Values | `[0, 255]` |
| Execution | `host` |
| Checkpointable | no |
| Backend | `agarcl-d256e7b6a61b` |
| Default drift | `stationary` |
| Driftable parameters | `reward_multiplier` |
| Default drift field | `reward_multiplier` |
| Privileged (Oracle) | `task_id`, `boundary`, drift values ([all](#privileged-fields)) |
| Upstream | [AgarCL](https://github.com/AgarCL/AgarCL) |

AgarCL's full game, the Grand Arena, with the world from the paper's Appendix G.1: 500 pellets, 10 viruses, 8 heuristic bots and 128x128 screens. Every setting is a keyword override.

## Definition

### Observation Space

float32 `(agents, 128, 128, 3)`: upstream's RGB screen, values 0 to 255.

### Action Space

float32 `(1, 3)`: x and y movement in [-1, 1], then a value the adapter rounds and clips to 0 (no-op), 1 (split) or 2 (feed). The Action Values row shows the overall range `[-1, 2]`.

### Rewards

float32 `(agents,)`: the change in the agent's mass over the tick, AgarCL's default reward.

### Respawns

Death respawns the agent.

### Drift

`reward_multiplier`.

## Arguments

| Argument | Default |
|---|---|
| `screen_len` | `128` |
| `agent_view` | `False` |
| `grid_size` | `128` |
| `num_bots` | `8` |
| `mode` | `0` |
| `respawn_after` | `None` |
| `arena_size` | `128` |
| `num_pellets` | `500` |
| `num_viruses` | `10` |
| `flatten` | `False` |
| `replay_capacity` | `None` |
| `drift` | `None` |
| `compiled` | `False` |

## Usage

```python
import jax
import jax.numpy as jnp
import perpetua as pp

env = pp.make("agarcl/grand-arena-v0")
state, obs, info = env.init(jax.random.key(0))
for tick in range(4):
    action = jnp.zeros(env.action_space.shape, dtype=jnp.float32)
    state, obs, reward, info = env.step(state, action, jax.random.key(1000 + tick))
print(obs.shape, reward, info["t"].to_int())

# Make it non-stationary: `reward_multiplier` alternates between two values every 1,000 ticks.
drifting = pp.make(
    "agarcl/grand-arena-v0",
    drift={"reward_multiplier": pp.drift.Cyclic([(1,), (0.5,)], period=1000)},
)
state, obs, info = drifting.init(jax.random.key(0))
drifting.close()
env.close()  # host simulators hold an external process or library
```

## Limits

Upstream's RGB render rather than the paper's four-channel tensor; the benchmark repository's committed config differs (arena 300, 600 pellets, 4 bots).

## See also

The [AgarCL setup, observations and replay](../agarcl.md) guide.

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

- `v0`: initial release.

## API

```{eval-rst}
.. autoclass:: perpetua.adapters.agarcl.AgarCLLifetime
   :no-index:
```
